"""JAX/Optax trainer for QLNN forecast models.

Mirrors the PyTorch classical trainer surface (`quantum_liquid_neuralode.training.trainer`)
so QLNN results land in the same `ForecastMetrics` shape and selection protocol
(best val MSE_norm checkpoint, configurable eval cadence + patience early-stop).

Model-agnostic: accepts any callable `eqx.Module` with the per-sample signature
`model(x: (T, F), t: (T,)) -> scalar`. The trainer batches with `jax.vmap`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.preprocessing import MinMaxScaler

from quantum_liquid_neuralode.evaluation.metrics import ForecastMetrics, compute_metrics


@dataclass(frozen=True)
class QLNNTrainerConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0  # adamw if > 0, else adam
    eval_every: int = 5
    patience: int = 5
    grad_clip_norm: float = 1.0  # 0 disables

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size > 0")
        if self.eval_every <= 0:
            raise ValueError("eval_every > 0")
        if self.patience <= 0:
            raise ValueError("patience > 0")
        if self.lr <= 0:
            raise ValueError("lr > 0")
        if self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm >= 0")


@dataclass(frozen=True)
class HistoryRow:
    epoch: int
    train_mse_norm: float
    val_mse_norm: float
    val_mae_raw: float
    val_rmse_raw: float
    val_r2_raw: float
    best_val_mse_norm: float
    best_epoch: int


@dataclass
class QLNNTrainResult:
    model: eqx.Module
    best_epoch: int
    history: list[HistoryRow]
    val_metrics: ForecastMetrics
    test_metrics: ForecastMetrics


def _build_optimizer(cfg: QLNNTrainerConfig) -> optax.GradientTransformation:
    """Construct the Optax optimizer chain from cfg.

    - grad_clip_norm > 0 -> prepend optax.clip_by_global_norm
    - weight_decay > 0   -> optax.adamw, else optax.adam
    """
    chain: list[optax.GradientTransformation] = []
    if cfg.grad_clip_norm > 0:
        chain.append(optax.clip_by_global_norm(cfg.grad_clip_norm))
    if cfg.weight_decay > 0:
        chain.append(optax.adamw(cfg.lr, weight_decay=cfg.weight_decay))
    else:
        chain.append(optax.adam(cfg.lr))
    return optax.chain(*chain)


def _loss_fn(model: eqx.Module, x_batch: jnp.ndarray, t_batch: jnp.ndarray, y_batch: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error in normalized OD space."""
    y_pred = jax.vmap(model)(x_batch, t_batch)  # (B,)
    return jnp.mean((y_pred - y_batch) ** 2)


def _predict_all(
    model: eqx.Module,
    x: np.ndarray,
    t: np.ndarray,
    batch_size: int,
    predict_batch: Callable[[eqx.Module, jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> np.ndarray:
    """Run model in mini-batches, returning numpy predictions."""
    preds: list[np.ndarray] = []
    n = x.shape[0]
    for i in range(0, n, batch_size):
        xb = jnp.asarray(x[i : i + batch_size])
        tb = jnp.asarray(t[i : i + batch_size])
        yb = predict_batch(model, xb, tb)
        preds.append(np.asarray(yb))
    return np.concatenate(preds, axis=0)


def train_one_qlnn(
    *,
    model: eqx.Module,
    x_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    t_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    t_test: np.ndarray,
    y_test: np.ndarray,
    od_scaler: MinMaxScaler,
    cfg: QLNNTrainerConfig,
    seed: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> QLNNTrainResult:
    """Train any Equinox model on (x, t) -> scalar via Optax Adam(W).

    See module docstring for the contract.
    """
    # --- Data as JAX arrays (kept on default device; trainer is device-agnostic).
    x_train_j = jnp.asarray(x_train)
    t_train_j = jnp.asarray(t_train)
    y_train_j = jnp.asarray(y_train)

    # --- Optimizer
    opt = _build_optimizer(cfg)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # --- JIT'd train step
    @eqx.filter_jit
    def train_step(
        model: eqx.Module,
        opt_state: optax.OptState,
        x_b: jnp.ndarray,
        t_b: jnp.ndarray,
        y_b: jnp.ndarray,
    ) -> tuple[eqx.Module, optax.OptState, jnp.ndarray]:
        loss, grads = eqx.filter_value_and_grad(_loss_fn)(model, x_b, t_b, y_b)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def predict_batch(model: eqx.Module, x_b: jnp.ndarray, t_b: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(model)(x_b, t_b)

    # --- Bookkeeping
    history: list[HistoryRow] = []
    best_val = float("inf")
    best_epoch = 0
    best_model: eqx.Module = model  # Equinox modules are PyTrees; rebinding is enough.
    bad_evals = 0

    key = jax.random.PRNGKey(int(seed))
    n_train = x_train_j.shape[0]

    for epoch in range(1, cfg.epochs + 1):
        # Shuffle indices with a fresh subkey each epoch
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_train)

        # Track training MSE across the epoch
        train_se_sum = 0.0
        train_n = 0

        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            x_b = x_train_j[idx]
            t_b = t_train_j[idx]
            y_b = y_train_j[idx]

            model, opt_state, loss = train_step(model, opt_state, x_b, t_b, y_b)

            bs = int(idx.shape[0])
            train_se_sum += float(loss) * bs
            train_n += bs

        do_eval = (epoch == 1) or (epoch % cfg.eval_every == 0) or (epoch == cfg.epochs)
        if do_eval:
            train_mse = float(train_se_sum / max(train_n, 1))

            val_preds = _predict_all(model, x_val, t_val, cfg.batch_size, predict_batch)
            val_m = compute_metrics(
                y_true_norm=np.asarray(y_val), y_pred_norm=val_preds, od_scaler=od_scaler
            )

            improved = val_m.mse_norm < best_val
            if improved:
                best_val = val_m.mse_norm
                best_epoch = epoch
                # Equinox modules are pytrees of arrays — rebinding suffices.
                best_model = model
                bad_evals = 0
            else:
                bad_evals += 1

            history.append(
                HistoryRow(
                    epoch=epoch,
                    train_mse_norm=train_mse,
                    val_mse_norm=val_m.mse_norm,
                    val_mae_raw=val_m.mae_raw,
                    val_rmse_raw=val_m.rmse_raw,
                    val_r2_raw=val_m.r2_raw,
                    best_val_mse_norm=best_val,
                    best_epoch=best_epoch,
                )
            )

            if log_fn is not None:
                log_fn(
                    f"epoch {epoch:4d} | train_mse={train_mse:.6f} | "
                    f"val_mse={val_m.mse_norm:.6f} mae={val_m.mae_raw:.4f} r2={val_m.r2_raw:.4f} | "
                    f"best={best_val:.6f}@{best_epoch}"
                )

            if bad_evals >= cfg.patience:
                if log_fn is not None:
                    log_fn(f"early stopping at epoch {epoch}")
                break

    # --- Final metrics on best checkpoint
    final_model = best_model

    val_preds = _predict_all(final_model, x_val, t_val, cfg.batch_size, predict_batch)
    val_metrics = compute_metrics(
        y_true_norm=np.asarray(y_val), y_pred_norm=val_preds, od_scaler=od_scaler
    )

    test_preds = _predict_all(final_model, x_test, t_test, cfg.batch_size, predict_batch)
    test_metrics = compute_metrics(
        y_true_norm=np.asarray(y_test), y_pred_norm=test_preds, od_scaler=od_scaler
    )

    return QLNNTrainResult(
        model=final_model,
        best_epoch=best_epoch,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def history_to_dicts(history: list[HistoryRow]) -> list[dict[str, float | int]]:
    return [
        {
            "epoch": int(h.epoch),
            "train_mse_norm": float(h.train_mse_norm),
            "val_mse_norm": float(h.val_mse_norm),
            "val_mae_raw": float(h.val_mae_raw),
            "val_rmse_raw": float(h.val_rmse_raw),
            "val_r2_raw": float(h.val_r2_raw),
            "best_val_mse_norm": float(h.best_val_mse_norm),
            "best_epoch": int(h.best_epoch),
        }
        for h in history
    ]
