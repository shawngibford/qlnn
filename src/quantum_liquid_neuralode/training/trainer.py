from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation.metrics import ForecastMetrics, compute_metrics
from .losses import logistic_growth_residual_loss, smoothness_loss


@dataclass(frozen=True)
class PhysicsLossConfig:
    """Weights for physics-informed regularizers. All default 0 (off)."""
    lambda_logistic: float = 0.0
    lambda_smooth: float = 0.0
    # Logistic-growth params (only used if lambda_logistic > 0).
    # These are in NORMALIZED OD space because the model emits normalized OD.
    mu_norm: float = 0.4       # growth rate (1/h)
    K_norm: float = 1.0        # carrying capacity in normalized OD


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    eval_every: int = 10
    patience: int = 10
    grad_clip_norm: float = 1.0
    physics: PhysicsLossConfig = field(default_factory=PhysicsLossConfig)


@dataclass(frozen=True)
class HistoryRow:
    epoch: int
    train_mse_norm: float
    train_loss_total: float
    val_mse_norm: float
    val_mae_raw: float
    val_rmse_raw: float
    val_r2_raw: float
    best_val_mse_norm: float
    best_epoch: int


@dataclass
class TrainResult:
    model_state: dict[str, Tensor]
    best_epoch: int
    history: list[HistoryRow]
    val_metrics: ForecastMetrics
    test_metrics: ForecastMetrics


def _to_loader(x: np.ndarray, t: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(t.astype(np.float32)),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def _predict_norm(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    yt: list[np.ndarray] = []
    yp: list[np.ndarray] = []
    for xb, tb, yb in loader:
        xb = xb.to(device)
        tb = tb.to(device)
        yp_b = model(xb, tb).detach().cpu().numpy()
        yp.append(yp_b)
        yt.append(yb.numpy())
    return np.concatenate(yt, axis=0), np.concatenate(yp, axis=0)


def _eval_metrics_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    od_scaler: MinMaxScaler,
) -> ForecastMetrics:
    y_true, y_pred = _predict_norm(model, loader, device)
    return compute_metrics(y_true_norm=y_true, y_pred_norm=y_pred, od_scaler=od_scaler)


def _physics_loss_terms(
    *,
    yp: Tensor,
    yb: Tensor,
    od_last: Tensor,
    horizon_hours: float,
    cfg: PhysicsLossConfig,
) -> Tensor:
    """Apply physics regularizers to the model's 1-step-ahead prediction.

    The regularizers are computed on the (od_last, yp) pair viewed as a 2-point
    "trajectory" over `horizon_hours`. This is a soft prior, not a hard constraint.

    - Logistic-growth residual penalizes deviations from the autonomous logistic
      ODE dOD/dt = mu * OD * (1 - OD/K).
    - Smoothness penalty regularizes against jumps that are large relative to
      both true and predicted deltas (here approximated as |delta_pred|^2; the
      smoothness loss API expects >=3 points so we use a simple delta penalty
      proxy when we only have a 1-step horizon).

    Returns scalar tensor.
    """
    total = yp.new_zeros(())

    if cfg.lambda_logistic > 0.0:
        # Two-point trajectory per sample: (od_last, yp) at times (0, h).
        traj = torch.stack([od_last, yp], dim=-1)  # (batch, 2)
        t_pts = torch.tensor(
            [0.0, float(horizon_hours)],
            device=yp.device,
            dtype=traj.dtype,
        )
        # logistic_growth_residual_loss expects (batch, T) or (T,)
        l_log = logistic_growth_residual_loss(
            traj,
            t_pts,
            mu=float(cfg.mu_norm),
            K=float(cfg.K_norm),
            reduction="mean",
        )
        total = total + cfg.lambda_logistic * l_log

    if cfg.lambda_smooth > 0.0:
        # With only a 1-step horizon we can't compute second differences;
        # use a delta-magnitude proxy that penalizes predicted jumps not
        # supported by the true delta.
        delta_pred = yp - od_last
        delta_true = yb - od_last
        excess = delta_pred - delta_true
        total = total + cfg.lambda_smooth * excess.pow(2).mean()

    return total


def train_one(
    *,
    model: nn.Module,
    x_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    od_last_train: np.ndarray,
    x_val: np.ndarray,
    t_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    t_test: np.ndarray,
    y_test: np.ndarray,
    od_scaler: MinMaxScaler,
    device: torch.device,
    cfg: TrainerConfig,
    horizon_hours: float,
    od_index: int,
    seed: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> TrainResult:
    """Train one model from scratch with a given seed.

    The seed is set immediately before model parameter init by the caller
    (or before instantiating `model`); we still set it here so the dataloader
    shuffle order is deterministic.
    """
    if cfg.eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if cfg.patience <= 0:
        raise ValueError("patience must be positive")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = _to_loader(x_train, t_train, y_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = _to_loader(x_val, t_val, y_val, batch_size=cfg.batch_size, shuffle=False)
    test_loader = _to_loader(x_test, t_test, y_test, batch_size=cfg.batch_size, shuffle=False)

    # od_last_train as a tensor we index alongside the dataloader. Easier: include
    # it inside x at runtime (od_last == x_last[:, od_index]); this is how the
    # forecaster already computes residuals, so we just read x_last[:, od_index].

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[HistoryRow] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    bad_evals = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        mse_sum = 0.0
        loss_sum = 0.0
        n_elems = 0

        for xb, tb, yb in train_loader:
            xb = xb.to(device)
            tb = tb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            yp = model(xb, tb)

            se = (yp - yb).pow(2)
            mse = se.mean()
            loss = mse

            if cfg.physics.lambda_logistic > 0.0 or cfg.physics.lambda_smooth > 0.0:
                od_last_b = xb[:, -1, od_index]
                loss = loss + _physics_loss_terms(
                    yp=yp,
                    yb=yb,
                    od_last=od_last_b,
                    horizon_hours=horizon_hours,
                    cfg=cfg.physics,
                )

            loss.backward()
            if cfg.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            opt.step()

            mse_sum += float(se.sum().detach().cpu().item())
            loss_sum += float(loss.detach().cpu().item()) * int(yb.numel())
            n_elems += int(yb.numel())

        if epoch == 1 or epoch % cfg.eval_every == 0:
            train_mse = float(mse_sum / max(n_elems, 1))
            train_loss = float(loss_sum / max(n_elems, 1))

            val_m = _eval_metrics_loader(model, val_loader, device, od_scaler)

            improved = val_m.mse_norm < best_val
            if improved:
                best_val = val_m.mse_norm
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_evals = 0
            else:
                bad_evals += 1

            history.append(
                HistoryRow(
                    epoch=epoch,
                    train_mse_norm=train_mse,
                    train_loss_total=train_loss,
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

    if best_state is None:
        # Take whatever we have if training never evaluated (epochs < eval_every and no epoch==1 path failed).
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = 0

    model.load_state_dict(best_state)

    val_metrics = _eval_metrics_loader(model, val_loader, device, od_scaler)
    test_metrics = _eval_metrics_loader(model, test_loader, device, od_scaler)

    return TrainResult(
        model_state=best_state,
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
            "train_loss_total": float(h.train_loss_total),
            "val_mse_norm": float(h.val_mse_norm),
            "val_mae_raw": float(h.val_mae_raw),
            "val_rmse_raw": float(h.val_rmse_raw),
            "val_r2_raw": float(h.val_r2_raw),
            "best_val_mse_norm": float(h.best_val_mse_norm),
            "best_epoch": int(h.best_epoch),
        }
        for h in history
    ]
