"""Tests for the QLNN JAX/Optax trainer.

These tests do NOT depend on the full QLNNForecaster. Instead, we define a
tiny Equinox model with the same per-sample signature (x:(T,F), t:(T,)) -> scalar.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from sklearn.preprocessing import MinMaxScaler

from qlnn_.training import (
    HistoryRow,
    QLNNTrainerConfig,
    QLNNTrainResult,
    history_to_dicts,
    train_one_qlnn,
)
from qlnn_.training.trainer import _build_optimizer


# --------------------------------------------------------------------------------------
# Tiny synthetic Equinox model. Per-sample signature (x:(T,F), t:(T,)) -> scalar.
# --------------------------------------------------------------------------------------
class TinyForecastModel(eqx.Module):
    """5-parameter Equinox model: residual around persistence.

    y = x[-1, 0] + tanh(w @ x[-1] + b) * 0.1
    """

    w: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, F: int, *, key: jax.Array) -> None:
        kw, _ = jax.random.split(key, 2)
        self.w = 0.1 * jax.random.normal(kw, (F,))
        self.b = jnp.zeros(())

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        del t
        return x[-1, 0] + jnp.tanh(self.w @ x[-1] + self.b) * 0.1


class LinearLastStepModel(eqx.Module):
    """y = w @ x[-1] + b — a model that CAN learn y = 0.5 * x[-1, 1].

    Used in convergence tests.
    """

    w: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, F: int, *, key: jax.Array) -> None:
        self.w = 0.05 * jax.random.normal(key, (F,))
        self.b = jnp.zeros(())

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        del t
        return self.w @ x[-1] + self.b


class FrozenModel(eqx.Module):
    """Model whose parameters are decoupled from the prediction — gradient is zero.

    Prediction is constant 0.5 (mid-OD) regardless of inputs or params; the
    optimizer cannot reduce val MSE for any non-trivial target.
    """

    w: jnp.ndarray

    def __init__(self, F: int, *, key: jax.Array) -> None:
        self.w = jax.random.normal(key, (F,))

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        del t
        # Use zero-multiplier so grads w.r.t. w are zero — model never moves.
        return jnp.array(0.5) + 0.0 * jnp.sum(self.w * x[-1])


# --------------------------------------------------------------------------------------
# Helpers to fabricate small datasets in normalized [0,1] OD space.
# --------------------------------------------------------------------------------------
def _make_od_scaler() -> MinMaxScaler:
    s = MinMaxScaler()
    # Match the project's locked OD range [0.0, 3.8].
    s.fit(np.array([[0.0], [3.8]], dtype=np.float64))
    return s


def _make_dataset(N: int, T: int, F: int, *, seed: int = 0, target: str = "noisy_persistence"):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9, size=(N, T, F)).astype(np.float32)
    t = np.broadcast_to(np.arange(T, dtype=np.float32), (N, T)).copy()
    if target == "noisy_persistence":
        # y near x[-1, 0] but offset
        y = (x[:, -1, 0] + 0.05 * rng.standard_normal(N)).astype(np.float32)
    elif target == "linear_feature1":
        y = (0.5 * x[:, -1, 1]).astype(np.float32)
    else:
        raise ValueError(target)
    # Clip to [0,1] just in case
    y = np.clip(y, 0.0, 1.0)
    return x, t, y


def _split(x, t, y, n_train, n_val):
    s_tr = slice(0, n_train)
    s_va = slice(n_train, n_train + n_val)
    s_te = slice(n_train + n_val, None)
    return (
        x[s_tr], t[s_tr], y[s_tr],
        x[s_va], t[s_va], y[s_va],
        x[s_te], t[s_te], y[s_te],
    )


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
def test_train_runs_and_returns_metrics():
    N, T, F = 24, 6, 3
    x, t, y = _make_dataset(N, T, F, seed=0)
    splits = _split(x, t, y, n_train=12, n_val=6)

    key = jax.random.PRNGKey(0)
    model = TinyForecastModel(F=F, key=key)

    cfg = QLNNTrainerConfig(epochs=2, batch_size=4, eval_every=1, patience=5)
    res = train_one_qlnn(
        model=model,
        x_train=splits[0], t_train=splits[1], y_train=splits[2],
        x_val=splits[3], t_val=splits[4], y_val=splits[5],
        x_test=splits[6], t_test=splits[7], y_test=splits[8],
        od_scaler=_make_od_scaler(),
        cfg=cfg,
        seed=0,
    )
    assert isinstance(res, QLNNTrainResult)
    assert isinstance(res.model, eqx.Module)
    assert len(res.history) >= 1
    for h in res.history:
        assert isinstance(h, HistoryRow)
    for m in (res.val_metrics, res.test_metrics):
        assert np.isfinite(m.mse_norm)
        assert np.isfinite(m.mae_raw)
        assert np.isfinite(m.rmse_raw)
        assert np.isfinite(m.r2_raw)


def test_train_improves_val_mse():
    N, T, F = 80, 5, 3
    x, t, y = _make_dataset(N, T, F, seed=1, target="linear_feature1")
    splits = _split(x, t, y, n_train=50, n_val=15)

    key = jax.random.PRNGKey(1)
    model = LinearLastStepModel(F=F, key=key)

    cfg = QLNNTrainerConfig(
        epochs=40, batch_size=10, eval_every=1, patience=100, lr=5e-2, grad_clip_norm=0.0
    )
    res = train_one_qlnn(
        model=model,
        x_train=splits[0], t_train=splits[1], y_train=splits[2],
        x_val=splits[3], t_val=splits[4], y_val=splits[5],
        x_test=splits[6], t_test=splits[7], y_test=splits[8],
        od_scaler=_make_od_scaler(),
        cfg=cfg,
        seed=1,
    )
    first_val = res.history[0].val_mse_norm
    last_val = res.history[-1].val_mse_norm
    assert last_val < first_val, f"val MSE did not improve: {first_val} -> {last_val}"


def test_best_checkpoint_is_returned():
    """After training, returned val_metrics should match the BEST history row, not the last."""
    N, T, F = 60, 4, 3
    x, t, y = _make_dataset(N, T, F, seed=2, target="linear_feature1")
    splits = _split(x, t, y, n_train=40, n_val=10)

    key = jax.random.PRNGKey(2)
    model = LinearLastStepModel(F=F, key=key)

    # Huge LR + many epochs -> the model will likely overshoot and degrade at some point.
    cfg = QLNNTrainerConfig(
        epochs=60, batch_size=8, eval_every=1, patience=1000, lr=2.0, grad_clip_norm=0.0
    )
    res = train_one_qlnn(
        model=model,
        x_train=splits[0], t_train=splits[1], y_train=splits[2],
        x_val=splits[3], t_val=splits[4], y_val=splits[5],
        x_test=splits[6], t_test=splits[7], y_test=splits[8],
        od_scaler=_make_od_scaler(),
        cfg=cfg,
        seed=2,
    )
    best_in_history = min(h.val_mse_norm for h in res.history)
    assert res.val_metrics.mse_norm == pytest.approx(best_in_history, rel=1e-5, abs=1e-7)
    # The reported best_epoch should match the argmin
    argmin_epoch = min(res.history, key=lambda h: h.val_mse_norm).epoch
    assert res.best_epoch == argmin_epoch


def test_multi_seed_determinism():
    """Same seed, two independent runs from scratch -> identical results.

    JAX is deterministic given the same PRNG key and the same op sequence, so
    fresh runs with seed=0 should produce bit-identical (or 1e-6 close)
    best-checkpoint metrics and parameter trees. Tolerated to 1e-6 to absorb
    any tiny JIT-compile-order driven variation.
    """
    N, T, F = 24, 5, 3

    def _run():
        x, t, y = _make_dataset(N, T, F, seed=10)
        splits = _split(x, t, y, n_train=14, n_val=5)
        model = TinyForecastModel(F=F, key=jax.random.PRNGKey(0))
        cfg = QLNNTrainerConfig(epochs=4, batch_size=4, eval_every=1, patience=100)
        return train_one_qlnn(
            model=model,
            x_train=splits[0], t_train=splits[1], y_train=splits[2],
            x_val=splits[3], t_val=splits[4], y_val=splits[5],
            x_test=splits[6], t_test=splits[7], y_test=splits[8],
            od_scaler=_make_od_scaler(),
            cfg=cfg,
            seed=0,
        )

    res_a = _run()
    res_b = _run()

    assert res_a.best_epoch == res_b.best_epoch
    assert res_a.val_metrics.mse_norm == pytest.approx(
        res_b.val_metrics.mse_norm, rel=1e-6, abs=1e-7
    )
    assert res_a.val_metrics.mae_raw == pytest.approx(
        res_b.val_metrics.mae_raw, rel=1e-6, abs=1e-7
    )
    assert res_a.test_metrics.mse_norm == pytest.approx(
        res_b.test_metrics.mse_norm, rel=1e-6, abs=1e-7
    )
    assert res_a.test_metrics.r2_raw == pytest.approx(
        res_b.test_metrics.r2_raw, rel=1e-6, abs=1e-7
    )

    # Param leaves must agree to numerical noise.
    leaves_a = jax.tree_util.tree_leaves(eqx.filter(res_a.model, eqx.is_array))
    leaves_b = jax.tree_util.tree_leaves(eqx.filter(res_b.model, eqx.is_array))
    assert len(leaves_a) == len(leaves_b)
    for la, lb in zip(leaves_a, leaves_b):
        assert la.shape == lb.shape
        assert jnp.allclose(la, lb, rtol=1e-6, atol=1e-7), "param leaves diverge"


def test_history_to_dicts_round_trips():
    N, T, F = 16, 4, 2
    x, t, y = _make_dataset(N, T, F, seed=3)
    splits = _split(x, t, y, n_train=8, n_val=4)

    model = TinyForecastModel(F=F, key=jax.random.PRNGKey(3))
    cfg = QLNNTrainerConfig(epochs=2, batch_size=4, eval_every=1, patience=5)
    res = train_one_qlnn(
        model=model,
        x_train=splits[0], t_train=splits[1], y_train=splits[2],
        x_val=splits[3], t_val=splits[4], y_val=splits[5],
        x_test=splits[6], t_test=splits[7], y_test=splits[8],
        od_scaler=_make_od_scaler(),
        cfg=cfg,
        seed=3,
    )
    dicts = history_to_dicts(res.history)
    assert len(dicts) == len(res.history)
    expected_keys = {
        "epoch",
        "train_mse_norm",
        "val_mse_norm",
        "val_mae_raw",
        "val_rmse_raw",
        "val_r2_raw",
        "best_val_mse_norm",
        "best_epoch",
    }
    for d in dicts:
        assert set(d.keys()) == expected_keys
        assert isinstance(d["epoch"], int)
        assert isinstance(d["best_epoch"], int)
        for k in expected_keys - {"epoch", "best_epoch"}:
            assert isinstance(d[k], float), f"{k} should be float, got {type(d[k])}"


def test_config_validates():
    with pytest.raises(ValueError):
        QLNNTrainerConfig(epochs=0)
    with pytest.raises(ValueError):
        QLNNTrainerConfig(batch_size=0)
    with pytest.raises(ValueError):
        QLNNTrainerConfig(eval_every=0)
    with pytest.raises(ValueError):
        QLNNTrainerConfig(patience=0)
    with pytest.raises(ValueError):
        QLNNTrainerConfig(lr=0)
    with pytest.raises(ValueError):
        QLNNTrainerConfig(grad_clip_norm=-0.5)


def test_no_weight_decay_uses_adam_with_weight_decay_uses_adamw():
    """_build_optimizer chooses adam vs adamw based on cfg.weight_decay."""
    cfg_adam = QLNNTrainerConfig(weight_decay=0.0, grad_clip_norm=0.0)
    cfg_adamw = QLNNTrainerConfig(weight_decay=1e-4, grad_clip_norm=0.0)
    cfg_clip = QLNNTrainerConfig(weight_decay=0.0, grad_clip_norm=1.0)

    opt_adam = _build_optimizer(cfg_adam)
    opt_adamw = _build_optimizer(cfg_adamw)
    opt_clip = _build_optimizer(cfg_clip)

    # Sanity: all are optax GradientTransformations
    for o in (opt_adam, opt_adamw, opt_clip):
        assert isinstance(o, optax.GradientTransformation)

    # Test functionally — adamw applies weight-decay drift even on zero grads;
    # adam does not. Build a dummy 1-param "model" and run one update.
    params = {"w": jnp.array([1.0, 1.0])}
    grads = {"w": jnp.zeros_like(params["w"])}

    state_adam = opt_adam.init(params)
    state_adamw = opt_adamw.init(params)

    upd_adam, _ = opt_adam.update(grads, state_adam, params)
    upd_adamw, _ = opt_adamw.update(grads, state_adamw, params)

    # adam on zero grads -> ~zero update
    assert float(jnp.max(jnp.abs(upd_adam["w"]))) == pytest.approx(0.0, abs=1e-8)
    # adamw on zero grads -> nonzero update from the weight decay term
    assert float(jnp.max(jnp.abs(upd_adamw["w"]))) > 1e-9


def test_grad_clipping_runs():
    """Train with tight grad-clipping; convergence should still happen."""
    N, T, F = 80, 5, 3
    x, t, y = _make_dataset(N, T, F, seed=4, target="linear_feature1")
    splits = _split(x, t, y, n_train=50, n_val=15)

    key = jax.random.PRNGKey(4)
    model = LinearLastStepModel(F=F, key=key)

    cfg = QLNNTrainerConfig(
        epochs=60, batch_size=10, eval_every=1, patience=1000, lr=5e-2, grad_clip_norm=0.1
    )
    res = train_one_qlnn(
        model=model,
        x_train=splits[0], t_train=splits[1], y_train=splits[2],
        x_val=splits[3], t_val=splits[4], y_val=splits[5],
        x_test=splits[6], t_test=splits[7], y_test=splits[8],
        od_scaler=_make_od_scaler(),
        cfg=cfg,
        seed=4,
    )
    first_val = res.history[0].val_mse_norm
    last_val = res.history[-1].val_mse_norm
    assert last_val < first_val, f"val MSE did not improve with clipping: {first_val} -> {last_val}"


def test_eval_every_and_patience_early_stops():
    """A model that cannot improve should trigger early stopping well before epochs end."""
    N, T, F = 60, 4, 3
    x, t, y = _make_dataset(N, T, F, seed=5, target="linear_feature1")
    splits = _split(x, t, y, n_train=40, n_val=10)

    key = jax.random.PRNGKey(5)
    model = FrozenModel(F=F, key=key)

    # patience=1 -> at most one non-improving evaluation tolerated.
    cfg = QLNNTrainerConfig(
        epochs=100, batch_size=8, eval_every=1, patience=1, lr=1e-3, grad_clip_norm=0.0
    )
    res = train_one_qlnn(
        model=model,
        x_train=splits[0], t_train=splits[1], y_train=splits[2],
        x_val=splits[3], t_val=splits[4], y_val=splits[5],
        x_test=splits[6], t_test=splits[7], y_test=splits[8],
        od_scaler=_make_od_scaler(),
        cfg=cfg,
        seed=5,
    )
    # First eval (epoch 1) sets baseline -> improved. Second eval (epoch 2) doesn't
    # improve -> bad_evals=1, equals patience, early-stop. So we should see ~2 history rows.
    assert res.history[-1].epoch < cfg.epochs
    assert len(res.history) <= 3  # generous upper bound
