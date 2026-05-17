import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from quantum_liquid_neuralode.models import LiquidODForecaster
from quantum_liquid_neuralode.training import (
    PhysicsLossConfig,
    TrainerConfig,
    train_one,
)


def _od_scaler() -> MinMaxScaler:
    sc = MinMaxScaler()
    sc.fit(np.array([[0.0], [1.0]]))
    return sc


def _toy_windows(n=24, T=4, F=3):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, T, F)).astype(np.float32)
    # Use a synthetic OD signal that's somewhat predictable from the last input.
    od_last = x[:, -1, 0].astype(np.float32)
    od_prev = x[:, -2, 0].astype(np.float32)
    y = (od_last + 0.1 * rng.standard_normal(n).astype(np.float32))
    # Bound to [0,1] roughly so the OD scaler stays sane.
    y = ((y - y.min()) / (np.ptp(y) + 1e-9)).astype(np.float32)
    od_last = ((od_last - od_last.min()) / (np.ptp(od_last) + 1e-9)).astype(np.float32)
    od_prev = ((od_prev - od_prev.min()) / (np.ptp(od_prev) + 1e-9)).astype(np.float32)
    t = np.tile(np.linspace(0.0, 0.5, T).astype(np.float32), (n, 1))
    return x, t, y, od_last, od_prev


def test_train_one_runs_and_improves_or_at_least_finishes():
    x, t, y, od_last, _ = _toy_windows(n=32, T=6, F=3)

    # Re-bake x so x[:, -1, 0] matches normalized od_last (the forecaster needs OD(t) as a feature).
    x = x.copy()
    x[:, -1, 0] = od_last

    model = LiquidODForecaster(
        input_size=3, hidden_size=8, horizon_hours=1.0, forecast_steps=1,
        od_index=0, delta_scale=0.2, ode_method="euler",
    )

    cfg = TrainerConfig(epochs=4, batch_size=8, lr=1e-2, eval_every=1, patience=10)
    result = train_one(
        model=model,
        x_train=x[:20], t_train=t[:20], y_train=y[:20], od_last_train=od_last[:20],
        x_val=x[20:26], t_val=t[20:26], y_val=y[20:26],
        x_test=x[26:], t_test=t[26:], y_test=y[26:],
        od_scaler=_od_scaler(), device=torch.device("cpu"),
        cfg=cfg, horizon_hours=1.0, od_index=0, seed=0,
    )

    assert result.best_epoch >= 1
    assert len(result.history) >= 1
    assert np.isfinite(result.val_metrics.mse_norm)
    assert np.isfinite(result.test_metrics.mae_raw)


def _build_and_train(seed: int, *, epochs: int = 4, lr: float = 1e-2):
    """Helper: build a model and train it from the given seed.

    Seeds are set both before model construction (so init weights are
    deterministic) and passed into train_one (so dataloader shuffles and
    in-trainer ops are deterministic).
    """
    x, t, y, od_last, _ = _toy_windows(n=32, T=6, F=3)
    x = x.copy()
    x[:, -1, 0] = od_last

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LiquidODForecaster(
        input_size=3, hidden_size=8, horizon_hours=1.0, forecast_steps=1,
        od_index=0, delta_scale=0.2, ode_method="euler",
    )

    cfg = TrainerConfig(epochs=epochs, batch_size=8, lr=lr, eval_every=1, patience=100)
    return train_one(
        model=model,
        x_train=x[:20], t_train=t[:20], y_train=y[:20], od_last_train=od_last[:20],
        x_val=x[20:26], t_val=t[20:26], y_val=y[20:26],
        x_test=x[26:], t_test=t[26:], y_test=y[26:],
        od_scaler=_od_scaler(), device=torch.device("cpu"),
        cfg=cfg, horizon_hours=1.0, od_index=0, seed=seed,
    )


def test_multi_seed_determinism():
    """Same seed, two independent runs from scratch -> identical results.

    Confirms the trainer's seeding (torch.manual_seed + np.random.seed +
    train_one(seed=...)) is sufficient for reproducible runs on CPU.
    Tolerated to 1e-6 to absorb any tiny floating-point variation across
    library versions.
    """
    res_a = _build_and_train(seed=0)
    res_b = _build_and_train(seed=0)

    # best_epoch is an integer, must match exactly.
    assert res_a.best_epoch == res_b.best_epoch

    # Final selected-checkpoint metrics agree to numerical noise.
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

    # Best-state tensors must agree.
    keys_a = set(res_a.model_state.keys())
    keys_b = set(res_b.model_state.keys())
    assert keys_a == keys_b
    for k in keys_a:
        ta = res_a.model_state[k]
        tb = res_b.model_state[k]
        assert ta.shape == tb.shape
        assert torch.allclose(ta, tb, rtol=1e-6, atol=1e-7), f"tensor {k!r} diverges"


def test_best_checkpoint_actually_returns_the_best():
    """The returned model must reflect the best-val checkpoint, not the final.

    Use a deliberately large LR over many epochs so the optimizer overshoots
    and val MSE degrades after some best point. Verify the final val_metrics
    matches the best history row (not the last).
    """
    res = _build_and_train(seed=0, epochs=20, lr=0.5)

    best_in_history = min(h.val_mse_norm for h in res.history)
    assert res.val_metrics.mse_norm == pytest.approx(
        best_in_history, rel=1e-5, abs=1e-7
    )

    argmin_epoch = min(res.history, key=lambda h: h.val_mse_norm).epoch
    assert res.best_epoch == argmin_epoch


def test_train_one_with_physics_loss_does_not_crash():
    x, t, y, od_last, _ = _toy_windows(n=24, T=4, F=3)
    x = x.copy()
    x[:, -1, 0] = od_last

    model = LiquidODForecaster(
        input_size=3, hidden_size=4, horizon_hours=1.0, forecast_steps=1,
        od_index=0, delta_scale=0.1, ode_method="euler",
    )
    cfg = TrainerConfig(
        epochs=2, batch_size=8, lr=1e-3, eval_every=1, patience=10,
        physics=PhysicsLossConfig(lambda_logistic=0.1),
    )
    result = train_one(
        model=model,
        x_train=x[:16], t_train=t[:16], y_train=y[:16], od_last_train=od_last[:16],
        x_val=x[16:20], t_val=t[16:20], y_val=y[16:20],
        x_test=x[20:], t_test=t[20:], y_test=y[20:],
        od_scaler=_od_scaler(), device=torch.device("cpu"),
        cfg=cfg, horizon_hours=1.0, od_index=0, seed=0,
    )
    assert np.isfinite(result.val_metrics.mse_norm)


def test_physics_loss_config_no_longer_accepts_lambda_smooth():
    """Reviewer R1 (BLOCKER B2) showed the old `lambda_smooth` branch reduced
    algebraically to a re-weighted data MSE — not a smoothness regularizer.
    The field has been removed from PhysicsLossConfig; passing it must raise
    TypeError so legacy callers fail loudly rather than silently re-introduce
    the mislabelled ablation.
    """
    with pytest.raises(TypeError):
        PhysicsLossConfig(lambda_smooth=0.05)  # type: ignore[call-arg]
