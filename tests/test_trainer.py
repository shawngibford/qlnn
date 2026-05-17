import numpy as np
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
        physics=PhysicsLossConfig(lambda_logistic=0.1, lambda_smooth=0.05),
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
