import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from quantum_liquid_neuralode.evaluation import (
    compute_metrics,
    linear_extrapolation_forecast,
    persistence_forecast,
)
from quantum_liquid_neuralode.evaluation.metrics import aggregate_seed_metrics


def _od_scaler(vmin: float = 0.0, vmax: float = 3.8) -> MinMaxScaler:
    sc = MinMaxScaler()
    sc.fit(np.array([[vmin], [vmax]]))
    return sc


def test_compute_metrics_perfect_prediction():
    sc = _od_scaler()
    y = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    m = compute_metrics(y_true_norm=y, y_pred_norm=y, od_scaler=sc)
    assert m.mse_norm == pytest.approx(0.0)
    assert m.mae_raw == pytest.approx(0.0)
    assert m.rmse_raw == pytest.approx(0.0)
    assert m.r2_raw == pytest.approx(1.0)


def test_compute_metrics_scales_to_raw_units():
    sc = _od_scaler(0.0, 4.0)  # easy math: 1.0 norm -> 4.0 raw
    y_true = np.array([0.0, 0.25, 0.5], dtype=np.float32)
    y_pred = np.array([0.0, 0.25, 0.75], dtype=np.float32)  # off by 1.0 raw on last sample
    m = compute_metrics(y_true_norm=y_true, y_pred_norm=y_pred, od_scaler=sc)
    # raw errors: 0, 0, 1.0
    assert m.mae_raw == pytest.approx(1.0 / 3.0)
    assert m.rmse_raw == pytest.approx(np.sqrt(1.0 / 3.0))


def test_persistence_baseline_is_identity():
    od_last = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    pred = persistence_forecast(od_last)
    np.testing.assert_allclose(pred, od_last)


def test_linear_extrapolation_basic():
    od_last = np.array([0.5, 0.6], dtype=np.float32)
    od_prev = np.array([0.4, 0.6], dtype=np.float32)
    dt = np.array([0.5, 0.5], dtype=np.float32)
    pred = linear_extrapolation_forecast(
        od_last=od_last, od_prev=od_prev, dt_last_hours=dt, horizon_hours=1.0
    )
    # sample 0: 0.5 + (0.5-0.4) * (1.0/0.5) = 0.5 + 0.2 = 0.7
    # sample 1: 0.6 + (0.6-0.6) * 2 = 0.6
    assert pred[0] == pytest.approx(0.7)
    assert pred[1] == pytest.approx(0.6)


def test_linear_extrapolation_handles_zero_dt():
    pred = linear_extrapolation_forecast(
        od_last=np.array([0.5], dtype=np.float32),
        od_prev=np.array([0.4], dtype=np.float32),
        dt_last_hours=np.array([0.0], dtype=np.float32),
        horizon_hours=1.0,
    )
    # Falls back to persistence when dt == 0.
    assert pred[0] == pytest.approx(0.5)


def test_aggregate_seed_metrics_mean_std():
    from quantum_liquid_neuralode.evaluation.metrics import ForecastMetrics

    ms = [
        ForecastMetrics(mse_norm=0.01, mae_raw=0.1, rmse_raw=0.12, r2_raw=0.90),
        ForecastMetrics(mse_norm=0.03, mae_raw=0.2, rmse_raw=0.22, r2_raw=0.80),
    ]
    agg = aggregate_seed_metrics(ms)
    assert agg["mae_raw"]["mean"] == pytest.approx(0.15)
    assert agg["mae_raw"]["min"] == pytest.approx(0.1)
    assert agg["mae_raw"]["max"] == pytest.approx(0.2)
    assert agg["mae_raw"]["n_seeds"] == 2
