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


def test_aggregate_seed_metrics_uses_unbiased_std():
    """Regression test for R1-B3: aggregate_seed_metrics must use unbiased
    sample std (ddof=1, Bessel's correction), not population std (ddof=0).

    Scientific convention is to report mean +/- sample std for finite seed
    populations. Hand-computed: for vals=[0.9, 1.0, 1.1] the mean is 1.0, and
    sample variance = (0.01 + 0 + 0.01) / (3 - 1) = 0.01 -> sample std = 0.1.
    Population std (ddof=0) would be sqrt(0.02/3) ~= 0.08165, which is wrong.
    """
    from quantum_liquid_neuralode.evaluation.metrics import ForecastMetrics

    ms = [
        ForecastMetrics(mse_norm=0.9, mae_raw=0.9, rmse_raw=0.9, r2_raw=0.9),
        ForecastMetrics(mse_norm=1.0, mae_raw=1.0, rmse_raw=1.0, r2_raw=1.0),
        ForecastMetrics(mse_norm=1.1, mae_raw=1.1, rmse_raw=1.1, r2_raw=1.1),
    ]
    agg = aggregate_seed_metrics(ms)

    expected_sample_std = 0.1  # hand-computed (ddof=1)
    population_std = float(np.sqrt(0.02 / 3.0))  # ~0.08165 (ddof=0) — bug value

    for field in ("mse_norm", "mae_raw", "rmse_raw", "r2_raw"):
        assert agg[field]["mean"] == pytest.approx(1.0, abs=1e-12)
        assert agg[field]["std"] == pytest.approx(expected_sample_std, abs=1e-9)
        # And make sure we are NOT computing the population std.
        assert abs(agg[field]["std"] - population_std) > 1e-3


def test_aggregate_seed_metrics_std_nan_for_single_seed():
    """With a single seed the unbiased std is undefined (division by zero).
    We surface NaN rather than silently emitting 0.0 (which ddof=0 would do).
    """
    from quantum_liquid_neuralode.evaluation.metrics import ForecastMetrics

    ms = [ForecastMetrics(mse_norm=0.5, mae_raw=0.5, rmse_raw=0.5, r2_raw=0.5)]
    agg = aggregate_seed_metrics(ms)
    for field in ("mse_norm", "mae_raw", "rmse_raw", "r2_raw"):
        assert agg[field]["mean"] == pytest.approx(0.5)
        assert np.isnan(agg[field]["std"])
        assert agg[field]["n_seeds"] == 1
