import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from quantum_liquid_neuralode.evaluation import (
    compute_metrics,
    linear_extrapolation_forecast,
    persistence_forecast,
    t_confidence_interval,
)
from quantum_liquid_neuralode.evaluation.metrics import (
    ForecastMetrics,
    aggregate_seed_metrics,
)

# Make the training scripts importable so we can unit-test helpers that live
# there (the clip-prediction utility lives in scripts/train_baseline.py until
# it's lifted into the package — R3 finding 6 follow-up).
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


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
    # Without od_last_norm, delta fields stay None and are dropped from the dict.
    d = m.to_dict()
    assert "delta_mae_raw" not in d
    assert "delta_rmse_raw" not in d
    assert "delta_r2_raw" not in d


def test_compute_metrics_scales_to_raw_units():
    sc = _od_scaler(0.0, 4.0)  # easy math: 1.0 norm -> 4.0 raw
    y_true = np.array([0.0, 0.25, 0.5], dtype=np.float32)
    y_pred = np.array([0.0, 0.25, 0.75], dtype=np.float32)  # off by 1.0 raw on last sample
    m = compute_metrics(y_true_norm=y_true, y_pred_norm=y_pred, od_scaler=sc)
    # raw errors: 0, 0, 1.0
    assert m.mae_raw == pytest.approx(1.0 / 3.0)
    assert m.rmse_raw == pytest.approx(np.sqrt(1.0 / 3.0))


def test_compute_metrics_delta_fields_filled_when_od_last_passed():
    """When od_last_norm is provided, delta_* fields are populated."""
    sc = _od_scaler(0.0, 4.0)
    y_true = np.array([0.0, 0.25, 0.5], dtype=np.float32)
    y_pred = np.array([0.0, 0.25, 0.75], dtype=np.float32)  # off by 1.0 raw on last sample
    od_last = np.array([0.0, 0.25, 0.5], dtype=np.float32)  # persistence == y_true

    m = compute_metrics(
        y_true_norm=y_true, y_pred_norm=y_pred, od_scaler=sc, od_last_norm=od_last
    )
    # od_last == y_true in raw -> true_delta = 0 for all samples.
    # pred_delta = y_pred_raw - od_last_raw -> [0, 0, 1.0]
    # delta_err = pred_delta - true_delta = pred_delta = [0, 0, 1.0]
    assert m.delta_mae_raw == pytest.approx(1.0 / 3.0)
    assert m.delta_rmse_raw == pytest.approx(np.sqrt(1.0 / 3.0))
    # var(true_delta) is zero -> R² undefined -> NaN by convention.
    assert m.delta_r2_raw is not None
    assert np.isnan(m.delta_r2_raw)

    # to_dict should now include the delta fields except NaN-handling note:
    # NaN is a valid value for r2 (not None), so it should appear.
    d = m.to_dict()
    assert "delta_mae_raw" in d
    assert "delta_rmse_raw" in d
    assert "delta_r2_raw" in d


def test_persistence_delta_metrics_expose_floor():
    """Persistence prediction => pred == od_last => pred_delta == 0.

    This is exactly the R3 reviewer's diagnostic:
    - delta_mae_raw == mean(|true_delta|) (i.e. naive zero-prediction error)
    - delta_r2_raw is strongly negative because predicting zero is worse than
      predicting the mean of true deltas (the r2 baseline).
    """
    sc = _od_scaler(0.0, 4.0)
    # Choose a case where true OD changes across samples.
    od_last = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32)
    y_true = np.array([0.15, 0.30, 0.20, 0.50], dtype=np.float32)
    y_pred = od_last.copy()  # persistence

    m = compute_metrics(
        y_true_norm=y_true, y_pred_norm=y_pred, od_scaler=sc, od_last_norm=od_last
    )
    # raw deltas: y_true_raw - od_last_raw scaled by (vmax-vmin)=4.0
    true_delta_raw = (y_true - od_last) * 4.0
    expected_dmae = float(np.mean(np.abs(true_delta_raw)))
    assert m.delta_mae_raw == pytest.approx(expected_dmae)
    # Persistence's pred_delta is 0 by construction -> R² < 0 unless true
    # deltas have zero variance (they don't here).
    assert m.delta_r2_raw is not None
    assert m.delta_r2_raw < 0.0


def test_compute_metrics_rejects_mismatched_od_last_shape():
    sc = _od_scaler()
    y = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    with pytest.raises(ValueError):
        compute_metrics(
            y_true_norm=y, y_pred_norm=y, od_scaler=sc,
            od_last_norm=np.array([0.1, 0.5], dtype=np.float32),
        )


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
    ms = [
        ForecastMetrics(mse_norm=0.01, mae_raw=0.1, rmse_raw=0.12, r2_raw=0.90),
        ForecastMetrics(mse_norm=0.03, mae_raw=0.2, rmse_raw=0.22, r2_raw=0.80),
    ]
    agg = aggregate_seed_metrics(ms)
    assert agg["mae_raw"]["mean"] == pytest.approx(0.15)
    assert agg["mae_raw"]["min"] == pytest.approx(0.1)
    assert agg["mae_raw"]["max"] == pytest.approx(0.2)
    assert agg["mae_raw"]["n_seeds"] == 2


def test_aggregate_seed_metrics_aggregates_delta_when_present_in_all():
    """When every seed reports delta_*, aggregate it normally."""
    ms = [
        ForecastMetrics(
            mse_norm=0.01, mae_raw=0.1, rmse_raw=0.12, r2_raw=0.90,
            delta_mae_raw=0.05, delta_rmse_raw=0.06, delta_r2_raw=0.50,
        ),
        ForecastMetrics(
            mse_norm=0.03, mae_raw=0.2, rmse_raw=0.22, r2_raw=0.80,
            delta_mae_raw=0.15, delta_rmse_raw=0.18, delta_r2_raw=0.30,
        ),
    ]
    agg = aggregate_seed_metrics(ms)
    assert "delta_mae_raw" in agg
    assert agg["delta_mae_raw"]["mean"] == pytest.approx(0.10)
    assert agg["delta_r2_raw"]["mean"] == pytest.approx(0.40)


def test_aggregate_seed_metrics_omits_delta_when_missing_in_any_seed():
    """If even one seed lacks a delta field, aggregate drops the field
    rather than crash or silently include a partial mean.
    """
    ms = [
        ForecastMetrics(
            mse_norm=0.01, mae_raw=0.1, rmse_raw=0.12, r2_raw=0.90,
            delta_mae_raw=0.05, delta_rmse_raw=0.06, delta_r2_raw=0.50,
        ),
        # Missing delta_*.
        ForecastMetrics(mse_norm=0.03, mae_raw=0.2, rmse_raw=0.22, r2_raw=0.80),
    ]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agg = aggregate_seed_metrics(ms)
        # At least one warning about the omitted field.
        assert any("delta" in str(rec.message) for rec in w)
    assert "mae_raw" in agg
    assert "delta_mae_raw" not in agg
    assert "delta_r2_raw" not in agg


def test_aggregate_seed_metrics_uses_unbiased_std():
    """Regression test for R1-B3: aggregate_seed_metrics must use unbiased
    sample std (ddof=1, Bessel's correction), not population std (ddof=0).

    Scientific convention is to report mean +/- sample std for finite seed
    populations. Hand-computed: for vals=[0.9, 1.0, 1.1] the mean is 1.0, and
    sample variance = (0.01 + 0 + 0.01) / (3 - 1) = 0.01 -> sample std = 0.1.
    Population std (ddof=0) would be sqrt(0.02/3) ~= 0.08165, which is wrong.
    """
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
    ms = [ForecastMetrics(mse_norm=0.5, mae_raw=0.5, rmse_raw=0.5, r2_raw=0.5)]
    agg = aggregate_seed_metrics(ms)
    for field in ("mse_norm", "mae_raw", "rmse_raw", "r2_raw"):
        assert agg[field]["mean"] == pytest.approx(0.5)
        assert np.isnan(agg[field]["std"])
        assert agg[field]["n_seeds"] == 1


# ---------------------------------------------------------------------------
# clip_predictions_norm tests — the four `test_clip_predictions_norm_*` tests
# that lived here imported `train_baseline.clip_predictions_norm`, which was
# moved to `archive/scripts/train_baseline.py` during the 2026-05-27 OD-era
# purge. The clip-predictions-norm helper itself is OD-program-only and is
# not used by any active post-pivot code path. The original tests are
# preserved at `archive/tests/test_metrics_clip_predictions_norm.py` (when
# present) for reproducibility of the archived OD-era results.
#
# Removed 2026-05-28 alongside the broader OD-purge test-debt cleanup.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# t_confidence_interval — R3 Tier 2.2.
# ---------------------------------------------------------------------------
def test_t_ci_basic():
    """For [1,2,3,4,5]: mean=3, sample std=sqrt(2.5), n=5, df=4.
    Critical t(0.025, df=4) ~= 2.776 -> half_width ~= 2.776 * sqrt(2.5)/sqrt(5)
    ~= 2.776 * 0.7071 ~= 1.9633."""
    mean, hw, df = t_confidence_interval([1, 2, 3, 4, 5], alpha=0.05)
    assert mean == pytest.approx(3.0)
    assert df == 4
    assert hw == pytest.approx(1.9633, abs=1e-3)


def test_t_ci_n1_returns_nan():
    mean, hw, df = t_confidence_interval([0.42])
    assert mean == pytest.approx(0.42)
    assert np.isnan(hw)
    assert df == 0


def test_t_ci_empty_returns_nan():
    mean, hw, df = t_confidence_interval([])
    assert np.isnan(mean)
    assert np.isnan(hw)
    assert df == 0


def test_aggregate_seed_metrics_emits_ci95():
    """The aggregator must surface CI fields alongside std (back-compat).

    For [0.9, 1.0, 1.1] across 3 seeds: sample std = 0.1, n=3, df=2,
    t(0.025, df=2) ~= 4.303 -> half_width ~= 4.303 * 0.1/sqrt(3) ~= 0.2484.
    """
    ms = [
        ForecastMetrics(mse_norm=0.9, mae_raw=0.9, rmse_raw=0.9, r2_raw=0.9),
        ForecastMetrics(mse_norm=1.0, mae_raw=1.0, rmse_raw=1.0, r2_raw=1.0),
        ForecastMetrics(mse_norm=1.1, mae_raw=1.1, rmse_raw=1.1, r2_raw=1.1),
    ]
    agg = aggregate_seed_metrics(ms)
    for field in ("mse_norm", "mae_raw", "rmse_raw", "r2_raw"):
        # Original fields preserved.
        assert agg[field]["mean"] == pytest.approx(1.0)
        assert agg[field]["std"] == pytest.approx(0.1, abs=1e-9)
        # New CI fields populated.
        assert agg[field]["ci95_half_width"] == pytest.approx(0.2484, abs=1e-3)
        assert agg[field]["ci95_low"] == pytest.approx(1.0 - 0.2484, abs=1e-3)
        assert agg[field]["ci95_high"] == pytest.approx(1.0 + 0.2484, abs=1e-3)


def test_aggregate_seed_metrics_ci_nan_for_single_seed():
    ms = [ForecastMetrics(mse_norm=0.5, mae_raw=0.5, rmse_raw=0.5, r2_raw=0.5)]
    agg = aggregate_seed_metrics(ms)
    for field in ("mse_norm", "mae_raw", "rmse_raw", "r2_raw"):
        assert np.isnan(agg[field]["ci95_half_width"])
        assert np.isnan(agg[field]["ci95_low"])
        assert np.isnan(agg[field]["ci95_high"])


# test_clip_predictions_norm_metric_impact_via_compute_metrics — removed
# 2026-05-28 alongside the other three clip_predictions_norm tests above.
# Same rationale: it imports `train_baseline.clip_predictions_norm` which
# moved to `archive/scripts/` during the 2026-05-27 OD-purge.
