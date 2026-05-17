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
# clip_predictions_norm — the R3 finding 6 raw-OD-range clip helper.
# Defined in scripts/train_baseline.py (and mirrored in scripts/train_qlnn.py).
# ---------------------------------------------------------------------------
def test_clip_predictions_norm_noop_when_max_is_none():
    from train_baseline import clip_predictions_norm

    sc = _od_scaler(0.0, 3.8)
    y = np.array([-0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    out = clip_predictions_norm(y, sc, clip_raw_max=None)
    # Pass-through (identity), same dtype and values.
    np.testing.assert_array_equal(out, y)


def test_clip_predictions_norm_clips_to_raw_range():
    """With scaler fit on [0, 4] and raw clip at 3.8, normalized clip = 0.95.
    Predictions outside [0, 0.95] in normalized space must be clamped."""
    from train_baseline import clip_predictions_norm

    sc = _od_scaler(0.0, 4.0)  # 1.0 norm == 4.0 raw
    y = np.array([-0.1, 0.0, 0.5, 0.95, 1.0, 1.5], dtype=np.float32)
    out = clip_predictions_norm(y, sc, clip_raw_max=3.8, clip_raw_min=0.0)
    # Normalized image of [0, 3.8] under fit [0, 4] is [0, 0.95].
    np.testing.assert_allclose(
        out, np.array([0.0, 0.0, 0.5, 0.95, 0.95, 0.95], dtype=np.float32), rtol=0, atol=1e-6
    )


def test_clip_predictions_norm_handles_train_only_scaler():
    """When the scaler is fit on a training slice whose max is below the
    physical clip ceiling, the normalized clip ceiling is > 1 — predictions
    in [0, 1] pass through, predictions above 1 are still clipped to the
    normalized image of clip_raw_max."""
    from train_baseline import clip_predictions_norm

    # Scaler fit on training range [0, 2.5]. Physical max prior = 3.8.
    sc = _od_scaler(0.0, 2.5)
    # 3.8 raw maps to (3.8 - 0) / (2.5 - 0) = 1.52 norm.
    y = np.array([0.5, 1.0, 1.4, 1.52, 1.8, 5.0], dtype=np.float32)
    out = clip_predictions_norm(y, sc, clip_raw_max=3.8, clip_raw_min=0.0)
    np.testing.assert_allclose(
        out, np.array([0.5, 1.0, 1.4, 1.52, 1.52, 1.52], dtype=np.float32), rtol=0, atol=1e-6
    )
    # Below-zero values get clipped up to 0.
    y2 = np.array([-1.0, -0.1, 0.0], dtype=np.float32)
    out2 = clip_predictions_norm(y2, sc, clip_raw_max=3.8, clip_raw_min=0.0)
    np.testing.assert_allclose(out2, np.array([0.0, 0.0, 0.0], dtype=np.float32))


def test_clip_predictions_norm_metric_impact_via_compute_metrics():
    """End-to-end: a wildly over-shooting prediction is clipped to the
    physical max before compute_metrics, so the raw-space error reflects the
    clipped value (not the original overshoot). This is the property that
    actually closes the leakage at the eval layer."""
    from train_baseline import clip_predictions_norm

    sc = _od_scaler(0.0, 4.0)  # 1.0 norm == 4.0 raw; 3.8 raw == 0.95 norm.
    y_true = np.array([0.5], dtype=np.float32)
    y_pred = np.array([5.0], dtype=np.float32)  # absurd overshoot
    # Without clipping: raw error = |5*4 - 0.5*4| = 18.0.
    m_unclipped = compute_metrics(y_true_norm=y_true, y_pred_norm=y_pred, od_scaler=sc)
    assert m_unclipped.mae_raw == pytest.approx(18.0)
    # With clipping at physical max 3.8: pred -> 0.95 norm -> 3.8 raw; error = 3.8 - 2.0 = 1.8.
    y_pred_c = clip_predictions_norm(y_pred, sc, clip_raw_max=3.8)
    m_clipped = compute_metrics(y_true_norm=y_true, y_pred_norm=y_pred_c, od_scaler=sc)
    assert m_clipped.mae_raw == pytest.approx(1.8)
