from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, fields
from typing import Any, Optional

import numpy as np
from scipy import stats as _scipy_stats
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


def t_confidence_interval(
    values, *, alpha: float = 0.05
) -> tuple[float, float, int]:
    """Compute (mean, half_width, df) for a t-distribution CI on a 1D sample.

    Reports a two-sided 100*(1-alpha)% confidence interval for the mean of a
    finite sample assuming approximate normality (or applicability of the
    Student-t distribution for small n). This is the standard convention for
    reporting "mean ± CI" in scientific papers when the population variance
    is unknown and n is small (per e.g. NIST/SEMATECH e-Handbook §1.3.6.7.2,
    Efron & Tibshirani 1993 §12.5).

    Formula::

        half_width = t_crit(alpha/2, df=n-1) * std(ddof=1) / sqrt(n)

    Args:
        values: 1D array-like sample (need not be a numpy array).
        alpha: significance level. Default 0.05 -> 95% CI.

    Returns:
        Tuple ``(mean, half_width, df)``:
          - ``mean``: the sample mean (NaN if n=0).
          - ``half_width``: t_crit * sample_std / sqrt(n). Undefined for n<2;
            we return NaN in that case.
          - ``df``: n - 1 (degrees of freedom). 0 when n<=1.

    Notes:
      - Uses ``scipy.stats.t.ppf(1 - alpha/2, df=n-1)`` for the critical value.
      - Sample std uses Bessel's correction (ddof=1), matching the rest of
        this module (R1-B3 fix).
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mean = float(arr.mean())
    if n < 2:
        # CI undefined for n=1: no degrees of freedom for the variance estimate.
        return (mean, float("nan"), 0)
    df = n - 1
    sample_std = float(arr.std(ddof=1))
    t_crit = float(_scipy_stats.t.ppf(1.0 - alpha / 2.0, df=df))
    half_width = t_crit * sample_std / math.sqrt(n)
    return (mean, half_width, df)


@dataclass(frozen=True)
class ForecastMetrics:
    """Canonical forecast metrics used by every model in this project.

    Legacy fields (REQUIRED):
        - `mse_norm` in normalized [0,1] OD space
        - `mae_raw`, `rmse_raw`, `r2_raw` in raw OD units

    Delta-of-OD fields (OPTIONAL, populated when `od_last_norm` is supplied to
    `compute_metrics`). These score the predicted delta `y_pred - od_last`
    against the true delta `y_true - od_last` and expose the "persistence floor"
    issue: a persistence model scores ~0 MAE on raw OD but its delta_r2_raw is
    strongly negative because predicting zero-delta is worse than the mean of
    true deltas as a variance-explanation baseline. Reviewer R3 asked for this
    diagnostic so the paper table can show how much real OD-change signal the
    model has actually captured.

    All four delta fields are `None` when `od_last_norm` was not provided —
    so legacy `to_dict()` output is byte-identical for callers that don't opt
    in.
    """
    mse_norm: float
    mae_raw: float
    rmse_raw: float
    r2_raw: float

    # Optional delta-of-OD metrics (filled when od_last_norm is provided).
    delta_mae_raw: Optional[float] = None
    delta_rmse_raw: Optional[float] = None
    delta_r2_raw: Optional[float] = None

    def to_dict(self) -> dict[str, float]:
        """Return a dict of fields, omitting any None-valued (optional) entries
        so legacy JSON outputs remain identical when delta metrics aren't used.
        """
        out: dict[str, float] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            out[f.name] = v
        return out


def compute_metrics(
    *,
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    od_scaler: MinMaxScaler,
    od_last_norm: Optional[np.ndarray] = None,
) -> ForecastMetrics:
    """Compute the canonical metric bundle.

    Inputs are normalized [0,1] OD values; we report MSE there (for loss-continuity)
    plus MAE/RMSE/R² in raw OD units (paper-table-ready).

    If `od_last_norm` is provided (the persistence anchor each prediction sits
    on top of), we additionally compute delta-of-OD metrics — scoring
    `y_pred - od_last` against `y_true - od_last`. Persistence has pred_delta
    identically 0 by construction, so its delta_r2_raw is strongly negative
    (predicting zero is worse than predicting the mean of true deltas). This
    makes the delta metrics the right diagnostic for "did the model learn
    real OD-change signal?", separate from the easy persistence floor.
    """
    if y_true_norm.shape != y_pred_norm.shape:
        raise ValueError(f"shape mismatch: y_true={y_true_norm.shape} y_pred={y_pred_norm.shape}")
    if y_true_norm.size == 0:
        raise ValueError("empty inputs")

    y_t = y_true_norm.reshape(-1).astype(np.float64)
    y_p = y_pred_norm.reshape(-1).astype(np.float64)

    mse_norm = float(np.mean((y_p - y_t) ** 2))

    y_t_raw = od_scaler.inverse_transform(y_t.reshape(-1, 1)).reshape(-1)
    y_p_raw = od_scaler.inverse_transform(y_p.reshape(-1, 1)).reshape(-1)

    err = y_p_raw - y_t_raw
    mae_raw = float(np.mean(np.abs(err)))
    rmse_raw = float(np.sqrt(np.mean(err ** 2)))
    r2_raw = float(r2_score(y_t_raw, y_p_raw))

    delta_mae_raw: Optional[float] = None
    delta_rmse_raw: Optional[float] = None
    delta_r2_raw: Optional[float] = None

    if od_last_norm is not None:
        if od_last_norm.shape != y_true_norm.shape:
            raise ValueError(
                f"od_last_norm shape {od_last_norm.shape} must match y_true_norm shape "
                f"{y_true_norm.shape}"
            )
        od_l = od_last_norm.reshape(-1).astype(np.float64)
        od_l_raw = od_scaler.inverse_transform(od_l.reshape(-1, 1)).reshape(-1)

        true_delta = y_t_raw - od_l_raw
        pred_delta = y_p_raw - od_l_raw
        derr = pred_delta - true_delta

        delta_mae_raw = float(np.mean(np.abs(derr)))
        delta_rmse_raw = float(np.sqrt(np.mean(derr ** 2)))

        # R² of pred_delta vs true_delta. r2_score is degenerate when the
        # true variance is zero (all deltas equal); guard so we don't crash.
        true_delta_var = float(np.var(true_delta))
        if true_delta_var <= 0.0:
            # Convention: when there's no variance to explain, R² is undefined.
            # Return NaN rather than crash; downstream aggregation handles it.
            delta_r2_raw = float("nan")
        else:
            delta_r2_raw = float(r2_score(true_delta, pred_delta))

    return ForecastMetrics(
        mse_norm=mse_norm,
        mae_raw=mae_raw,
        rmse_raw=rmse_raw,
        r2_raw=r2_raw,
        delta_mae_raw=delta_mae_raw,
        delta_rmse_raw=delta_rmse_raw,
        delta_r2_raw=delta_r2_raw,
    )


# Fields aggregate_seed_metrics knows how to summarize. Order matters for the
# JSON layout; required fields first, then optional delta fields (which are
# skipped if absent from every seed).
_AGGREGATABLE_FIELDS: tuple[str, ...] = (
    "mse_norm",
    "mae_raw",
    "rmse_raw",
    "r2_raw",
    "delta_mae_raw",
    "delta_rmse_raw",
    "delta_r2_raw",
)


def aggregate_seed_metrics(metrics_list: list[ForecastMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate per-seed metrics into mean/std/min/max per field.

    None-safety: any optional (delta_*) field that is None in *any* seed is
    omitted from the aggregate (with a warning). This keeps mixed-mode runs
    (some seeds reporting deltas, others not) from silently producing
    inconsistent aggregates. If a field is present in every seed, it's
    aggregated normally.

    Returns:
        {
            "mae_raw":  {"mean": ..., "std": ..., "min": ..., "max": ...},
            ...
        }
    """
    if not metrics_list:
        raise ValueError("metrics_list is empty")

    out: dict[str, dict[str, float]] = {}
    for f in _AGGREGATABLE_FIELDS:
        raw = [getattr(m, f) for m in metrics_list]
        # Skip aggregating fields where any seed is missing the value.
        if any(v is None for v in raw):
            # If at least one seed *did* report this field, warn — the user
            # likely intended for all seeds to provide it.
            if any(v is not None for v in raw):
                warnings.warn(
                    f"aggregate_seed_metrics: field {f!r} is None in some seeds "
                    f"but not others; omitting from aggregate.",
                    stacklevel=2,
                )
            continue

        vals = np.asarray(raw, dtype=np.float64)
        # ddof=1: unbiased sample std (Bessel's correction). This is the
        # standard convention for reporting mean +/- std over finite seed
        # populations in ML/scientific journals. numpy's default ddof=0 is the
        # population std and underestimates by a factor sqrt(n/(n-1)) for
        # small n (e.g. ~12% too small at n=5).
        std_val = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
        # 95% t-CI half-width on the mean. Reported alongside std so the paper
        # table can switch from bare-std error bars to proper CIs without
        # breaking back-compat for callers that only consume mean/std/min/max.
        # (R3 Tier 2.2 — reviewer asked for CI-based reporting instead of std.)
        mean_val, ci95_hw, _df = t_confidence_interval(vals, alpha=0.05)
        if math.isnan(ci95_hw):
            ci95_low = float("nan")
            ci95_high = float("nan")
        else:
            ci95_low = mean_val - ci95_hw
            ci95_high = mean_val + ci95_hw
        out[f] = {
            "mean": float(vals.mean()),
            "std": std_val,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n_seeds": int(vals.size),
            "ci95_half_width": ci95_hw,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
        }
    return out


def metrics_bundle_to_jsonable(metrics: ForecastMetrics) -> dict[str, Any]:
    return metrics.to_dict()
