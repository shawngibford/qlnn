"""Paired bootstrap for head-to-head forecast-model comparison.

The classical mean ± std comparison across seeds has very little power on this
task: at 1h horizon the headroom above persistence is only ~5% of variance,
seed std bars overlap heavily, and we have at most n=5 seeds. The
power-maximizing alternative is a **paired bootstrap on the per-window
residuals**: resampling the (relatively large) pool of test windows holds
the model pair fixed and lets us read CI / p-value directly off the
diff distribution.

Convention: Efron & Tibshirani 1993 (*An Introduction to the Bootstrap*) §10
("hypothesis testing") and §13 ("the bootstrap estimate of standard error").
For a two-sided p-value against H0: diff == 0 we report
``2 * min(P(diff>0), P(diff<0))`` over the bootstrap distribution.
"""
from __future__ import annotations

from typing import Literal

import numpy as np


_MetricName = Literal["mae", "rmse", "r2"]


def _metric(y_true: np.ndarray, y_pred: np.ndarray, metric: _MetricName) -> float:
    if metric == "mae":
        return float(np.mean(np.abs(y_pred - y_true)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    if metric == "r2":
        # 1 - SS_res / SS_tot; degenerate when SS_tot == 0 (all y_true equal).
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        if ss_tot <= 0.0:
            return float("nan")
        ss_res = float(np.sum((y_pred - y_true) ** 2))
        return 1.0 - ss_res / ss_tot
    raise ValueError(f"unknown metric: {metric!r}")


def paired_bootstrap_diff(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y_true: np.ndarray,
    *,
    metric: _MetricName = "mae",
    n_iter: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap on a per-window basis (Efron & Tibshirani 1993).

    Resamples WINDOW INDICES with replacement (not seeds) and on each
    resample evaluates the chosen metric for both predictors against the
    same resampled ground truth. The difference distribution ``diff = M(a) -
    M(b)`` yields the point estimate, an empirical 95% CI from the
    [alpha/2, 1-alpha/2] percentiles, and a two-sided p-value
    ``2 * min(frac_above_0, frac_below_0)``.

    For lower-is-better metrics (``mae``, ``rmse``) a negative ``mean_diff``
    means model A is better than model B; for ``r2`` higher is better and a
    positive ``mean_diff`` favors A.

    Args:
        pred_a, pred_b: 1D arrays of predictions (same length, same units).
        y_true: 1D ground-truth array (same length as the predictions).
        metric: one of ``"mae"``, ``"rmse"``, ``"r2"``.
        n_iter: number of bootstrap iterations.
        alpha: two-sided significance level for the CI. Default 0.05.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys:
          - ``metric``: the metric used.
          - ``n_iter``: bootstrap iterations.
          - ``n_windows``: number of test windows.
          - ``metric_a``, ``metric_b``: point estimates on the full set.
          - ``mean_diff``: bootstrap mean of (M(a) - M(b)).
          - ``ci_low``, ``ci_high``: empirical bootstrap CI bounds.
          - ``ci_half_width``: (ci_high - ci_low) / 2 (convenience).
          - ``p_value``: two-sided p-value against H0: diff == 0.
          - ``alpha``: the alpha used.
    """
    a = np.asarray(pred_a, dtype=np.float64).reshape(-1)
    b = np.asarray(pred_b, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if not (a.shape == b.shape == y.shape):
        raise ValueError(f"shape mismatch: a={a.shape} b={b.shape} y={y.shape}")
    n = a.size
    if n == 0:
        raise ValueError("empty inputs")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        diffs[i] = _metric(y[idx], a[idx], metric) - _metric(y[idx], b[idx], metric)

    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    ci_low = float(np.quantile(diffs, lo_q))
    ci_high = float(np.quantile(diffs, hi_q))

    # Two-sided p-value (Efron-Tibshirani achieved-significance-level
    # convention). For lower-is-better metrics: ``p = 2*min(frac_above_0,
    # frac_below_0)`` — counts how often the bootstrap diff has the
    # "wrong" sign for the observed point estimate.
    #
    # Edge case 1: every bootstrap iteration is exactly zero (e.g. pred_a ==
    # pred_b). frac_above = frac_below = 0 -> 2*min = 0. Interpreting that
    # as "extremely significant" is wrong; the correct reading is "no
    # signal of any direction", i.e. p = 1.0. We special-case it.
    #
    # Edge case 2: directional rejection observed (e.g. every iteration has
    # diff < 0). 2*min(0, 1) = 0; we floor at 1/n_iter so the reported p is
    # finite (any finite bootstrap only shows the resolution it can afford).
    frac_above = float(np.mean(diffs > 0.0))
    frac_below = float(np.mean(diffs < 0.0))
    if frac_above == 0.0 and frac_below == 0.0:
        # Degenerate distribution at zero -> H0 cannot be rejected.
        p_two_sided = 1.0
    else:
        p_two_sided = 2.0 * min(frac_above, frac_below)
        p_two_sided = max(p_two_sided, 1.0 / n_iter)
        p_two_sided = min(p_two_sided, 1.0)

    return {
        "metric": metric,
        "n_iter": int(n_iter),
        "n_windows": int(n),
        "metric_a": _metric(y, a, metric),
        "metric_b": _metric(y, b, metric),
        "mean_diff": float(diffs.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half_width": float((ci_high - ci_low) / 2.0),
        "p_value": float(p_two_sided),
        "alpha": float(alpha),
    }
