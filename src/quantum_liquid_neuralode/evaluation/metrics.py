from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class ForecastMetrics:
    """Canonical forecast metrics used by every model in this project.

    `mse_norm` is in normalized [0,1] OD space; everything else is in raw OD units.
    """
    mse_norm: float
    mae_raw: float
    rmse_raw: float
    r2_raw: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(
    *,
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    od_scaler: MinMaxScaler,
) -> ForecastMetrics:
    """Compute the canonical metric bundle.

    Inputs are normalized [0,1] OD values; we report MSE there (for loss-continuity)
    plus MAE/RMSE/R² in raw OD units (paper-table-ready).
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

    return ForecastMetrics(
        mse_norm=mse_norm,
        mae_raw=mae_raw,
        rmse_raw=rmse_raw,
        r2_raw=r2_raw,
    )


def aggregate_seed_metrics(metrics_list: list[ForecastMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate per-seed metrics into mean/std/min/max per field.

    Returns:
        {
            "mae_raw":  {"mean": ..., "std": ..., "min": ..., "max": ...},
            ...
        }
    """
    if not metrics_list:
        raise ValueError("metrics_list is empty")

    fields = ("mse_norm", "mae_raw", "rmse_raw", "r2_raw")
    out: dict[str, dict[str, float]] = {}
    for f in fields:
        vals = np.asarray([getattr(m, f) for m in metrics_list], dtype=np.float64)
        # ddof=1: unbiased sample std (Bessel's correction). This is the
        # standard convention for reporting mean +/- std over finite seed
        # populations in ML/scientific journals. numpy's default ddof=0 is the
        # population std and underestimates by a factor sqrt(n/(n-1)) for
        # small n (e.g. ~12% too small at n=5).
        std_val = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
        out[f] = {
            "mean": float(vals.mean()),
            "std": std_val,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n_seeds": int(vals.size),
        }
    return out


def metrics_bundle_to_jsonable(metrics: ForecastMetrics) -> dict[str, Any]:
    return metrics.to_dict()
