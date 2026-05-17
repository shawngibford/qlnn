from __future__ import annotations

import numpy as np


def persistence_forecast(od_last: np.ndarray) -> np.ndarray:
    """OD(t+h) = OD(t). Trivially returns od_last unchanged.

    This is the most important baseline for highly autocorrelated bioreactor
    OD: beating persistence is the bar a real model must clear.
    """
    return np.asarray(od_last, dtype=np.float32).copy()


def linear_extrapolation_forecast(
    *,
    od_last: np.ndarray,
    od_prev: np.ndarray,
    dt_last_hours: np.ndarray,
    horizon_hours: float,
) -> np.ndarray:
    """OD(t+h) = OD(t) + (OD(t) - OD(t-1)) * (h / dt_last).

    Replaces non-finite results (dt_last == 0 etc.) with the persistence value.
    """
    if not (od_last.shape == od_prev.shape == dt_last_hours.shape):
        raise ValueError("od_last, od_prev, dt_last_hours must share shape")
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be > 0")

    safe_dt = np.where(dt_last_hours > 0, dt_last_hours, np.nan)
    pred = od_last + (od_last - od_prev) * (horizon_hours / safe_dt)
    pred = np.where(np.isfinite(pred), pred, od_last)
    return pred.astype(np.float32)
