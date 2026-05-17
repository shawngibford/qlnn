from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class SplitIdx:
    train_end: int
    val_end: int


def split_indices(n: int, *, train_ratio: float, val_ratio: float) -> SplitIdx:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end < 3:
        raise ValueError("train split too small")
    if val_end <= train_end:
        raise ValueError("val split too small")

    return SplitIdx(train_end=train_end, val_end=val_end)


def fit_minmax(
    df: pd.DataFrame,
    cols: list[str],
    *,
    fit_end: int,
    fixed_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, MinMaxScaler]:
    """Fit per-column MinMax scalers.

    If a column appears in fixed_bounds, the scaler is fit on those bounds
    instead of the training slice — useful for OD where train-only bounds
    can push later rows outside [0,1].
    """
    scalers: dict[str, MinMaxScaler] = {}
    fixed_bounds = fixed_bounds or {}

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

        sc = MinMaxScaler()

        if col in fixed_bounds:
            vmin, vmax = fixed_bounds[col]
            if vmax <= vmin:
                raise ValueError(f"Invalid fixed bounds for {col}: min={vmin}, max={vmax}")
            sc.fit(pd.DataFrame({col: [vmin, vmax]}))
        else:
            sc.fit(df[[col]].iloc[:fit_end])

        scalers[col] = sc

    return scalers


def apply_minmax(df: pd.DataFrame, cols: list[str], scalers: dict[str, MinMaxScaler]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = scalers[col].transform(out[[col]])
    return out


@dataclass(frozen=True)
class HorizonWindows:
    """Supervised (history-window -> target-at-horizon) examples.

    Shapes:
        x:           (n_windows, window_size, n_features)
        t:           (n_windows, window_size)         window-relative time in hours
        y:           (n_windows,)                     OD at +horizon (normalized)
        od_last:     (n_windows,)                     OD at t_end (normalized)
        od_prev:     (n_windows,)                     OD at t_end - 1 step
        dt_last:     (n_windows,)                     last in-window dt (hours)
        end_idx:     (n_windows,)                     global row index of t_end
        target_idx:  (n_windows,)                     global row index of (t_end + horizon)
    """
    x: np.ndarray
    t: np.ndarray
    y: np.ndarray
    od_last: np.ndarray
    od_prev: np.ndarray
    dt_last: np.ndarray
    end_idx: np.ndarray
    target_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])


def make_horizon_windows(
    *,
    features: np.ndarray,
    od: np.ndarray,
    time_hours: np.ndarray,
    window_size: int,
    stride: int,
    horizon_hours: float,
    horizon_tolerance_hours: float,
    index_offset: int = 0,
) -> HorizonWindows:
    """Create supervised (history, OD-at-horizon) windows from a contiguous segment.

    A window is kept only if it has a real observation at (t_end + horizon_hours),
    within ±horizon_tolerance_hours. This makes the task a strict fixed-horizon
    forecast rather than a learned interpolation.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be > 0")
    if horizon_tolerance_hours < 0:
        raise ValueError("horizon_tolerance_hours must be >= 0")

    if not (features.shape[0] == od.shape[0] == time_hours.shape[0]):
        raise ValueError("features/od/time_hours must have matching first dimension")

    xs: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    ys: list[float] = []
    od_last_list: list[float] = []
    od_prev_list: list[float] = []
    dt_last_list: list[float] = []
    end_idx_list: list[int] = []
    target_idx_list: list[int] = []

    n = features.shape[0]
    for start in range(0, n - window_size, stride):
        end = start + window_size
        end_idx = end - 1

        target_time = time_hours[end_idx] + horizon_hours
        target_idx = int(np.searchsorted(time_hours, target_time, side="left"))
        if target_idx >= n:
            continue

        realized = float(time_hours[target_idx] - time_hours[end_idx])
        if abs(realized - horizon_hours) > horizon_tolerance_hours:
            continue

        xw = features[start:end]
        tw = time_hours[start:end]
        tw = tw - tw[0]

        xs.append(xw)
        ts.append(tw.astype(np.float32))
        ys.append(float(od[target_idx]))

        od_last_list.append(float(od[end_idx]))
        od_prev_list.append(float(od[end_idx - 1]))
        dt_last_list.append(float(tw[-1] - tw[-2]))

        end_idx_list.append(int(index_offset + end_idx))
        target_idx_list.append(int(index_offset + target_idx))

    if not xs:
        raise ValueError("No windows created (check window_size/stride/horizon vs data length)")

    return HorizonWindows(
        x=np.stack(xs, axis=0),
        t=np.stack(ts, axis=0),
        y=np.asarray(ys, dtype=np.float32),
        od_last=np.asarray(od_last_list, dtype=np.float32),
        od_prev=np.asarray(od_prev_list, dtype=np.float32),
        dt_last=np.asarray(dt_last_list, dtype=np.float32),
        end_idx=np.asarray(end_idx_list, dtype=np.int64),
        target_idx=np.asarray(target_idx_list, dtype=np.int64),
    )
