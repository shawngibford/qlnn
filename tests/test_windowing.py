import numpy as np
import pytest

from quantum_liquid_neuralode.data_processing import (
    apply_minmax,
    fit_minmax,
    make_horizon_windows,
    split_indices,
)
from quantum_liquid_neuralode.data_processing.windowing import HorizonWindows


def test_split_indices_basic():
    s = split_indices(100, train_ratio=0.7, val_ratio=0.15)
    assert s.train_end == 70
    assert s.val_end == 85


def test_split_indices_rejects_invalid_ratios():
    with pytest.raises(ValueError):
        split_indices(100, train_ratio=0.7, val_ratio=0.3)
    with pytest.raises(ValueError):
        split_indices(100, train_ratio=0.0, val_ratio=0.15)


def test_make_horizon_windows_shapes_and_endpoints():
    n = 50
    time_hours = np.arange(n, dtype=np.float64) / 6.0  # 10-minute spacing
    od = np.linspace(0.1, 0.9, n).astype(np.float32)
    feats = np.stack([od, np.zeros(n, dtype=np.float32)], axis=1)

    win = make_horizon_windows(
        features=feats,
        od=od,
        time_hours=time_hours,
        window_size=6,
        stride=1,
        horizon_hours=1.0,
        horizon_tolerance_hours=1e-3,
    )

    assert isinstance(win, HorizonWindows)
    assert win.x.shape[1] == 6
    assert win.x.shape[2] == 2
    assert win.y.shape == (len(win),)
    assert win.t.shape == (len(win), 6)
    # First window should start at t=0
    assert win.t[0, 0] == pytest.approx(0.0)
    # All target indices must be valid into the input array
    assert win.target_idx.max() < n


def test_make_horizon_windows_drops_misaligned_horizons():
    # Irregular sampling — most horizon points won't land exactly at 1h.
    n = 50
    time_hours = np.cumsum(np.full(n, 0.1, dtype=np.float64))  # 0.1h apart
    od = np.linspace(0.1, 0.9, n).astype(np.float32)
    feats = od.reshape(-1, 1)

    # Horizon of 1.0h is exactly 10 steps -> all windows should have a real target.
    w_aligned = make_horizon_windows(
        features=feats, od=od, time_hours=time_hours,
        window_size=6, stride=1, horizon_hours=1.0, horizon_tolerance_hours=1e-6,
    )
    assert len(w_aligned) > 0

    # Horizon of 1.05h falls between samples -> tight tolerance should drop all.
    with pytest.raises(ValueError):
        make_horizon_windows(
            features=feats, od=od, time_hours=time_hours,
            window_size=6, stride=1, horizon_hours=1.05, horizon_tolerance_hours=1e-6,
        )


def test_fit_apply_minmax_fixed_bounds():
    import pandas as pd

    df = pd.DataFrame({"OD": [0.5, 1.0, 2.0, 3.0]})
    scalers = fit_minmax(df, ["OD"], fit_end=2, fixed_bounds={"OD": (0.0, 4.0)})
    out = apply_minmax(df, ["OD"], scalers)
    # OD=0 -> 0, OD=4 -> 1, OD=2 -> 0.5
    assert out["OD"].iloc[2] == pytest.approx(0.5)
    assert out["OD"].iloc[3] == pytest.approx(0.75)
