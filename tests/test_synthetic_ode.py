"""Tests for the synthetic ODE benchmark generator.

Covers the properties that make the suite scientifically trustworthy:
determinism, dynamical-regime sanity (conserved quantity / boundedness /
chaos), and — most important — bit-exact compatibility with the existing
qZETA loader so the trainer pipeline consumes synthetic data unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantum_liquid_neuralode.data_processing import (
    load_qzeta,
    make_horizon_windows,
    split_indices,
    time_hours_from_date,
)
from quantum_liquid_neuralode.data_processing.synthetic_ode import (
    SYSTEMS,
    get_system,
    make_ode_dataframe,
    simulate,
)


def test_all_five_systems_registered():
    assert SYSTEMS == ["lotka_volterra", "fitzhugh_nagumo", "van_der_pol",
                       "lorenz", "kuramoto"]
    for name in SYSTEMS:
        get_system(name)  # must not raise


def test_unknown_system_raises():
    with pytest.raises(ValueError, match="unknown system"):
        get_system("rossler")


@pytest.mark.parametrize("name", SYSTEMS)
def test_simulation_is_deterministic(name):
    t1, Y1, _ = simulate(name, n_points=300, noise_std=0.0, seed=0)
    t2, Y2, _ = simulate(name, n_points=300, noise_std=0.0, seed=0)
    assert np.array_equal(t1, t2)
    assert np.array_equal(Y1, Y2)


@pytest.mark.parametrize("name", SYSTEMS)
def test_trajectory_finite_and_right_shape(name):
    sys = get_system(name)
    t, Y, _ = simulate(name, n_points=500, noise_std=0.0, seed=0)
    assert Y.shape == (500, len(sys.state_names))
    assert np.all(np.isfinite(Y))
    # Not a fixed point — every benchmark system must actually move.
    assert Y.std(axis=0).max() > 1e-3


def test_lotka_volterra_conserved_quantity_is_stable():
    """LV conserves V = δx − γ ln x + β y − α ln y. Under fixed-step RK4
    it should drift only slightly over a bounded run.
    """
    _, Y, sys = simulate("lotka_volterra", n_points=2000, noise_std=0.0)
    a, b = sys.params["alpha"], sys.params["beta"]
    d, g = sys.params["delta"], sys.params["gamma"]
    x, y = Y[:, 0], Y[:, 1]
    V = d * x - g * np.log(x) + b * y - a * np.log(y)
    rel_drift = (V.max() - V.min()) / abs(np.median(V))
    assert rel_drift < 0.05  # < 5% drift ⇒ integration is faithful


def test_lorenz_is_bounded_and_chaotic():
    """Lorenz stays on the attractor (bounded) yet shows sensitive
    dependence (nearby IC diverge)."""
    _, Y, _ = simulate("lorenz", n_points=3000, noise_std=0.0)
    assert np.abs(Y).max() < 100.0          # on the attractor, not blowing up
    base = get_system("lorenz")
    # Perturb IC by 1e-6 and re-integrate by hand for a few hundred steps.
    from quantum_liquid_neuralode.data_processing.synthetic_ode import _rk4
    import dataclasses
    pert = dataclasses.replace(base, y0=base.y0 + 1e-6)
    _, A = _rk4(base, 1500)
    _, B = _rk4(pert, 1500)
    assert np.linalg.norm(A[-1] - B[-1]) > 1.0  # divergence ⇒ chaos


def test_van_der_pol_limit_cycle_oscillates():
    _, Y, _ = simulate("van_der_pol", n_points=2000, noise_std=0.0)
    x = Y[:, 0]
    # A relaxation oscillator crosses zero many times.
    sign_changes = np.sum(np.diff(np.sign(x)) != 0)
    assert sign_changes > 4


@pytest.mark.parametrize("name", SYSTEMS)
def test_qzeta_schema_roundtrip_is_bit_exact(name, tmp_path):
    """make_ode_dataframe → CSV → load_qzeta must give a monotonic 1.0-hour
    step axis with t[-1] == n_points-1 (no dayfirst misparse / reorder).
    This is the property the whole trainer pipeline depends on.
    """
    n = 600
    df, target = make_ode_dataframe(name, n_points=n, noise_std=0.0, seed=0)
    assert "DATE" in df.columns and target in df.columns
    csv = tmp_path / f"{name}.csv"
    df.to_csv(csv, index=False)

    d2 = load_qzeta(csv)
    t = time_hours_from_date(d2)
    assert len(d2) == n
    assert np.allclose(np.diff(t), 1.0)            # exact 1-hour steps
    assert t[-1] == pytest.approx(n - 1)           # no scramble
    # And the locked h=3 windowing yields a sane window count.
    s = split_indices(n, train_ratio=0.7, val_ratio=0.15)
    feat = [c for c in d2.columns if c != "DATE"]
    arr = d2[feat].iloc[: s.train_end].to_numpy(dtype=np.float32)
    w = make_horizon_windows(
        features=arr, od=arr[:, feat.index(target)],
        time_hours=t[: s.train_end].astype(np.float64),
        window_size=24, stride=1, horizon_hours=3.0,
        horizon_tolerance_hours=0.0835)
    assert w.x.shape[0] > 100 and w.x.shape[1] == 24


def test_observation_noise_is_seeded_and_optional():
    _, clean, _ = simulate("lorenz", n_points=300, noise_std=0.0, seed=0)
    _, noisy_a, _ = simulate("lorenz", n_points=300, noise_std=0.1, seed=0)
    _, noisy_b, _ = simulate("lorenz", n_points=300, noise_std=0.1, seed=0)
    assert not np.array_equal(clean, noisy_a)        # noise applied
    assert np.array_equal(noisy_a, noisy_b)          # but seeded ⇒ repeatable
