"""P4 commit 2 — rollout metric suite tests.

Verifies each metric against known-trivial inputs and analytic
references. Per pre-reg §5 each metric is locked; these tests guard
the lock.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qlnn_.evaluation.rollout_metrics import (
    LYAPUNOV_EXPONENT,
    RolloutMetrics,
    VPTResult,
    invariant_drift,
    kdv_energy,
    kdv_mass,
    lotka_volterra_invariant,
    normalize_to_lyapunov_time,
    relative_l2_error,
    relative_l2_over_time,
    spectral_error,
    valid_prediction_time,
)


# ---------- relative-L2 -----------------------------------------------------

def test_relative_l2_zero_on_identity():
    u = np.random.randn(50, 3)
    assert relative_l2_error(u, u) == pytest.approx(0.0, abs=1e-12)


def test_relative_l2_unit_on_orthogonal():
    """If u_pred = 0 and u_ref is non-zero, the relative-L2 is exactly 1.
    This is the 'predict-zero floor' the pre-reg references."""
    u_pred = np.zeros((20, 2))
    u_ref = np.ones((20, 2)) * 0.7
    assert relative_l2_error(u_pred, u_ref) == pytest.approx(1.0, rel=1e-10)


def test_relative_l2_scale_invariant_in_ref():
    """Scaling BOTH u_pred and u_ref by same factor doesn't change the
    relative error — relative-L2 is scale-invariant in the reference."""
    rng = np.random.default_rng(42)
    u_ref = rng.standard_normal((30, 4))
    u_pred = u_ref + 0.1 * rng.standard_normal((30, 4))
    e1 = relative_l2_error(u_pred, u_ref)
    e2 = relative_l2_error(100.0 * u_pred, 100.0 * u_ref)
    assert e1 == pytest.approx(e2, rel=1e-10)


def test_relative_l2_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        relative_l2_error(np.zeros((10, 3)), np.zeros((10, 4)))


def test_relative_l2_over_time_per_timestep_shape():
    rng = np.random.default_rng(0)
    u_ref = rng.standard_normal((25, 3))
    u_pred = u_ref + 0.05 * rng.standard_normal((25, 3))
    curve = relative_l2_over_time(u_pred, u_ref)
    assert curve.shape == (25,)
    assert np.all(np.isfinite(curve))
    assert np.all(curve >= 0)


# ---------- VPT -------------------------------------------------------------

def test_vpt_never_exceeded_returns_minus_one():
    """A model that perfectly tracks the reference has VPT = -1 (sentinel)
    and vpt_time = full rollout duration."""
    u = np.ones((40, 2))
    res = valid_prediction_time(u, u, dt=0.1, threshold=0.3)
    assert isinstance(res, VPTResult)
    assert res.vpt_step == -1
    assert res.vpt_time == pytest.approx(40 * 0.1)
    assert res.vpt_lyapunov is None
    assert res.rel_l2_curve.shape == (40,)


def test_vpt_step_index_and_time():
    """Construct a trajectory whose first deviation hits the threshold
    at a known step."""
    # u_ref is constant ones; u_pred matches for first 5 steps then
    # deviates such that relative-L2 jumps above 0.3.
    u_ref = np.ones((10, 1))
    u_pred = u_ref.copy()
    u_pred[5:] = 0.0  # large deviation starting at step 5
    res = valid_prediction_time(u_pred, u_ref, dt=0.5, threshold=0.3)
    assert res.vpt_step == 5
    assert res.vpt_time == pytest.approx(5 * 0.5)


def test_vpt_with_lyapunov_normalization():
    """For Lorenz (λ=0.906), vpt_lyapunov = vpt_time × λ."""
    u_ref = np.ones((10, 3))
    u_pred = u_ref.copy()
    u_pred[3:] = 0.0
    res = valid_prediction_time(
        u_pred, u_ref, dt=0.1, threshold=0.3,
        lyapunov_exponent=LYAPUNOV_EXPONENT["lorenz"])
    assert res.vpt_step == 3
    assert res.vpt_time == pytest.approx(0.3)
    assert res.vpt_lyapunov == pytest.approx(0.3 * 0.906, rel=1e-6)


def test_vpt_rejects_non_positive_lyapunov():
    u = np.ones((5, 1))
    with pytest.raises(ValueError, match="lyapunov_exponent"):
        valid_prediction_time(u, u, dt=0.1, lyapunov_exponent=-1.0)


# ---------- spectral error --------------------------------------------------

def test_spectral_error_zero_on_identity():
    rng = np.random.default_rng(7)
    u = rng.standard_normal((128, 3))
    assert spectral_error(u, u) == pytest.approx(0.0, abs=1e-12)


def test_spectral_error_unit_on_orthogonal():
    """If u_pred is zero and u_ref is non-zero, the spectral error is
    exactly 1 (PSD(0) = 0 ⇒ ‖0 − PSD(u)‖ / ‖PSD(u)‖ = 1)."""
    u_ref = np.sin(2 * np.pi * np.linspace(0, 1, 128))[:, None] * np.ones((1, 2))
    u_pred = np.zeros_like(u_ref)
    err = spectral_error(u_pred, u_ref)
    assert err == pytest.approx(1.0, rel=1e-10)


def test_spectral_error_sine_vs_sine_shifted():
    """A pure sine at frequency f vs the same sine shifted in phase
    should give a spectral error of ~0 (PSD ignores phase).

    Uses `endpoint=False` so the sampling grid completes an exact
    integer number of cycles (4π / period 2π = 2 cycles in N samples)
    — without endpoint=False the implicit DFT period is N·dt slightly
    off-integer and the windowing leakage differs slightly between
    the two phase-shifted signals."""
    t = np.linspace(0, 4 * np.pi, 256, endpoint=False)
    u_ref = np.sin(t)[:, None]
    u_pred = np.sin(t + np.pi / 3)[:, None]  # same PSD, different phase
    err = spectral_error(u_pred, u_ref)
    assert err < 1e-6, (
        f"sine-vs-phase-shifted PSDs should match; got err={err}")


def test_spectral_error_different_frequencies_nonzero():
    """A sine at f=1 and a sine at f=4 have very different PSDs."""
    t = np.linspace(0, 4 * np.pi, 256, endpoint=False)
    u_low = np.sin(t)[:, None]
    u_high = np.sin(4 * t)[:, None]
    err = spectral_error(u_low, u_high)
    assert err > 0.5, (
        f"f=1 vs f=4 PSDs should differ significantly; got err={err}")


# ---------- invariant drift -------------------------------------------------

def test_invariant_drift_constant_invariant_is_zero():
    """If invariant_fn returns a constant, drift is exactly 0 everywhere."""
    trajectory = np.random.randn(20, 3)
    drift = invariant_drift(trajectory, lambda x: 1.0)
    assert drift.shape == (20,)
    assert np.allclose(drift, 0.0)


def test_invariant_drift_first_step_is_zero():
    """By construction, drift[0] = 0 (no drift at the reference point)."""
    trajectory = np.random.randn(15, 2) + 5.0   # away from origin
    drift = invariant_drift(trajectory, lambda x: float(np.sum(x ** 2)))
    assert drift[0] == pytest.approx(0.0, abs=1e-12)


def test_lotka_volterra_invariant_preserved_on_lv_orbit():
    """Numerically integrate the canonical LV ODE for a short time and
    confirm the invariant H = u + v - ln(u*v) drifts very little.

    We use 4th-order Runge-Kutta (the same scheme synthetic_ode uses
    when n_points is large enough). The canonical RK4 has O(dt^5)
    local truncation error; with dt=0.01 the per-step drift is
    O(10^-10), so over ~100 steps the cumulative drift is well below
    10^-5.
    """
    def rhs(state):
        u, v = state[0], state[1]
        return np.array([u * (1.0 - v), v * (u - 1.0)])

    def rk4(state, dt):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    state = np.array([1.5, 0.7])
    traj = [state.copy()]
    for _ in range(200):
        state = rk4(state, dt=0.01)
        traj.append(state.copy())
    traj = np.asarray(traj)

    drift = invariant_drift(traj, lotka_volterra_invariant)
    # Final drift should be very small for RK4 with dt=0.01.
    assert drift[-1] < 1e-5, (
        f"LV invariant drifted {drift[-1]} over 200 RK4 steps — "
        f"expected drift < 1e-5; either the invariant function is "
        f"wrong or the integrator regressed.")


def test_lotka_volterra_invariant_rejects_non_positive():
    with pytest.raises(ValueError, match="u, v > 0"):
        lotka_volterra_invariant(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="u, v > 0"):
        lotka_volterra_invariant(np.array([1.0, -0.5]))


def test_kdv_mass_and_energy_on_a_sine_field():
    """Sanity: KdV mass = ∫ u dx and energy = ∫ u² dx via Riemann sum.

    For u(x) = sin(x) on [0, 2π], n_x = 256 points, dx = 2π/256:
    - mass ≈ 0 (sin integrates to zero over a full period)
    - energy ≈ ∫sin²dx = π (half the period)
    """
    n_x = 256
    x = np.linspace(0, 2 * np.pi, n_x, endpoint=False)
    u = np.sin(x)
    dx = 2 * np.pi / n_x
    m = kdv_mass(u, dx=dx)
    e = kdv_energy(u, dx=dx)
    assert abs(m) < 1e-10, f"mass should be ≈ 0 over a full period, got {m}"
    assert e == pytest.approx(np.pi, rel=1e-3), (
        f"energy of sin² over [0, 2π] should be ≈ π, got {e}")


# ---------- Lyapunov-time normalization ------------------------------------

def test_normalize_to_lyapunov_time_scalar():
    assert normalize_to_lyapunov_time(1.0, 0.906) == pytest.approx(0.906)
    assert normalize_to_lyapunov_time(10.0, 0.906) == pytest.approx(9.06)


def test_normalize_to_lyapunov_time_array():
    t = np.array([0.5, 1.0, 2.0])
    expect = t * 0.906
    out = normalize_to_lyapunov_time(t, 0.906)
    assert np.allclose(out, expect)


def test_normalize_rejects_non_positive_le():
    with pytest.raises(ValueError, match="lyapunov_exponent"):
        normalize_to_lyapunov_time(1.0, 0.0)


# ---------- LYAPUNOV_EXPONENT table ----------------------------------------

def test_lyapunov_exponent_table_has_lorenz_with_canonical_value():
    """Lock the canonical Lorenz LE to 3 sig figs for reproducibility.
    Pre-reg §5: 'computed from the reference trajectory.'"""
    assert LYAPUNOV_EXPONENT["lorenz"] == 0.906
    # Other listed systems explicitly have no LE (non-chaotic).
    for sys in ("lotka_volterra", "van_der_pol",
                "fitzhugh_nagumo", "kuramoto"):
        assert LYAPUNOV_EXPONENT[sys] is None


# ---------- RolloutMetrics bundle smoke ------------------------------------

def test_rollout_metrics_dataclass_smoke():
    m = RolloutMetrics(
        relative_l2=0.05,
        vpt_step=42,
        vpt_time=4.2,
        vpt_lyapunov=3.8,
        spectral_error=0.12,
        invariant_drift_final=0.003,
        threshold=0.3,
    )
    assert m.relative_l2 == 0.05
    assert m.vpt_step == 42
    assert m.vpt_lyapunov == 3.8
    assert m.invariant_drift_final == 0.003


def test_rollout_metrics_allows_none_for_optional_fields():
    """Non-chaotic systems have vpt_lyapunov=None; dissipative systems
    have invariant_drift_final=None."""
    m = RolloutMetrics(
        relative_l2=0.05, vpt_step=-1, vpt_time=4.0,
        vpt_lyapunov=None, spectral_error=0.01,
        invariant_drift_final=None, threshold=0.3,
    )
    assert m.vpt_lyapunov is None
    assert m.invariant_drift_final is None
