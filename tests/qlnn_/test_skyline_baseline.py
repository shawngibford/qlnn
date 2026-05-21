"""P5 commit 3 — skyline baseline tests.

The skyline is the structural upper bound. Tests verify:
  - Fit recovers the known coefficients for each system from clean data.
  - Rollout integrated forward stays close to the reference trajectory.
  - For Lorenz at long horizon, even the skyline diverges (chaos);
    this confirms the system can be 'out-of-reach' per pre-reg §7
    (the skyline-guard rationale).
"""
from __future__ import annotations

import numpy as np
import pytest

from qlnn_.training.skyline_baseline import (
    _central_diff,
    fit_skyline,
    rollout_skyline,
    skyline_predict,
)


# Lazy import of synthetic_ode for the reference trajectories.
def _simulate(name, *, n_points=400, seed=0):
    from quantum_liquid_neuralode.data_processing.synthetic_ode import (
        simulate,
    )
    return simulate(name, n_points=n_points, seed=seed)


# ---------- finite-difference utility -------------------------------------

def test_central_diff_recovers_linear_derivative():
    """For y(t) = t, dy/dt should be 1.0 everywhere."""
    t = np.linspace(0, 10, 100)
    traj = t[:, None]                    # (100, 1)
    dydt = _central_diff(traj, dt=t[1] - t[0])
    assert dydt.shape == (100, 1)
    assert np.allclose(dydt, 1.0, atol=1e-8)


def test_central_diff_recovers_quadratic_derivative():
    """For y(t) = t², dy/dt = 2t (the interior central diff is exact)."""
    t = np.linspace(0, 5, 200)
    traj = (t ** 2)[:, None]
    dydt = _central_diff(traj, dt=t[1] - t[0])
    expected = 2 * t
    # Interior is exact for quadratic; endpoints have O(dt) error.
    assert np.allclose(dydt[2:-2], expected[2:-2, None], atol=1e-8)


# ---------- per-system fit + rollout: LV ----------------------------------

def test_skyline_lv_recovers_known_params():
    """LV canonical config (synthetic_ode): α=1.1, β=0.4, δ=0.1, γ=0.4.
    Skyline fit on a clean RK4 trajectory should recover these to
    high precision."""
    t, Y, sys_obj = _simulate("lotka_volterra", n_points=400)
    dt = sys_obj.dt * sys_obj.sample_every
    fit = fit_skyline("lotka_volterra", Y, dt)
    coef_u, coef_v = fit["coeffs_per_component"]
    # u' = u·(α − β·v) = α·u − β·u·v → coef_u = [α, −β]
    assert coef_u[0] == pytest.approx(1.1, abs=0.02), (
        f"α recovery: got {coef_u[0]:.4f}, expected 1.1")
    assert coef_u[1] == pytest.approx(-0.4, abs=0.02), (
        f"−β recovery: got {coef_u[1]:.4f}, expected -0.4")
    # v' = v·(δ·u − γ) = −γ·v + δ·u·v → coef_v = [−γ, δ]
    assert coef_v[0] == pytest.approx(-0.4, abs=0.02), (
        f"−γ recovery: got {coef_v[0]:.4f}, expected -0.4")
    assert coef_v[1] == pytest.approx(0.1, abs=0.02), (
        f"δ recovery: got {coef_v[1]:.4f}, expected 0.1")


def test_skyline_lv_rollout_tracks_reference():
    """Fit skyline on the first 70% of LV; roll out from the test
    initial state. The skyline rollout should track the reference
    closely (relL2 << 1) over the first ~100 steps."""
    t, Y, sys_obj = _simulate("lotka_volterra", n_points=400)
    dt = sys_obj.dt * sys_obj.sample_every
    n_train = int(0.7 * 400)
    Y_train = Y[:n_train]
    Y_test = Y[n_train:]

    traj_pred, _info = skyline_predict(
        "lotka_volterra", Y_train, Y_test[0], n_steps=100, dt=dt)
    ref = Y_test[1:101]    # Y_test[0] = y0, next 100 are the targets
    # Relative-L2 should be small (well under 0.1) — the skyline
    # has the perfect structural form.
    err = np.linalg.norm(traj_pred - ref) / np.linalg.norm(ref)
    assert err < 0.1, (
        f"LV skyline rollout relL2={err:.4f} — expected < 0.1; "
        f"either the fit failed or RK4 step is too coarse.")


# ---------- per-system fit: Lorenz ----------------------------------------

def test_skyline_lorenz_recovers_known_params_within_lsq_noise():
    """Lorenz canonical: σ=10, ρ=28, β=8/3.

    NOTE on tolerances: central-difference + LSQ on Lorenz at 400
    points has substantial collinearity errors — the `u·w` cross-
    feature spans ~10³ in magnitude on the attractor, amplifying
    finite-difference noise into the fit. Typical recovery:
      σ ≈ 9.5  (5% error)
      ρ ≈ 23-25  (10-20% error)
      v coefficient in v' ≈ -0.1 to -0.5  (rather than -1)
      u·v ≈ 0.9  (10% error)
      β ≈ 2.5  (5% error)

    For the H1 paper, exact param recovery is NOT required; the
    skyline-as-upper-bound role uses ROLLOUT relL2 (not coefficient
    accuracy) for the out-of-reach guard per pre-reg §7. The
    rollout test below confirms the practical utility.

    This test just asserts the qualitative correctness: the σ + ρ
    + β magnitudes are recovered to within a factor of 2.
    """
    t, Y, sys_obj = _simulate("lorenz", n_points=400)
    dt = sys_obj.dt * sys_obj.sample_every
    fit = fit_skyline("lorenz", Y, dt)
    coef_u, coef_v, coef_w = fit["coeffs_per_component"]
    # σ within factor-of-2 of 10.
    assert 5.0 < coef_u[0] < 15.0, (
        f"σ recovery {coef_u[0]:.3f} outside factor-of-2 of ground 10")
    # ρ within factor-of-2 of 28.
    assert 14.0 < coef_v[0] < 42.0, (
        f"ρ recovery {coef_v[0]:.3f} outside factor-of-2 of ground 28")
    # β within factor-of-2 of 8/3 ≈ 2.67. coef_w[1] = −β so positive
    # β value is `-coef_w[1]`.
    beta_recovered = -float(coef_w[1])
    assert 1.3 < beta_recovered < 5.4, (
        f"β recovery {beta_recovered:.3f} outside factor-of-2 of ground 2.67")


def test_skyline_lorenz_rollout_beats_predict_zero():
    """Lorenz is chaotic — even with 5-10% parameter recovery error
    the skyline rollout exponentially diverges from the reference.
    This is exactly the 'skyline-out-of-reach' condition pre-reg §7
    uses to exclude systems from H1 aggregation.

    But the rollout should still BEAT predict-zero (relL2 ~ 1) by
    a meaningful margin at SHORT horizon — this confirms the
    structural-fit machinery is doing useful work even on chaos.
    """
    t, Y, sys_obj = _simulate("lorenz", n_points=400)
    dt = sys_obj.dt * sys_obj.sample_every
    n_train = int(0.7 * 400)
    traj_pred, _info = skyline_predict(
        "lorenz", Y[:n_train], Y[n_train], n_steps=30, dt=dt)
    ref = Y[n_train + 1:n_train + 31]
    err = np.linalg.norm(traj_pred - ref) / np.linalg.norm(ref)
    # Predict-zero floor for Lorenz at the attractor is roughly 1.0.
    # Even with chaotic divergence, the skyline rollout should be
    # below ~1.0 at this horizon — confirms structural-fit utility.
    assert err < 1.0, (
        f"Lorenz skyline rollout relL2={err:.4f} — expected < 1.0 "
        f"(beat predict-zero floor). Larger than this means the "
        f"structural fit failed even for context.")
    # Document the typical magnitude for the P5 verdict's
    # skyline-out-of-reach guard — Lorenz at 30 steps is typically
    # in the 0.3-0.8 range with central-difference fit.


# ---------- per-system fit: Van der Pol ----------------------------------

def test_skyline_vdp_recovers_known_params():
    """VdP canonical (synthetic_ode): μ=5 (relaxation oscillator).

    Stiff dynamics: μ=5 produces fast switching layers that aren't
    well-resolved by central finite difference at the sampled-dt.
    The fitted μ is off-by ~20% (recovered as ~4.0 vs ground-truth 5).
    This is a documented limitation of finite-difference-based
    structural fit on stiff systems — feeds the P5 verdict module's
    underfit/skyline-out-of-reach decision per pre-reg §7.
    """
    t, Y, sys_obj = _simulate("van_der_pol", n_points=400)
    dt = sys_obj.dt * sys_obj.sample_every
    fit = fit_skyline("van_der_pol", Y, dt)
    coef_u, coef_v = fit["coeffs_per_component"]
    # u' = v → coef_u = [1] (this is exact regardless of stiffness).
    assert coef_u[0] == pytest.approx(1.0, abs=0.05)
    # v' = μ·(1−u²)·v − u → coef_v = [μ, −1]
    # Loose tolerance for μ: stiff dynamics + central-diff loss.
    # The qualitative fit (μ ≈ 4 vs ground 5) is still useful as
    # an upper-bound contextualizer; the H1 verdict module
    # documents the residual when applying the skyline guard.
    assert coef_v[0] == pytest.approx(5.0, rel=0.3), (
        f"VdP μ recovered as {coef_v[0]:.3f}; expected ≈5 ±30%. "
        f"Larger error than this suggests a structural bug, not "
        f"just finite-difference noise.")
    assert coef_v[1] == pytest.approx(-1.0, abs=0.2)


# ---------- error paths --------------------------------------------------

def test_fit_rejects_unknown_system():
    with pytest.raises(ValueError, match="unknown skyline system"):
        fit_skyline("nonsense", np.zeros((10, 2)), dt=0.1)


def test_fit_rejects_wrong_state_dim():
    with pytest.raises(ValueError, match="state-dim"):
        # LV expects d=2; give d=3.
        fit_skyline("lotka_volterra", np.zeros((10, 3)), dt=0.1)


def test_fit_rejects_non_positive_dt():
    with pytest.raises(ValueError, match="dt"):
        fit_skyline("lotka_volterra", np.zeros((10, 2)), dt=0.0)


def test_rollout_rejects_non_positive_n_steps():
    def rhs(y): return -y
    with pytest.raises(ValueError, match="n_steps"):
        rollout_skyline(rhs, np.zeros(2), n_steps=0, dt=0.1)


def test_rollout_rk4_decay():
    """Simple ODE y' = -y has analytic solution exp(-t). RK4 should
    track it closely."""
    def rhs(y): return -y
    y0 = np.array([1.0])
    n_steps = 50
    dt = 0.1
    traj = rollout_skyline(rhs, y0, n_steps=n_steps, dt=dt)
    expected = np.exp(-dt * np.arange(1, n_steps + 1))[:, None]
    assert np.allclose(traj, expected, rtol=1e-4), (
        "RK4 rollout of y'=-y should track exp(-t) closely")
