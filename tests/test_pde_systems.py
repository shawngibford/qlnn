"""Tests for the synthetic PDE benchmark generators.

Same trust standard as `test_synthetic_ode.py`: determinism, shape/finite
sanity, the npz field-artifact round-trip, the H1 regime tags being bound
to the pre-registration — and, most important, each system validated
against its KNOWN ANALYTIC behavior:

  - viscous Burgers  : gradient catastrophe near the inviscid shock time
                       t* = -1/min u0' = 1 (shock regime) vs no
                       catastrophe (smooth regime).
  - Allen-Cahn       : an over-narrow front relaxes toward the
                       equilibrium interface width √2·eps while staying
                       stationary; the Ginzburg-Landau energy is a
                       non-increasing Lyapunov functional.
  - KdV              : an exact sech² soliton conserves mass ∫u dx and
                       momentum ∫u² dx and translates at its speed c.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantum_liquid_neuralode.data_processing.pde_systems import (
    PDE_SYSTEMS,
    allen_cahn_front_width,
    burgers_inviscid_shock_time,
    compute_invariants,
    get_pde_system,
    kdv_soliton,
    make_pde_npz,
    simulate_pde,
)


def test_all_pde_systems_registered():
    assert PDE_SYSTEMS == ["burgers_smooth", "burgers_shock",
                           "allen_cahn", "kdv"]
    for name in PDE_SYSTEMS:
        get_pde_system(name)  # must not raise


def test_unknown_pde_raises():
    with pytest.raises(ValueError, match="unknown PDE system"):
        get_pde_system("navier_stokes")


def test_regime_tags_match_pre_registration():
    # H1 partition (ODE_PDE_PRE_REG.md §2) — bound in code so a drift
    # from the pre-registration fails the suite.
    assert get_pde_system("burgers_smooth").regime == "smooth_periodic"
    for name in ("burgers_shock", "allen_cahn", "kdv"):
        assert get_pde_system(name).regime == "broadband_multiscale"


@pytest.mark.parametrize("name", PDE_SYSTEMS)
def test_simulation_is_deterministic(name):
    t1, x1, U1, _ = simulate_pde(name)
    t2, x2, U2, _ = simulate_pde(name)
    np.testing.assert_array_equal(U1, U2)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(x1, x2)


@pytest.mark.parametrize("name", PDE_SYSTEMS)
def test_field_shape_and_finite(name):
    t, x, U, sys = simulate_pde(name)
    assert U.shape == (sys.n_frames, sys.n_x)
    assert x.shape == (sys.n_x,)
    assert t.shape == (sys.n_frames,)
    assert np.all(np.isfinite(U))
    assert np.isrealobj(U)
    # IC is reproduced exactly in frame 0.
    np.testing.assert_allclose(U[0], sys.ic(x), rtol=0, atol=0)


def _max_abs_gradient_over_time(x, U):
    dx = x[1] - x[0]
    # periodic central difference, per frame
    gx = (np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2.0 * dx)
    return np.max(np.abs(gx), axis=1)


def test_burgers_shock_formation_time():
    # Analytic: inviscid Burgers with u0=sin(x) shocks at t* = 1.
    assert burgers_inviscid_shock_time(-1.0) == pytest.approx(1.0)
    assert np.isinf(burgers_inviscid_shock_time(0.5))

    t_s, x_s, U_s, _ = simulate_pde("burgers_shock")
    t_m, x_m, U_m, _ = simulate_pde("burgers_smooth")

    g_s = _max_abs_gradient_over_time(x_s, U_s)
    g_m = _max_abs_gradient_over_time(x_m, U_m)

    init_grad = g_s[0]  # = max|cos x| ≈ 1
    # SHOCK regime: a genuine gradient catastrophe — the peak gradient
    # blows up far past the initial slope, and it happens near t*=1.
    assert g_s.max() / init_grad > 10.0
    t_peak = t_s[int(np.argmax(g_s))]
    assert 0.7 <= t_peak <= 1.7, f"shock peak at t={t_peak}, expect ≈1"

    # SMOOTH regime: strong viscosity prevents the catastrophe.
    assert g_m.max() / init_grad < 5.0
    # And the regimes are well separated.
    assert g_s.max() / g_m.max() > 4.0


def test_burgers_mass_conserved():
    # Conservative flux form ⇒ ∫u dx invariant to spectral round-off.
    for name in ("burgers_smooth", "burgers_shock"):
        _, x, U, _ = simulate_pde(name)
        mass = compute_invariants(name, x, U)["mass"]
        drift = np.max(np.abs(mass - mass[0]))
        assert drift < 1e-8, f"{name} mass drift {drift}"


def test_allen_cahn_front_relaxation_and_stationarity():
    eps = get_pde_system("allen_cahn").params["eps"]
    w_eq = allen_cahn_front_width(eps)            # √2·eps
    assert w_eq == pytest.approx(np.sqrt(2.0) * eps)

    t, x, U, sys = simulate_pde("allen_cahn")
    Lx = sys.domain_length

    def up_front_width(u):
        # near the up-front at Lx/4, for u≈tanh((x-x0)/w): max u_x = 1/w
        sel = (x > Lx / 8.0) & (x < 3.0 * Lx / 8.0)
        dx = x[1] - x[0]
        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
        return 1.0 / np.max(np.abs(ux[sel]))

    def up_front_center(u):
        sel = (x > Lx / 8.0) & (x < 3.0 * Lx / 8.0)
        xs, us = x[sel], u[sel]
        i = int(np.argmin(np.abs(us)))            # zero crossing
        return xs[i]

    w0 = up_front_width(U[0])
    wf = up_front_width(U[-1])
    # Starts deliberately too narrow (0.6·√2eps), must RELAX toward √2eps.
    assert w0 < 0.8 * w_eq
    assert abs(wf - w_eq) / w_eq < 0.20
    assert wf > w0                                # widened, didn't shrink

    # Equal-well ⇒ the (symmetric) front stays put.
    drift = abs(up_front_center(U[-1]) - up_front_center(U[0]))
    assert drift < 5.0 * (x[1] - x[0])

    # Ginzburg-Landau energy is a non-increasing Lyapunov functional,
    # and it actually decreased (we started out of equilibrium).
    E = compute_invariants("allen_cahn", x, U)["ginzburg_landau_energy"]
    assert np.all(np.diff(E) <= 1e-7 * abs(E[0]))
    assert E[-1] < E[0]


def test_kdv_soliton_conservation_and_speed():
    t, x, U, sys = simulate_pde("kdv")
    c = sys.params["c"]
    x0 = sys.params["x0"]
    Lx = sys.domain_length

    inv = compute_invariants("kdv", x, U)
    for q in ("mass", "momentum"):
        s = inv[q]
        rel = np.max(np.abs(s - s[0])) / abs(s[0])
        assert rel < 5e-3, f"KdV {q} relative drift {rel}"

    # Soliton retains its amplitude c/2.
    amp = U[-1].max()
    assert abs(amp - 0.5 * c) / (0.5 * c) < 0.05

    # …and translated at speed c: peak position ≈ (x0 + c·T) mod L.
    T = t[-1]
    peak_x = x[int(np.argmax(U[-1]))]
    expected = (x0 + c * T) % Lx
    err = abs((peak_x - expected + Lx / 2.0) % Lx - Lx / 2.0)
    assert err < 1.0, f"KdV soliton at {peak_x}, expected {expected}"

    # Final field matches the exact analytic soliton (phase-aligned).
    ref = kdv_soliton(x, T, c, x0, Lx)
    rel_l2 = np.linalg.norm(U[-1] - ref) / np.linalg.norm(ref)
    assert rel_l2 < 0.10, f"KdV shape rel-L2 {rel_l2}"


def test_npz_field_artifact_roundtrip(tmp_path):
    p = tmp_path / "kdv.npz"
    meta = make_pde_npz("kdv", str(p))
    assert p.exists()
    d = np.load(str(p))
    for key in ("u", "x", "t", "u0", "meta_json", "invariants_json"):
        assert key in d
    _, _, U, sys = simulate_pde("kdv")
    np.testing.assert_array_equal(d["u"], U)
    np.testing.assert_array_equal(d["u0"], U[0])
    assert d["u"].shape == (sys.n_frames, sys.n_x)
    assert meta["bc"] == "periodic"
    assert meta["regime"] == "broadband_multiscale"
    assert "mass" in meta["invariant_names"]
    # provenance hash is deterministic
    assert meta["sha256"] == make_pde_npz("kdv", str(tmp_path / "k2.npz"))["sha256"]
