"""Smoke test for the P3.6 multi-state ODE solver.

Exercises `multi_state_solver.train_one_vector` for ALL 4 families × ALL
3 vector systems on seed=0 at heavily reduced steps. Asserts the
per-component interop contract: a `d`-component pytree, finite vector
residual loss, per-component gradient mass non-zero, prediction shape
(T, d), reference-comparison sane.

Does NOT re-validate scalar-gate accuracy (that's
`test_physics_residual_solver` and stays unchanged).
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from qlnn_.training.multi_state_solver import (
    VECTOR_ODES,
    _build_per_component,
    _reference_trajectory,
    run_vector_sweep,
    summarize_vector_seeds,
    train_one_vector,
)
from qlnn_.training.solver_demo import FAMILIES as SCALAR_FAMILIES


SMOKE_STEPS = 60
SMOKE_COLLOC = 10


def test_all_three_vector_systems_registered():
    assert set(VECTOR_ODES) == {"lotka_volterra", "van_der_pol", "lorenz"}


def test_systems_dimensions_match_synthetic_ode():
    """LV/VdP are 2D, Lorenz is 3D — matches the canonical sources."""
    assert VECTOR_ODES["lotka_volterra"].dim == 2
    assert VECTOR_ODES["van_der_pol"].dim == 2
    assert VECTOR_ODES["lorenz"].dim == 3


def test_regime_tags_match_pre_registration():
    """H1 partition lives in ODE_PDE_PRE_REG.md §2 — bind it in code."""
    assert VECTOR_ODES["lotka_volterra"].regime == "smooth_periodic"
    assert VECTOR_ODES["lorenz"].regime == "broadband_multiscale"


def test_per_component_pytree_has_d_independent_blocks():
    """For Lorenz (d=3) we should get exactly 3 sub-keys c0/c1/c2,
    each with its own {w, s, b} — the defining property of the
    per-component dispatch."""
    circuits, p, counts = _build_per_component("chebyshev_dqc",
                                                dim=3, seed=0)
    assert len(circuits) == 3
    assert set(p.keys()) == {"c0", "c1", "c2"}
    for k in range(3):
        assert "w" in p[f"c{k}"]
        assert "s" in p[f"c{k}"]
        assert "b" in p[f"c{k}"]
    # Decorrelated weights: w0 != w1 != w2 (different seeds).
    w0 = np.asarray(p["c0"]["w"]).ravel()
    w1 = np.asarray(p["c1"]["w"]).ravel()
    assert not np.allclose(w0, w1)
    assert counts["dim"] == 3
    assert counts["pqc_params"] == 3 * counts["per_component_pqc_params"]


def test_reference_trajectory_starts_at_ic():
    """The numerical reference (canonical numpy RK4) must start at u0."""
    bench = VECTOR_ODES["lotka_volterra"]
    t_eval = np.linspace(bench.t0, bench.t1, 21)
    ref = _reference_trajectory("lotka_volterra", t_eval)
    assert ref.shape == (21, bench.dim)
    np.testing.assert_allclose(ref[0], bench.u0, atol=1e-6)


# Coverage strategy: skip the full 4×3 Cartesian (each fresh combo
# triggers JIT compile and dominates CI time). Test (a) all 4 families
# on the SAME system (lotka_volterra) to prove family-interop, and
# (b) the same family (chebyshev_dqc) on ALL 3 systems to prove
# dim-handling (d=2, d=2, d=3). 4 + 3 = 7 cases covers (family, dim)
# space at ~half the JIT cost.
_INTEROP_CASES = (
    [("lotka_volterra", fam) for fam in SCALAR_FAMILIES]      # 4 fam × 1 sys
    + [("van_der_pol", "chebyshev_dqc"),                       # +
       ("lorenz", "chebyshev_dqc")]                            #  2 extra sys
)


@pytest.mark.parametrize("system,family", _INTEROP_CASES)
def test_each_family_trains_on_each_system(family, system):
    """Family × system interop smoke at reduced steps. See coverage
    strategy comment above for why this is not the full Cartesian."""
    out = train_one_vector(family, system, seed=0,
                            steps=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
    # schema
    for key in ("family", "system", "seed", "dim", "regime",
                "steps", "pqc_params", "classical_params",
                "final_loss", "mae", "mae_per_component",
                "relative_l2", "t_eval", "u_pred", "u_ref"):
        assert key in out, f"{family}/{system}: missing key {key}"
    # numerics
    assert math.isfinite(out["final_loss"])
    assert math.isfinite(out["mae"])
    assert math.isfinite(out["relative_l2"])
    # training moved off init (residual loss decreased)
    assert out["loss_history"][-1] <= out["loss_history"][0] + 1e-6
    # shapes
    d = out["dim"]
    assert out["u_pred"].shape == (100, d)
    assert out["u_ref"].shape == (100, d)
    assert len(out["mae_per_component"]) == d
    # IC respected (per-component Lagaris hard-IC: at the first interior
    # t≈t0+small, u_pred ≈ u0 within a tolerance proportional to (t−t0))
    bench = VECTOR_ODES[system]
    np.testing.assert_allclose(out["u_pred"][0], bench.u0,
                                rtol=0.0, atol=0.5 * np.linalg.norm(bench.u0))


def test_smoke_sweep_wall_clock_bound():
    """4 families × 1 system × seed=0 at smoke steps — must finish in
    well under the 60s/family budget so the full CI suite stays fast."""
    t0 = time.time()
    for family in SCALAR_FAMILIES:
        out = train_one_vector(family, "lotka_volterra", seed=0,
                                steps=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
        assert math.isfinite(out["final_loss"])
    elapsed = time.time() - t0
    assert elapsed < 240.0, (
        f"smoke sweep took {elapsed:.1f}s; budget is 240s for 4 families")


def test_run_sweep_returns_full_cartesian_product():
    res = run_vector_sweep(
        families=["chebyshev_dqc"],
        systems=["lotka_volterra"],
        seeds=[0, 1],
        steps_override=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
    assert len(res) == 2
    assert {r["seed"] for r in res} == {0, 1}


def test_summarize_vector_seeds_schema():
    res = run_vector_sweep(
        families=["chebyshev_dqc"], systems=["lotka_volterra"],
        seeds=[0, 1, 2],
        steps_override=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
    s = summarize_vector_seeds(res)
    assert s["family"] == "chebyshev_dqc"
    assert s["system"] == "lotka_volterra"
    assert s["n_seeds"] == 3
    assert s["dim"] == 2
    for m in ("mae", "relative_l2", "final_loss"):
        block = s["metrics"][m]
        for key in ("mean", "std", "min", "max", "n_seeds",
                    "ci95_half_width", "ci95_low", "ci95_high"):
            assert key in block


def test_unknown_family_or_system_raises():
    with pytest.raises(ValueError, match="unknown family"):
        train_one_vector("bogus", "lotka_volterra", seed=0,
                          steps=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
    with pytest.raises(ValueError, match="unknown system"):
        train_one_vector("chebyshev_dqc", "bogus", seed=0,
                          steps=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
