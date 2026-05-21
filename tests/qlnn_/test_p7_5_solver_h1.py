"""P7.5 commit 1 — Solver-task H1 module tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from qlnn_.training.classical_pinn_solver import (
    MLPConfig, init_classical_pinn_weights,
)
from qlnn_.training.p7_5_solver_h1 import (
    _make_classical_pinn_vector_residual_loss,
    load_p36_qlnn_best,
    train_classical_pinn_solver_one_cell,
)
from qlnn_.training.multi_state_solver import VECTOR_ODES


# ---------- vector-residual loss schema -----------------------------------

def test_vector_residual_loss_returns_scalar():
    """The residual loss must reduce (T_colloc, d) field errors to a scalar."""
    import jax
    import jax.numpy as jnp
    cfg = MLPConfig(input_dim=1, output_dim=2, hidden_layers=2,
                    target_param_count=40)
    w = init_classical_pinn_weights(cfg, seed=0)
    bench = VECTOR_ODES["lotka_volterra"]
    loss_fn, u_of_t = _make_classical_pinn_vector_residual_loss(
        cfg, bench.rhs_jax,
        t0=bench.t0, t1=bench.t1, u0=jnp.asarray(bench.u0))
    t_c = jnp.linspace(bench.t0, bench.t1, 12)[1:-1]
    L = loss_fn(w, t_c)
    assert jnp.ndim(L) == 0
    assert math.isfinite(float(L))


def test_lagaris_trial_at_t0_returns_u0():
    """The Lagaris hard-IC u(t0) = u₀ must hold structurally."""
    import jax.numpy as jnp
    cfg = MLPConfig(input_dim=1, output_dim=2, hidden_layers=2)
    w = init_classical_pinn_weights(cfg, seed=42)
    bench = VECTOR_ODES["lotka_volterra"]
    _, u_of_t = _make_classical_pinn_vector_residual_loss(
        cfg, bench.rhs_jax,
        t0=bench.t0, t1=bench.t1, u0=jnp.asarray(bench.u0))
    u_at_t0 = u_of_t(jnp.asarray(bench.t0), w)
    assert jnp.allclose(u_at_t0, jnp.asarray(bench.u0), atol=1e-10), (
        f"Lagaris hard-IC violated at t0: u={u_at_t0}, u₀={bench.u0}")


# ---------- training smoke ------------------------------------------------

@pytest.mark.parametrize("system", [
    "lotka_volterra", "van_der_pol", "lorenz",
])
def test_classical_pinn_solver_runs_end_to_end(system):
    """Smoke: tiny budget — confirm the trainer runs and produces
    finite output. Numerical accuracy is NOT asserted here (real
    convergence is the sweep's job)."""
    r = train_classical_pinn_solver_one_cell(
        system, seed=0, n_colloc=10, steps=20, lr=0.02,
        target_param_count=20)
    assert r["family"] == "classical_pinn"
    assert r["system"] == system
    assert r["dim"] == VECTOR_ODES[system].dim
    assert math.isfinite(r["relative_l2"])
    assert math.isfinite(r["train_relative_l2"])
    assert r["u_pred"].shape == r["u_ref"].shape
    assert len(r["loss_history"]) == 20


def test_classical_pinn_solver_reports_train_side_relL2():
    """The underfit guard requires train-side relative-L2 to be
    reported (R2 fix from P7.5)."""
    r = train_classical_pinn_solver_one_cell(
        "lotka_volterra", seed=0, n_colloc=10, steps=10,
        target_param_count=20)
    assert "train_relative_l2" in r
    assert math.isfinite(r["train_relative_l2"])
    assert r["train_relative_l2"] >= 0


# ---------- P3.6 QLNN best loader ----------------------------------------

@pytest.mark.parametrize("system", [
    "lotka_volterra", "van_der_pol", "lorenz",
])
def test_load_p36_qlnn_best_returns_finite_relL2(system):
    """The P3.6 multi-state QLNN solver data is on disk; the loader
    must return a finite relative-L2 value for every seed of every
    system in our P7.5 sweep."""
    for seed in (0, 1, 2):
        v, fam = load_p36_qlnn_best(system, seed)
        assert v is not None, (
            f"P3.6 data missing for {system} seed={seed} — required for "
            f"the solver-task H1 verdict")
        assert math.isfinite(v)
        assert fam in ("chebyshev_dqc", "te_qpinn_fnn",
                       "te_qpinn_qnn", "qcpinn")


def test_load_p36_qlnn_best_returns_minimum_across_families():
    """The loader returns the BEST (lowest relL2) family per cell."""
    v, fam = load_p36_qlnn_best("lotka_volterra", 0)
    # Verify by manually loading all 4 families and computing min.
    import json
    from qlnn_.training.p7_5_solver_h1 import P36_RESULTS
    manual_min_v = float("inf")
    manual_min_fam = None
    for f in ("chebyshev_dqc", "te_qpinn_fnn", "te_qpinn_qnn", "qcpinn"):
        p = P36_RESULTS / f"{f}_lotka_volterra" / "seed_0" / "metrics.json"
        if p.exists():
            mv = float(json.loads(p.read_text())["relative_l2"])
            if mv < manual_min_v:
                manual_min_v = mv
                manual_min_fam = f
    assert v == manual_min_v
    assert fam == manual_min_fam


# ---------- error paths ---------------------------------------------------

def test_rejects_unknown_system():
    with pytest.raises(ValueError, match="unknown system"):
        train_classical_pinn_solver_one_cell(
            "nonsense", seed=0, n_colloc=5, steps=5)
