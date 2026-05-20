"""P3.7 PDE solver acceptance gate — TWO complementary checks.

The make-or-break of the entire PDE side of the H1 hypothesis space:

1. **Mechanism gate** (deterministic, ≤15s):
   `test_mixed_jacrev_through_qnode_is_finite_and_nontrivial` — asserts
   that `jax.jacrev(jax.jacrev(QNode))` returns finite, non-trivial
   second-derivative values at random init. If this fails, that is the
   REAL Risk-#2-redux confirmation: nested mixed-mode autodiff through
   PennyLane's JAX QNode is structurally broken, and PDE work stops.

2. **Convergence gate** (heat equation, lives in this file, separate
   test): trains the 2D solver on `u_t = ν u_xx` with `u(0,x)=sin(x)`,
   asserts interior MAE < 0.10 vs the analytic `e^{−νt}·sin(x)` at
   seed 0. If (1) passes but (2) fails, that's a tuning problem fixed
   within the phase (qubit count / steps / lr), NOT a phase blocker.

Plus the standard schema / smoke tests for the new module's contract.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qlnn_.training.pde_residual_loss import (
    ChebyshevDQC2DConfig,
    build_chebyshev_dqc_2d,
    init_pde_solver_params,
    make_pde_residual_loss,
    pde_solver_gate_heat,
    train_pde_solver,
)


# ---------- schema + shape contract ----------------------------------------

def test_config_validates():
    with pytest.raises(ValueError):
        ChebyshevDQC2DConfig(n_t_qubits=0)
    with pytest.raises(ValueError):
        ChebyshevDQC2DConfig(n_x_qubits=0)
    with pytest.raises(ValueError):
        ChebyshevDQC2DConfig(num_layers=0)
    with pytest.raises(ValueError):
        ChebyshevDQC2DConfig(entanglement="bogus")


def test_2d_circuit_weight_shape_and_scalar_output():
    cfg = ChebyshevDQC2DConfig(n_t_qubits=3, n_x_qubits=3, num_layers=2)
    assert cfg.num_qubits == 6
    assert cfg.weight_shape == (2, 6, 3)
    circ = build_chebyshev_dqc_2d(cfg)
    p = init_pde_solver_params(cfg.weight_shape, seed=0)
    y = circ(jnp.asarray(0.3), jnp.asarray(-0.4), p["w"])
    assert jnp.ndim(y) == 0
    # Total magnetization Σ⟨Z⟩ ∈ [-n, n]
    assert -cfg.num_qubits - 1e-6 <= float(y) <= cfg.num_qubits + 1e-6


# ---------- THE MECHANISM GATE (Risk-#2-redux) -----------------------------

def test_mixed_jacrev_through_qnode_is_finite_and_nontrivial():
    """The Risk-#2-redux check. If this fails, PDE work stops.

    Asserts that `jax.jacrev(jax.jacrev(QNode, argnums=1), argnums=1)`
    returns finite, non-trivial values. PennyLane's JAX interface uses
    `vjp` (not Diffrax's `custom_vjp`), so reverse-over-reverse should
    compose, but this is the first test of it in this repo.
    """
    cfg = ChebyshevDQC2DConfig(n_t_qubits=3, n_x_qubits=3, num_layers=2)
    circ = build_chebyshev_dqc_2d(cfg)
    p = init_pde_solver_params(cfg.weight_shape, seed=0)

    def f(t, x):
        return circ(t, x, p["w"])

    # First spatial derivative.
    fx = jax.jacrev(f, argnums=1)
    val_fx = fx(jnp.asarray(0.3), jnp.asarray(-0.4))
    assert np.isfinite(float(val_fx))

    # Second spatial derivative — reverse-over-reverse.
    fxx = jax.jacrev(fx, argnums=1)
    val_fxx = fxx(jnp.asarray(0.3), jnp.asarray(-0.4))
    assert np.isfinite(float(val_fxx)), (
        "jacrev(jacrev(QNode, argnums=1), argnums=1) returned non-finite "
        "— Risk-#2-redux confirmed; PDE work blocked.")

    # The 2nd derivative must be NON-TRIVIAL (a constant circuit gives 0).
    # Probe at a couple of points and assert |fxx| > tiny tol somewhere.
    probes = [(jnp.asarray(0.3), jnp.asarray(-0.4)),
              (jnp.asarray(-0.5), jnp.asarray(0.2)),
              (jnp.asarray(0.7), jnp.asarray(-0.1))]
    max_abs = max(float(jnp.abs(fxx(t, x))) for t, x in probes)
    assert max_abs > 1e-6, (
        f"jacrev∘jacrev returns ~0 everywhere (max |fxx| = {max_abs}) — "
        f"the Chebyshev tower may be degenerate or the autodiff path "
        f"may be silently dropping the 2nd-order term.")


def test_mixed_grad_through_loss_with_jacrev_jacrev_is_finite():
    """Inner check: not just the 2nd derivative, but ∂/∂θ of a loss
    that USES the 2nd derivative. This is the actual pattern
    train_pde_solver runs."""
    cfg = ChebyshevDQC2DConfig(n_t_qubits=3, n_x_qubits=3, num_layers=2)
    circ = build_chebyshev_dqc_2d(cfg)
    p = init_pde_solver_params(cfg.weight_shape, seed=0)

    def rhs_heat(t, x, u, ut, ux, uxx):
        return ut - 0.1 * uxx

    def ic_sin(x):
        return jnp.sin(x)

    loss_fn, _ = make_pde_residual_loss(
        circ, rhs_heat, ic_sin,
        t0=0.0, t1=1.0, x0=0.0, x1=2.0 * np.pi)

    # Tiny collocation grid for speed.
    ts = jnp.linspace(0.1, 0.9, 4)
    xs = jnp.linspace(0.5, 5.5, 4)
    T, X = jnp.meshgrid(ts, xs, indexing="ij")
    tx = jnp.stack([T.ravel(), X.ravel()], axis=1)

    val, grads = jax.value_and_grad(loss_fn)(p, tx)
    assert np.isfinite(float(val))
    for leaf in jax.tree_util.tree_leaves(grads):
        arr = np.asarray(leaf)
        assert np.all(np.isfinite(arr))
    gnorm = sum(float(jnp.sum(jnp.abs(g)))
                for g in jax.tree_util.tree_leaves(grads))
    assert gnorm > 1e-6, "grad through loss-with-jacrev∘jacrev is trivial"


# ---------- IC structural constraint smoke ---------------------------------

def test_ic_is_hard_constrained_via_lagaris_form():
    """At t=t0, the trial solution must equal u₀(x) exactly (no soft
    penalty needed). The Lagaris form u(t,x)=u₀(x)+(t-t0)·N(t,x)
    enforces this structurally."""
    cfg = ChebyshevDQC2DConfig(n_t_qubits=2, n_x_qubits=2, num_layers=1)
    circ = build_chebyshev_dqc_2d(cfg)
    p = init_pde_solver_params(cfg.weight_shape, seed=42)
    _, u_of_tx = make_pde_residual_loss(
        circ, lambda *args: jnp.asarray(0.0), lambda x: jnp.sin(x),
        t0=0.0, t1=1.0, x0=0.0, x1=2.0 * np.pi)

    # At t=t0=0, must equal sin(x) exactly for any x and any weights.
    for x in jnp.linspace(0.5, 5.5, 5):
        u_at_t0 = float(u_of_tx(jnp.asarray(0.0), x, p))
        expected = float(jnp.sin(x))
        assert abs(u_at_t0 - expected) < 1e-5, (
            f"IC violated at x={float(x):.2f}: u(0,x)={u_at_t0:.4f} "
            f"≠ sin(x)={expected:.4f}")


# ---------- train_pde_solver schema (does NOT run the gate) ----------------

def test_train_pde_solver_returns_correct_shapes_at_smoke_steps():
    """Smoke: train_pde_solver runs end-to-end at tiny budget."""
    cfg = ChebyshevDQC2DConfig(n_t_qubits=2, n_x_qubits=2, num_layers=1)
    circ = build_chebyshev_dqc_2d(cfg)

    def rhs(t, x, u, ut, ux, uxx):
        return ut - 0.1 * uxx

    res = train_pde_solver(
        circ, rhs, lambda x: jnp.sin(x),
        t0=0.0, t1=1.0, x0=0.0, x1=2.0 * np.pi,
        weight_shape=cfg.weight_shape,
        n_t_colloc=6, n_x_colloc=6, n_t_eval=10, n_x_eval=10,
        steps=20, lr=0.02, seed=0)
    assert res.u_pred.shape == (10, 10)
    assert math.isfinite(res.final_loss)
    assert len(res.loss_history) == 20


# ---------- THE CONVERGENCE GATE (heat equation MAE < 0.10) ----------------

def test_heat_equation_gate_recovers_analytic_at_seed_0():
    """P3.7 ACCEPTANCE GATE — convergence half.

    Solve u_t = ν u_xx with u(0,x)=sin(x), x ∈ [0, 2π), t ∈ [0, 1].
    Exact solution: u(t,x) = e^{−νt}·sin(x). Trained purely by PDE
    residual; assert interior MAE < 0.10 vs analytic at seed 0
    (deterministic on CPU).

    Threshold rationale: the 2D split-qubit feature map has ~half
    the per-coord expressivity of the 1D gate (0.02 threshold), so
    0.10 in 2D is comparable in stringency. 0.10 is still strong
    evidence of recovery — a constant baseline on the e^{−νt}·sin(x)
    target has MAE ≈ 0.3.
    """
    res, mae = pde_solver_gate_heat(
        nu=0.1, n_t_qubits=4, n_x_qubits=4, num_layers=5,
        steps=1200, seed=0)
    # Sanity: training actually moved the residual.
    assert res.loss_history[-1] < res.loss_history[0]
    assert math.isfinite(res.final_loss)
    # The gate.
    assert mae < 0.10, (
        f"heat-equation convergence gate FAILED: MAE={mae:.4f} >= 0.10. "
        f"If the mechanism gate passes, this is a tuning issue (steps / "
        f"qubit count / lr) to fix within P3.7 — NOT a Risk-#2 blocker.")
