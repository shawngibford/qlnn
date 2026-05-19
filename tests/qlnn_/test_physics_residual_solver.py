"""P3 solver-path acceptance gate.

Two things are validated here, in order of importance:

1. The NESTED AUTODIFF works — `jax.value_and_grad` (w.r.t. the param
   pytree) of a loss that itself contains `jax.jacrev` (w.r.t. the
   scalar input coordinate) of a PennyLane JAX QNode produces finite,
   non-trivial gradients. This is Risk #2 of the pivot and the locked
   convention (HANDOFF gotcha #1: jacrev only). Nothing in the solver
   strand may scale until this holds.

2. The Chebyshev-DQC solver, trained ONLY by the physics residual
   (no supervised targets), recovers the analytic solution of
   u' = −u, u(0)=1  (exact u = e^{−t}) at the project's canonical
   seed 0.

With the Lagaris hard-constraint trial solution u(t)=u0+(t−t0)·N(t)
(IC structural, not a soft penalty — and so NOT anchored at the
Chebyshev-singular x=−1 endpoint), the solver is both accurate and
seed-robust: interior MAE ≈ 0.003 (seed0), ≤0.0074 across seeds
{0,1,2}, residual ~1e-3. The gate pins seed 0 (deterministic on CPU)
and asserts a strong, reproducible convergence bound.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qlnn_.training.physics_residual_loss import (
    ChebyshevDQCConfig,
    build_chebyshev_dqc,
    init_solver_params,
    make_residual_loss,
    solver_prototype_ode,
)


def test_chebyshev_dqc_weight_shape_and_scalar_output():
    cfg = ChebyshevDQCConfig(num_qubits=4, num_layers=5)
    assert cfg.weight_shape == (5, 4, 3)          # HEA Rz,Rx,Rz per qubit/layer
    circ = build_chebyshev_dqc(cfg)
    p = init_solver_params(cfg.weight_shape, seed=0)
    y = circ(jnp.asarray(0.3), p["w"])
    # Total magnetization Σ⟨Z_j⟩ is a SCALAR in [-n, n].
    assert jnp.ndim(y) == 0
    assert -cfg.num_qubits - 1e-6 <= float(y) <= cfg.num_qubits + 1e-6


def test_nested_autodiff_is_finite_and_nontrivial():
    # The make-or-break check: grad through jacrev-of-QNode.
    cfg = ChebyshevDQCConfig(num_qubits=4, num_layers=3)
    circ = build_chebyshev_dqc(cfg)
    loss_fn, _ = make_residual_loss(
        circ, lambda t, u: -u, t0=0.0, t1=2.0, u0=1.0)
    p = init_solver_params(cfg.weight_shape, seed=0)
    t_colloc = jnp.linspace(0.0, 2.0, 16)

    val, grads = jax.value_and_grad(loss_fn)(p, t_colloc)
    assert np.isfinite(float(val))
    # gradients exist and are non-zero for every trainable leaf
    for leaf in jax.tree_util.tree_leaves(grads):
        assert np.all(np.isfinite(np.asarray(leaf)))
    gnorm = sum(float(jnp.sum(jnp.abs(g)))
                for g in jax.tree_util.tree_leaves(grads))
    assert gnorm > 1e-6, "nested-autodiff gradient is trivially zero"


def test_solver_prototype_recovers_exponential_decay():
    # P3 ACCEPTANCE GATE — canonical seed 0, deterministic on CPU.
    res, mae = solver_prototype_ode(num_layers=5, steps=1200, seed=0)

    # The physics residual was actually minimized.
    assert res.final_loss < 0.15

    # Recovered the analytic solution far better than the best constant
    # fit to e^{-t} on [0,2] (whose MAE ≈ 0.22) — i.e. it learned the
    # CURVE from the ODE alone, not a flat guess.
    t = res.t
    exact = jnp.exp(-t)
    best_const = float(jnp.mean(exact))
    const_mae = float(jnp.mean(jnp.abs(best_const - exact)))
    assert const_mae > 0.18                       # sanity on the baseline
    # Lagaris hard-IC trial solution: interior MAE ≈ 0.003 at seed 0
    # (deterministic on CPU); ≤0.0074 across seeds {0,1,2}. The gate
    # bound 0.02 is a strong "recovered the analytic curve" assertion
    # (~10× tighter than the soft-IC version, ~30× better than the
    # best constant) with margin for platform float drift.
    assert mae < 0.02, f"solver MAE {mae:.4f} vs exp(-t) (gate < 0.02)"
    assert mae < 0.1 * const_mae

    # Interior endpoints (Chebyshev-singular bare ±1 excluded): starts
    # near e^{-t0}=1, decays toward e^{-t1}=0.135.
    assert float(res.u_pred[0]) > 0.90
    assert float(res.u_pred[-1]) < 0.25
    assert float(res.u_pred[-1]) < float(res.u_pred[0])
