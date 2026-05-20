"""P3.8 — Classical MLP-PINN smoke + drop-in interop tests.

Validates that the classical PINN module:
1. Produces a capacity-matched MLP within 2× of the target param count
   (the pre-reg's matched-comparison rule).
2. Is a DROP-IN replacement for the quantum circuit in both
   `make_residual_loss` (1D ODE) and `make_pde_residual_loss` (2D PDE)
   — same `{w, s, b}` pytree contract, same Lagaris hard-IC.
3. The IC structural property holds (u(t0, x) = u₀(x) exactly).
4. Nested jax.jacrev through the MLP returns finite values (the
   classical analog of P3.7's mechanism gate; should be trivially
   true for an MLP but worth asserting).
5. The classical PINN can actually train on the same gate task
   (u' = -u; assert MAE < 0.02 — the same threshold as the
   1D quantum gate).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from qlnn_.training.classical_pinn_solver import (
    MLPConfig,
    build_classical_pinn_1d,
    build_classical_pinn_2d,
    init_classical_pinn_weights,
    matched_mlp_config,
)


# ---------- config + shape contract ----------------------------------------

def test_config_validates():
    with pytest.raises(ValueError):
        MLPConfig(input_dim=0)
    with pytest.raises(ValueError):
        MLPConfig(hidden_layers=0)
    with pytest.raises(ValueError):
        MLPConfig(activation="elu")
    with pytest.raises(ValueError):
        MLPConfig(target_param_count=2)


def test_mlp_total_params_grows_with_target():
    """Sanity: the hidden_width heuristic should produce monotonically
    larger MLPs as the target grows."""
    counts = [
        matched_mlp_config(t, input_dim=1).total_params()
        for t in (20, 60, 200, 800)
    ]
    assert counts == sorted(counts)


def test_matched_mlp_within_2x_of_target():
    """Pre-reg's matched-comparison rule: actual params in [target/2, 2·target]."""
    for target in (60, 120, 200, 500):
        for input_dim in (1, 2):
            cfg = matched_mlp_config(target, input_dim=input_dim)
            actual = cfg.total_params()
            assert target / 2.0 <= actual <= 2.0 * target, (
                f"target={target}, input_dim={input_dim}: "
                f"actual={actual} not in [{target/2}, {2*target}]")


# ---------- 1D drop-in interop with physics_residual_loss ------------------

def test_1d_drop_in_runs_through_make_residual_loss():
    """The classical PINN must be a drop-in for the quantum circuit
    inside `make_residual_loss`. Same {w, s, b} pytree contract."""
    from qlnn_.training.physics_residual_loss import make_residual_loss

    cfg = matched_mlp_config(60, input_dim=1)
    circuit = build_classical_pinn_1d(cfg)
    w = init_classical_pinn_weights(cfg, seed=0)
    p = {"w": w, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}

    loss_fn, u_of_t = make_residual_loss(
        circuit, lambda t, u: -u, t0=0.0, t1=2.0, u0=1.0)
    t_colloc = jnp.linspace(0.0, 2.0, 16)
    val, grads = jax.value_and_grad(loss_fn)(p, t_colloc)
    assert math.isfinite(float(val))
    # Gradient mass in the MLP pytree must be non-trivial
    leaves = jax.tree_util.tree_leaves(grads["w"])
    gnorm = sum(float(jnp.sum(jnp.abs(g))) for g in leaves)
    assert gnorm > 1e-6, "MLP-PINN gradient is trivially zero through make_residual_loss"


# ---------- 2D drop-in interop with pde_residual_loss ----------------------

def test_2d_drop_in_runs_through_make_pde_residual_loss():
    from qlnn_.training.pde_residual_loss import make_pde_residual_loss

    cfg = matched_mlp_config(120, input_dim=2)
    circuit = build_classical_pinn_2d(cfg)
    w = init_classical_pinn_weights(cfg, seed=0)
    p = {"w": w, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}

    def rhs_heat(t, x, u, ut, ux, uxx):
        return ut - 0.1 * uxx

    loss_fn, _ = make_pde_residual_loss(
        circuit, rhs_heat, lambda x: jnp.sin(x),
        t0=0.0, t1=1.0, x0=0.0, x1=2.0 * float(np.pi))

    ts = jnp.linspace(0.1, 0.9, 4)
    xs = jnp.linspace(0.5, 5.5, 4)
    T, X = jnp.meshgrid(ts, xs, indexing="ij")
    tx = jnp.stack([T.ravel(), X.ravel()], axis=1)

    val, grads = jax.value_and_grad(loss_fn)(p, tx)
    assert math.isfinite(float(val))
    leaves = jax.tree_util.tree_leaves(grads["w"])
    gnorm = sum(float(jnp.sum(jnp.abs(g))) for g in leaves)
    assert gnorm > 1e-6, "MLP-PINN gradient through 2D PDE residual is trivially zero"


# ---------- IC structural check (Lagaris hard-constraint preserved) --------

def test_ic_is_hard_constrained_for_classical_pinn_1d():
    """At t=t0 the trial solution u₀ + (t-t0)·N must equal u₀ exactly
    for ANY MLP weights."""
    from qlnn_.training.physics_residual_loss import make_residual_loss

    cfg = matched_mlp_config(60, input_dim=1)
    circuit = build_classical_pinn_1d(cfg)
    w = init_classical_pinn_weights(cfg, seed=42)
    p = {"w": w, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}

    _, u_of_t = make_residual_loss(
        circuit, lambda t, u: -u, t0=0.0, t1=2.0, u0=1.0)
    # IC at t=t0=0 must be u0=1.0 exactly.
    assert abs(float(u_of_t(jnp.asarray(0.0), p)) - 1.0) < 1e-5


def test_ic_is_hard_constrained_for_classical_pinn_2d():
    """At t=t0, u(0, x) must equal u₀(x) for any x."""
    from qlnn_.training.pde_residual_loss import make_pde_residual_loss

    cfg = matched_mlp_config(120, input_dim=2)
    circuit = build_classical_pinn_2d(cfg)
    w = init_classical_pinn_weights(cfg, seed=42)
    p = {"w": w, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}

    _, u_of_tx = make_pde_residual_loss(
        circuit, lambda t, x, u, ut, ux, uxx: ut - 0.1 * uxx,
        lambda x: jnp.sin(x),
        t0=0.0, t1=1.0, x0=0.0, x1=2.0 * float(np.pi))

    for x in jnp.linspace(0.5, 5.5, 5):
        u_at_t0 = float(u_of_tx(jnp.asarray(0.0), x, p))
        expected = float(jnp.sin(x))
        assert abs(u_at_t0 - expected) < 1e-5


# ---------- Mechanism: jacrev through MLP (classical analog of the gate) ---

def test_jacrev_jacrev_through_2d_classical_pinn_is_finite():
    """The MLP's mixed 2nd derivative should be trivially finite (no
    custom_vjp concerns), but assert it explicitly for symmetry with
    the quantum mechanism gate."""
    cfg = matched_mlp_config(120, input_dim=2)
    circuit = build_classical_pinn_2d(cfg)
    w = init_classical_pinn_weights(cfg, seed=0)

    def f(t, x):
        return circuit(t, x, w)

    fxx = jax.jacrev(jax.jacrev(f, argnums=1), argnums=1)
    val = fxx(jnp.asarray(0.3), jnp.asarray(-0.4))
    assert np.isfinite(float(val))


# ---------- Convergence: classical PINN trains on u'=-u --------------------

def test_classical_pinn_trains_on_exp_decay_to_within_quantum_gate_threshold():
    """End-to-end: train a capacity-matched classical PINN on u'=-u
    (the same gate task as the quantum solver) and assert MAE < 0.02
    (the 1D quantum gate's threshold). Establishes the classical
    baseline performance on a smooth ODE.
    """
    from qlnn_.training.physics_residual_loss import make_residual_loss

    # Capacity match the 1D quantum gate (60 PQC params).
    cfg = matched_mlp_config(60, input_dim=1)
    circuit = build_classical_pinn_1d(cfg)
    w = init_classical_pinn_weights(cfg, seed=0)
    p = {"w": w, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}

    loss_fn, u_of_t = make_residual_loss(
        circuit, lambda t, u: -u, t0=0.0, t1=2.0, u0=1.0)
    t_colloc = jnp.linspace(0.0, 2.0, 60 + 2)[1:-1]

    opt = optax.adam(0.02)
    opt_state = opt.init(p)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    for _ in range(1200):
        val, grads = loss_and_grad(p, t_colloc)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)

    t_eval = jnp.linspace(0.0, 2.0, 102)[1:-1]
    u_pred = jax.vmap(lambda tt: u_of_t(tt, p))(t_eval)
    exact = jnp.exp(-t_eval)
    mae = float(jnp.mean(jnp.abs(u_pred - exact)))
    assert mae < 0.02, f"classical PINN MAE={mae:.4f} exceeds 0.02 threshold"
