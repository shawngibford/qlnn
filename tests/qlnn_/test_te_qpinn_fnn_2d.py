"""P3.9 commit 2 — te_qpinn_fnn_2d acceptance gates.

Same gate structure as test_qcpinn_2d.py: schema/shape contract,
mechanism gate (`jacrev(jacrev(QNode))` finite + non-trivial), Lagaris
IC hard constraint, train_pde_solver smoke. Faithfulness hook from
1D: N_rot = 3·n·L (paper Eq. 12) — preserved under split-qubit 2D
port (depends on n=n_t+n_x and L only).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from qlnn_.circuits.pde_2d.te_qpinn_fnn_2d import (
    TEQPINNFnn2DConfig,
    build_te_qpinn_fnn_2d,
    init_te_qpinn_fnn_2d_solver_params,
)
from qlnn_.training.pde_residual_loss import make_pde_residual_loss


# ---------- schema + shape contract ----------------------------------------

def test_config_validates():
    with pytest.raises(ValueError):
        TEQPINNFnn2DConfig(n_t_qubits=0)
    with pytest.raises(ValueError):
        TEQPINNFnn2DConfig(n_x_qubits=0)
    with pytest.raises(ValueError):
        TEQPINNFnn2DConfig(num_layers=0)
    with pytest.raises(ValueError):
        TEQPINNFnn2DConfig(fnn_hidden_dim=0)


def test_config_derived_shapes():
    cfg = TEQPINNFnn2DConfig(n_t_qubits=3, n_x_qubits=2, num_layers=4)
    assert cfg.num_qubits == 5
    assert cfg.pqc_weight_shape == (4, 5, 3)
    # Berger Eq. 12 unit-test hook — survives the 2D port.
    assert cfg.n_pqc_rotations == 3 * 5 * 4


def test_circuit_returns_scalar_in_minus_1_to_1():
    cfg = TEQPINNFnn2DConfig(n_t_qubits=2, n_x_qubits=2, num_layers=2)
    circ = build_te_qpinn_fnn_2d(cfg)
    p = init_te_qpinn_fnn_2d_solver_params(cfg, seed=0)
    y = circ(jnp.asarray(0.3), jnp.asarray(-0.4), p["w"])
    assert jnp.ndim(y) == 0
    # ⟨⊗ Z_k⟩ output range (faithful to paper Eq. 13).
    assert -1.0 - 1e-6 <= float(y) <= 1.0 + 1e-6


def test_init_returns_lagaris_outer_pytree_and_two_fnn_heads():
    """The 2D port has TWO FNN heads (split-qubit declared design
    choice). Confirm both are present and the outer {w, s, b}
    contract holds."""
    cfg = TEQPINNFnn2DConfig(n_t_qubits=2, n_x_qubits=2, num_layers=2,
                              fnn_hidden_dim=8)
    p = init_te_qpinn_fnn_2d_solver_params(cfg, seed=0)
    assert set(p.keys()) == {"w", "s", "b"}
    assert isinstance(p["w"], dict)
    # Both FNN heads with the right shapes.
    for prefix in ("fnn_t_", "fnn_x_"):
        for k in ("W1", "b1", "W2", "b2"):
            assert (prefix + k) in p["w"], f"missing {prefix}{k}"
    assert p["w"]["fnn_t_W1"].shape == (2, 8)
    assert p["w"]["fnn_x_W1"].shape == (2, 8)
    assert p["w"]["fnn_t_W2"].shape == (8, 2)   # → n_t = 2
    assert p["w"]["fnn_x_W2"].shape == (8, 2)   # → n_x = 2
    assert p["w"]["pqc_W"].shape == cfg.pqc_weight_shape


# ---------- THE FAITHFULNESS HOOK (Berger Eq. 12 unchanged) ----------------

@pytest.mark.parametrize("n_t,n_x,L", [
    (2, 2, 5), (3, 1, 4), (1, 3, 2), (4, 4, 3),
])
def test_pqc_rotation_count_eq_3nl(n_t, n_x, L):
    """Paper Eq. 12: PQC has N_rot = 3·n·L rotations.

    Survives the 2D port (depends on n=n_t+n_x and L only). This is
    the Berger paper's binding unit-test hook from
    CIRCUIT_SPECS.md §1.
    """
    cfg = TEQPINNFnn2DConfig(n_t_qubits=n_t, n_x_qubits=n_x, num_layers=L)
    n = n_t + n_x
    assert cfg.n_pqc_rotations == 3 * n * L
    p = init_te_qpinn_fnn_2d_solver_params(cfg, seed=0)
    # The PQC weight tensor must have exactly N_rot scalar leaves.
    assert int(p["w"]["pqc_W"].size) == 3 * n * L


# ---------- THE MECHANISM GATE (Risk-#2-redux for te_qpinn_fnn_2d) ---------

def test_jacrev_jacrev_finite_and_nontrivial():
    cfg = TEQPINNFnn2DConfig(n_t_qubits=2, n_x_qubits=2, num_layers=2)
    circ = build_te_qpinn_fnn_2d(cfg)
    p = init_te_qpinn_fnn_2d_solver_params(cfg, seed=0)

    def f(t, x):
        return circ(t, x, p["w"])

    fx = jax.jacrev(f, argnums=1)
    val_fx = fx(jnp.asarray(0.3), jnp.asarray(-0.4))
    assert math.isfinite(float(val_fx))

    fxx = jax.jacrev(fx, argnums=1)
    val_fxx = fxx(jnp.asarray(0.3), jnp.asarray(-0.4))
    assert math.isfinite(float(val_fxx))

    probes = [(jnp.asarray(0.3), jnp.asarray(-0.4)),
              (jnp.asarray(-0.5), jnp.asarray(0.2)),
              (jnp.asarray(0.7), jnp.asarray(-0.1))]
    max_abs = max(float(jnp.abs(fxx(t, x))) for t, x in probes)
    assert max_abs > 1e-8, (
        f"te_qpinn_fnn_2d jacrev∘jacrev is ~0 (max |fxx|={max_abs}). "
        f"The tensor-product Z readout at n=4 random init may be "
        f"too close to zero — try larger init scale or smaller n.")


def test_grad_through_pde_loss_is_finite():
    cfg = TEQPINNFnn2DConfig(n_t_qubits=2, n_x_qubits=2, num_layers=2)
    circ = build_te_qpinn_fnn_2d(cfg)
    p = init_te_qpinn_fnn_2d_solver_params(cfg, seed=0)

    def rhs_heat(t, x, u, ut, ux, uxx):
        return ut - 0.1 * uxx

    loss_fn, _ = make_pde_residual_loss(
        circ, rhs_heat, lambda x: jnp.sin(x),
        t0=0.0, t1=1.0, x0=0.0, x1=2.0 * np.pi)

    ts = jnp.linspace(0.1, 0.9, 4)
    xs = jnp.linspace(0.5, 5.5, 4)
    T, X = jnp.meshgrid(ts, xs, indexing="ij")
    tx = jnp.stack([T.ravel(), X.ravel()], axis=1)

    val, grads = jax.value_and_grad(loss_fn)(p, tx)
    assert math.isfinite(float(val))
    for leaf in jax.tree_util.tree_leaves(grads):
        assert np.all(np.isfinite(np.asarray(leaf)))
    gnorm = sum(float(jnp.sum(jnp.abs(g)))
                for g in jax.tree_util.tree_leaves(grads))
    assert gnorm > 1e-6


# ---------- LAGARIS IC HARD CONSTRAINT --------------------------------------

def test_ic_is_hard_constrained():
    cfg = TEQPINNFnn2DConfig(n_t_qubits=2, n_x_qubits=2, num_layers=1)
    circ = build_te_qpinn_fnn_2d(cfg)
    p = init_te_qpinn_fnn_2d_solver_params(cfg, seed=42)
    _, u_of_tx = make_pde_residual_loss(
        circ, lambda *args: jnp.asarray(0.0), lambda x: jnp.sin(x),
        t0=0.0, t1=1.0, x0=0.0, x1=2.0 * np.pi)
    for x in jnp.linspace(0.5, 5.5, 5):
        u_at_t0 = float(u_of_tx(jnp.asarray(0.0), x, p))
        expected = float(jnp.sin(x))
        assert abs(u_at_t0 - expected) < 1e-5


# ---------- smoke train ----------------------------------------------------

def test_train_smoke_at_tiny_budget():
    cfg = TEQPINNFnn2DConfig(n_t_qubits=2, n_x_qubits=2, num_layers=1,
                              fnn_hidden_dim=8)
    circ = build_te_qpinn_fnn_2d(cfg)

    def rhs(t, x, u, ut, ux, uxx):
        return ut - 0.1 * uxx

    p0 = init_te_qpinn_fnn_2d_solver_params(cfg, seed=0)
    loss_fn, u_of_tx = make_pde_residual_loss(
        circ, rhs, lambda x: jnp.sin(x),
        t0=0.0, t1=1.0, x0=0.0, x1=2.0 * np.pi)

    ts = jnp.linspace(0.0, 1.0, 8)[1:-1]
    xs = jnp.linspace(0.0, 2.0 * np.pi, 8)[1:-1]
    T, X = jnp.meshgrid(ts, xs, indexing="ij")
    tx = jnp.stack([T.ravel(), X.ravel()], axis=1)

    opt = optax.adam(0.02)
    opt_state = opt.init(p0)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    p = p0
    last = 0.0
    for _ in range(15):
        last, grads = loss_and_grad(p, tx)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)

    assert math.isfinite(float(last))
    assert not jnp.allclose(p0["w"]["pqc_W"], p["w"]["pqc_W"])
