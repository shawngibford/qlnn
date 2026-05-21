"""P3.9 commit 3 — te_qpinn_qnn_2d acceptance gates.

Same gate structure as test_qcpinn_2d.py / test_te_qpinn_fnn_2d.py.
The faithfulness hook from the 1D CIRCUIT_SPECS §2 is the LINEAR
parameter scaling in n·(K+L); preserved under the split-qubit
U_embed declared design choice (the trained-param count formula
n_total·(2K + 3L) is linear in n_total·(K+L) by construction).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from qlnn_.circuits.pde_2d.te_qpinn_qnn_2d import (
    TEQPINNQnn2DConfig,
    build_te_qpinn_qnn_2d,
    init_te_qpinn_qnn_2d_solver_params,
)
from qlnn_.training.pde_residual_loss import make_pde_residual_loss


# ---------- schema + shape contract ----------------------------------------

def test_config_validates():
    with pytest.raises(ValueError):
        TEQPINNQnn2DConfig(n_t_qubits=0)
    with pytest.raises(ValueError):
        TEQPINNQnn2DConfig(n_x_qubits=0)
    with pytest.raises(ValueError):
        TEQPINNQnn2DConfig(num_layers=0)
    with pytest.raises(ValueError):
        TEQPINNQnn2DConfig(num_embed_layers=0)


def test_config_derived_shapes():
    cfg = TEQPINNQnn2DConfig(n_t_qubits=3, n_x_qubits=2,
                              num_layers=4, num_embed_layers=2)
    assert cfg.num_qubits == 5
    assert cfg.embed_t_weight_shape == (2, 3, 2)
    assert cfg.embed_x_weight_shape == (2, 2, 2)
    assert cfg.var_weight_shape == (4, 5, 3)
    # Total trained = 2·n·K + 3·n·L = 2·5·2 + 3·5·4 = 20 + 60 = 80.
    assert cfg.n_trained_params == 80


def test_circuit_returns_scalar_in_minus_n_to_n():
    cfg = TEQPINNQnn2DConfig(n_t_qubits=2, n_x_qubits=2,
                              num_layers=2, num_embed_layers=2)
    circ = build_te_qpinn_qnn_2d(cfg)
    p = init_te_qpinn_qnn_2d_solver_params(cfg, seed=0)
    y = circ(jnp.asarray(0.3), jnp.asarray(-0.4), p["w"])
    assert jnp.ndim(y) == 0
    # Σ⟨Z⟩ output range — paper Eq. 26.
    assert -cfg.num_qubits - 1e-6 <= float(y) <= cfg.num_qubits + 1e-6


def test_init_returns_lagaris_outer_pytree_and_three_weight_tensors():
    cfg = TEQPINNQnn2DConfig(n_t_qubits=2, n_x_qubits=2,
                              num_layers=2, num_embed_layers=2)
    p = init_te_qpinn_qnn_2d_solver_params(cfg, seed=0)
    assert set(p.keys()) == {"w", "s", "b"}
    assert isinstance(p["w"], dict)
    assert set(p["w"].keys()) == {"embed_t_W", "embed_x_W", "var_W"}
    assert p["w"]["embed_t_W"].shape == cfg.embed_t_weight_shape
    assert p["w"]["embed_x_W"].shape == cfg.embed_x_weight_shape
    assert p["w"]["var_W"].shape == cfg.var_weight_shape


# ---------- THE FAITHFULNESS HOOK (linear param scaling) -------------------

@pytest.mark.parametrize("n_t,n_x,K,L", [
    (2, 2, 2, 3), (3, 1, 3, 5), (2, 3, 1, 4), (4, 4, 2, 2),
])
def test_trained_param_count_is_linear_in_n_times_K_plus_L(n_t, n_x, K, L):
    """CIRCUIT_SPECS §2 hook: trained-param count grows linearly in
    n·(K_embed + L_var). The 2D port preserves this — the split-qubit
    decomposition splits the embedding params per-axis but the SUM
    is still 2·n·K + 3·n·L (linear in n·(K+L)).
    """
    cfg = TEQPINNQnn2DConfig(n_t_qubits=n_t, n_x_qubits=n_x,
                              num_layers=L, num_embed_layers=K)
    n = n_t + n_x
    assert cfg.n_trained_params == 2 * n * K + 3 * n * L

    p = init_te_qpinn_qnn_2d_solver_params(cfg, seed=0)
    # Confirm the leaves match the count.
    leaf_size = sum(int(jnp.asarray(leaf).size)
                    for leaf in jax.tree_util.tree_leaves(p["w"]))
    assert leaf_size == cfg.n_trained_params


# ---------- THE MECHANISM GATE (Risk-#2-redux for te_qpinn_qnn_2d) ---------

def test_jacrev_jacrev_finite_and_nontrivial():
    """Most rigorous of the 3 ports — te_qpinn_qnn_2d composes TWO
    QNodes (U_embed → α → U_var). If nested autodiff through the
    composition works, the rest of the family is structurally safe."""
    cfg = TEQPINNQnn2DConfig(n_t_qubits=2, n_x_qubits=2,
                              num_layers=2, num_embed_layers=2)
    circ = build_te_qpinn_qnn_2d(cfg)
    p = init_te_qpinn_qnn_2d_solver_params(cfg, seed=0)

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
        f"te_qpinn_qnn_2d jacrev∘jacrev through TWO QNodes is ~0 "
        f"(max |fxx|={max_abs}). The composition pattern (U_embed → "
        f"⟨Z⟩ → α → U_var) may be silently dropping the 2nd-order "
        f"contribution via the intermediate stack.")


def test_grad_through_pde_loss_is_finite():
    cfg = TEQPINNQnn2DConfig(n_t_qubits=2, n_x_qubits=2,
                              num_layers=2, num_embed_layers=2)
    circ = build_te_qpinn_qnn_2d(cfg)
    p = init_te_qpinn_qnn_2d_solver_params(cfg, seed=0)

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
    cfg = TEQPINNQnn2DConfig(n_t_qubits=2, n_x_qubits=2,
                              num_layers=1, num_embed_layers=1)
    circ = build_te_qpinn_qnn_2d(cfg)
    p = init_te_qpinn_qnn_2d_solver_params(cfg, seed=42)
    _, u_of_tx = make_pde_residual_loss(
        circ, lambda *args: jnp.asarray(0.0), lambda x: jnp.sin(x),
        t0=0.0, t1=1.0, x0=0.0, x1=2.0 * np.pi)
    for x in jnp.linspace(0.5, 5.5, 5):
        u_at_t0 = float(u_of_tx(jnp.asarray(0.0), x, p))
        expected = float(jnp.sin(x))
        assert abs(u_at_t0 - expected) < 1e-5


# ---------- smoke train ----------------------------------------------------

def test_train_smoke_at_tiny_budget():
    cfg = TEQPINNQnn2DConfig(n_t_qubits=2, n_x_qubits=2,
                              num_layers=1, num_embed_layers=1)
    circ = build_te_qpinn_qnn_2d(cfg)

    def rhs(t, x, u, ut, ux, uxx):
        return ut - 0.1 * uxx

    p0 = init_te_qpinn_qnn_2d_solver_params(cfg, seed=0)
    loss_fn, u_of_tx = make_pde_residual_loss(
        circ, rhs, lambda x: jnp.sin(x),
        t0=0.0, t1=1.0, x0=0.0, x1=2.0 * np.pi)

    ts = jnp.linspace(0.0, 1.0, 6)[1:-1]
    xs = jnp.linspace(0.0, 2.0 * np.pi, 6)[1:-1]
    T, X = jnp.meshgrid(ts, xs, indexing="ij")
    tx = jnp.stack([T.ravel(), X.ravel()], axis=1)

    opt = optax.adam(0.02)
    opt_state = opt.init(p0)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    p = p0
    last = 0.0
    for _ in range(10):
        last, grads = loss_and_grad(p, tx)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)

    assert math.isfinite(float(last))
    assert not jnp.allclose(p0["w"]["var_W"], p["w"]["var_W"])
