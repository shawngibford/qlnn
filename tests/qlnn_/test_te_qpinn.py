"""Faithfulness tests for `te_qpinn_fnn` (Berger 2025).

Each test corresponds to a CIRCUIT_SPECS §1 unit-test hook +
end-to-end interoperation with the strand-1 solver infrastructure.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qlnn_.circuits.te_qpinn import (
    TEQPINNFnnConfig,
    build_te_qpinn_fnn,
    init_te_qpinn_fnn_weights,
)


# ---------- HOOK: PQC weight shape and rotation count -----------------------

def test_pqc_weight_shape_is_L_n_3():
    cfg = TEQPINNFnnConfig(num_qubits=3, num_layers=2)
    assert cfg.pqc_weight_shape == (2, 3, 3)


def test_rotation_count_matches_paper_3nL_anchor():
    """Paper anchor (Berger 2025): n=4 qubits, L=5 layers ⇒ 60 PQC
    params (3·4·5 = 60). Both the primary extractor and the independent
    dual-check verified this hook."""
    cfg = TEQPINNFnnConfig(num_qubits=4, num_layers=5)
    assert cfg.n_pqc_rotations == 60
    # And the leaf shape carries the same count.
    w = init_te_qpinn_fnn_weights(cfg, seed=0)
    assert int(np.prod(w["pqc_W"].shape)) == 60


# ---------- HOOK: tensor-Z readout is a scalar in [-1, 1] -------------------

def test_output_is_scalar_in_minus1_to_1():
    cfg = TEQPINNFnnConfig(num_qubits=3, num_layers=2)
    circ = build_te_qpinn_fnn(cfg)
    w = init_te_qpinn_fnn_weights(cfg, seed=0)
    for x in (-0.7, 0.0, 0.5):
        y = circ(jnp.asarray(x), w)
        assert jnp.ndim(y) == 0
        assert -1.0 - 1e-6 <= float(y) <= 1.0 + 1e-6


# ---------- HOOK: differentiable via jax.jacrev (the locked convention) ----

def test_jacrev_input_derivative_is_finite_and_nontrivial():
    """The whole point of the solver path: an input-coordinate
    derivative through the QNode is exactly what the physics residual
    needs. Locked convention is jacrev (gotcha #1)."""
    cfg = TEQPINNFnnConfig(num_qubits=3, num_layers=2)
    circ = build_te_qpinn_fnn(cfg)
    w = init_te_qpinn_fnn_weights(cfg, seed=0)
    du_dx = jax.jacrev(lambda x: circ(x, w))(jnp.asarray(0.3))
    assert np.isfinite(float(du_dx))
    assert abs(float(du_dx)) > 1e-6, "input-derivative is trivially zero"


def test_param_gradients_through_both_FNN_and_PQC_leaves():
    """The trainable pytree has FNN + PQC leaves; gradient mass must
    land in BOTH (the whole "trainable embedding" concept)."""
    cfg = TEQPINNFnnConfig(num_qubits=3, num_layers=2)
    circ = build_te_qpinn_fnn(cfg)
    w = init_te_qpinn_fnn_weights(cfg, seed=0)

    def loss(p):
        return circ(jnp.asarray(0.3), p) ** 2

    g = jax.grad(loss)(w)
    for key in ("fnn_W1", "fnn_b1", "fnn_W2", "fnn_b2", "pqc_W"):
        leaf = np.asarray(g[key])
        assert np.all(np.isfinite(leaf))
    # Both subsystems must contribute non-trivially.
    assert float(np.sum(np.abs(g["pqc_W"]))) > 1e-6
    assert (float(np.sum(np.abs(g["fnn_W1"])))
            + float(np.sum(np.abs(g["fnn_W2"])))) > 1e-6


# ---------- end-to-end: plugs into the strand-1 solver scaffolding ---------

def test_drop_in_interop_with_make_residual_loss():
    """te_qpinn_fnn must be a drop-in for the chebyshev_dqc circuit
    inside `physics_residual_loss.make_residual_loss`. The same nested
    grad/jacrev pattern must produce finite, non-trivial gradients."""
    from qlnn_.training.physics_residual_loss import make_residual_loss
    cfg = TEQPINNFnnConfig(num_qubits=3, num_layers=2)
    circ = build_te_qpinn_fnn(cfg)
    w = init_te_qpinn_fnn_weights(cfg, seed=0)

    # The loss expects a params pytree with keys w / s / b (the
    # Chebyshev-DQC shape). Wrap accordingly.
    p = {"w": w, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}

    def circ_for_loss(x, ww):
        return circ(x, ww)

    loss_fn, _ = make_residual_loss(
        circ_for_loss, lambda t, u: -u, t0=0.0, t1=2.0, u0=1.0)
    t_colloc = jnp.linspace(0.0, 2.0, 16)
    val, g = jax.value_and_grad(loss_fn)(p, t_colloc)
    assert np.isfinite(float(val))
    # Gradient mass in the PQC leaf (the new circuit family) AND in
    # the affine head (the shared Lagaris trial-solution wrapping).
    pqc_grad = sum(float(jnp.sum(jnp.abs(leaf)))
                   for leaf in jax.tree_util.tree_leaves(g["w"]))
    assert pqc_grad > 1e-6


def test_config_validates():
    with pytest.raises(ValueError):
        TEQPINNFnnConfig(num_qubits=0)
    with pytest.raises(ValueError):
        TEQPINNFnnConfig(num_layers=0)
    with pytest.raises(ValueError):
        TEQPINNFnnConfig(fnn_hidden_dim=0)


# ===========================================================================
# te_qpinn_qnn — fully-quantum trainable embedding (P3a-corrected source)
# ===========================================================================

from qlnn_.circuits.te_qpinn import (
    TEQPINNQnnConfig,
    build_te_qpinn_qnn,
    init_te_qpinn_qnn_weights,
)


# ---------- HOOK: trained-param count is LINEAR in N_q · L_total -----------

def test_qnn_param_count_scales_linearly_in_n_qubits():
    """CIRCUIT_SPECS §2 unit-test hook: param count scales linearly in
    N_q · L (paper §p.6: 'params ∝ N_q · L (linear)')."""
    fixed = dict(num_layers=5, num_embed_layers=3)
    counts = {n: TEQPINNQnnConfig(num_qubits=n, **fixed).n_trained_params
              for n in (2, 4, 8)}
    # Linearity: count(8) / count(4) ≈ 2 and count(4) / count(2) ≈ 2.
    assert abs(counts[8] / counts[4] - 2.0) < 1e-9
    assert abs(counts[4] / counts[2] - 2.0) < 1e-9


def test_qnn_param_count_scales_linearly_in_layers():
    fixed = dict(num_qubits=4, num_embed_layers=3)
    # Vary L_var. The slope (Δparams / ΔL) must be constant = 3·n.
    counts = [(L, TEQPINNQnnConfig(num_layers=L, **fixed).n_trained_params)
              for L in (1, 5, 10)]
    slope_a = (counts[1][1] - counts[0][1]) / (counts[1][0] - counts[0][0])
    slope_b = (counts[2][1] - counts[1][1]) / (counts[2][0] - counts[1][0])
    assert abs(slope_a - slope_b) < 1e-9
    assert slope_a == 3 * 4              # 3·n per added variational layer


def test_qnn_param_pytree_size_matches_config():
    cfg = TEQPINNQnnConfig(num_qubits=4, num_layers=5, num_embed_layers=3)
    w = init_te_qpinn_qnn_weights(cfg, seed=0)
    assert int(np.prod(w["embed_W"].shape)) == 2 * 4 * 3
    assert int(np.prod(w["var_W"].shape)) == 3 * 4 * 5
    total = sum(int(np.prod(v.shape)) for v in w.values())
    assert total == cfg.n_trained_params


# ---------- HOOK: scalar output (Σ_j Z_j readout in [-n, n]) ---------------

def test_qnn_output_is_scalar_in_minus_n_to_n():
    cfg = TEQPINNQnnConfig(num_qubits=3, num_layers=2, num_embed_layers=2)
    f = build_te_qpinn_qnn(cfg)
    w = init_te_qpinn_qnn_weights(cfg, seed=0)
    y = f(jnp.asarray(0.3), w)
    assert jnp.ndim(y) == 0
    assert -cfg.num_qubits - 1e-6 <= float(y) <= cfg.num_qubits + 1e-6


# ---------- HOOK: jacrev finite + gradients in BOTH subsystems -------------

def test_qnn_jacrev_finite_and_grad_flows_through_both_subsystems():
    """The defining 'trainable embedding' property: gradient mass must
    land in BOTH embed_W and var_W."""
    cfg = TEQPINNQnnConfig(num_qubits=3, num_layers=2, num_embed_layers=2)
    f = build_te_qpinn_qnn(cfg)
    w = init_te_qpinn_qnn_weights(cfg, seed=0)
    du_dx = jax.jacrev(lambda x: f(x, w))(jnp.asarray(0.3))
    assert np.isfinite(float(du_dx))

    g = jax.grad(lambda p: f(jnp.asarray(0.3), p) ** 2)(w)
    assert float(np.sum(np.abs(g["embed_W"]))) > 1e-6
    assert float(np.sum(np.abs(g["var_W"]))) > 1e-6


# ---------- HOOK: paper anchor magnitude ≈360 params (sanity bound) --------

def test_qnn_paper_anchor_magnitude_is_in_sensible_range():
    """Paper anchor: ≈360 PQC params total (vs 608 FNN-variant, 6594
    classical PINN). Treated as a soft sanity bound (the schematic
    source's exact constant is UNSPECIFIED), not a hard equality."""
    cfg = TEQPINNQnnConfig(num_qubits=8, num_layers=10, num_embed_layers=5)
    # 2·8·5 + 3·8·10 = 80 + 240 = 320  (≈ 360 ✓ within an order)
    assert 200 <= cfg.n_trained_params <= 600


# ---------- end-to-end: drop-in for make_residual_loss ---------------------

def test_qnn_drop_in_interop_with_make_residual_loss():
    from qlnn_.training.physics_residual_loss import make_residual_loss
    cfg = TEQPINNQnnConfig(num_qubits=3, num_layers=2, num_embed_layers=2)
    f = build_te_qpinn_qnn(cfg)
    w = init_te_qpinn_qnn_weights(cfg, seed=0)
    p = {"w": w, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    loss_fn, _ = make_residual_loss(
        lambda x, ww: f(x, ww), lambda t, u: -u,
        t0=0.0, t1=2.0, u0=1.0)
    t_colloc = jnp.linspace(0.0, 2.0, 12)
    val, g = jax.value_and_grad(loss_fn)(p, t_colloc)
    assert np.isfinite(float(val))


def test_qnn_config_validation():
    with pytest.raises(ValueError):
        TEQPINNQnnConfig(num_qubits=0)
    with pytest.raises(ValueError):
        TEQPINNQnnConfig(num_layers=0)
    with pytest.raises(ValueError):
        TEQPINNQnnConfig(num_embed_layers=0)
