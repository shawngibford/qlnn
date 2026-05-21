"""P5 commit 4 — Classical PINN extended to vector ODE.

Tests the new `_mlp_apply_vector`, `build_classical_pinn_vector_ode`,
`vector_ode_pinn_trial`, and `matched_mlp_config_vector_ode`
additions for the H1 verdict's solver-task contrast.

Also verifies the existing P3.8 1D/2D PINN paths still work
(backward compatibility — output_dim default = 1).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qlnn_.training.classical_pinn_solver import (
    MLPConfig,
    _mlp_apply,
    _mlp_apply_vector,
    build_classical_pinn_1d,
    build_classical_pinn_2d,
    build_classical_pinn_vector_ode,
    init_classical_pinn_weights,
    matched_mlp_config,
    matched_mlp_config_vector_ode,
    vector_ode_pinn_trial,
)


# ---------- config: output_dim ---------------------------------------------

def test_config_rejects_zero_or_negative_output_dim():
    with pytest.raises(ValueError, match="output_dim"):
        MLPConfig(input_dim=1, output_dim=0)


def test_config_default_output_dim_is_one():
    cfg = MLPConfig(input_dim=1)
    assert cfg.output_dim == 1


def test_weight_shapes_scale_with_output_dim():
    cfg = MLPConfig(input_dim=1, output_dim=3, hidden_layers=2,
                    target_param_count=60)
    shapes = cfg.weight_shapes()
    # Last two shapes are (H, output_dim) and (output_dim,).
    assert shapes[-2][-1] == 3
    assert shapes[-1] == (3,)


def test_total_params_grows_with_output_dim():
    cfg_scalar = MLPConfig(input_dim=1, output_dim=1, target_param_count=60)
    cfg_vector = MLPConfig(input_dim=1, output_dim=3, target_param_count=60)
    # Vector PINN has more output params (H·3 vs H·1 + 3 vs 1 bias).
    assert cfg_vector.total_params() > cfg_scalar.total_params()


# ---------- _mlp_apply backward compatibility (output_dim=1) --------------

def test_mlp_apply_scalar_path_preserved():
    cfg = MLPConfig(input_dim=1, output_dim=1)
    weights = init_classical_pinn_weights(cfg, seed=0)
    out = _mlp_apply(jnp.asarray(0.5), weights, cfg)
    assert out.shape == ()      # scalar
    assert jnp.isfinite(out)


def test_mlp_apply_rejects_vector_output():
    """_mlp_apply must error out when output_dim != 1 — protects the
    P3.8 contract."""
    cfg = MLPConfig(input_dim=1, output_dim=3)
    weights = init_classical_pinn_weights(cfg, seed=0)
    with pytest.raises(ValueError, match="output_dim must be 1"):
        _mlp_apply(jnp.asarray(0.5), weights, cfg)


# ---------- _mlp_apply_vector (P5 commit 4) -------------------------------

def test_mlp_apply_vector_shape():
    cfg = MLPConfig(input_dim=1, output_dim=3, hidden_layers=2,
                    target_param_count=60)
    weights = init_classical_pinn_weights(cfg, seed=0)
    out = _mlp_apply_vector(jnp.asarray(0.5), weights, cfg)
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))


def test_mlp_apply_vector_for_output_dim_1():
    """When output_dim=1, the vector apply returns shape (1,) (vs
    _mlp_apply which squeezes to scalar). Confirm both pathways
    work in their own niche."""
    cfg = MLPConfig(input_dim=1, output_dim=1)
    weights = init_classical_pinn_weights(cfg, seed=0)
    s = _mlp_apply(jnp.asarray(0.5), weights, cfg)
    v = _mlp_apply_vector(jnp.asarray(0.5), weights, cfg)
    assert s.shape == ()
    assert v.shape == (1,)
    assert jnp.allclose(v[0], s)


# ---------- build_classical_pinn_vector_ode ------------------------------

def test_build_vector_ode_returns_d_vector():
    cfg = MLPConfig(input_dim=1, output_dim=2, hidden_layers=2)
    fwd = build_classical_pinn_vector_ode(cfg)
    weights = init_classical_pinn_weights(cfg, seed=0)
    out = fwd(jnp.asarray(0.3), weights)
    assert out.shape == (2,)


def test_build_vector_ode_rejects_bad_input_dim():
    cfg = MLPConfig(input_dim=2, output_dim=2)
    with pytest.raises(ValueError, match="input_dim=1"):
        build_classical_pinn_vector_ode(cfg)


# ---------- Lagaris hard-IC trial solution ---------------------------------

def test_lagaris_trial_solution_exact_at_t0():
    """The Lagaris hard-IC trial `u(t) = u₀ + (t − t₀) · MLP(t)` must
    return EXACTLY `u₀` at t = t₀ for any weights (the structural IC
    enforcement that's the entire point of Lagaris 1998)."""
    cfg = MLPConfig(input_dim=1, output_dim=3, hidden_layers=2)
    fwd = build_classical_pinn_vector_ode(cfg)
    weights = init_classical_pinn_weights(cfg, seed=42)
    u0 = jnp.array([1.5, -0.7, 2.3])
    t0 = 0.0
    u_of_t = vector_ode_pinn_trial(fwd, u0, t0=t0)
    u_at_t0 = u_of_t(jnp.asarray(t0), weights)
    assert jnp.allclose(u_at_t0, u0, atol=1e-12), (
        f"Lagaris hard-IC violated: u({t0}) = {u_at_t0} ≠ u₀ = {u0}")


def test_lagaris_trial_solution_works_at_nonzero_t0():
    """The hard-IC works for any t0, not just 0."""
    cfg = MLPConfig(input_dim=1, output_dim=2)
    fwd = build_classical_pinn_vector_ode(cfg)
    weights = init_classical_pinn_weights(cfg, seed=0)
    u0 = jnp.array([5.0, -3.0])
    t0 = 1.5
    u_of_t = vector_ode_pinn_trial(fwd, u0, t0=t0)
    assert jnp.allclose(u_of_t(jnp.asarray(t0), weights), u0, atol=1e-12)


def test_lagaris_trial_solution_evolves_for_t_neq_t0():
    """At t ≠ t0, the prediction should differ from u₀ (the MLP
    actually contributes)."""
    cfg = MLPConfig(input_dim=1, output_dim=2, hidden_layers=2)
    fwd = build_classical_pinn_vector_ode(cfg)
    weights = init_classical_pinn_weights(cfg, seed=7)
    u0 = jnp.array([1.0, 2.0])
    u_of_t = vector_ode_pinn_trial(fwd, u0, t0=0.0)
    u_later = u_of_t(jnp.asarray(0.5), weights)
    # Should differ from u0 (modulo a near-zero MLP coincidence).
    assert not jnp.allclose(u_later, u0, atol=1e-6)


# ---------- gradient flow through vector-ODE PINN -------------------------

def test_gradient_flows_through_vector_pinn():
    cfg = MLPConfig(input_dim=1, output_dim=2, hidden_layers=2)
    fwd = build_classical_pinn_vector_ode(cfg)
    weights = init_classical_pinn_weights(cfg, seed=0)
    u0 = jnp.array([1.0, 2.0])
    u_of_t = vector_ode_pinn_trial(fwd, u0)

    def loss_fn(w):
        u = u_of_t(jnp.asarray(0.5), w)
        return jnp.sum(u ** 2)

    grad = jax.grad(loss_fn)(weights)
    total = sum(float(jnp.sum(jnp.abs(g)))
                for g in jax.tree_util.tree_leaves(grad))
    assert total > 1e-6, (
        f"gradient through vector PINN is trivial (total={total})")


# ---------- matched_mlp_config_vector_ode --------------------------------

def test_matched_mlp_config_vector_ode_within_factor_2():
    target = 80
    cfg = matched_mlp_config_vector_ode(target, output_dim=2)
    actual = cfg.total_params()
    assert target / 2.0 <= actual <= 2.0 * target, (
        f"matched-MLP capacity {actual} outside factor-of-2 of "
        f"target {target}")


def test_matched_mlp_config_vector_ode_preserves_output_dim():
    cfg = matched_mlp_config_vector_ode(60, output_dim=3)
    assert cfg.output_dim == 3
    assert cfg.input_dim == 1


# ---------- backward compatibility: P3.8 1D + 2D paths still work --------

def test_p3_8_1d_path_still_works():
    """The original P3.8 1D ODE solver path must keep working
    unchanged (output_dim defaults to 1)."""
    cfg = MLPConfig(input_dim=1, target_param_count=60)
    circuit = build_classical_pinn_1d(cfg)
    weights = init_classical_pinn_weights(cfg, seed=0)
    out = circuit(jnp.asarray(0.3), weights)
    assert out.shape == ()


def test_p3_8_2d_path_still_works():
    """The original P3.8 2D PDE solver path must keep working
    unchanged."""
    cfg = MLPConfig(input_dim=2, target_param_count=120)
    circuit = build_classical_pinn_2d(cfg)
    weights = init_classical_pinn_weights(cfg, seed=0)
    out = circuit(jnp.asarray(0.3), jnp.asarray(-0.4), weights)
    assert out.shape == ()


def test_matched_mlp_config_p3_8_path_still_works():
    """Pre-P5 matched_mlp_config (without output_dim) still returns
    output_dim=1 configs."""
    cfg = matched_mlp_config(60, input_dim=1)
    assert cfg.output_dim == 1


# ---------- numerical sanity: vector PINN on a known linear ODE ----------

def test_vector_pinn_can_fit_linear_decay():
    """For y' = -y with y(0) = 1 the analytic solution is exp(-t).
    A vector-ODE PINN of capacity ~80 trained for ~500 steps should
    achieve relL2 << 0.1 on this trivial case. Skipped here at the
    schema-only level — the actual P5 verdict sweep will exercise
    this with the full physics-residual loss + adam optimizer."""
    # Smoke: just confirm the build + trial-solution composition
    # produces a finite trajectory.
    cfg = MLPConfig(input_dim=1, output_dim=1, hidden_layers=1,
                    target_param_count=20)
    fwd = build_classical_pinn_vector_ode(cfg)
    weights = init_classical_pinn_weights(cfg, seed=0)
    u_of_t = vector_ode_pinn_trial(fwd, jnp.array([1.0]), t0=0.0)
    ts = jnp.linspace(0.0, 1.0, 10)
    traj = jnp.stack([u_of_t(t, weights) for t in ts])
    assert traj.shape == (10, 1)
    assert jnp.all(jnp.isfinite(traj))
    # First state is exactly u0 = 1.
    assert jnp.allclose(traj[0], 1.0, atol=1e-10)
