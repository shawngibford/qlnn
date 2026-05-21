"""P5 commit 1 — plain non-liquid Neural-ODE forecaster tests.

Verifies the MANDATORY H1 contrast baseline mirrors VectorForecaster
structurally (same shape contract, same residual-around-persistence
behavior, same Diffrax integration) but differs in two locked ways:
  1. Cell is an MLP, not a quantum circuit.
  2. NO learnable time-constants (no τ parameter in the cell).

Test plan:
  - Config validation (matches VectorForecasterConfig's gates)
  - Cell vector field shape contract
  - Cell has NO τ parameter (structural confirmation of "non-liquid")
  - Forecaster output shape: (T, d) → (d,)
  - Residual-around-persistence default at near-zero head init
  - Gradient flow end-to-end
  - Plugs into the OneStepForecaster adapter for autoregressive rollout
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qlnn_.evaluation.forecaster_adapters import (
    make_vector_forecaster_adapter,
)
from qlnn_.evaluation.rollout import autoregressive_rollout
from qlnn_.models.plain_neuralode_forecaster import (
    NeuralODECell, NeuralODECellConfig,
    PlainNeuralODEForecaster, PlainNeuralODEForecasterConfig,
    _inv_softplus,
)


# ---------- config validation ---------------------------------------------

def test_cell_config_rejects_bad_inputs():
    with pytest.raises(ValueError, match="input_dim"):
        NeuralODECellConfig(input_dim=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        NeuralODECellConfig(input_dim=2, hidden_dim=0)
    with pytest.raises(ValueError, match="activation"):
        NeuralODECellConfig(input_dim=2, activation="sigmoid")


def test_forecaster_config_rejects_bad_inputs():
    with pytest.raises(ValueError, match="input_dim"):
        PlainNeuralODEForecasterConfig(input_dim=0)
    with pytest.raises(ValueError, match="step_dt"):
        PlainNeuralODEForecasterConfig(input_dim=2, step_dt=0)
    with pytest.raises(ValueError, match="delta_scale_init"):
        PlainNeuralODEForecasterConfig(
            input_dim=2, delta_scale_init=0.005, delta_scale_min=0.01)


# ---------- cell vector field ---------------------------------------------

def test_cell_vector_field_shape():
    cfg = NeuralODECellConfig(input_dim=3, hidden_dim=5)
    cell = NeuralODECell(cfg, key=jax.random.PRNGKey(0))
    h = jnp.ones((5,))
    x = jnp.array([0.5, -0.2, 0.1])
    dh = cell(0.0, h, x)
    assert dh.shape == (5,)
    assert jnp.all(jnp.isfinite(dh))


def test_cell_rejects_wrong_h_shape():
    cell = NeuralODECell(NeuralODECellConfig(input_dim=2, hidden_dim=4),
                         key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="h must have shape"):
        cell(0.0, jnp.ones(3), jnp.ones(2))


def test_cell_rejects_wrong_x_shape():
    cell = NeuralODECell(NeuralODECellConfig(input_dim=2, hidden_dim=4),
                         key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="x must have shape"):
        cell(0.0, jnp.ones(4), jnp.ones(5))


def test_cell_has_NO_tau_parameter():
    """CRITICAL: the cell must NOT have a learnable τ — that's the
    'non-liquid' property pre-reg §6 requires for the H1 contrast."""
    import equinox as eqx
    cell = NeuralODECell(NeuralODECellConfig(input_dim=2, hidden_dim=4),
                         key=jax.random.PRNGKey(0))
    leaves_dict = {
        k for k in vars(cell).keys()
        if not k.startswith("_") and k != "config"
    }
    assert "tau_unconstrained" not in leaves_dict, (
        "NeuralODECell has tau_unconstrained — that's a LIQUID property, "
        "violates pre-reg §6's 'plain Neural-ODE baseline' requirement.")
    assert "A" not in leaves_dict, (
        "NeuralODECell has 'A' (per-qubit amplitude) — that's a QUANTUM "
        "property, violates pre-reg §6.")
    # Confirm the four MLP leaves are the ONLY trainable ones.
    expected = {"mlp_W1", "mlp_b1", "mlp_W2", "mlp_b2"}
    assert leaves_dict == expected, (
        f"NeuralODECell has unexpected leaves {leaves_dict - expected}")


def test_cell_param_count_is_minimum_faithful():
    """For Q=3, d=2, H=Q=3: params = (Q+d)·H + H + H·Q + Q
                                   = 5·3 + 3 + 3·3 + 3 = 30."""
    cfg = NeuralODECellConfig(input_dim=2, hidden_dim=3)
    cell = NeuralODECell(cfg, key=jax.random.PRNGKey(0))
    assert cell.num_parameters() == 30


# ---------- forecaster output shape ---------------------------------------

def test_forecaster_output_is_d_vector():
    cfg = PlainNeuralODEForecasterConfig(input_dim=2, hidden_dim=4)
    model = PlainNeuralODEForecaster(cfg, key=jax.random.PRNGKey(0))
    hist = jnp.array([[0.5, 0.3], [0.6, 0.35], [0.7, 0.4]])
    y = model(hist)
    assert y.shape == (2,)
    assert jnp.all(jnp.isfinite(y))


def test_forecaster_rejects_wrong_input_dim():
    cfg = PlainNeuralODEForecasterConfig(input_dim=3, hidden_dim=4)
    model = PlainNeuralODEForecaster(cfg, key=jax.random.PRNGKey(0))
    bad = jnp.ones((4, 2))     # d=2 but cfg says 3
    with pytest.raises(ValueError, match="input_dim"):
        model(bad)


def test_forecaster_rejects_single_timestep():
    cfg = PlainNeuralODEForecasterConfig(input_dim=2, hidden_dim=4)
    model = PlainNeuralODEForecaster(cfg, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="2 time points"):
        model(jnp.ones((1, 2)))


# ---------- residual-around-persistence default ---------------------------

def test_init_prediction_is_near_persistence():
    cfg = PlainNeuralODEForecasterConfig(
        input_dim=2, hidden_dim=3, step_dt=0.05,
        delta_scale_init=0.02, delta_scale_min=0.01,
        init_head_std=0.01)
    model = PlainNeuralODEForecaster(cfg, key=jax.random.PRNGKey(123))
    hist = jnp.array([[1.0, 2.0], [1.1, 2.05], [1.15, 2.08]])
    y = model(hist)
    assert jnp.allclose(y, hist[-1], atol=0.1), (
        f"prediction {y} should be near persistence {hist[-1]} at init")


# ---------- delta_scale is positive ---------------------------------------

def test_delta_scale_above_floor():
    cfg = PlainNeuralODEForecasterConfig(
        input_dim=2, hidden_dim=3,
        delta_scale_init=0.5, delta_scale_min=0.1)
    model = PlainNeuralODEForecaster(cfg, key=jax.random.PRNGKey(0))
    ds = float(model.delta_scale())
    assert ds >= 0.1 - 1e-6


def test_inv_softplus_round_trip():
    for y in (0.5, 1.0, 2.0):
        u = _inv_softplus(y)
        assert abs(float(jax.nn.softplus(u)) - y) < 1e-5


# ---------- gradient flow -------------------------------------------------

def test_gradient_flows_end_to_end():
    cfg = PlainNeuralODEForecasterConfig(
        input_dim=2, hidden_dim=3, step_dt=0.05)
    model = PlainNeuralODEForecaster(cfg, key=jax.random.PRNGKey(0))
    hist = jnp.array([[0.5, 0.3], [0.55, 0.32]])
    target = jnp.array([0.6, 0.34])

    def loss_fn(m):
        return jnp.sum((m(hist) - target) ** 2)

    grad = jax.grad(loss_fn)(model)
    leaves = jax.tree_util.tree_leaves(grad)
    total_norm = sum(float(jnp.sum(jnp.abs(g))) for g in leaves
                     if hasattr(g, 'shape'))
    assert total_norm > 1e-8, (
        "no gradient signal through PlainNeuralODEForecaster")


# ---------- end-to-end: adapter + autoregressive rollout ------------------

def test_forecaster_plugs_into_autoregressive_rollout():
    """The plain Neural-ODE forecaster must drop into the same
    OneStepForecaster adapter machinery that VectorForecaster uses.
    The H1 verdict module then treats both identically when computing
    rollout relative-L2."""
    cfg = PlainNeuralODEForecasterConfig(
        input_dim=2, hidden_dim=3, step_dt=0.05)
    model = PlainNeuralODEForecaster(cfg, key=jax.random.PRNGKey(0))

    # The plain Neural-ODE forecaster has the SAME (T, d) → (d,) call
    # signature as VectorForecaster, so the same adapter works.
    adapter = make_vector_forecaster_adapter(model)
    hist0 = jnp.array([[1.0, 2.0], [1.05, 2.02]])
    traj = autoregressive_rollout(
        adapter, hist0, n_steps=5, dt=0.05)
    assert traj.shape == (5, 2)
    assert jnp.all(jnp.isfinite(traj))


# ---------- structural confirmation: shape parity with VectorForecaster --

def test_weight_shapes_match_vector_forecaster_structure():
    """The plain Neural-ODE forecaster must have the same
    encoder/decoder structure as VectorForecaster so the head-to-head
    comparison is structurally fair."""
    cfg = PlainNeuralODEForecasterConfig(input_dim=3, hidden_dim=5)
    model = PlainNeuralODEForecaster(cfg, key=jax.random.PRNGKey(0))
    assert model.initial_h_W.shape == (3, 5)    # (input_dim, hidden_dim)
    assert model.initial_h_b.shape == (5,)
    assert model.delta_head_W.shape == (5, 3)   # (hidden_dim, input_dim)
    assert model.delta_head_b.shape == (3,)
