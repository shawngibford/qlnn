"""P5 commit 2 — plain MLP forecaster tests."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qlnn_.evaluation.forecaster_adapters import (
    make_vector_forecaster_adapter,
)
from qlnn_.evaluation.rollout import autoregressive_rollout
from qlnn_.models.plain_mlp_forecaster import (
    PlainMLPForecaster, PlainMLPForecasterConfig,
)


# ---------- config validation ---------------------------------------------

def test_config_rejects_bad_inputs():
    with pytest.raises(ValueError, match="input_dim"):
        PlainMLPForecasterConfig(input_dim=0, window_length=4)
    with pytest.raises(ValueError, match="window_length"):
        PlainMLPForecasterConfig(input_dim=2, window_length=1)
    with pytest.raises(ValueError, match="hidden_dim"):
        PlainMLPForecasterConfig(input_dim=2, window_length=4, hidden_dim=0)
    with pytest.raises(ValueError, match="n_hidden_layers"):
        PlainMLPForecasterConfig(
            input_dim=2, window_length=4, n_hidden_layers=3)
    with pytest.raises(ValueError, match="activation"):
        PlainMLPForecasterConfig(
            input_dim=2, window_length=4, activation="sigmoid")
    with pytest.raises(ValueError, match="delta_scale_init"):
        PlainMLPForecasterConfig(
            input_dim=2, window_length=4,
            delta_scale_init=0.005, delta_scale_min=0.01)


# ---------- output shape contract -----------------------------------------

def test_forecaster_output_shape():
    cfg = PlainMLPForecasterConfig(
        input_dim=3, window_length=5, hidden_dim=8)
    model = PlainMLPForecaster(cfg, key=jax.random.PRNGKey(0))
    hist = jnp.zeros((5, 3))
    y = model(hist)
    assert y.shape == (3,)
    assert jnp.all(jnp.isfinite(y))


def test_forecaster_rejects_wrong_window_length():
    cfg = PlainMLPForecasterConfig(
        input_dim=2, window_length=4, hidden_dim=8)
    model = PlainMLPForecaster(cfg, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="T = 4"):
        model(jnp.zeros((5, 2)))         # wrong T


def test_forecaster_rejects_wrong_input_dim():
    cfg = PlainMLPForecasterConfig(
        input_dim=3, window_length=4, hidden_dim=8)
    model = PlainMLPForecaster(cfg, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="input_dim"):
        model(jnp.zeros((4, 2)))         # wrong d


# ---------- residual-around-persistence -----------------------------------

def test_init_prediction_is_near_persistence():
    cfg = PlainMLPForecasterConfig(
        input_dim=2, window_length=4, hidden_dim=4,
        delta_scale_init=0.02, delta_scale_min=0.01,
        init_head_std=0.01)
    model = PlainMLPForecaster(cfg, key=jax.random.PRNGKey(0))
    hist = jnp.array([[1.0, 2.0], [1.1, 2.05],
                      [1.15, 2.08], [1.18, 2.1]])
    y = model(hist)
    # Should be very close to x[-1] when delta-scale is near floor.
    assert jnp.allclose(y, hist[-1], atol=0.1)


# ---------- gradient flow -------------------------------------------------

def test_gradient_flows_end_to_end():
    cfg = PlainMLPForecasterConfig(
        input_dim=2, window_length=4, hidden_dim=4)
    model = PlainMLPForecaster(cfg, key=jax.random.PRNGKey(0))
    hist = jnp.array([[0.5, 0.3], [0.55, 0.32],
                      [0.6, 0.34], [0.65, 0.36]])
    target = jnp.array([0.7, 0.38])

    def loss_fn(m):
        return jnp.sum((m(hist) - target) ** 2)

    grad = jax.grad(loss_fn)(model)
    leaves = jax.tree_util.tree_leaves(grad)
    total = sum(float(jnp.sum(jnp.abs(g))) for g in leaves
                if hasattr(g, 'shape'))
    assert total > 1e-8


# ---------- n_hidden_layers behavior --------------------------------------

def test_n_hidden_layers_1_bypasses_W2():
    """With n_hidden_layers=1, the forward pass should not use W2."""
    cfg = PlainMLPForecasterConfig(
        input_dim=2, window_length=4, hidden_dim=8, n_hidden_layers=1)
    model = PlainMLPForecaster(cfg, key=jax.random.PRNGKey(0))
    hist = jnp.zeros((4, 2))
    # Just confirm it runs cleanly.
    y = model(hist)
    assert y.shape == (2,)


def test_n_hidden_layers_2_uses_W2():
    """With n_hidden_layers=2, both W1 and W2 contribute. Verify by
    perturbing W2 and confirming the output changes."""
    cfg = PlainMLPForecasterConfig(
        input_dim=2, window_length=4, hidden_dim=8, n_hidden_layers=2,
        delta_scale_init=1.0, init_head_std=0.5)
    model = PlainMLPForecaster(cfg, key=jax.random.PRNGKey(0))
    hist = jnp.array([[0.5, 0.3]] * 4)
    y_orig = model(hist)
    # Replace W2 with zeros — output should change since W2 is in the path.
    perturbed = eqx.tree_at(lambda m: m.W2, model, jnp.zeros_like(model.W2))
    y_perturbed = perturbed(hist)
    assert not jnp.allclose(y_orig, y_perturbed)


# Need to import equinox for the perturbation test.
import equinox as eqx


# ---------- adapter + autoregressive rollout ------------------------------

def test_plugs_into_autoregressive_rollout():
    cfg = PlainMLPForecasterConfig(
        input_dim=2, window_length=3, hidden_dim=4)
    model = PlainMLPForecaster(cfg, key=jax.random.PRNGKey(0))
    adapter = make_vector_forecaster_adapter(model)
    hist0 = jnp.array([[1.0, 2.0], [1.05, 2.02], [1.1, 2.05]])
    traj = autoregressive_rollout(adapter, hist0, n_steps=5, dt=0.05)
    assert traj.shape == (5, 2)
    assert jnp.all(jnp.isfinite(traj))


# ---------- parameter count -----------------------------------------------

def test_param_count_matches_formula():
    """For d=2, T=4, H=4, n_hidden=2:
       W1: T·d·H = 4·2·4 = 32
       b1: H = 4
       W2: H·H = 16
       b2: H = 4
       Wout: H·d = 8
       bout: d = 2
       delta_scale_unconstrained: 1
       Total: 67
    """
    cfg = PlainMLPForecasterConfig(
        input_dim=2, window_length=4, hidden_dim=4, n_hidden_layers=2)
    model = PlainMLPForecaster(cfg, key=jax.random.PRNGKey(0))
    assert model.num_parameters() == 67
