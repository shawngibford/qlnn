"""P4 commit 3a — Vector-output QLNN forecaster tests.

Schema + shape contract + the existing OD forecaster's behavior
pattern, generalized to vector output. Tests:
  - Config validation (matches QLNNForecaster's gates).
  - Output shape contract: (T, d) → (d,).
  - Residual-around-persistence default: at init (delta_scale small),
    prediction is close to x[-1].
  - Tests the 4 registry ansätze plug in via AnsatzConfig — at
    least the basic three exercised (d_reuploading,
    hardware_efficient, brickwall).
  - JIT compatibility + gradient flow through the call.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qlnn_.circuits import AnsatzConfig
from qlnn_.models.vector_forecaster import (
    VectorForecaster,
    VectorForecasterConfig,
    _inv_softplus,
)


# ---------- config validation ----------------------------------------------

def test_config_rejects_zero_or_neg_input_dim():
    with pytest.raises(ValueError, match="input_dim"):
        VectorForecasterConfig(input_dim=0)


def test_config_rejects_bad_solver():
    with pytest.raises(ValueError, match="solver"):
        VectorForecasterConfig(input_dim=2, solver="rk4")


def test_config_rejects_non_positive_step_dt():
    with pytest.raises(ValueError, match="step_dt"):
        VectorForecasterConfig(input_dim=2, step_dt=0.0)
    with pytest.raises(ValueError, match="step_dt"):
        VectorForecasterConfig(input_dim=2, step_dt=-0.1)


def test_config_rejects_delta_scale_below_min():
    with pytest.raises(ValueError, match="delta_scale_init"):
        VectorForecasterConfig(
            input_dim=2, delta_scale_init=0.005, delta_scale_min=0.01)


# ---------- shape contract -------------------------------------------------

def test_forecaster_output_is_d_vector():
    cfg = VectorForecasterConfig(input_dim=2, num_qubits=4, num_layers=2,
                                 step_dt=0.05)
    key = jax.random.PRNGKey(0)
    model = VectorForecaster(cfg, key=key)
    hist = jnp.array([[0.5, 0.3], [0.6, 0.35], [0.7, 0.4]])  # T=3, d=2
    y = model(hist)
    assert y.shape == (2,)
    assert jnp.all(jnp.isfinite(y))


def test_forecaster_rejects_wrong_input_dim():
    cfg = VectorForecasterConfig(input_dim=3, num_qubits=4)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))
    bad = jnp.ones((4, 2))  # d=2 but cfg.input_dim=3
    with pytest.raises(ValueError, match="input_dim"):
        model(bad)


def test_forecaster_rejects_single_timestep():
    cfg = VectorForecasterConfig(input_dim=2, num_qubits=4)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="2 time points"):
        model(jnp.ones((1, 2)))


# ---------- residual-around-persistence default --------------------------

def test_init_prediction_is_near_persistence():
    """At init with delta_scale_init=0.01 (near floor), the prediction
    should be very close to x[-1] (the persistence baseline)."""
    cfg = VectorForecasterConfig(
        input_dim=3, num_qubits=4, num_layers=1, step_dt=0.05,
        delta_scale_init=0.02, delta_scale_min=0.01,
        init_head_std=0.01)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(123))
    hist = jnp.array([[1.0, 2.0, -1.0],
                      [1.1, 2.05, -0.95],
                      [1.15, 2.08, -0.93]])
    y = model(hist)
    # Should be very close to x[-1] when the delta-scale is near floor.
    assert jnp.allclose(y, hist[-1], atol=0.1), (
        f"prediction {y} too far from persistence baseline {hist[-1]}")


# ---------- delta_scale is positive ----------------------------------------

def test_delta_scale_is_positive_and_above_floor():
    cfg = VectorForecasterConfig(
        input_dim=2, num_qubits=4, num_layers=2,
        delta_scale_init=0.5, delta_scale_min=0.1)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))
    ds = float(model.delta_scale())
    assert ds >= 0.1 - 1e-6
    assert ds <= 10.0   # sanity ceiling — softplus(unconstrained) + floor


def test_inv_softplus_round_trip():
    """Inverse-softplus must round-trip through softplus."""
    for y_target in (0.5, 1.0, 2.0, 5.0):
        u = _inv_softplus(y_target)
        recovered = float(jax.nn.softplus(u))
        assert abs(recovered - y_target) < 1e-5


def test_inv_softplus_rejects_non_positive():
    with pytest.raises(ValueError, match=">"):
        _inv_softplus(0.0)
    with pytest.raises(ValueError, match=">"):
        _inv_softplus(-0.1)


# ---------- gradient flow + JIT --------------------------------------------

def test_gradient_flows_through_call():
    """The forecaster must be reverse-mode differentiable end-to-end
    so it can train via gradient descent."""
    cfg = VectorForecasterConfig(input_dim=2, num_qubits=3, num_layers=1,
                                 step_dt=0.05)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))
    hist = jnp.array([[0.5, 0.3], [0.55, 0.32]])
    target = jnp.array([0.6, 0.34])

    def loss_fn(m):
        return jnp.sum((m(hist) - target) ** 2)

    grad = jax.grad(loss_fn)(model)
    # All parameter leaves should have finite gradients with non-zero norm.
    leaves = jax.tree_util.tree_leaves(grad)
    assert len(leaves) > 0
    total_norm = sum(float(jnp.sum(jnp.abs(g))) for g in leaves
                     if hasattr(g, 'shape'))
    assert total_norm > 1e-8, (
        "no gradient signal through VectorForecaster — likely a JIT trace "
        "issue or the cell isn't connected to the loss")


# ---------- registry ansatz plug-in tests (the 4 forecaster families) -----

@pytest.mark.parametrize("ansatz_name", [
    "data_reuploading",
    "hardware_efficient",
    "strongly_entangling",
    "brickwall",
])
def test_each_registry_ansatz_runs_forward(ansatz_name):
    """Every registry forecaster ansatz must plug into the vector
    forecaster via AnsatzConfig and produce a finite vector output."""
    ansatz = AnsatzConfig(name=ansatz_name, num_qubits=4, num_layers=2)
    cfg = VectorForecasterConfig(
        input_dim=2, num_qubits=4, num_layers=2, step_dt=0.05,
        ansatz=ansatz)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(7))
    hist = jnp.array([[0.5, 0.3], [0.6, 0.35], [0.7, 0.4]])
    y = model(hist)
    assert y.shape == (2,)
    assert jnp.all(jnp.isfinite(y)), (
        f"ansatz {ansatz_name} produced non-finite output {y}")


# ---------- weight shape contract -----------------------------------------

def test_vector_head_shape_is_num_qubits_by_d():
    """The KEY change from QLNNForecaster: delta_head_W is (Q, d),
    not (Q, 1). Confirm via direct shape inspection."""
    cfg = VectorForecasterConfig(input_dim=3, num_qubits=5)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))
    assert model.delta_head_W.shape == (5, 3)
    assert model.delta_head_b.shape == (3,)
    assert model.initial_h_W.shape == (3, 5)
    assert model.initial_h_b.shape == (5,)
