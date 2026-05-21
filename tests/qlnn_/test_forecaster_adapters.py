"""P4 commit 3c — per-family adapter tests.

Verifies that each adapter:
  1. Satisfies the `OneStepForecaster` protocol signature.
  2. Plugs into `autoregressive_rollout` end-to-end.
  3. Produces a `(d,)` output shape from a `(T, d)` history.

Plus floor-adapter correctness (persistence + linear-extrapolation).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qlnn_.circuits.rf_qrc import RFQRCConfig, RFQRCForecaster
from qlnn_.evaluation.forecaster_adapters import (
    make_classical_mlp_adapter,
    make_linear_extrapolation_adapter,
    make_persistence_adapter,
    make_rf_qrc_adapter,
    make_vector_forecaster_adapter,
)
from qlnn_.evaluation.rollout import (
    OneStepForecaster,
    autoregressive_rollout,
    autoregressive_rollout_python_loop,
)
from qlnn_.models.vector_forecaster import (
    VectorForecaster, VectorForecasterConfig,
)


# ---------- adapters return OneStepForecaster-compliant callables ---------

def test_persistence_adapter_returns_callable():
    adapter = make_persistence_adapter()
    assert isinstance(adapter, OneStepForecaster)


def test_linear_extrapolation_adapter_returns_callable():
    adapter = make_linear_extrapolation_adapter()
    assert isinstance(adapter, OneStepForecaster)


# ---------- persistence + linear-extrapolation correctness ----------------

def test_persistence_predicts_last_row():
    adapter = make_persistence_adapter()
    hist = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = adapter(hist, jnp.asarray(0.1))
    assert jnp.allclose(out, jnp.array([5.0, 6.0]))


def test_linear_extrapolation_predicts_linear_step():
    adapter = make_linear_extrapolation_adapter()
    hist = jnp.array([[0.0], [1.0]])
    # 2*x[-1] - x[-2] = 2*1 - 0 = 2
    out = adapter(hist, jnp.asarray(0.1))
    assert jnp.allclose(out, jnp.array([2.0]))


def test_linear_extrapolation_on_constant_trajectory_is_constant():
    """If x[-1] = x[-2] = c, then 2c - c = c (no extrapolation needed)."""
    adapter = make_linear_extrapolation_adapter()
    hist = jnp.array([[3.7, 1.1], [3.7, 1.1]])
    out = adapter(hist, jnp.asarray(0.5))
    assert jnp.allclose(out, jnp.array([3.7, 1.1]))


# ---------- VectorForecaster adapter ---------------------------------------

def test_vector_forecaster_adapter_shape_and_finite():
    cfg = VectorForecasterConfig(
        input_dim=2, num_qubits=4, num_layers=2, step_dt=0.05,
        delta_scale_init=0.05, delta_scale_min=0.01)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))
    adapter = make_vector_forecaster_adapter(model)
    assert isinstance(adapter, OneStepForecaster)
    hist = jnp.array([[1.0, 2.0], [1.1, 2.05], [1.2, 2.1]])
    out = adapter(hist, jnp.asarray(0.05))
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))


def test_vector_forecaster_adapter_end_to_end_rollout():
    """The adapter plugs into autoregressive_rollout cleanly."""
    cfg = VectorForecasterConfig(
        input_dim=2, num_qubits=3, num_layers=1, step_dt=0.05,
        delta_scale_init=0.05, delta_scale_min=0.01)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))
    adapter = make_vector_forecaster_adapter(model)
    hist0 = jnp.array([[1.0, 2.0], [1.1, 2.05]])
    traj = autoregressive_rollout(adapter, hist0, n_steps=5, dt=0.05)
    assert traj.shape == (5, 2)
    assert jnp.all(jnp.isfinite(traj))


# ---------- rf_qrc adapter -------------------------------------------------

def test_rf_qrc_adapter_shape_after_fit():
    """rf_qrc must be fit before predict; the adapter wraps the
    fitted version. Use a tiny synthetic dataset for the fit."""
    cfg = RFQRCConfig(num_qubits=3, input_dim=2,
                       leak_rate=0.5, alpha_seed=0)
    forecaster = RFQRCForecaster(cfg)

    # Tiny synthetic series — y_t = 0.5·x_{t-1} (one-step shift).
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    Y = 0.5 * X
    forecaster.fit(X, Y)

    adapter = make_rf_qrc_adapter(forecaster)
    assert isinstance(adapter, OneStepForecaster)
    hist = jnp.asarray(X[:5])
    out = adapter(hist, jnp.asarray(0.1))
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))


def test_rf_qrc_adapter_rollout_smoke():
    cfg = RFQRCConfig(num_qubits=2, input_dim=2,
                       leak_rate=0.5, alpha_seed=1)
    forecaster = RFQRCForecaster(cfg)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((15, 2))
    Y = 0.3 * X
    forecaster.fit(X, Y)

    adapter = make_rf_qrc_adapter(forecaster)
    hist0 = jnp.asarray(X[:4])
    # rf_qrc's predict is pure numpy → use python-loop rollout
    # (jax.lax.scan would trace through and fail on np.asarray).
    traj = autoregressive_rollout_python_loop(
        adapter, hist0, n_steps=3, dt=0.1)
    assert traj.shape == (3, 2)
    assert jnp.all(jnp.isfinite(traj))


# ---------- classical MLP adapter (placeholder for P5) --------------------

def test_classical_mlp_adapter_flatten_history():
    """The default flatten=True path passes a flat (T*d,) vector."""
    def fake_mlp(x_flat):
        # Identity-like: return the last d=2 entries.
        return x_flat[-2:]
    adapter = make_classical_mlp_adapter(fake_mlp, flatten_history=True)
    hist = jnp.array([[1.0, 2.0], [3.0, 4.0]])     # T=2, d=2
    out = adapter(hist, jnp.asarray(0.1))
    assert jnp.allclose(out, jnp.array([3.0, 4.0]))


def test_classical_mlp_adapter_no_flatten():
    """flatten_history=False passes the (T, d) 2-D array unchanged."""
    def fake_mlp(x_2d):
        return x_2d[-1]
    adapter = make_classical_mlp_adapter(fake_mlp, flatten_history=False)
    hist = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = adapter(hist, jnp.asarray(0.1))
    assert jnp.allclose(out, jnp.array([3.0, 4.0]))


# ---------- floor adapters in autoregressive_rollout ----------------------

def test_persistence_rollout_is_constant():
    adapter = make_persistence_adapter()
    hist0 = jnp.array([[1.5, -0.3]])
    traj = autoregressive_rollout(adapter, hist0, n_steps=8, dt=0.05)
    # Every prediction = persistence of x[-1] = [1.5, -0.3]
    expected = jnp.broadcast_to(jnp.array([1.5, -0.3]), (8, 2))
    assert jnp.allclose(traj, expected)


def test_linear_extrapolation_rollout_is_linear():
    """linear-extrap on [0, 1] produces 2, 3, 4, ... (already covered
    in test_rollout.py, but this confirms the adapter wrapper)."""
    adapter = make_linear_extrapolation_adapter()
    hist0 = jnp.array([[0.0], [1.0]])
    traj = autoregressive_rollout(adapter, hist0, n_steps=4, dt=1.0)
    expected = jnp.array([[2.0], [3.0], [4.0], [5.0]])
    assert jnp.allclose(traj, expected)
