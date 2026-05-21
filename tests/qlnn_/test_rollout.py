"""P4 commit 1 — autoregressive rollout helper tests.

Tests:
  - Schema / shape contract
  - Identity-model determinism (model that returns its last history
    entry must produce a constant trajectory)
  - Known-trivial sliding-window correctness
  - Linear-extrapolation model produces correct linear trajectory
  - JIT compatibility (the rollout loop is jax.lax.scan-based)
  - make_history_slider matches autoregressive_rollout step-for-step
  - Error paths (bad shape, n_steps<1, dt<=0)

NOTE: this commit does NOT test rollout metrics (relative-L2, VPT,
spectral, invariant drift) — those land in P4 commit 2 alongside
`rollout_metrics.py`.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qlnn_.evaluation.rollout import (
    OneStepForecaster,
    autoregressive_rollout,
    autoregressive_rollout_python_loop,
    autoregressive_rollout_with_history,
    make_history_slider,
)


# ---------- shape contract -------------------------------------------------

def test_rollout_returns_correct_shape():
    def model(history, dt):
        return history[-1]                              # identity
    hist0 = jnp.zeros((5, 3))
    traj = autoregressive_rollout(model, hist0, n_steps=10, dt=0.1)
    assert traj.shape == (10, 3)


def test_rollout_with_history_prepends_initial_window():
    def model(history, dt):
        return history[-1]
    T, d, n_steps = 4, 2, 7
    hist0 = jnp.arange(T * d, dtype=jnp.float32).reshape(T, d)
    full = autoregressive_rollout_with_history(
        model, hist0, n_steps=n_steps, dt=0.1)
    assert full.shape == (T + n_steps, d)
    # First T rows must equal the initial history exactly.
    assert jnp.allclose(full[:T], hist0)


# ---------- identity & sliding-window correctness --------------------------

def test_identity_model_produces_constant_trajectory():
    """A model that returns history[-1] should produce a constant
    rollout = the last entry of the initial history."""
    def model(history, dt):
        return history[-1]
    hist0 = jnp.array([[0.0, 0.0],
                       [1.0, 2.0],
                       [3.0, 5.0]])  # last row = [3, 5]
    traj = autoregressive_rollout(model, hist0, n_steps=8, dt=0.5)
    # Every step's prediction must equal [3, 5].
    expected = jnp.broadcast_to(jnp.array([3.0, 5.0]), (8, 2))
    assert jnp.allclose(traj, expected)


def test_oldest_dropped_newest_appended():
    """If the model emits the FIRST history entry each step (a
    rotation-style predictor), the trajectory should be predictable
    from the sliding-window dynamics."""
    # model: return history[0] (oldest). Then at step 1 history becomes
    # [old[1], old[2], old[0]] (oldest dropped, predicted appended).
    def model(history, dt):
        return history[0]
    hist0 = jnp.array([[1.0], [2.0], [3.0]])
    traj = autoregressive_rollout(model, hist0, n_steps=6, dt=1.0)
    # Step-by-step trace:
    #   t=0 history=[1,2,3] → pred=1, new=[2,3,1]
    #   t=1 history=[2,3,1] → pred=2, new=[3,1,2]
    #   t=2 history=[3,1,2] → pred=3, new=[1,2,3]
    #   t=3 history=[1,2,3] → pred=1, new=[2,3,1]
    #   ...period-3 cycle: 1, 2, 3, 1, 2, 3
    expected = jnp.array([[1.0], [2.0], [3.0], [1.0], [2.0], [3.0]])
    assert jnp.allclose(traj, expected)


def test_linear_extrapolation_model():
    """A model that predicts history[-1] + (history[-1] - history[-2])
    is a discrete linear extrapolator. Initial history = [0, 1] → at
    step 0 emit 2, new history = [1, 2]; step 1 emit 3; step 2 emit 4."""
    def model(history, dt):
        return 2.0 * history[-1] - history[-2]
    hist0 = jnp.array([[0.0], [1.0]])
    traj = autoregressive_rollout(model, hist0, n_steps=5, dt=1.0)
    expected = jnp.arange(2, 7, dtype=jnp.float32).reshape(5, 1)
    assert jnp.allclose(traj, expected), f"got {traj.flatten()}"


# ---------- JIT compatibility ----------------------------------------------

def test_rollout_jit_compatible():
    """The rollout loop is jax.lax.scan-based — it must JIT-compile."""
    def model(history, dt):
        return history[-1] + dt * 0.1
    hist0 = jnp.ones((3, 2))
    fn = jax.jit(lambda h: autoregressive_rollout(
        model, h, n_steps=20, dt=0.1))
    traj_jit = fn(hist0)
    assert traj_jit.shape == (20, 2)
    assert jnp.all(jnp.isfinite(traj_jit))


def test_rollout_grad_well_defined():
    """The rollout must be reverse-mode differentiable through the
    initial history. This is the property that lets P6+ train end-to-
    end on rollout loss (not just 1-step loss)."""
    def model(history, dt):
        # Smooth, differentiable. Avoid using jnp.tanh on Diffrax.
        return jnp.sin(history[-1]) + jnp.cos(history[-2])
    hist0 = jnp.array([[0.5, -0.2], [0.1, 0.3]])

    def total_l2(h0):
        traj = autoregressive_rollout(model, h0, n_steps=5, dt=0.1)
        return jnp.sum(traj ** 2)

    grad = jax.grad(total_l2)(hist0)
    assert grad.shape == hist0.shape
    assert jnp.all(jnp.isfinite(grad))
    # Non-trivial: rollout couples history0 to all subsequent steps.
    assert jnp.sum(jnp.abs(grad)) > 1e-6


# ---------- make_history_slider mirrors autoregressive_rollout ------------

def test_history_slider_matches_scan_rollout():
    def model(history, dt):
        return 0.5 * history[-1] - 0.25 * history[-2]
    hist0 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    n_steps = 6

    traj_scan = autoregressive_rollout(model, hist0, n_steps, dt=0.1)
    step = make_history_slider(model, hist0, dt=0.1)
    manual = jnp.stack([step() for _ in range(n_steps)])

    assert jnp.allclose(traj_scan, manual, atol=1e-6)


# ---------- protocol acceptance --------------------------------------------

def test_oneStepForecaster_protocol_accepts_lambda():
    """The Protocol is runtime_checkable; any callable matching the
    signature is acceptable."""
    f = lambda h, dt: h[-1]
    assert isinstance(f, OneStepForecaster)


def test_oneStepForecaster_protocol_rejects_wrong_signature():
    """A function with the wrong signature still passes the runtime
    isinstance check (Protocol is structural, not nominal). This is
    a documented limit — the protocol provides docs, not enforcement.
    Adapters are responsible for calling-convention discipline."""
    # Just confirm the test runs without error; structural typing
    # doesn't reject by signature.
    f = lambda: None
    assert isinstance(f, OneStepForecaster)  # passes (no shape check)


# ---------- error paths ----------------------------------------------------

def test_rejects_1d_history():
    def model(h, dt):
        return h[-1]
    with pytest.raises(ValueError, match="2-D"):
        autoregressive_rollout(model, jnp.zeros(5), n_steps=3, dt=0.1)


def test_rejects_n_steps_below_one():
    def model(h, dt):
        return h[-1]
    with pytest.raises(ValueError, match="n_steps"):
        autoregressive_rollout(model, jnp.zeros((3, 2)), n_steps=0, dt=0.1)


def test_rejects_non_positive_dt():
    def model(h, dt):
        return h[-1]
    with pytest.raises(ValueError, match="dt"):
        autoregressive_rollout(model, jnp.zeros((3, 2)), n_steps=5, dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        autoregressive_rollout(model, jnp.zeros((3, 2)), n_steps=5, dt=-0.5)


# ---------- physical-time sanity (a tiny Euler-step ODE rollout) -----------

def test_python_loop_rollout_matches_scan_on_pure_jax_model():
    """The python-loop variant must produce IDENTICAL results to the
    scan variant for pure-JAX models. Locks the equivalence so we can
    use either path interchangeably depending on model JAX-purity."""
    def model(history, dt):
        return 0.6 * history[-1] + 0.2 * history[-2] + 0.1 * dt
    hist0 = jnp.array([[1.0, -0.3], [0.5, 0.1]])

    traj_scan = autoregressive_rollout(model, hist0, n_steps=10, dt=0.05)
    traj_loop = autoregressive_rollout_python_loop(
        model, hist0, n_steps=10, dt=0.05)
    assert jnp.allclose(traj_scan, traj_loop, atol=1e-6)


def test_python_loop_rollout_handles_numpy_model():
    """The python-loop variant supports adapters that return numpy
    arrays (rf_qrc's predict returns np.ndarray) — the scan variant
    would fail with a TracerArrayConversionError. The python-loop
    variant accepts the numpy and converts back to JAX inside."""
    def numpy_model(history, dt):
        # Returns a numpy array, NOT a JAX array.
        h_np = np.asarray(history)
        return np.asarray(h_np[-1] * 0.9, dtype=np.float64)

    hist0 = jnp.array([[1.0], [2.0]])
    # The scan variant would raise here — confirms our adapter design.
    traj = autoregressive_rollout_python_loop(
        numpy_model, hist0, n_steps=4, dt=0.1)
    assert traj.shape == (4, 1)
    expected = np.array([[2.0 * 0.9],
                         [2.0 * 0.9 ** 2],
                         [2.0 * 0.9 ** 3],
                         [2.0 * 0.9 ** 4]])
    assert jnp.allclose(traj, jnp.asarray(expected), atol=1e-6)


def test_euler_step_rollout_recovers_linear_ode():
    """Forward-Euler step y_{n+1} = y_n + dt · f(y_n), with f(y) = -y
    yields y_n = (1 - dt)^n · y_0. Verify rollout matches this exactly."""
    decay_rate = 1.0
    dt = 0.1

    def euler_decay(history, dt_arr):
        y = history[-1]
        return y - dt_arr * decay_rate * y

    hist0 = jnp.array([[1.0]])           # single-step window, y0=1
    n_steps = 20
    traj = autoregressive_rollout(euler_decay, hist0, n_steps, dt=dt)
    expected = jnp.array(
        [(1.0 - dt) ** (i + 1) for i in range(n_steps)],
        dtype=jnp.float32).reshape(n_steps, 1)
    assert jnp.allclose(traj, expected, atol=1e-5), (
        f"Euler rollout diverges from analytic; "
        f"got {traj.flatten()[:5]}, expected {expected.flatten()[:5]}")
