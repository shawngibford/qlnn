"""P4 commit 3b — forecaster training-loop tests.

Verifies prepare_windows, train_test_split, mse_loss, and
train_vector_forecaster. Tests:
  - Windowing produces correct shapes + correct row alignment.
  - Train-test split is chronological (no row mixing).
  - MSE loss is zero on perfect predictions.
  - Training reduces MSE on a tiny synthetic ODE-like signal.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qlnn_.models.vector_forecaster import (
    VectorForecaster, VectorForecasterConfig,
)
from qlnn_.training.forecaster_training import (
    mse_loss,
    prepare_windows,
    train_test_split,
    train_vector_forecaster,
)


# ---------- prepare_windows ------------------------------------------------

def test_prepare_windows_shapes_and_alignment():
    # Make a trajectory whose value at row t is just `t` (along all dims).
    N, d = 10, 2
    traj = np.outer(np.arange(N, dtype=np.float64), np.ones(d))
    T = 3
    X, Y = prepare_windows(traj, T)
    assert X.shape == (N - T, T, d)
    assert Y.shape == (N - T, d)
    # First window is rows 0..2, target is row 3.
    np.testing.assert_array_equal(X[0], traj[0:3])
    np.testing.assert_array_equal(Y[0], traj[3])
    # Last window is rows (N-T-1)..(N-2), target is row N-1.
    np.testing.assert_array_equal(X[-1], traj[N - T - 1:N - 1])
    np.testing.assert_array_equal(Y[-1], traj[N - 1])


def test_prepare_windows_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        prepare_windows(np.arange(10), window_length=3)


def test_prepare_windows_rejects_short_traj():
    traj = np.zeros((3, 2))
    with pytest.raises(ValueError, match="exceed"):
        prepare_windows(traj, window_length=3)
    with pytest.raises(ValueError, match="exceed"):
        prepare_windows(traj, window_length=5)


def test_prepare_windows_rejects_tiny_window():
    traj = np.zeros((10, 2))
    with pytest.raises(ValueError, match="window_length"):
        prepare_windows(traj, window_length=1)
    with pytest.raises(ValueError, match="window_length"):
        prepare_windows(traj, window_length=0)


# ---------- train_test_split -----------------------------------------------

def test_train_test_split_chronological():
    traj = np.arange(100, dtype=np.float64).reshape(50, 2)
    train, test = train_test_split(traj, train_frac=0.6)
    assert train.shape[0] == 30   # 60% of 50
    assert test.shape[0] == 20
    # Chronological: train is rows 0..29, test is rows 30..49.
    np.testing.assert_array_equal(train, traj[:30])
    np.testing.assert_array_equal(test, traj[30:])


def test_train_test_split_rejects_bad_frac():
    traj = np.zeros((50, 2))
    with pytest.raises(ValueError, match="train_frac"):
        train_test_split(traj, train_frac=0.0)
    with pytest.raises(ValueError, match="train_frac"):
        train_test_split(traj, train_frac=1.0)
    with pytest.raises(ValueError, match="train_frac"):
        train_test_split(traj, train_frac=1.5)


def test_train_test_split_rejects_tiny_train_size():
    traj = np.zeros((4, 2))
    with pytest.raises(ValueError, match="train"):
        # 0.3 * 4 = 1.2 → 1 train sample, below minimum 2
        train_test_split(traj, train_frac=0.3)


# ---------- mse_loss -------------------------------------------------------

def test_mse_loss_zero_on_perfect_predictions():
    """A model that returns x[-1] (perfect persistence) on a constant
    trajectory has zero MSE if targets equal x[-1]."""
    cfg = VectorForecasterConfig(
        input_dim=2, num_qubits=4, num_layers=1, step_dt=0.05,
        delta_scale_init=0.011, delta_scale_min=0.01,
        init_head_std=1e-6)  # near-zero head → delta ≈ 0 → output ≈ x[-1]
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))

    # Constant trajectory: every row [1.0, 2.0].
    constant = jnp.array([[1.0, 2.0]] * 5)              # T=5 history
    X = jnp.stack([constant])                            # (1, 5, 2)
    Y = jnp.array([[1.0, 2.0]])                          # target = persistence

    loss = float(mse_loss(model, X, Y))
    # With near-zero init head + small delta_scale, loss should be tiny
    # (delta ≈ 0 → prediction ≈ x[-1] = [1, 2] = target).
    assert loss < 0.01, (
        f"persistence prediction on constant trajectory should give tiny "
        f"loss; got {loss}")


# ---------- training reduces loss ------------------------------------------

def test_train_reduces_mse_on_synthetic_linear_trajectory():
    """Simple test: train on a 1-D ramp y_t = α·t. Persistence is a
    biased estimator (predicts y_t when truth is y_{t+1}), so a
    well-trained model should improve over the persistence baseline.

    NOTE: this is a smoke test — we only require that training
    monotonically reduces loss over a short budget, not that the
    model fully recovers the ramp."""
    np.random.seed(0)
    alpha = 0.05
    N = 50
    trajectory = (alpha * np.arange(N, dtype=np.float64))[:, None]  # (N, 1)
    T_window = 4

    X, Y = prepare_windows(trajectory, T_window)
    X_jax = jnp.asarray(X, dtype=jnp.float32)
    Y_jax = jnp.asarray(Y, dtype=jnp.float32)

    cfg = VectorForecasterConfig(
        input_dim=1, num_qubits=3, num_layers=1, step_dt=0.05,
        delta_scale_init=0.05, delta_scale_min=0.01,
        init_head_std=0.01)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(42))

    loss_before = float(mse_loss(model, X_jax, Y_jax))
    trained, hist = train_vector_forecaster(
        model, X, Y, steps=50, lr=1e-2, log_every=10)
    loss_after = float(mse_loss(trained, X_jax, Y_jax))

    # Loss should decrease (training is doing something).
    assert loss_after < loss_before, (
        f"training didn't reduce loss: before={loss_before:.4e}, "
        f"after={loss_after:.4e}, history={hist}")


def test_train_rejects_zero_steps():
    cfg = VectorForecasterConfig(input_dim=2, num_qubits=3)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))
    X = np.zeros((3, 4, 2))
    Y = np.zeros((3, 2))
    with pytest.raises(ValueError, match="steps"):
        train_vector_forecaster(model, X, Y, steps=0, lr=1e-3)


def test_train_rejects_non_positive_lr():
    cfg = VectorForecasterConfig(input_dim=2, num_qubits=3)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))
    X = np.zeros((3, 4, 2))
    Y = np.zeros((3, 2))
    with pytest.raises(ValueError, match="lr"):
        train_vector_forecaster(model, X, Y, steps=5, lr=0.0)


def test_train_returns_loss_history_at_log_cadence():
    """If log_every=k, history has steps/k entries."""
    cfg = VectorForecasterConfig(input_dim=1, num_qubits=2, num_layers=1)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))
    X = np.random.randn(5, 3, 1)
    Y = np.random.randn(5, 1)
    _, hist = train_vector_forecaster(
        model, X, Y, steps=20, lr=1e-3, log_every=5)
    assert len(hist) == 4  # steps 5, 10, 15, 20


def test_train_minibatch_runs():
    """Mini-batch training also runs end-to-end."""
    cfg = VectorForecasterConfig(input_dim=2, num_qubits=3, num_layers=1)
    model = VectorForecaster(cfg, key=jax.random.PRNGKey(0))
    X = np.random.randn(10, 3, 2).astype(np.float32)
    Y = np.random.randn(10, 2).astype(np.float32)
    trained, hist = train_vector_forecaster(
        model, X, Y, steps=10, lr=1e-3, batch_size=4, log_every=5)
    assert len(hist) == 2
    # Trained model is still callable.
    pred = trained(jnp.asarray(X[0]))
    assert pred.shape == (2,)
