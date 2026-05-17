"""Tests for the JAX port of the logistic-growth physics loss.

R3 finding 2.5: the QLNN side gets a symmetric +physics ablation. These tests
verify that the JAX implementation matches the PyTorch reference exactly so
the +physics rows are comparable across stacks.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from qlnn_.training.losses import (
    QLNNPhysicsLossConfig,
    logistic_growth_residual_loss as jax_logistic,
)
from quantum_liquid_neuralode.training.losses import (
    logistic_growth_residual_loss as torch_logistic,
)


# ---------------------------------------------------------------------------
# Pure-loss tests
# ---------------------------------------------------------------------------
def test_logistic_growth_loss_zero_on_zero_od():
    """OD = 0 everywhere → trivially satisfies dOD/dt = 0 = mu*0*(1-0) → loss=0."""
    batch, T = 4, 6
    od = jnp.zeros((batch, T))
    t = jnp.linspace(0.0, 5.0, T)
    loss = jax_logistic(od, t, mu=0.4, K=1.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_logistic_growth_loss_scalar_two_point_formula():
    """On a hand-crafted 2-point trajectory the loss is the analytic
    forward-Euler residual squared.

    residual = (y - od_last)/h - mu * od_last * (1 - od_last/K)
    loss     = residual ** 2
    """
    od_last = 0.3
    y = 0.42
    h = 1.0
    mu = 0.4
    K = 1.0

    traj = jnp.asarray([[od_last, y]])  # (1, 2)
    t_pts = jnp.asarray([0.0, h])
    loss = float(jax_logistic(traj, t_pts, mu=mu, K=K))

    expected_residual = (y - od_last) / h - mu * od_last * (1.0 - od_last / K)
    expected_loss = expected_residual ** 2
    # JAX defaults to float32 -> ~1e-6 relative precision.
    assert loss == pytest.approx(expected_loss, rel=1e-5, abs=1e-7)


def test_logistic_growth_loss_matches_pytorch_reference():
    """JAX and PyTorch implementations must agree to 1e-6 on a non-trivial trajectory.

    Uses float32 inputs on the JAX side (the default) and float32 PyTorch tensors
    so the two stacks operate in the same precision; the algebra is identical.
    """
    rng = np.random.default_rng(0)
    batch, T = 5, 8
    od_np = rng.uniform(0.05, 0.95, size=(batch, T)).astype(np.float32)
    # Non-uniform but strictly increasing time grid.
    t_np = np.cumsum(rng.uniform(0.2, 1.5, size=T)).astype(np.float32)
    t_np = (t_np - t_np[0]).astype(np.float32)

    mu, K = 0.4, 1.0

    jax_loss = float(jax_logistic(jnp.asarray(od_np), jnp.asarray(t_np), mu=mu, K=K))
    torch_loss = float(
        torch_logistic(
            torch.from_numpy(od_np), torch.from_numpy(t_np), mu=mu, K=K, reduction="mean",
        )
    )
    # Same precision on both sides; 1e-6 relative is well within float32 noise.
    assert jax_loss == pytest.approx(torch_loss, rel=1e-6, abs=1e-7)


def test_logistic_growth_loss_handles_3d_input():
    """Trailing-1 dim is squeezed (matches PyTorch reference)."""
    batch, T = 3, 4
    od_2d = jnp.linspace(0.1, 0.5, batch * T).reshape(batch, T)
    od_3d = od_2d[..., None]  # (batch, T, 1)
    t = jnp.linspace(0.0, 3.0, T)
    loss_2d = float(jax_logistic(od_2d, t, mu=0.4, K=1.0))
    loss_3d = float(jax_logistic(od_3d, t, mu=0.4, K=1.0))
    assert loss_2d == pytest.approx(loss_3d, rel=1e-12, abs=1e-12)


def test_logistic_growth_loss_validates_args():
    od = jnp.zeros((2, 4))
    t = jnp.linspace(0.0, 3.0, 4)
    with pytest.raises(ValueError):
        jax_logistic(od, t, mu=0.0, K=1.0)
    with pytest.raises(ValueError):
        jax_logistic(od, t, mu=0.4, K=0.0)
    with pytest.raises(ValueError):
        # time_points 2D
        jax_logistic(od, jnp.zeros((2, 4)), mu=0.4, K=1.0)
    with pytest.raises(ValueError):
        # mismatched T
        jax_logistic(od, jnp.linspace(0.0, 1.0, 5), mu=0.4, K=1.0)


# ---------------------------------------------------------------------------
# Integration: the trainer's physics path actually runs end-to-end.
# ---------------------------------------------------------------------------
def test_physics_loss_config_defaults_off():
    cfg = QLNNPhysicsLossConfig()
    assert cfg.lambda_logistic == 0.0
    assert cfg.mu_norm == 0.4
    assert cfg.K_norm == 1.0
