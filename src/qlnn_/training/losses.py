"""Physics-informed losses for the JAX-side trainer.

Mirrors the PyTorch logistic-growth residual loss in
``quantum_liquid_neuralode.training.losses.logistic_growth_residual_loss`` so
the +physics ablation row is symmetric across the two stacks (R3 finding 2.5).

Discretization is **left-endpoint forward Euler** to match the PyTorch
reference exactly:

    r_i = (od[i+1] - od[i]) / (t[i+1] - t[i]) - mu * od[i] * (1 - od[i] / K)

so that for a 2-point trajectory (od_last, y_pred) at times (0, h) the loss
collapses to a single scalar residual squared.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class QLNNPhysicsLossConfig:
    """Weights for physics-informed regularizers on the JAX trainer.

    Mirrors ``quantum_liquid_neuralode.training.trainer.PhysicsLossConfig`` so a
    +physics ablation row at the QLNN side is symmetric with the classical row.

    All defaults are off (lambda_logistic=0.0) — physics is opt-in per config.
    """
    lambda_logistic: float = 0.0
    # Logistic-growth params (only used if lambda_logistic > 0).
    # These are in NORMALIZED OD space because the model emits normalized OD.
    mu_norm: float = 0.4       # growth rate (1/h)
    K_norm: float = 1.0        # carrying capacity in normalized OD


def _validate_time_points(time_points: jnp.ndarray) -> None:
    if time_points.ndim != 1:
        raise ValueError(f"time_points must be 1D, got shape={tuple(time_points.shape)}")
    if time_points.shape[0] < 2:
        raise ValueError("time_points must have at least 2 elements")


def logistic_growth_residual_loss(
    od: jnp.ndarray,
    time_points: jnp.ndarray,
    *,
    mu: float,
    K: float,
) -> jnp.ndarray:
    """JAX port of the PyTorch logistic-growth residual loss.

    Residual:
        r(t) = dOD/dt - mu * OD * (1 - OD/K)

    Discretization (left-endpoint forward Euler, matches the PyTorch reference):
        r_i = (od[i+1] - od[i]) / (t[i+1] - t[i]) - mu * od[i] * (1 - od[i]/K)

    Args:
        od: shape (T,), (batch, T), or (batch, T, 1)
        time_points: shape (T,) in hours (or any consistent unit)
        mu: growth rate (>0)
        K: carrying capacity (>0)

    Returns:
        scalar mean-squared residual.
    """
    if mu <= 0:
        raise ValueError(f"mu must be > 0, got {mu}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")

    _validate_time_points(time_points)

    # Squeeze a trailing-1 dimension to support (batch, T, 1) inputs.
    if od.ndim == 3 and od.shape[-1] == 1:
        od_ = jnp.squeeze(od, axis=-1)
    else:
        od_ = od

    if od_.ndim not in (1, 2):
        raise ValueError(
            f"od must be 1D or 2D (or 3D with trailing 1), got shape={tuple(od.shape)}"
        )

    T = time_points.shape[0]
    if od_.shape[-1] != T:
        raise ValueError(
            f"od last dimension must match time_points (T={T}), got {od_.shape[-1]}"
        )

    dt = time_points[1:] - time_points[:-1]  # (T-1,)
    dod_dt = (od_[..., 1:] - od_[..., :-1]) / dt  # (..., T-1)

    # Left-endpoint sampling of the vector field (forward Euler residual).
    od_left = od_[..., :-1]
    expected = mu * od_left * (1.0 - (od_left / K))

    residual = dod_dt - expected
    return jnp.mean(residual ** 2)
