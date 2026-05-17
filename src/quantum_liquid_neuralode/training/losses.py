from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


Reduction = Literal["mean", "sum", "none"]


def _validate_time_points(time_points: Tensor) -> None:
    if time_points.ndim != 1:
        raise ValueError(f"time_points must be 1D, got shape={tuple(time_points.shape)}")
    if time_points.numel() < 2:
        raise ValueError("time_points must have at least 2 elements")
    if not torch.all(time_points[1:] > time_points[:-1]):
        raise ValueError("time_points must be strictly increasing")


def logistic_growth_residual_loss(
    od: Tensor,
    time_points: Tensor,
    *,
    mu: float,
    K: float,
    reduction: Reduction = "mean",
) -> Tensor:
    """Physics loss based on logistic growth residual.

    Residual:
        r(t) = dOD/dt - mu * OD * (1 - OD/K)

    Args:
        od: shape (T,), (batch, T), or (batch, T, 1)
        time_points: shape (T,) in hours (or any consistent unit)
        mu: growth rate (>0)
        K: carrying capacity (>0)

    Returns:
        loss tensor (scalar if reduction != "none")
    """
    if mu <= 0:
        raise ValueError(f"mu must be > 0, got {mu}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")

    _validate_time_points(time_points)

    if od.ndim == 3 and od.shape[-1] == 1:
        od_ = od.squeeze(-1)
    else:
        od_ = od

    if od_.ndim not in (1, 2):
        raise ValueError(f"od must be 1D or 2D (or 3D with trailing 1), got shape={tuple(od.shape)}")

    T = time_points.shape[0]
    if od_.shape[-1] != T:
        raise ValueError(f"od last dimension must match time_points (T={T}), got {od_.shape[-1]}")

    dt = time_points[1:] - time_points[:-1]  # (T-1,)
    dod_dt = (od_[..., 1:] - od_[..., :-1]) / dt  # (..., T-1)

    od_mid = od_[..., :-1]
    expected = mu * od_mid * (1.0 - (od_mid / K))

    residual = dod_dt - expected
    loss = residual.pow(2)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"Unknown reduction: {reduction}")


def smoothness_loss(
    sequence: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Second-difference smoothness penalty.

    Computes mean squared second differences along the last dimension.

    Args:
        sequence: shape (T,) or (batch, T)

    Returns:
        loss tensor (scalar if reduction != "none")
    """
    if sequence.ndim not in (1, 2):
        raise ValueError(f"sequence must be 1D or 2D, got shape={tuple(sequence.shape)}")
    if sequence.shape[-1] < 3:
        raise ValueError("sequence length must be at least 3 to compute second differences")

    first_diff = sequence[..., 1:] - sequence[..., :-1]
    second_diff = first_diff[..., 1:] - first_diff[..., :-1]

    loss = second_diff.pow(2)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"Unknown reduction: {reduction}")
