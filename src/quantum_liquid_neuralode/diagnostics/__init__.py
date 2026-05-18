"""Diagnostics for the classical (PyTorch) Liquid-ODE side
(effective dimension, ...)."""

from .effective_dimension import (
    empirical_fisher,
    normalized_effective_dimension,
    effective_dimension_curve,
    flatten_model_params,
    classical_forward_from_flat,
)

__all__ = [
    "empirical_fisher",
    "normalized_effective_dimension",
    "effective_dimension_curve",
    "flatten_model_params",
    "classical_forward_from_flat",
]
