"""JAX/Optax training utilities for the QLNN subpackage."""

from .losses import QLNNPhysicsLossConfig, logistic_growth_residual_loss
from .trainer import (
    HistoryRow,
    QLNNTrainerConfig,
    QLNNTrainResult,
    history_to_dicts,
    train_one_qlnn,
)

__all__ = [
    "QLNNTrainerConfig",
    "QLNNPhysicsLossConfig",
    "HistoryRow",
    "QLNNTrainResult",
    "train_one_qlnn",
    "history_to_dicts",
    "logistic_growth_residual_loss",
]
