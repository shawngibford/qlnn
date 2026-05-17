"""qlnn_ — Quantum / JAX side of the quantum-liquid neural network project.

Built with JAX + Equinox + PennyLane (default.qubit, JAX interface). Coexists
with the PyTorch `quantum_liquid_neuralode` package; data crosses the boundary
as numpy arrays so the locked evaluation protocol is identical across stacks.

Modules:
- circuits/ — Parameterized quantum circuits (data re-uploading).
- encoders/ — QuantumFeatureEncoder (linear → angles → PQC → ⟨Z⟩).
- cells/    — LiquidQuantumCell (Liquid CT-RNN with quantum-modulated VF).
- models/   — QLNNForecaster (full hybrid forecaster with Diffrax integration).
- training/ — Optax-driven trainer producing ForecastMetrics shared with
              the classical baseline.
"""

from .circuits.reuploading import DataReuploadingCircuit, DataReuploadingConfig
from .encoders.quantum_feature_encoder import (
    QuantumFeatureEncoder,
    QuantumFeatureEncoderConfig,
)
from .cells.liquid_quantum_cell import LiquidQuantumCell, LiquidQuantumCellConfig
from .models.qlnn_forecaster import QLNNForecaster, QLNNForecasterConfig
from .training.trainer import (
    QLNNTrainerConfig,
    HistoryRow,
    QLNNTrainResult,
    train_one_qlnn,
    history_to_dicts,
)

__all__ = [
    # circuits
    "DataReuploadingCircuit",
    "DataReuploadingConfig",
    # encoders
    "QuantumFeatureEncoder",
    "QuantumFeatureEncoderConfig",
    # cells
    "LiquidQuantumCell",
    "LiquidQuantumCellConfig",
    # models
    "QLNNForecaster",
    "QLNNForecasterConfig",
    # training
    "QLNNTrainerConfig",
    "HistoryRow",
    "QLNNTrainResult",
    "train_one_qlnn",
    "history_to_dicts",
]
