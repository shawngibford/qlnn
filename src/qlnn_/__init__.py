"""qlnn_ — Quantum / JAX side of the quantum-liquid neural network project.

Built with JAX + Equinox + PennyLane (default.qubit.jax device). Coexists with
the PyTorch `quantum_liquid_neuralode` package and shares its data preprocessing
and evaluation protocol via numpy-array hand-off at module boundaries.

Step 2 deliverable: QuantumFeatureEncoder (data-reuploading PQC).
"""

from .encoders.quantum_feature_encoder import (
    QuantumFeatureEncoder,
    QuantumFeatureEncoderConfig,
)
from .circuits.reuploading import (
    DataReuploadingCircuit,
    DataReuploadingConfig,
)

__all__ = [
    "QuantumFeatureEncoder",
    "QuantumFeatureEncoderConfig",
    "DataReuploadingCircuit",
    "DataReuploadingConfig",
]
