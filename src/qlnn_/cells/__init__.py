"""Liquid quantum cells — vector fields conditioned on a quantum feature encoder.

Step 3 deliverable: LiquidQuantumCell — the continuous-time hidden-state vector
field used inside the JAX-side Liquid ODE forecaster. It wraps a
`QuantumFeatureEncoder` so the data conditioning of the dynamics is genuinely
quantum-circuit-derived.
"""

from .liquid_quantum_cell import (
    LiquidQuantumCell,
    LiquidQuantumCellConfig,
)

__all__ = [
    "LiquidQuantumCell",
    "LiquidQuantumCellConfig",
]
