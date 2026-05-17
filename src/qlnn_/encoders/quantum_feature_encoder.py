"""Quantum feature encoder.

Maps a feature vector x ∈ ℝ^F to a quantum-derived latent ∈ ℝ^Q via:

    x  --linear F→Q-->  angles π·tanh(...)  --re-uploading PQC-->  ⟨Z⟩_i  ∈ [-1, 1]

The pre-projection lets us absorb any feature count F into the fixed-Q
qubit space of the circuit and bounds the embedding angles into a clean
[-π, π] range (so the periodicity of RX is not an issue).

This is the standalone deliverable of step 2 — it's used in step 3 by the
Liquid Quantum Cell as the vector field's data conditioning.

Design choices:
- Built as an `equinox.Module` so it lives in JAX PyTree land cleanly
  (gradients, JIT, vmap all work without bookkeeping).
- The PennyLane QNode is held as a *static* (non-PyTree-leaf) attribute via
  `eqx.field(static=True)`, while the trainable arrays (linear weights,
  circuit weights) are PyTree leaves.
- Single-sample forward; use `jax.vmap` over a batch dimension at the call
  site to stay JIT-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ..circuits.reuploading import DataReuploadingCircuit, DataReuploadingConfig


@dataclass(frozen=True)
class QuantumFeatureEncoderConfig:
    input_dim: int = 7
    num_qubits: int = 4
    num_layers: int = 3
    ring_entanglement: bool = True
    # Std-dev of the Gaussian used to init the linear projection W. Small
    # values keep the initial embedding angles in a benign part of [-π, π].
    init_w_std: float = 0.1
    # Std-dev used to init circuit weights (around 0). Tiny std keeps the
    # initial circuit close to identity → ⟨Z⟩ ≈ 1, gradients well-defined.
    init_circuit_std: float = 0.05

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")


class QuantumFeatureEncoder(eqx.Module):
    """Feature-vector → quantum-latent encoder.

    Parameters (PyTree leaves):
        W : (input_dim, num_qubits)   linear projection weights
        b : (num_qubits,)             projection bias
        circuit_weights : (num_layers, num_qubits, 3)   PQC rotation params

    Static (non-PyTree) state:
        config  — QuantumFeatureEncoderConfig
        circuit — DataReuploadingCircuit (owns the QNode)
    """

    W: jnp.ndarray
    b: jnp.ndarray
    circuit_weights: jnp.ndarray

    # Static fields — not differentiated, not JIT-traced as leaves.
    config: QuantumFeatureEncoderConfig = eqx.field(static=True)
    circuit: DataReuploadingCircuit = eqx.field(static=True)

    def __init__(
        self,
        config: QuantumFeatureEncoderConfig,
        *,
        key: jax.Array,
    ) -> None:
        self.config = config

        circuit_cfg = DataReuploadingConfig(
            num_qubits=config.num_qubits,
            num_layers=config.num_layers,
            ring_entanglement=config.ring_entanglement,
        )
        self.circuit = DataReuploadingCircuit(circuit_cfg)

        k_w, k_b, k_c = jax.random.split(key, 3)

        self.W = config.init_w_std * jax.random.normal(
            k_w, (config.input_dim, config.num_qubits)
        )
        self.b = jnp.zeros((config.num_qubits,))
        # circuit weights ~ N(0, init_circuit_std^2)
        self.circuit_weights = config.init_circuit_std * jax.random.normal(
            k_c, self.circuit.weight_shape
        )

    @property
    def output_dim(self) -> int:
        return int(self.config.num_qubits)

    def _project_to_angles(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project a single sample to bounded rotation angles.

        x : (input_dim,)  ->  angles : (num_qubits,) in (-π, π).

        We use π · tanh(W x + b) so the angle is smoothly bounded into the
        natural period of RX. This keeps gradients well-defined regardless
        of the input scale.
        """
        pre = x @ self.W + self.b
        return jnp.pi * jnp.tanh(pre)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode a single feature vector.

        x : (input_dim,) -> latent : (num_qubits,), each entry in [-1, 1].

        For a batched call, wrap this with `jax.vmap`:
            batched_encoder = jax.vmap(encoder)
            batched_encoder(X)   # X : (batch, input_dim) -> (batch, num_qubits)
        """
        if x.shape != (self.config.input_dim,):
            raise ValueError(
                f"x must have shape ({self.config.input_dim},), got {tuple(x.shape)}"
            )
        angles = self._project_to_angles(x)
        return self.circuit(angles, self.circuit_weights)

    def num_parameters(self) -> int:
        """Total trainable parameter count (linear + circuit)."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        return sum(int(jnp.size(leaf)) for leaf in leaves)


def encoder_apply_batched(encoder: QuantumFeatureEncoder, x_batch: jnp.ndarray) -> jnp.ndarray:
    """Convenience: vmap-applied encoder for a (batch, input_dim) array.

    Returns (batch, num_qubits).
    """
    if x_batch.ndim != 2:
        raise ValueError(f"x_batch must be 2D (batch, input_dim), got {tuple(x_batch.shape)}")
    if x_batch.shape[-1] != encoder.config.input_dim:
        raise ValueError(
            f"x_batch last dim must equal input_dim={encoder.config.input_dim}, got {x_batch.shape[-1]}"
        )
    return jax.vmap(encoder)(x_batch)
