"""Quantum feature encoder.

Maps a feature vector x ∈ ℝ^F to a quantum-derived latent ∈ ℝ^Q via:

    x  --linear F→Q-->  angles π·tanh(...)  --PQC-->  ⟨Z⟩_i  ∈ [-1, 1]

The pre-projection lets us absorb any feature count F into the fixed-Q
qubit space of the circuit and bounds the embedding angles into a clean
[-π, π] range (so the periodicity of RX is not an issue).

The PQC itself is pluggable via the `ansatz` field of
`QuantumFeatureEncoderConfig`: any ansatz registered in
`src/qlnn_/circuits/protocol.py` (data_reuploading, hardware_efficient,
strongly_entangling, brickwall, or custom) can be slotted in by name.
Backward compatibility: when `ansatz=None`, the encoder builds the
historical data-reuploading circuit so existing checkpoints and configs
keep working unchanged.

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

from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ..circuits import AnsatzConfig, AnsatzProtocol, build as build_ansatz


@dataclass(frozen=True)
class QuantumFeatureEncoderConfig:
    input_dim: int = 7
    num_qubits: int = 4
    num_layers: int = 3
    # Legacy bool — used only when `ansatz` is None (the historical
    # data_reuploading path). Kept for backward compat with existing
    # checkpoints and YAML configs.
    ring_entanglement: bool = True
    # Std-dev of the Gaussian used to init the linear projection W. Small
    # values keep the initial embedding angles in a benign part of [-π, π].
    init_w_std: float = 0.1
    # Std-dev used to init circuit weights (around 0). Tiny std keeps the
    # initial circuit close to identity → ⟨Z⟩ ≈ 1, gradients well-defined.
    init_circuit_std: float = 0.05
    # New: declarative ansatz spec. When None (default), the encoder builds
    # the historical `data_reuploading` circuit using `num_qubits`,
    # `num_layers`, and `ring_entanglement` so all pre-existing call sites
    # keep behaving identically.
    ansatz: AnsatzConfig | None = None

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.ansatz is not None:
            # When ansatz is set, num_qubits/num_layers must match — the
            # encoder's downstream consumers depend on these for shape contracts.
            if self.ansatz.num_qubits != self.num_qubits:
                raise ValueError(
                    f"ansatz.num_qubits ({self.ansatz.num_qubits}) must match "
                    f"encoder num_qubits ({self.num_qubits})"
                )
            if self.ansatz.num_layers != self.num_layers:
                raise ValueError(
                    f"ansatz.num_layers ({self.ansatz.num_layers}) must match "
                    f"encoder num_layers ({self.num_layers})"
                )

    def resolved_ansatz(self) -> AnsatzConfig:
        """Return the ansatz config that will actually be built, applying
        the backward-compat default if `ansatz` is None.
        """
        if self.ansatz is not None:
            return self.ansatz
        return AnsatzConfig(
            name="data_reuploading",
            num_qubits=self.num_qubits,
            num_layers=self.num_layers,
            params={"ring_entanglement": self.ring_entanglement},
        )


class QuantumFeatureEncoder(eqx.Module):
    """Feature-vector → quantum-latent encoder.

    Parameters (PyTree leaves):
        W : (input_dim, num_qubits)   linear projection weights
        b : (num_qubits,)             projection bias
        circuit_weights : ansatz.weight_shape (e.g. (num_layers, num_qubits, 3))

    Static (non-PyTree) state:
        config  — QuantumFeatureEncoderConfig
        circuit — the built ansatz (owns the QNode)
    """

    W: jnp.ndarray
    b: jnp.ndarray
    circuit_weights: jnp.ndarray

    # Static fields — not differentiated, not JIT-traced as leaves.
    config: QuantumFeatureEncoderConfig = eqx.field(static=True)
    circuit: AnsatzProtocol = eqx.field(static=True)

    def __init__(
        self,
        config: QuantumFeatureEncoderConfig,
        *,
        key: jax.Array,
    ) -> None:
        self.config = config

        self.circuit = build_ansatz(config.resolved_ansatz())

        k_w, _k_b, k_c = jax.random.split(key, 3)

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
