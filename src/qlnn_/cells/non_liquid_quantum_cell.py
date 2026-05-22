"""Non-Liquid Quantum Cell — continuous-time vector field WITHOUT liquid τ.

Implements the τ-ablated variant of `LiquidQuantumCell` for the P7.11
2×2-completion sprint. The pre-registered forecaster H1 contrast
confounds the quantum circuit with the liquid-τ machinery. P7.10 closed
the classical side of the 2×2 with `ClassicalLTCForecaster`; this module
closes the quantum side with the analogous τ-ablated quantum cell.

The hidden state h ∈ ℝ^Q (Q := num_qubits) evolves under the ODE

    dh/dt = -q(x) ⊙ h + A ⊙ q(x)

i.e. the LiquidQuantumCell dynamics

    dh/dt = -(1/τ + q(x)) ⊙ h + A ⊙ q(x)

with the `1/τ` leak coefficient REMOVED. The only structural change
from `LiquidQuantumCell` is the loss of the `tau_unconstrained` parameter
and its contribution to the leak. The same `QuantumFeatureEncoder`, the
same per-qubit amplitude `A`, the same `(t, h, x) → dh/dt` Diffrax-ready
signature.

Contractivity note: with τ removed, the leak coefficient is `q(x) ∈
[-1, 1]`. When q(x) < 0 the cell becomes locally expansive (the hidden
state amplifies). This is the standard non-liquid Neural-ODE
behavior — the network has no built-in stability prior. The Diffrax
integrator handles this without issue; the trial-solution dynamics
either learn to stay in a bounded regime or diverge, and the rollout
metric (relative-L²) penalizes divergence directly.

This module is the minimum-faithful mirror of `LiquidQuantumCell`
required for the 2×2 ablation: same quantum readout, same A, same
Diffrax signature, ONLY the τ-leak removed. Any further architectural
change would confound the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ..circuits import AnsatzConfig
from ..encoders.quantum_feature_encoder import (
    QuantumFeatureEncoder,
    QuantumFeatureEncoderConfig,
)


@dataclass(frozen=True)
class NonLiquidQuantumCellConfig:
    """Configuration for `NonLiquidQuantumCell`.

    Mirrors `LiquidQuantumCellConfig` field-for-field with the two τ
    scalars (`tau_min`, `tau_init`) REMOVED — the cell has no τ.

    Attributes:
        input_dim: Raw feature count F (matches the dataset feature width).
        num_qubits: Hidden dimension Q. The hidden state width is locked to
            this value by design — it matches the quantum readout dim.
        num_layers: Re-uploading layers in the encoder PQC.
        ring_entanglement: Pass-through to `QuantumFeatureEncoderConfig`.
        init_w_std: Pass-through to `QuantumFeatureEncoderConfig`.
        init_circuit_std: Pass-through to `QuantumFeatureEncoderConfig`.
        ansatz: Optional declarative ansatz spec. When None (default),
            the encoder builds the historical data-reuploading circuit.
    """

    input_dim: int
    num_qubits: int = 4
    num_layers: int = 3

    # Encoder init knobs (pass-through to the same QuantumFeatureEncoder).
    ring_entanglement: bool = True
    init_w_std: float = 0.1
    init_circuit_std: float = 0.05

    ansatz: AnsatzConfig | None = None

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")


class NonLiquidQuantumCell(eqx.Module):
    """Non-Liquid quantum cell vector field — τ-ablated LiquidQuantumCell.

    Parameters (PyTree leaves):
        encoder : `QuantumFeatureEncoder` (its W/b/circuit_weights are leaves)
        A       : (num_qubits,) — per-qubit input amplitude

    Static:
        config : `NonLiquidQuantumCellConfig`

    Vector field:

        dh/dt = -q(x) ⊙ h + A ⊙ q(x)

    where q(x) ∈ ℝ^Q is the per-qubit PauliZ-expectation output of the
    encoder. The `LiquidQuantumCell` dynamics has an additional `-h/τ`
    leak; that term is removed here.

    The vector field is single-sample; batch via
    `jax.vmap(in_axes=(None, 0, 0))`.
    """

    encoder: QuantumFeatureEncoder
    A: jnp.ndarray

    config: NonLiquidQuantumCellConfig = eqx.field(static=True)

    def __init__(self, config: NonLiquidQuantumCellConfig, *,
                 key: jax.Array) -> None:
        self.config = config

        k_encoder, _k_reserved = jax.random.split(key, 2)

        encoder_cfg = QuantumFeatureEncoderConfig(
            input_dim=config.input_dim,
            num_qubits=config.num_qubits,
            num_layers=config.num_layers,
            ring_entanglement=config.ring_entanglement,
            init_w_std=config.init_w_std,
            init_circuit_std=config.init_circuit_std,
            ansatz=config.ansatz,
        )
        self.encoder = QuantumFeatureEncoder(encoder_cfg, key=k_encoder)

        # Per-qubit input amplitude — initialized to ones per the
        # LiquidQuantumCell convention so the non-liquid cell starts
        # with the same drive scaling as its liquid counterpart.
        self.A = jnp.ones((config.num_qubits,))

    def __call__(self, t: Any, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Compute dh/dt for ONE sample.

        Args:
            t: Scalar time. Accepted for Diffrax compatibility; unused
               because the vector field is autonomous given the held-
               constant feature vector x.
            h: (num_qubits,) hidden state.
            x: (input_dim,) raw feature vector — held constant on the
               integration interval (zero-order-hold).

        Returns:
            dh_dt: (num_qubits,) time derivative of h.
        """
        Q = self.config.num_qubits
        F = self.config.input_dim

        if h.shape != (Q,):
            raise ValueError(
                f"h must have shape ({Q},), got {tuple(h.shape)}"
            )
        if x.shape != (F,):
            raise ValueError(
                f"x must have shape ({F},), got {tuple(x.shape)}"
            )

        q = self.encoder(x)              # (Q,) — PauliZ expectations in [-1, 1]
        # Non-liquid form: drop the `1/τ` term from the LiquidQuantumCell
        # leak. Only q(x) contributes to the leak coefficient now.
        leak = q * h
        drive = self.A * q
        return -leak + drive

    def num_parameters(self) -> int:
        """Total trainable parameter count (encoder + A; NO tau)."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        return sum(int(jnp.size(leaf)) for leaf in leaves)
