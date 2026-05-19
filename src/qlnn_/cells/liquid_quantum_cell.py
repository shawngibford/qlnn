"""Liquid Quantum Cell — continuous-time vector field with a quantum encoder.

The hidden state h ∈ ℝ^Q (Q := num_qubits) evolves under the ODE

    dh/dt = -(1/tau + q(x)) ⊙ h + A ⊙ q(x)

where q(x) ∈ ℝ^Q is the per-qubit PauliZ-expectation output of a
`QuantumFeatureEncoder` evaluated on the (held-constant) feature vector x,
tau ∈ ℝ_{>0}^Q is a vector of learnable time constants, and A ∈ ℝ^Q is a
learnable per-qubit input amplitude.

This is the JAX-side analog of `quantum_liquid_neuralode.models.LiquidCell`,
adapted to a true Liquid-CT-RNN form (Hasani et al., 2021) in which the
input also modulates the leak term — q(x) acts as a synaptic conductance.
By construction the hidden width equals the number of qubits, so the quantum
readout slots directly into the dynamics without a learned projection.

Design choices mirror `QuantumFeatureEncoder`:
- `equinox.Module` so the cell lives cleanly in JAX PyTree land. Trainable
  arrays (`tau_unconstrained`, `A`, and the encoder's `W`, `b`,
  `circuit_weights`) are PyTree leaves; the `config` is static.
- Single-sample vector field. Batch via
  `jax.vmap(cell, in_axes=(None, 0, 0))`.
- The signature `(t, h, x)` matches the Diffrax `ODETerm` convention so the
  forecaster can wrap this directly without an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp

from ..circuits import AnsatzConfig
from ..encoders.quantum_feature_encoder import (
    QuantumFeatureEncoder,
    QuantumFeatureEncoderConfig,
)


@dataclass(frozen=True)
class LiquidQuantumCellConfig:
    """Configuration for `LiquidQuantumCell`.

    Attributes:
        input_dim: Raw feature count F (matches the dataset feature width).
        num_qubits: Hidden dimension Q. The hidden state width is locked to
            this value by design — it matches the quantum readout dim.
        num_layers: Re-uploading layers in the encoder PQC.
        tau_min: Lower bound on each time constant (softplus offset).
        tau_init: Initial value of `cell.tau()`. Must satisfy
            `tau_init > tau_min` so the inverse-softplus is well-defined.
        ring_entanglement: Pass-through to `QuantumFeatureEncoderConfig`
            (used only when `ansatz` is None — historical default path).
        init_w_std: Pass-through to `QuantumFeatureEncoderConfig`.
        init_circuit_std: Pass-through to `QuantumFeatureEncoderConfig`.
        ansatz: Optional declarative ansatz spec. When None (default),
            the encoder builds the historical data-reuploading circuit so
            existing checkpoints and YAML configs keep working unchanged.
    """

    input_dim: int
    num_qubits: int = 4
    num_layers: int = 3
    tau_min: float = 0.1
    tau_init: float = 1.0

    # Encoder init knobs (pass-through).
    ring_entanglement: bool = True
    init_w_std: float = 0.1
    init_circuit_std: float = 0.05

    # New: ansatz spec (None = backward-compat default).
    ansatz: AnsatzConfig | None = None

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.tau_min <= 0:
            raise ValueError(f"tau_min must be > 0, got {self.tau_min}")
        # Contractivity guard. The cell's vector field is
        #     dh/dt = -(1/tau + q(x)) ⊙ h + A ⊙ q(x)
        # with q(x) ∈ [-1, 1] (PauliZ expectations). For any tau >= tau_min,
        # 1/tau <= 1/tau_min. If tau_min > 1 then 1/tau_min < 1, so q(x) = -1
        # makes the leak coefficient (1/tau + q(x)) negative and the cell
        # flips from contractive to exponentially growing — a silent stability
        # landmine under HPO. Enforce tau_min <= 1 so 1/tau >= 1 >= |q(x)|
        # and the leak stays non-negative.
        if self.tau_min > 1.0:
            raise ValueError(
                f"tau_min must be <= 1.0 for the cell to stay contractive "
                f"(leak coefficient 1/tau + q(x) is guaranteed non-negative "
                f"only when 1/tau_min >= 1 >= |q(x)|), got tau_min={self.tau_min}"
            )
        # softplus(value) + tau_min == tau_init requires tau_init > tau_min so
        # that softplus(value) = tau_init - tau_min > 0 has a real solution.
        if self.tau_init <= self.tau_min:
            raise ValueError(
                f"tau_init must be > tau_min ({self.tau_min}), got tau_init={self.tau_init}"
            )


class LiquidQuantumCell(eqx.Module):
    """Liquid quantum cell vector field.

    Parameters (PyTree leaves):
        encoder           : `QuantumFeatureEncoder` (its W/b/circuit_weights are leaves)
        tau_unconstrained : (num_qubits,) — softplus → positive time constants
        A                 : (num_qubits,) — per-qubit input amplitude

    Static:
        config : `LiquidQuantumCellConfig`

    The vector field is single-sample; batch via `jax.vmap(in_axes=(None, 0, 0))`.
    """

    encoder: QuantumFeatureEncoder
    tau_unconstrained: jnp.ndarray
    A: jnp.ndarray

    config: LiquidQuantumCellConfig = eqx.field(static=True)

    def __init__(self, config: LiquidQuantumCellConfig, *, key: jax.Array) -> None:
        self.config = config

        # Split the key so the encoder is initialized deterministically from
        # one sub-key; the second sub-key is reserved for any future random
        # parameter initialization (currently tau/A are deterministic).
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

        # Inverse softplus: solve softplus(v) + tau_min = tau_init for v.
        #   softplus(v) = log(1 + exp(v)) = tau_init - tau_min
        #   exp(v) = exp(tau_init - tau_min) - 1
        #   v = log(exp(tau_init - tau_min) - 1)
        # __post_init__ guarantees tau_init - tau_min > 0 so the argument is > 0.
        delta = float(config.tau_init - config.tau_min)
        init_unconstrained = float(jnp.log(jnp.expm1(jnp.asarray(delta))))
        self.tau_unconstrained = jnp.full((config.num_qubits,), init_unconstrained)

        # Per-qubit input amplitude — initialized to ones per the spec.
        self.A = jnp.ones((config.num_qubits,))

    def tau(self) -> jnp.ndarray:
        """Positive per-qubit time constants, shape (num_qubits,).

        Uses softplus + `tau_min` to keep tau strictly above `tau_min`.
        """
        return jnn.softplus(self.tau_unconstrained) + self.config.tau_min

    def __call__(self, t: Any, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Compute dh/dt for ONE sample.

        Args:
            t: Scalar time. Accepted for Diffrax compatibility; unused
               because the vector field is autonomous given the held-constant
               feature vector x.
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
        inv_tau = 1.0 / self.tau()       # (Q,) — positive
        # Liquid-CT-RNN form: input acts both as a leak conductance and as a
        # driving signal scaled by the per-qubit amplitude A.
        leak = (inv_tau + q) * h
        drive = self.A * q
        return -leak + drive

    def num_parameters(self) -> int:
        """Total trainable parameter count (encoder + tau_unconstrained + A)."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        return sum(int(jnp.size(leaf)) for leaf in leaves)
