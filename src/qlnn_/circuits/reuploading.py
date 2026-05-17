"""Data-reuploading parameterized quantum circuit (PQC).

Implements the data re-uploading universal classifier pattern from
Pérez-Salinas et al., 2020 (arXiv:1907.02085) — interleaves angle-encoding
data layers with parameterized variational layers and entangling layers.

This is the heart of the QLNN's "quantum feature encoder": few qubits, many
re-uploading layers, exponential expressivity in fitting truncated Fourier
series (see Schuld et al. 2021, arXiv:2008.08605).

The circuit is built lazily as a PennyLane `QNode` with the JAX interface, so
we can JIT-compile and take `jax.grad` through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import pennylane as qml


@dataclass(frozen=True)
class DataReuploadingConfig:
    num_qubits: int = 4
    num_layers: int = 3
    # If True, the entanglement stage adds a final wrap-around CNOT to close
    # the linear chain into a ring. Tends to help with all-to-all coupling on
    # small qubit counts.
    ring_entanglement: bool = True
    # Pennylane device name. "default.qubit" is the modern unified device that
    # picks the JAX interface automatically when called from JAX code.
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")


def _build_qnode(cfg: DataReuploadingConfig) -> Callable:
    """Return a JAX-interfaced QNode implementing the re-uploading circuit.

    The returned function has signature (inputs, weights) -> jnp.ndarray of
    shape (num_qubits,), each entry the PauliZ expectation on a wire.

    inputs : (num_qubits,)
        Bioprocess feature vector for one sample/time-step. Caller is
        responsible for normalizing into the angle-encoding range.
    weights : (num_layers, num_qubits, 3)
        Rotation parameters per layer / qubit (Rot = RZ ∘ RY ∘ RZ).
    """
    dev = qml.device(cfg.device_name, wires=cfg.num_qubits)

    @qml.qnode(dev, interface="jax")
    def circuit(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        for layer in range(cfg.num_layers):
            # Data re-uploading: re-inject the inputs at every layer (this is
            # what gives the circuit expressivity in the truncated-Fourier
            # sense — without it, only the first layer sees the data).
            for i in range(cfg.num_qubits):
                qml.RX(inputs[i], wires=i)

            # Variational block.
            for i in range(cfg.num_qubits):
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

            # Entangling block — linear chain (optionally closed into a ring).
            if cfg.num_qubits >= 2:
                for i in range(cfg.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if cfg.ring_entanglement and cfg.num_qubits > 2:
                    qml.CNOT(wires=[cfg.num_qubits - 1, 0])

        # Return a tuple of measurement processes; PennyLane wraps the tuple as
        # a sequence of scalar outputs. The wrapper below stacks them into a
        # single JAX array so downstream code sees shape (num_qubits,).
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(cfg.num_qubits))

    return circuit


class DataReuploadingCircuit:
    """Thin wrapper that owns a JAX-interfaced QNode and exposes the
    weight-shape contract.

    Holds no parameters — those are stored externally (e.g. in
    `QuantumFeatureEncoder`) so JIT/grad transformations see them as
    leaves of the PyTree.
    """

    def __init__(self, config: DataReuploadingConfig | None = None) -> None:
        self.config = config or DataReuploadingConfig()
        self._qnode = _build_qnode(self.config)

    @property
    def weight_shape(self) -> tuple[int, int, int]:
        c = self.config
        return (c.num_layers, c.num_qubits, 3)

    @property
    def output_dim(self) -> int:
        return self.config.num_qubits

    def __call__(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        if inputs.shape[-1] != self.config.num_qubits:
            raise ValueError(
                f"inputs last-dim must equal num_qubits ({self.config.num_qubits}), "
                f"got {inputs.shape[-1]}"
            )
        if weights.shape != self.weight_shape:
            raise ValueError(
                f"weights must have shape {self.weight_shape}, got {tuple(weights.shape)}"
            )
        out = self._qnode(inputs, weights)
        # PennyLane returns a tuple of per-wire expectations; stack into a single
        # (num_qubits,) array so it composes cleanly with JAX transforms.
        if isinstance(out, tuple):
            return jnp.stack(out)
        return out
