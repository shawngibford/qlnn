"""Brickwall ansatz.

Alternating-pair CNOT entanglement (even-odd / odd-even pairs each layer)
with RY+RZ single-qubit rotations. Common in variational quantum
eigensolver work; produces a "brickwall" gate-pattern when drawn. Useful
for the search because its entanglement reach is *layer-bounded* (depth-d
brickwall mixes d wires), giving a different topological signature than
ring or all-to-all.

Registered as ``"brickwall"``. Supported ``AnsatzConfig.params``:

    encoding    : "rx" | "ry"   (default "rx")
    reupload    : bool           (default False — set True to re-inject
                                  inputs at every layer like data_reuploading)
    device_name : str            (default "default.qubit")

Weight shape: ``(num_layers, num_qubits, 2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import pennylane as qml

from .protocol import AnsatzConfig, AnsatzProtocol, register


@dataclass(frozen=True)
class BrickwallConfig:
    num_qubits: int = 4
    num_layers: int = 3
    encoding: str = "rx"
    reupload: bool = False
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.encoding not in ("rx", "ry"):
            raise ValueError(f"encoding must be 'rx' or 'ry', got {self.encoding!r}")


def _brick(num_qubits: int, even: bool) -> None:
    """Emit the even-pair or odd-pair CNOT layer for a brickwall pattern."""
    start = 0 if even else 1
    for i in range(start, num_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def _build_qnode(cfg: BrickwallConfig) -> Callable:
    dev = qml.device(cfg.device_name, wires=cfg.num_qubits)
    encode_gate = qml.RX if cfg.encoding == "rx" else qml.RY

    @qml.qnode(dev, interface="jax")
    def circuit(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        # First-layer encoding.
        for i in range(cfg.num_qubits):
            encode_gate(inputs[i], wires=i)

        for layer in range(cfg.num_layers):
            if cfg.reupload and layer > 0:
                for i in range(cfg.num_qubits):
                    encode_gate(inputs[i], wires=i)

            for i in range(cfg.num_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)

            # Brickwall: even-pair on even layers, odd-pair on odd layers.
            _brick(cfg.num_qubits, even=(layer % 2 == 0))

        return tuple(qml.expval(qml.PauliZ(i)) for i in range(cfg.num_qubits))

    return circuit


class BrickwallCircuit:
    def __init__(self, config: BrickwallConfig | None = None) -> None:
        self.config = config or BrickwallConfig()
        self._qnode = _build_qnode(self.config)

    @property
    def weight_shape(self) -> tuple[int, int, int]:
        c = self.config
        return (c.num_layers, c.num_qubits, 2)

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
        if isinstance(out, tuple):
            return jnp.stack(out)
        return out


def _factory(cfg: AnsatzConfig) -> AnsatzProtocol:
    p = cfg.params or {}
    return BrickwallCircuit(BrickwallConfig(
        num_qubits=cfg.num_qubits,
        num_layers=cfg.num_layers,
        encoding=str(p.get("encoding", "rx")),
        reupload=bool(p.get("reupload", False)),
        device_name=str(p.get("device_name", "default.qubit")),
    ))


register("brickwall", _factory, overwrite=True)
