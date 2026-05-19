"""Hardware-efficient ansatz (HEA).

The canonical QML baseline ansatz (Kandala et al. 2017, arXiv:1704.05018):
alternating single-qubit RY+RZ rotations followed by an entangling block.
Unlike `data_reuploading`, the data is encoded ONCE at the start (no
re-uploading), so this exposes how much of the QLNN's expressivity is owed
to re-uploading.

Registered as ``"hardware_efficient"``. Supported ``AnsatzConfig.params``:

    entanglement : "linear" | "ring" | "all_to_all"  (default "ring")
    encoding     : "rx" | "ry"                       (default "rx")
    device_name  : str                               (default "default.qubit")

Weight shape: ``(num_layers, num_qubits, 2)`` — two angles (RY, RZ) per
qubit per layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import pennylane as qml

from .protocol import AnsatzConfig, AnsatzProtocol, register
from .reuploading import _entangle


@dataclass(frozen=True)
class HardwareEfficientConfig:
    num_qubits: int = 4
    num_layers: int = 3
    entanglement: str = "ring"
    encoding: str = "rx"
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.entanglement not in ("linear", "ring", "all_to_all"):
            raise ValueError(f"entanglement must be linear/ring/all_to_all, got {self.entanglement!r}")
        if self.encoding not in ("rx", "ry"):
            raise ValueError(f"encoding must be 'rx' or 'ry', got {self.encoding!r}")


def _build_qnode(cfg: HardwareEfficientConfig) -> Callable:
    dev = qml.device(cfg.device_name, wires=cfg.num_qubits)
    encode_gate = qml.RX if cfg.encoding == "rx" else qml.RY

    @qml.qnode(dev, interface="jax")
    def circuit(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        # One-shot input encoding.
        for i in range(cfg.num_qubits):
            encode_gate(inputs[i], wires=i)

        for layer in range(cfg.num_layers):
            for i in range(cfg.num_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            _entangle(cfg.num_qubits, cfg.entanglement)

        return tuple(qml.expval(qml.PauliZ(i)) for i in range(cfg.num_qubits))

    return circuit


class HardwareEfficientCircuit:
    def __init__(self, config: HardwareEfficientConfig | None = None) -> None:
        self.config = config or HardwareEfficientConfig()
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
    return HardwareEfficientCircuit(HardwareEfficientConfig(
        num_qubits=cfg.num_qubits,
        num_layers=cfg.num_layers,
        entanglement=str(p.get("entanglement", "ring")),
        encoding=str(p.get("encoding", "rx")),
        device_name=str(p.get("device_name", "default.qubit")),
    ))


register("hardware_efficient", _factory, overwrite=True)
