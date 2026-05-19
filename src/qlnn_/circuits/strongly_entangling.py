"""Strongly-entangling-layers ansatz.

Thin wrapper around PennyLane's built-in ``qml.StronglyEntanglingLayers``
template (Schuld et al. 2020, arXiv:1804.00633). Each layer applies a Rot
(RZ·RY·RZ) gate per qubit followed by long-range CNOT-with-range
entanglement, providing strong mixing on a small qubit budget.

Registered as ``"strongly_entangling"``. Supported ``AnsatzConfig.params``:

    encoding    : "rx" | "ry"               (default "rx")
    ranges      : list[int] | None          (default None — PennyLane's default 1..n-1)
    device_name : str                       (default "default.qubit")

Weight shape: ``(num_layers, num_qubits, 3)`` — PennyLane's template signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp
import pennylane as qml

from .protocol import AnsatzConfig, AnsatzProtocol, register


@dataclass(frozen=True)
class StronglyEntanglingConfig:
    num_qubits: int = 4
    num_layers: int = 3
    encoding: str = "rx"
    ranges: tuple[int, ...] | None = None
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.encoding not in ("rx", "ry"):
            raise ValueError(f"encoding must be 'rx' or 'ry', got {self.encoding!r}")
        if self.ranges is not None:
            if len(self.ranges) != self.num_layers:
                raise ValueError(
                    f"ranges must have length num_layers={self.num_layers}, "
                    f"got {len(self.ranges)}"
                )
            for r in self.ranges:
                if not (1 <= r < self.num_qubits):
                    raise ValueError(
                        f"each range must satisfy 1 <= r < num_qubits, got {r}"
                    )


def _build_qnode(cfg: StronglyEntanglingConfig) -> Callable:
    dev = qml.device(cfg.device_name, wires=cfg.num_qubits)
    encode_gate = qml.RX if cfg.encoding == "rx" else qml.RY
    ranges = list(cfg.ranges) if cfg.ranges is not None else None

    @qml.qnode(dev, interface="jax")
    def circuit(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        for i in range(cfg.num_qubits):
            encode_gate(inputs[i], wires=i)
        # PennyLane template handles the entanglement internally.
        qml.StronglyEntanglingLayers(
            weights=weights, wires=range(cfg.num_qubits), ranges=ranges
        )
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(cfg.num_qubits))

    return circuit


class StronglyEntanglingCircuit:
    def __init__(self, config: StronglyEntanglingConfig | None = None) -> None:
        self.config = config or StronglyEntanglingConfig()
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
        if isinstance(out, tuple):
            return jnp.stack(out)
        return out


def _factory(cfg: AnsatzConfig) -> AnsatzProtocol:
    p = cfg.params or {}
    ranges = p.get("ranges")
    if ranges is not None:
        ranges = tuple(int(r) for r in ranges)
    return StronglyEntanglingCircuit(StronglyEntanglingConfig(
        num_qubits=cfg.num_qubits,
        num_layers=cfg.num_layers,
        encoding=str(p.get("encoding", "rx")),
        ranges=ranges,
        device_name=str(p.get("device_name", "default.qubit")),
    ))


register("strongly_entangling", _factory, overwrite=True)
