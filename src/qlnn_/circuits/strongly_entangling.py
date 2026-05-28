"""Strongly-entangling-layers ansatz.

Thin wrapper around PennyLane's built-in ``qml.StronglyEntanglingLayers``
template (Schuld et al. 2020, arXiv:1804.00633). Each layer applies a Rot
(RZ·RY·RZ) gate per qubit followed by long-range CNOT-with-range
entanglement, providing strong mixing on a small qubit budget.

Registered as ``"strongly_entangling"``. Supported ``AnsatzConfig.params``:

    encoding    : "rx" | "ry"               (default "rx")
    ranges      : list[int] | None          (default: long-range — see below)
    device_name : str                       (default "default.qubit")

Weight shape: ``(num_layers, num_qubits, 3)`` — PennyLane's template signature.

**Aliasing fix (2026-05-28).** Prior behavior was ``ranges=None`` →
PennyLane's per-layer fallback ``r = l mod num_qubits``. At the project's
forecaster config (num_qubits=3, num_layers=1) this gives layer-0 r = 0,
which PennyLane silently rewrites to r=1 (nearest-neighbor ring CNOT) —
making this circuit's unitary BIT-IDENTICAL to ``data_reuploading`` at
that config. Audit report (2026-05-28) confirmed 16-digit identical
relL² across all P4 forecaster cells.

The fix: when ``ranges`` is left unset, default to long-range
``r = num_qubits - 1`` on every layer. This realizes the "strongly
entangling" name — every layer's CNOT spans the maximum non-trivial
range, distinct from data-reuploading's nearest-neighbor pattern. At
num_qubits=3 this gives r=2 (skip-one CNOT); at larger n it gives the
longest-range entangler available.

Callers who want PennyLane's per-layer-modulo behavior can still pass
explicit ranges (e.g. ``params={"ranges": [1, 2, 1]}``).
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


def _default_long_range_ranges(num_qubits: int, num_layers: int) -> list[int]:
    """Default ranges for StronglyEntanglingLayers when caller leaves it
    unset: every layer uses the maximum non-trivial range r = n-1.

    This is the aliasing-fix described in the module docstring. PennyLane's
    own fallback (r = l mod n) reduces to nearest-neighbor (r=1) at small
    configs and produces a unitary identical to data_reuploading's ring
    CNOT. Explicit r = n-1 gives a genuinely distinct entangling pattern
    (skip-(n-2) CNOT) at every n ≥ 3.

    Validation in StronglyEntanglingConfig requires 1 <= r < num_qubits,
    so n=2 cannot use r=1 (which is the only valid range there); in that
    degenerate case the long-range default equals the only-possible
    default, with no behavior change.
    """
    if num_qubits < 2:
        # n=1 has no valid range at all; PennyLane's template is a no-op
        # entangler. Return [] so the validator accepts the empty tuple.
        return []
    r = num_qubits - 1
    return [r] * num_layers


def _build_qnode(cfg: StronglyEntanglingConfig) -> Callable:
    dev = qml.device(cfg.device_name, wires=cfg.num_qubits)
    encode_gate = qml.RX if cfg.encoding == "rx" else qml.RY
    # Aliasing fix: if caller didn't supply explicit ranges, fall back to
    # max-range CNOT per layer rather than PennyLane's `l mod n` default
    # (which aliases data_reuploading at n=3, L=1). See module docstring.
    if cfg.ranges is None:
        ranges: list[int] | None = _default_long_range_ranges(
            cfg.num_qubits, cfg.num_layers)
        if not ranges:                    # degenerate n=1 case
            ranges = None
    else:
        ranges = list(cfg.ranges)

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
