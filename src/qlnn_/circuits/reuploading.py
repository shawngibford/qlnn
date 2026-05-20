"""Data-reuploading parameterized quantum circuit (PQC).

**Architecture (the form the code emits):** the alternating
encoding/variational construction

    U(x) = W^{(L)} S(x) W^{(L-1)} … S(x) W^{(1)}        (default)
    U(x) = W^{(L+1)} S(x) W^{(L)} … S(x) W^{(1)}        (terminal_block=True)

per **Schuld, Sweke, Meyer**, *"The effect of data encoding on the
expressive power of variational quantum machine learning models"*,
**PRA 103, 032430 (2021)** (arXiv:2008.08605), Eq. 4 — the foundational
truncated-Fourier-series result that hypothesis H1 of
`ODE_PDE_PRE_REG.md` rests on. With a Pauli (RX) data generator each
encoding block contributes one integer to the accessible frequency
spectrum, so an L-layer single-qubit model is band-limited to
|ω| ≤ L (Schuld §II.2, Eqs. 10–11). The earlier **Pérez-Salinas et al.,
2020 (arXiv:1907.02085)** "universal classifier" is the predecessor
lineage of the re-uploading idea, NOT the architecture this code
implements — the P3a dual-check (refs/CIRCUIT_SPECS.md §7) corrected
that citation drift.

**Backward-compat note on the terminal block.** The historical code
(and every committed OD checkpoint / baseline lock) ends each of the
L layers with the entangler, omitting the trailing variational
W^{(L+1)} of canonical Schuld Eq. 4. The dual-check confirmed this is
a *spectrum-equivalent reduced form* — the accessible Ω that H1
depends on is unchanged. To preserve checkpoint compatibility the
default is the historical form (`terminal_block=False`); new code may
opt into the canonical Eq. 4 form by passing `terminal_block=True`
(weight shape grows from (L, Q, 3) to (L+1, Q, 3) accordingly).

This is the heart of the QLNN's "quantum feature encoder": few qubits,
many re-uploading layers, exponential expressivity in fitting
truncated Fourier series — H1's theoretical backbone.

Registered as ``"data_reuploading"`` in the ansatz registry. The
supported ``AnsatzConfig.params`` keys are:

    ring_entanglement : bool   (default True; legacy alias for entanglement="ring")
    entanglement      : str    one of "linear", "ring", "all_to_all" (default "ring")
    terminal_block    : bool   (default False; True ⇒ canonical Schuld Eq.4)
    device_name       : str    PennyLane device (default "default.qubit")

Backward compatibility: the historical ``DataReuploadingCircuit`` /
``DataReuploadingConfig`` API is preserved so existing call-sites keep
working. New code should prefer
``circuits.build(AnsatzConfig(name="data_reuploading", ...))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import pennylane as qml

from .protocol import AnsatzConfig, AnsatzProtocol, register


@dataclass(frozen=True)
class DataReuploadingConfig:
    num_qubits: int = 4
    num_layers: int = 3
    # If True, the entanglement stage adds a final wrap-around CNOT to close
    # the linear chain into a ring. Tends to help with all-to-all coupling on
    # small qubit counts.
    ring_entanglement: bool = True
    # Optional finer-grained entanglement override. If None, derives from
    # `ring_entanglement` (True → "ring", False → "linear").
    entanglement: str | None = None
    # If True, append a final variational `Rot` layer AFTER the last
    # entangler (no data re-upload after it) — the canonical Schuld
    # Eq. 4 form. Default False preserves OD-checkpoint compatibility;
    # see module docstring.
    terminal_block: bool = False
    # Pennylane device name. "default.qubit" is the modern unified device that
    # picks the JAX interface automatically when called from JAX code.
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.entanglement is not None and self.entanglement not in (
            "linear", "ring", "all_to_all"
        ):
            raise ValueError(
                f"entanglement must be one of 'linear', 'ring', 'all_to_all', "
                f"got {self.entanglement!r}"
            )

    @property
    def total_variational_layers(self) -> int:
        """Number of W blocks in the circuit: L (historical) or L+1
        (canonical Schuld Eq. 4 with `terminal_block=True`)."""
        return self.num_layers + (1 if self.terminal_block else 0)

    @property
    def resolved_entanglement(self) -> str:
        """The effective entanglement pattern, accounting for the legacy
        `ring_entanglement` bool alias."""
        if self.entanglement is not None:
            return self.entanglement
        return "ring" if self.ring_entanglement else "linear"


def _entangle(num_qubits: int, pattern: str) -> None:
    """Emit the entangling block for `pattern` on `num_qubits` wires.

    Patterns:
        linear      — chain CNOTs i -> i+1
        ring        — chain CNOTs + wrap-around (n-1) -> 0 (only if n > 2)
        all_to_all  — CNOTs between every ordered pair (i, j), i < j
    """
    if num_qubits < 2:
        return
    if pattern == "linear":
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return
    if pattern == "ring":
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        if num_qubits > 2:
            qml.CNOT(wires=[num_qubits - 1, 0])
        return
    if pattern == "all_to_all":
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qml.CNOT(wires=[i, j])
        return
    raise ValueError(f"unknown entanglement pattern: {pattern!r}")


def _build_qnode(cfg: DataReuploadingConfig) -> Callable:
    """Return a JAX-interfaced QNode implementing the re-uploading circuit.

    The returned function has signature (inputs, weights) -> jnp.ndarray of
    shape (num_qubits,), each entry the PauliZ expectation on a wire.

    inputs : (num_qubits,)
        Bioprocess feature vector for one sample/time-step. Caller is
        responsible for normalizing into the angle-encoding range.
    weights : (total_variational_layers, num_qubits, 3)
        Rotation parameters per W block (Rot = RZ ∘ RY ∘ RZ). The shape's
        first dim is `cfg.num_layers` historically, or `num_layers + 1`
        when `cfg.terminal_block=True` (canonical Schuld Eq. 4).
    """
    dev = qml.device(cfg.device_name, wires=cfg.num_qubits)
    entanglement = cfg.resolved_entanglement
    use_terminal = cfg.terminal_block

    @qml.qnode(dev, interface="jax")
    def circuit(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        for layer in range(cfg.num_layers):
            # Data re-uploading: re-inject the inputs at every layer (this is
            # what gives the circuit expressivity in the truncated-Fourier
            # sense — without it, only the first layer sees the data).
            for i in range(cfg.num_qubits):
                qml.RX(inputs[i], wires=i)

            # Variational W^{(layer+1)} block.
            for i in range(cfg.num_qubits):
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

            # Entangling block.
            _entangle(cfg.num_qubits, entanglement)

        if use_terminal:
            # Canonical Schuld Eq. 4: a final W^{(L+1)} variational block
            # AFTER the last entangler, with no data re-upload after it.
            tl = cfg.num_layers
            for i in range(cfg.num_qubits):
                qml.Rot(weights[tl, i, 0], weights[tl, i, 1], weights[tl, i, 2], wires=i)

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
        return (c.total_variational_layers, c.num_qubits, 3)

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


# ---------------------------------------------------------------------------
# Registry hook
# ---------------------------------------------------------------------------
def _factory(cfg: AnsatzConfig) -> AnsatzProtocol:
    """Build a `DataReuploadingCircuit` from an `AnsatzConfig`."""
    params = cfg.params or {}
    drc = DataReuploadingConfig(
        num_qubits=cfg.num_qubits,
        num_layers=cfg.num_layers,
        ring_entanglement=bool(params.get("ring_entanglement", True)),
        entanglement=params.get("entanglement"),
        terminal_block=bool(params.get("terminal_block", False)),
        device_name=str(params.get("device_name", "default.qubit")),
    )
    return DataReuploadingCircuit(drc)


register("data_reuploading", _factory, overwrite=True)
