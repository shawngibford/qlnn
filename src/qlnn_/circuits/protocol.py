"""Ansatz protocol + registry — the swap-in interface for the QLNN PQC.

The QLNN's downstream consumers (`QuantumFeatureEncoder`, `LiquidQuantumCell`,
`QLNNForecaster`) treat the parameterized quantum circuit as a single
black-box callable that obeys this contract:

    ansatz(inputs, weights) -> jnp.ndarray of shape (num_qubits,)

with every entry in `[-1, 1]` (a PauliZ expectation). Beyond that, the ansatz
declares its `weight_shape` (so the encoder can initialize circuit weights
of the right shape) and its `output_dim` (= num_qubits).

Any new ansatz module (hardware-efficient, brickwall, strongly-entangling, ...)
implements the `AnsatzProtocol` and registers itself under a unique name via
`register()`. The training pipeline then looks the ansatz up by name from
the YAML config (`model.ansatz.name`) and constructs it via `build()`.

Backward compatibility: if `AnsatzConfig` is left at its default, `build()`
returns the original data-reuploading circuit so existing checkpoints and
configs continue to work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import jax.numpy as jnp


@runtime_checkable
class AnsatzProtocol(Protocol):
    """Contract every QLNN PQC implementation must honor."""

    @property
    def output_dim(self) -> int: ...

    @property
    def weight_shape(self) -> tuple[int, ...]: ...

    def __call__(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        ...


@dataclass(frozen=True)
class AnsatzConfig:
    """Declarative spec for which ansatz to build and with what hyperparameters.

    `name` selects the registered factory. `params` is a free-form dict of
    factory-specific options (e.g. ``ring_entanglement: True``,
    ``entanglement: "all_to_all"``, ``variational: "ry_rz"``). The factory
    is responsible for validating its own `params`.

    The default `name="data_reuploading"` reproduces the historical 4-qubit,
    3-layer, ring-entanglement, RX-encoding circuit so configs without an
    `ansatz` section get identical behavior to before the refactor.
    """

    name: str = "data_reuploading"
    num_qubits: int = 4
    num_layers: int = 3
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not isinstance(self.params, dict):
            raise ValueError(f"params must be a dict, got {type(self.params).__name__}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# A "factory" is a plain callable `(AnsatzConfig) -> AnsatzProtocol`. We hold
# the factories in a module-level dict so import-time side effects in each
# ansatz module register themselves automatically.

_REGISTRY: dict[str, Callable[[AnsatzConfig], AnsatzProtocol]] = {}


def register(
    name: str, factory: Callable[[AnsatzConfig], AnsatzProtocol], *, overwrite: bool = False
) -> None:
    """Register an ansatz factory under `name`.

    Raises:
        ValueError if `name` is already registered and `overwrite=False`.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string, got {name!r}")
    if not callable(factory):
        raise ValueError("factory must be callable")
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"ansatz {name!r} already registered; pass overwrite=True to replace it"
        )
    _REGISTRY[name] = factory


def build(cfg: AnsatzConfig) -> AnsatzProtocol:
    """Construct an ansatz from a config by registry lookup."""
    if cfg.name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"unknown ansatz {cfg.name!r}; registered ansätze: {available}"
        )
    return _REGISTRY[cfg.name](cfg)


def available() -> list[str]:
    """Return the sorted list of registered ansatz names."""
    return sorted(_REGISTRY.keys())
