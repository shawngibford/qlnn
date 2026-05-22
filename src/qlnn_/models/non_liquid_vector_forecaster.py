"""Non-liquid vector-output QLNN forecaster (P7.11 commit 2).

Mirrors `VectorForecaster` field-for-field with `LiquidQuantumCell`
swapped for `NonLiquidQuantumCell`. Same Diffrax integration, same
encoder + decoder scaffold, same per-step training loop and adapter
interface — only the cell differs.

Together with `ClassicalLTCForecaster` (P7.10) and the existing
`VectorForecaster` (liquid quantum) + `PlainNeuralODEForecaster` (P5),
this completes the 2×2 fairness matrix on the forecaster task:

                  non-liquid (no τ)        liquid (learnable τ)
    Classical     PlainNeuralODEForecaster ClassicalLTCForecaster
    Quantum       NonLiquidVectorForecaster ⟵ THIS MODULE
                                            VectorForecaster

The pre-registered H1 contrast is QLNN_with_τ − Neural-ODE; the LTC
decomposition isolates the liquid-τ contribution on the classical
side; this module's variant isolates the same on the quantum side.
The 4 ansätze (data_reuploading, hardware_efficient,
strongly_entangling, brickwall) all share the underlying cell — one
NonLiquidVectorForecaster covers all 4 by swapping the `ansatz`
field, matching the existing VectorForecaster convention.

rf_qrc is intentionally NOT covered here: per the audit
(`src/qlnn_/circuits/rf_qrc.py`), its reservoir has a fixed
non-learnable `leak_rate` hyperparameter, so it is already non-liquid
in the QLNN sense. Including a rf_qrc "ablation" would be a no-op.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from qlnn_.circuits import AnsatzConfig
from qlnn_.cells.non_liquid_quantum_cell import (
    NonLiquidQuantumCell,
    NonLiquidQuantumCellConfig,
)


def _inv_softplus(y: float) -> float:
    if y <= 0:
        raise ValueError(f"inverse softplus requires y > 0, got {y}")
    return float(math.log(math.expm1(y)))


@dataclass(frozen=True)
class NonLiquidVectorForecasterConfig:
    """Config for the non-liquid vector-output QLNN forecaster.

    Mirrors `VectorForecasterConfig` field-for-field with the two τ
    scalars (`tau_min`, `tau_init`) REMOVED — the underlying cell has
    no τ.

    Diffrax integration settings are intentionally identical to
    `VectorForecasterConfig` so the head-to-head differs only in the
    cell's τ-presence.
    """

    input_dim: int
    num_qubits: int = 4
    num_layers: int = 3
    step_dt: float = 0.05
    delta_scale_init: float = 1.0
    delta_scale_min: float = 0.01
    solver: str = "tsit5"
    rtol: float = 1e-3
    atol: float = 1e-4
    dt0: float = 0.05
    max_steps: int = 4096
    init_head_std: float = 0.1
    init_circuit_std: float = 0.05
    ansatz: AnsatzConfig | None = None

    def __post_init__(self) -> None:
        if self.solver not in ("tsit5", "dopri5"):
            raise ValueError(
                f"solver must be 'tsit5' or 'dopri5', got {self.solver!r}")
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.num_qubits < 1:
            raise ValueError(
                f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(
                f"num_layers must be >= 1, got {self.num_layers}")
        if self.step_dt <= 0:
            raise ValueError(f"step_dt must be > 0, got {self.step_dt}")
        if self.delta_scale_init <= self.delta_scale_min:
            raise ValueError(
                f"delta_scale_init ({self.delta_scale_init}) must exceed "
                f"delta_scale_min ({self.delta_scale_min})")
        if self.delta_scale_min <= 0:
            raise ValueError(
                f"delta_scale_min must be > 0, got {self.delta_scale_min}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.dt0 <= 0:
            raise ValueError(f"dt0 must be > 0, got {self.dt0}")


def _build_solver(name: str):
    if name == "tsit5":
        return diffrax.Tsit5()
    if name == "dopri5":
        return diffrax.Dopri5()
    raise ValueError(f"unknown solver: {name}")  # pragma: no cover


class NonLiquidVectorForecaster(eqx.Module):
    """Non-liquid vector-output QLNN forecaster.

    Parameters (PyTree leaves):
        cell                       : NonLiquidQuantumCell
        initial_h_W                : (input_dim, num_qubits)
        initial_h_b                : (num_qubits,)
        delta_head_W               : (num_qubits, input_dim)
        delta_head_b               : (input_dim,)
        delta_scale_unconstrained  : scalar (learnable; softplus + floor)
    """

    cell: NonLiquidQuantumCell
    initial_h_W: jnp.ndarray
    initial_h_b: jnp.ndarray
    delta_head_W: jnp.ndarray
    delta_head_b: jnp.ndarray
    delta_scale_unconstrained: jnp.ndarray

    config: NonLiquidVectorForecasterConfig = eqx.field(static=True)

    def __init__(self, config: NonLiquidVectorForecasterConfig,
                 *, key: jax.Array) -> None:
        self.config = config
        k_cell, k_hW, k_dW, _k_extra = jax.random.split(key, 4)

        cell_cfg = NonLiquidQuantumCellConfig(
            input_dim=config.input_dim,
            num_qubits=config.num_qubits,
            num_layers=config.num_layers,
            init_circuit_std=config.init_circuit_std,
            ansatz=config.ansatz,
        )
        self.cell = NonLiquidQuantumCell(cell_cfg, key=k_cell)

        self.initial_h_W = config.init_head_std * jax.random.normal(
            k_hW, (config.input_dim, config.num_qubits))
        self.initial_h_b = jnp.zeros((config.num_qubits,))
        self.delta_head_W = config.init_head_std * jax.random.normal(
            k_dW, (config.num_qubits, config.input_dim))
        self.delta_head_b = jnp.zeros((config.input_dim,))
        self.delta_scale_unconstrained = jnp.asarray(
            _inv_softplus(
                config.delta_scale_init - config.delta_scale_min),
            dtype=jnp.float32)

    def delta_scale(self) -> jnp.ndarray:
        return (jax.nn.softplus(self.delta_scale_unconstrained)
                + self.config.delta_scale_min)

    def _integrate(self, h: jnp.ndarray, x_const: jnp.ndarray,
                   dt: jnp.ndarray) -> jnp.ndarray:
        cfg = self.config
        cell = self.cell

        def vf(t, y, args):
            return cell(t, y, args)

        term = diffrax.ODETerm(vf)
        solver = _build_solver(cfg.solver)
        controller = diffrax.PIDController(rtol=cfg.rtol, atol=cfg.atol)
        dt0 = jnp.minimum(
            jnp.asarray(cfg.dt0, dtype=jnp.float32), dt * 0.5)

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=dt, dt0=dt0,
            y0=h, args=x_const,
            stepsize_controller=controller,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=cfg.max_steps,
        )
        return sol.ys[-1]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Single sample. (T, d) history → (d,) next-state prediction.

        Mirrors `VectorForecaster.__call__` exactly; the only difference
        is that the cell has no τ-leak.
        """
        cfg = self.config

        if x.ndim != 2 or x.shape[-1] != cfg.input_dim:
            raise ValueError(
                f"x must have shape (T, input_dim={cfg.input_dim}), "
                f"got {tuple(x.shape)}")
        if x.shape[0] < 2:
            raise ValueError(
                f"need at least 2 time points, got T={x.shape[0]}")

        T = x.shape[0]
        dt = jnp.asarray(cfg.step_dt, dtype=jnp.float32)

        h = jnp.tanh(x[0] @ self.initial_h_W + self.initial_h_b)
        for i in range(T - 1):
            h = self._integrate(h, x[i], dt)
        h = self._integrate(h, x[-1], dt)

        delta_raw = jnp.tanh(h @ self.delta_head_W + self.delta_head_b)
        delta = delta_raw * self.delta_scale()
        return x[-1] + delta
