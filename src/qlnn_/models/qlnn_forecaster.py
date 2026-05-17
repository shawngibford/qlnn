"""Quantum-liquid neural ODE forecaster (JAX / Equinox).

JAX-side mirror of `quantum_liquid_neuralode.models.LiquidODForecaster`.

Pipeline for one sample (x : (T, F), t_hours : (T,)):
    1. h0 = tanh(W_h @ x[0] + b_h)                          # (Q,)
    2. For i in 0..T-2: integrate dh/dt = cell(t, h, x[i])
       over [0, t_hours[i+1] - t_hours[i]] with Diffrax.
    3. Integrate dh/dt = cell(t, h, x[-1]) over [0, horizon_hours].
    4. delta = tanh(W_d @ h + b_d) * delta_scale            # scalar
    5. y = x[-1, od_index] + delta                           # residual around persistence

Semantics intentionally mirror the classical PyTorch forecaster: the only
thing that changes when we swap classical → quantum is the vector field
inside the cell. Everything else (initial state encoding, residual head,
ODE-over-history-then-horizon protocol) is identical so head-to-head
comparison isolates the effect of the quantum dynamics.

Diffrax notes:
- We hold the input vector constant over each integration interval (matches
  the asynchronous-sampling property of the original neural ODE) by passing
  `args=x_i` and reading `args` inside the term lambda.
- `SaveAt(t1=True)` returns only the endpoint of each solve — we don't need
  dense trajectories, and skipping them saves memory & avoids gradient bloat.
- `dt0` is clipped to half the interval so the very first proposed step
  never overshoots `t1` on short history intervals (~10 min ≈ 0.167 h).
"""

from __future__ import annotations

from dataclasses import dataclass

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from ..cells.liquid_quantum_cell import LiquidQuantumCell, LiquidQuantumCellConfig


@dataclass(frozen=True)
class QLNNForecasterConfig:
    input_dim: int
    num_qubits: int = 4
    num_layers: int = 3
    horizon_hours: float = 1.0
    od_index: int = 0
    delta_scale: float = 0.1
    tau_min: float = 0.1
    tau_init: float = 1.0
    # Diffrax solver knobs.
    solver: str = "tsit5"           # "tsit5" or "dopri5"
    rtol: float = 1e-3
    atol: float = 1e-4
    dt0: float = 0.05
    max_steps: int = 4096
    # Init scale for initial-state encoder and delta head.
    init_head_std: float = 0.1

    def __post_init__(self) -> None:
        if self.solver not in ("tsit5", "dopri5"):
            raise ValueError(f"solver must be 'tsit5' or 'dopri5', got {self.solver!r}")
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if not (0 <= self.od_index < self.input_dim):
            raise ValueError(
                f"od_index must be in [0, input_dim={self.input_dim}), got {self.od_index}"
            )
        if self.delta_scale <= 0:
            raise ValueError(f"delta_scale must be > 0, got {self.delta_scale}")
        if self.horizon_hours <= 0:
            raise ValueError(f"horizon_hours must be > 0, got {self.horizon_hours}")
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
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


class QLNNForecaster(eqx.Module):
    """Hybrid QLNN forecaster — JAX/Equinox analog of LiquidODForecaster.

    Parameters (PyTree leaves):
        cell            : LiquidQuantumCell (whose own params are leaves)
        initial_h_W     : (input_dim, num_qubits)
        initial_h_b     : (num_qubits,)
        delta_head_W    : (num_qubits, 1)
        delta_head_b    : (1,)
    """

    cell: LiquidQuantumCell
    initial_h_W: jnp.ndarray
    initial_h_b: jnp.ndarray
    delta_head_W: jnp.ndarray
    delta_head_b: jnp.ndarray

    config: QLNNForecasterConfig = eqx.field(static=True)

    def __init__(self, config: QLNNForecasterConfig, *, key: jax.Array) -> None:
        self.config = config

        k_cell, k_hW, k_dW, _k_extra = jax.random.split(key, 4)

        cell_cfg = LiquidQuantumCellConfig(
            input_dim=config.input_dim,
            num_qubits=config.num_qubits,
            num_layers=config.num_layers,
            tau_min=config.tau_min,
            tau_init=config.tau_init,
        )
        self.cell = LiquidQuantumCell(cell_cfg, key=k_cell)

        self.initial_h_W = config.init_head_std * jax.random.normal(
            k_hW, (config.input_dim, config.num_qubits)
        )
        self.initial_h_b = jnp.zeros((config.num_qubits,))
        self.delta_head_W = config.init_head_std * jax.random.normal(
            k_dW, (config.num_qubits, 1)
        )
        self.delta_head_b = jnp.zeros((1,))

    # ------------------------------------------------------------------
    # Diffrax integration of dh/dt = cell(t, h, x_const) over [0, dt].
    # ------------------------------------------------------------------
    def _integrate(self, h: jnp.ndarray, x_const: jnp.ndarray, dt) -> jnp.ndarray:
        cfg = self.config
        cell = self.cell

        def vf(t, y, args):
            return cell(t, y, args)

        term = diffrax.ODETerm(vf)
        solver = _build_solver(cfg.solver)
        controller = diffrax.PIDController(rtol=cfg.rtol, atol=cfg.atol)
        # Clip dt0 so the first proposed step never overshoots a short interval.
        dt0 = jnp.minimum(jnp.asarray(cfg.dt0, dtype=jnp.float32), dt * 0.5)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=dt,
            dt0=dt0,
            y0=h,
            args=x_const,
            stepsize_controller=controller,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=cfg.max_steps,
        )
        # ys has shape (1, num_qubits) because SaveAt(t1=True).
        return sol.ys[-1]

    def __call__(self, x: jnp.ndarray, t_hours: jnp.ndarray) -> jnp.ndarray:
        """Single sample. Returns scalar OD prediction (normalized).

        x       : (T, input_dim)
        t_hours : (T,)
        """
        cfg = self.config

        if x.ndim != 2 or x.shape[-1] != cfg.input_dim:
            raise ValueError(
                f"x must have shape (T, input_dim={cfg.input_dim}), got {tuple(x.shape)}"
            )
        if t_hours.shape != (x.shape[0],):
            raise ValueError(
                f"t_hours must have shape ({x.shape[0]},), got {tuple(t_hours.shape)}"
            )
        if x.shape[0] < 2:
            raise ValueError(f"need at least 2 time points, got T={x.shape[0]}")

        T = x.shape[0]

        # Strict-monotonicity guard on t_hours. Each per-step integration solves
        # over [0, dt] with dt = t_hours[i+1] - t_hours[i]; dt == 0 makes the
        # solve a silent no-op and dt < 0 produces NaN. Catch both even under
        # JIT (eqx.error_if traces through jit).
        t_hours = eqx.error_if(
            t_hours,
            jnp.any(jnp.diff(t_hours) <= 0),
            "t_hours must be strictly increasing (every dt > 0).",
        )

        # 1. Initial state encoder.
        h = jnp.tanh(x[0] @ self.initial_h_W + self.initial_h_b)

        # 2. Evolve over the history. Python for-loop is fine — T = 24 in
        # production, so trace cost is amortized after the first JIT.
        for i in range(T - 1):
            dt = t_hours[i + 1] - t_hours[i]
            h = self._integrate(h, x[i], dt)

        # 3. Forecast horizon: hold the last observed input constant.
        h = self._integrate(h, x[-1], jnp.asarray(cfg.horizon_hours, dtype=h.dtype))

        # 4. Residual delta around persistence.
        delta = jnp.tanh(h @ self.delta_head_W + self.delta_head_b).squeeze(-1) * cfg.delta_scale
        od_last = x[-1, cfg.od_index]
        return od_last + delta
