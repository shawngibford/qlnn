"""P4 commit 3a — Vector-output QLNN forecaster for ODE state-vector rollout.

Purpose-built sibling to `qlnn_forecaster.py`. The OD forecaster
predicts a SCALAR (delta around persistence in the OD coordinate);
this module predicts a VECTOR (the full state at the next rollout
step). For ODE systems where the state is multi-dimensional
(Lotka-Volterra 2D, Van der Pol 2D, Lorenz 3D), the registry
forecaster ansätze need a vector-output head — that's exactly this
file.

**Reuses the existing LiquidQuantumCell unchanged.** Only the wrapper
around the cell changes. The 4 registry forecaster ansätze
(`data_reuploading`, `hardware_efficient`, `strongly_entangling`,
`brickwall`) plug in via the cell's `AnsatzConfig` parameter — no
architecture change per family.

Pipeline for one sample (history `x: (T, d)`, equispaced step `dt`):

    1. h0 = tanh(W_h @ x[0] + b_h)                            # (Q,)
    2. For i in 0..T-2:
         h = integrate(cell, h, x[i], dt)                     # Diffrax
    3. h = integrate(cell, h, x[-1], dt)                      # forecast step
    4. delta = tanh(W_d @ h + b_d) * delta_scale              # (d,)
    5. y = x[-1] + delta                                       # residual around
                                                                # persistence

Differences from the OD forecaster:
- `delta_head_W` shape is `(num_qubits, output_dim)` (was `(num_qubits, 1)`).
- The residual is around persistence of THE STATE VECTOR (not the OD scalar):
  `y = x[-1] + delta` where both are length-d vectors.
- `dt` is a single equispaced step (set by the training data's cadence),
  not `t_hours[i+1] - t_hours[i]`.
- No `od_index` field — vector residual replaces scalar residual.

The forecaster's autoregressive rollout adapter (commit 3c) wraps
this into the `OneStepForecaster` protocol:

    forecaster_eval = bound_vector_forecaster(model, dt)
    def model_for_rollout(history, dt_arr):
        return forecaster_eval(history)
    traj = autoregressive_rollout(model_for_rollout, hist0, n_steps, dt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from ..cells.liquid_quantum_cell import LiquidQuantumCell, LiquidQuantumCellConfig
from ..circuits import AnsatzConfig


def _inv_softplus(y: float) -> float:
    """Inverse of softplus, used for initializing the unconstrained
    delta-scale parameter so that softplus(x) + min == requested_init."""
    if y <= 0:
        raise ValueError(f"inverse softplus requires y > 0, got {y}")
    return float(math.log(math.expm1(y)))


@dataclass(frozen=True)
class VectorForecasterConfig:
    """Config for the vector-output QLNN forecaster.

    Args:
      input_dim   : d — state-vector dimension. Used for both the
                    initial-state encoder's input AND the residual
                    head's output (the forecaster predicts a same-
                    dim vector around persistence).
      num_qubits  : Q — hidden cell dimension (the quantum circuit's
                    qubit count + the cell's latent dim).
      num_layers  : L — circuit depth.
      step_dt     : positive scalar — physical-time advance per
                    rollout step. Set by the training data's
                    sampling cadence (typically 0.01-0.1).
      delta_scale_init, delta_scale_min, tau_min, tau_init,
      solver, rtol, atol, dt0, max_steps, init_head_std,
      init_circuit_std, ansatz : forwarded to LiquidQuantumCell;
                                  same semantics as QLNNForecaster.
    """

    input_dim: int
    num_qubits: int = 4
    num_layers: int = 3
    step_dt: float = 0.05
    delta_scale_init: float = 1.0
    delta_scale_min: float = 0.01
    tau_min: float = 0.1
    tau_init: float = 1.0
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
                f"delta_scale_min ({self.delta_scale_min}) so the softplus "
                f"pre-image is well defined")
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


class VectorForecaster(eqx.Module):
    """Vector-output QLNN forecaster.

    Parameters (PyTree leaves):
        cell                       : LiquidQuantumCell (own leaves)
        initial_h_W                : (input_dim, num_qubits)
        initial_h_b                : (num_qubits,)
        delta_head_W               : (num_qubits, input_dim)   <-- VECTOR HEAD
        delta_head_b               : (input_dim,)               <-- VECTOR BIAS
        delta_scale_unconstrained  : scalar — learnable
    """

    cell: LiquidQuantumCell
    initial_h_W: jnp.ndarray
    initial_h_b: jnp.ndarray
    delta_head_W: jnp.ndarray
    delta_head_b: jnp.ndarray
    delta_scale_unconstrained: jnp.ndarray

    config: VectorForecasterConfig = eqx.field(static=True)

    def __init__(self, config: VectorForecasterConfig, *,
                 key: jax.Array) -> None:
        self.config = config
        k_cell, k_hW, k_dW, _k_extra = jax.random.split(key, 4)

        cell_cfg = LiquidQuantumCellConfig(
            input_dim=config.input_dim,
            num_qubits=config.num_qubits,
            num_layers=config.num_layers,
            tau_min=config.tau_min,
            tau_init=config.tau_init,
            init_circuit_std=config.init_circuit_std,
            ansatz=config.ansatz,
        )
        self.cell = LiquidQuantumCell(cell_cfg, key=k_cell)

        self.initial_h_W = config.init_head_std * jax.random.normal(
            k_hW, (config.input_dim, config.num_qubits))
        self.initial_h_b = jnp.zeros((config.num_qubits,))
        # Vector-output head: (Q, d) — the KEY change from the OD forecaster.
        self.delta_head_W = config.init_head_std * jax.random.normal(
            k_dW, (config.num_qubits, config.input_dim))
        self.delta_head_b = jnp.zeros((config.input_dim,))
        self.delta_scale_unconstrained = jnp.asarray(
            _inv_softplus(
                config.delta_scale_init - config.delta_scale_min),
            dtype=jnp.float32)

    def delta_scale(self) -> jnp.ndarray:
        """Constrained positive learnable delta-scale (softplus + floor)."""
        return (jax.nn.softplus(self.delta_scale_unconstrained)
                + self.config.delta_scale_min)

    def _integrate(self, h: jnp.ndarray, x_const: jnp.ndarray,
                    dt: jnp.ndarray) -> jnp.ndarray:
        """Diffrax-integrate the cell over a single step `dt`,
        holding the input vector `x_const` constant (matches the
        async-sampling convention of the original Neural ODE).
        """
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
        """Single sample. Predicts the next state vector.

        Args:
          x : (T, input_dim) history window of state vectors,
              equispaced in time at the config's `step_dt`.

        Returns:
          y : (input_dim,) predicted state at `t = T·step_dt`.

        Implements the pipeline in the module docstring:
        encoder → cell-integrated-over-history → forecast-step
        integration → vector delta head → residual around
        persistence of x[-1].
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

        # 1. Initial state encoder.
        h = jnp.tanh(x[0] @ self.initial_h_W + self.initial_h_b)

        # 2. Evolve over the history at constant step_dt.
        for i in range(T - 1):
            h = self._integrate(h, x[i], dt)

        # 3. Forecast step: hold last observed input constant.
        h = self._integrate(h, x[-1], dt)

        # 4. Vector delta around persistence. tanh(...) yields a value
        #    in [-1, 1] per-coordinate; delta_scale scales overall.
        delta_raw = jnp.tanh(h @ self.delta_head_W + self.delta_head_b)
        delta = delta_raw * self.delta_scale()           # (input_dim,)
        return x[-1] + delta                              # (input_dim,)
