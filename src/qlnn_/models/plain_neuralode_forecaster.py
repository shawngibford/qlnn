"""P5 commit 1 — Plain non-liquid Neural-ODE forecaster (MANDATORY H1 contrast).

Per `ODE_PDE_PRE_REG.md` §6: "MANDATORY: a non-liquid plain
Neural-ODE baseline. It is the primary H1 contrast — H1 is stated
as QLNN-minus-NeuralODE advantage, not QLNN-minus-persistence. This
baseline is non-optional; the program is invalid without it."

This module ships a **fair head-to-head** with `VectorForecaster`
(P4 commit 3a). Same Diffrax integration, same initial-state
encoder, same vector residual decoder, same training loop. The
ONLY differences:

  1. Cell is an **MLP** (no quantum circuit).
  2. NO **learnable time-constants** — the cell is purely
     `dh/dt = MLP([h, x])`, which is the canonical Neural-ODE
     (Chen et al. NeurIPS 2018, arXiv:1806.07366) form WITHOUT the
     "liquid" τ that makes a network liquid (Hasani et al. AAAI
     2021, arXiv:2006.04439).

The two structural differences together isolate the H1 contrast:
  - Plain Neural-ODE vs QLNN  → quantum-vs-classical-cell test.
  - Non-liquid vs liquid       → τ-learning impact.

Architecture (one sample, history `x: (T, d)`, equispaced `step_dt`):

    1. h0 = tanh(W_h @ x[0] + b_h)                        # (Q,)
    2. For i in 0..T-2:
         h = integrate(NeuralODECell, h, x[i], dt)         # Diffrax
    3. h = integrate(NeuralODECell, h, x[-1], dt)          # forecast step
    4. delta = tanh(W_d @ h + b_d) * delta_scale           # (d,)
    5. y = x[-1] + delta                                    # residual around
                                                             # persistence

NeuralODECell vector field:
    dh/dt = MLP_W2 · tanh(MLP_W1 · [h, x] + b1) + b2

Trainable parameters per cell:
  - mlp_W1 : (Q + d, H)        first MLP layer
  - mlp_b1 : (H,)
  - mlp_W2 : (H, Q)            output layer (back to hidden dim Q)
  - mlp_b2 : (Q,)

Default H = num_qubits (matches the LiquidQuantumCell's hidden
dimension) — a minimum-faithful capacity-matched cell.

**Capacity matching note:** the LiquidQuantumCell has roughly
`3·n_qubits·n_layers + 2·n_qubits + encoder_overhead` trainable
parameters (PQC + tau + A + classical pre-encoder). The MLP cell
has `(Q+d)·H + H + H·Q + Q ≈ 2·Q·H + d·H + 2·Q` parameters. At
default n_qubits=3, n_layers=1, d=2-3, the QLNN has ~30 params and
the plain MLP cell at H=Q=3 has ~30 params too — capacity-matched
within a factor of 2 per pre-reg §6.

The pre-reg's underfit guard (a model that can't fit the training
trajectory is reported as underfit, not as a null) is checked by
the P5 verdict module against the train-side relative-L2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp


def _inv_softplus(y: float) -> float:
    if y <= 0:
        raise ValueError(f"inverse softplus requires y > 0, got {y}")
    return float(math.log(math.expm1(y)))


# ---------------------------------------------------------------------------
# Plain MLP cell (the non-liquid Neural-ODE cell)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeuralODECellConfig:
    """Config for the plain non-liquid MLP cell.

    Args:
      input_dim   : d — input feature dim (the state vector's dim).
      hidden_dim  : Q — cell hidden dim.
      mlp_hidden  : H — MLP intermediate hidden width (default = Q).
      activation  : "tanh" (default) or "relu".
    """

    input_dim: int
    hidden_dim: int = 4
    mlp_hidden: int = 0           # 0 means "default to hidden_dim"
    activation: str = "tanh"

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.hidden_dim < 1:
            raise ValueError(
                f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.activation not in ("tanh", "relu"):
            raise ValueError(
                f"activation must be 'tanh' or 'relu', "
                f"got {self.activation!r}")
        # Default mlp_hidden = hidden_dim (capacity-matched).
        if self.mlp_hidden < 0:
            raise ValueError(
                f"mlp_hidden must be >= 0, got {self.mlp_hidden}")

    @property
    def H(self) -> int:
        return self.mlp_hidden if self.mlp_hidden > 0 else self.hidden_dim


class NeuralODECell(eqx.Module):
    """Plain MLP cell — the non-liquid Neural-ODE vector field.

    Parameters (PyTree leaves):
        mlp_W1 : (Q + d, H)
        mlp_b1 : (H,)
        mlp_W2 : (H, Q)
        mlp_b2 : (Q,)

    Static:
        config : NeuralODECellConfig

    NO learnable time-constants — that's the "non-liquid" property
    pre-reg §6 requires for the H1 contrast.
    """

    mlp_W1: jnp.ndarray
    mlp_b1: jnp.ndarray
    mlp_W2: jnp.ndarray
    mlp_b2: jnp.ndarray

    config: NeuralODECellConfig = eqx.field(static=True)

    def __init__(self, config: NeuralODECellConfig, *,
                 key: jax.Array) -> None:
        self.config = config
        Q = config.hidden_dim
        d = config.input_dim
        H = config.H

        k1, k2 = jax.random.split(key, 2)
        # Glorot-like init scaled by 1/sqrt(in_dim) so initial dh/dt is
        # bounded ~ O(1) and the integrator doesn't blow up.
        scale1 = 1.0 / float(jnp.sqrt(Q + d))
        scale2 = 1.0 / float(jnp.sqrt(H))
        self.mlp_W1 = scale1 * jax.random.normal(k1, (Q + d, H))
        self.mlp_b1 = jnp.zeros((H,))
        self.mlp_W2 = scale2 * jax.random.normal(k2, (H, Q))
        self.mlp_b2 = jnp.zeros((Q,))

    def __call__(self, t, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """dh/dt for ONE sample.

        Args:
          t : scalar time (ignored — vector field is autonomous given
              the held-constant input x; same convention as
              LiquidQuantumCell so Diffrax can swap-in either cell).
          h : (Q,) hidden state.
          x : (d,) input feature vector — held constant on the
              integration interval (zero-order-hold, same as LQC).

        Returns: (Q,) time derivative.
        """
        cfg = self.config
        if h.shape != (cfg.hidden_dim,):
            raise ValueError(
                f"h must have shape ({cfg.hidden_dim},), got {h.shape}")
        if x.shape != (cfg.input_dim,):
            raise ValueError(
                f"x must have shape ({cfg.input_dim},), got {x.shape}")
        hx = jnp.concatenate([h, x])                          # (Q + d,)
        hidden = hx @ self.mlp_W1 + self.mlp_b1                # (H,)
        if cfg.activation == "tanh":
            hidden = jnp.tanh(hidden)
        else:
            hidden = jax.nn.relu(hidden)
        return hidden @ self.mlp_W2 + self.mlp_b2              # (Q,)

    def num_parameters(self) -> int:
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        return sum(int(jnp.size(leaf)) for leaf in leaves)


# ---------------------------------------------------------------------------
# The full forecaster (mirrors VectorForecaster structure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlainNeuralODEForecasterConfig:
    """Config for the plain Neural-ODE vector-output forecaster.

    Mirrors `VectorForecasterConfig` field-for-field so the head-to-
    head comparison is structurally identical. The Diffrax solver +
    integrator settings are intentionally the same.
    """

    input_dim: int
    hidden_dim: int = 4
    mlp_hidden: int = 0           # 0 → defaults to hidden_dim
    activation: str = "tanh"
    step_dt: float = 0.05
    delta_scale_init: float = 1.0
    delta_scale_min: float = 0.01
    solver: str = "tsit5"
    rtol: float = 1e-3
    atol: float = 1e-4
    dt0: float = 0.05
    max_steps: int = 4096
    init_head_std: float = 0.1

    def __post_init__(self) -> None:
        if self.solver not in ("tsit5", "dopri5"):
            raise ValueError(
                f"solver must be 'tsit5' or 'dopri5', got {self.solver!r}")
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.hidden_dim < 1:
            raise ValueError(
                f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.step_dt <= 0:
            raise ValueError(f"step_dt must be > 0, got {self.step_dt}")
        if self.delta_scale_init <= self.delta_scale_min:
            raise ValueError(
                f"delta_scale_init ({self.delta_scale_init}) must exceed "
                f"delta_scale_min ({self.delta_scale_min})")
        if self.delta_scale_min <= 0:
            raise ValueError(f"delta_scale_min must be > 0")
        if self.activation not in ("tanh", "relu"):
            raise ValueError(
                f"activation must be 'tanh' or 'relu', "
                f"got {self.activation!r}")
        if self.max_steps < 1 or self.dt0 <= 0:
            raise ValueError("max_steps>=1 and dt0>0 required")


def _build_solver(name: str):
    if name == "tsit5":
        return diffrax.Tsit5()
    if name == "dopri5":
        return diffrax.Dopri5()
    raise ValueError(f"unknown solver: {name}")  # pragma: no cover


class PlainNeuralODEForecaster(eqx.Module):
    """Non-liquid Neural-ODE forecaster — the MANDATORY H1 baseline.

    Parameters:
        cell                       : NeuralODECell
        initial_h_W                : (input_dim, hidden_dim)
        initial_h_b                : (hidden_dim,)
        delta_head_W               : (hidden_dim, input_dim)
        delta_head_b               : (input_dim,)
        delta_scale_unconstrained  : scalar (learnable; softplus + floor)
    """

    cell: NeuralODECell
    initial_h_W: jnp.ndarray
    initial_h_b: jnp.ndarray
    delta_head_W: jnp.ndarray
    delta_head_b: jnp.ndarray
    delta_scale_unconstrained: jnp.ndarray

    config: PlainNeuralODEForecasterConfig = eqx.field(static=True)

    def __init__(self, config: PlainNeuralODEForecasterConfig,
                 *, key: jax.Array) -> None:
        self.config = config
        k_cell, k_hW, k_dW = jax.random.split(key, 3)

        cell_cfg = NeuralODECellConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            mlp_hidden=config.mlp_hidden,
            activation=config.activation,
        )
        self.cell = NeuralODECell(cell_cfg, key=k_cell)

        self.initial_h_W = config.init_head_std * jax.random.normal(
            k_hW, (config.input_dim, config.hidden_dim))
        self.initial_h_b = jnp.zeros((config.hidden_dim,))
        self.delta_head_W = config.init_head_std * jax.random.normal(
            k_dW, (config.hidden_dim, config.input_dim))
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
            term, solver, t0=0.0, t1=dt, dt0=dt0,
            y0=h, args=x_const,
            stepsize_controller=controller,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=cfg.max_steps)
        return sol.ys[-1]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Single sample. (T, d) history → (d,) next-state prediction.

        Mirrors VectorForecaster.__call__ exactly except the cell is
        the plain MLP (no quantum, no learnable τ).
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
