"""P7.10 commit 1 — Classical Liquid-Time-Constant (LTC) forecaster.

Closes the LTC-decomposition fairness gap surfaced mid-session
2026-05-21. The pre-registered forecaster H1 contrast is
``QLNN_forecaster − Neural-ODE`` where QLNN has learnable per-qubit
``tau`` (`LiquidQuantumCell`) and Neural-ODE does not
(`PlainNeuralODEForecaster`). A reviewer running the Bowles/Schuld
2024 "remove the quantum component" ablation will correctly ask
which part of the gap is the quantum circuit and which part is
the liquid τ machinery itself.

This module provides the missing fourth quadrant: a CLASSICAL
forecaster with learnable τ but no quantum circuit. Letting:
  - ``Δ_combined = QLNN − Neural-ODE``       (pre-reg-mandated)
  - ``Δ_quantum  = QLNN − classical_LTC``    (isolated quantum contribution)
  - ``Δ_liquid   = classical_LTC − Neural-ODE`` (isolated liquid-τ contribution)
We decompose the forecaster verdict cleanly, with
``Δ_combined ≈ Δ_quantum + Δ_liquid`` (paired-bootstrap CI on each).

The implementation MIRRORS `plain_neuralode_forecaster.py` field
for field, with two minimal additions:

  1. Cell adds ``tau_unconstrained`` (per-hidden-unit learnable τ
     via softplus + tau_min, identical pattern to
     `LiquidQuantumCell.tau_unconstrained` and
     `qzeta.models.LiquidCell.tau_unconstrained`).
  2. Dynamics modulated by τ-leak:
        ``dh/dt = -(1/tau) ⊙ h + MLP([h, x])``
     This is Hasani et al. 2021 (AAAI, arXiv:2006.04439) LTC form
     in the simplest input-independent-τ variant, isolated from the
     extra "input-as-conductance" term that the LiquidQuantumCell
     uses for the quantum encoder. Removing that input-conductance
     term gives the cleanest "liquid-τ-only" baseline.

Capacity: matches `PlainNeuralODEForecaster` cell within Q
additional parameters (the `tau_unconstrained` vector). At
default Q=4, the LTC cell adds 4 params on top of the ~30-param
plain Neural-ODE cell — well within pre-reg §6's factor-of-2
matched-capacity rule.

The training adapter (``forecaster_adapters.make_vector_forecaster_adapter``)
accepts this model unchanged: same ``(T, d) → (d,)`` call shape.
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
# Classical LTC cell (the liquid Neural-ODE cell, no quantum)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassicalLTCCellConfig:
    """Config for the classical LTC cell.

    Mirrors `NeuralODECellConfig` plus the two τ-init scalars (matched
    to `LiquidQuantumCellConfig` defaults so the comparison is fair).
    """

    input_dim: int
    hidden_dim: int = 4
    mlp_hidden: int = 0           # 0 → default to hidden_dim
    activation: str = "tanh"
    tau_init: float = 1.0         # initial τ value (matches LiquidQuantumCell)
    tau_min: float = 0.1          # floor on τ (matches LiquidQuantumCell)

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
        if self.mlp_hidden < 0:
            raise ValueError(
                f"mlp_hidden must be >= 0, got {self.mlp_hidden}")
        if self.tau_min <= 0:
            raise ValueError(f"tau_min must be > 0, got {self.tau_min}")
        if self.tau_init <= self.tau_min:
            raise ValueError(
                f"tau_init must be > tau_min ({self.tau_min}), "
                f"got tau_init={self.tau_init}")

    @property
    def H(self) -> int:
        return self.mlp_hidden if self.mlp_hidden > 0 else self.hidden_dim


class ClassicalLTCCell(eqx.Module):
    """Classical LTC vector field — non-quantum, liquid-τ.

    Parameters (PyTree leaves):
        mlp_W1            : (Q + d, H)
        mlp_b1            : (H,)
        mlp_W2            : (H, Q)
        mlp_b2            : (Q,)
        tau_unconstrained : (Q,)  — softplus → positive τ per unit

    Static:
        config : ClassicalLTCCellConfig

    Vector field (Hasani 2021 LTC, simplest input-independent-τ form):

        dh/dt = -(1/τ) ⊙ h + MLP_W2 · σ(MLP_W1 · [h, x] + b1) + b2

    where σ is tanh or relu per `cfg.activation`. The leak
    ``-h / τ`` is the LTC's defining feature; the MLP drive term is
    structurally identical to `NeuralODECell` so the only addition
    over the non-liquid baseline is the learnable τ.
    """

    mlp_W1: jnp.ndarray
    mlp_b1: jnp.ndarray
    mlp_W2: jnp.ndarray
    mlp_b2: jnp.ndarray
    tau_unconstrained: jnp.ndarray

    config: ClassicalLTCCellConfig = eqx.field(static=True)

    def __init__(self, config: ClassicalLTCCellConfig, *,
                 key: jax.Array) -> None:
        self.config = config
        Q = config.hidden_dim
        d = config.input_dim
        H = config.H

        k1, k2 = jax.random.split(key, 2)
        # Same Glorot-like scaling as PlainNeuralODECell so the MLP
        # contribution to dh/dt starts on the same scale.
        scale1 = 1.0 / float(jnp.sqrt(Q + d))
        scale2 = 1.0 / float(jnp.sqrt(H))
        self.mlp_W1 = scale1 * jax.random.normal(k1, (Q + d, H))
        self.mlp_b1 = jnp.zeros((H,))
        self.mlp_W2 = scale2 * jax.random.normal(k2, (H, Q))
        self.mlp_b2 = jnp.zeros((Q,))

        # τ-init: inverse softplus so that softplus(v) + tau_min == tau_init
        # at init. Identical pattern to LiquidQuantumCell.tau_unconstrained.
        delta = float(config.tau_init - config.tau_min)
        init_unconstrained = float(jnp.log(jnp.expm1(jnp.asarray(delta))))
        self.tau_unconstrained = jnp.full((Q,), init_unconstrained)

    def tau(self) -> jnp.ndarray:
        """Positive per-unit time constants, shape (Q,)."""
        return (jax.nn.softplus(self.tau_unconstrained)
                + self.config.tau_min)

    def __call__(self, t, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """dh/dt for ONE sample.

        Args:
            t : scalar time (ignored — vector field is autonomous given
                held-constant input x; matches NeuralODECell + LQC API).
            h : (Q,) hidden state.
            x : (d,) input feature vector — held constant on the
                integration interval (zero-order-hold).

        Returns: (Q,) time derivative.
        """
        cfg = self.config
        if h.shape != (cfg.hidden_dim,):
            raise ValueError(
                f"h must have shape ({cfg.hidden_dim},), got {h.shape}")
        if x.shape != (cfg.input_dim,):
            raise ValueError(
                f"x must have shape ({cfg.input_dim},), got {x.shape}")

        # MLP drive term — identical to PlainNeuralODECell.
        hx = jnp.concatenate([h, x])
        hidden = hx @ self.mlp_W1 + self.mlp_b1
        if cfg.activation == "tanh":
            hidden = jnp.tanh(hidden)
        else:
            hidden = jax.nn.relu(hidden)
        drive = hidden @ self.mlp_W2 + self.mlp_b2                 # (Q,)

        # Liquid-τ leak — the ONLY structural difference from the non-liquid
        # baseline. Per Hasani 2021 LTC (input-independent-τ variant).
        leak = h / self.tau()                                       # (Q,)

        return -leak + drive

    def num_parameters(self) -> int:
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        return sum(int(jnp.size(leaf)) for leaf in leaves)


# ---------------------------------------------------------------------------
# Full forecaster (mirrors PlainNeuralODEForecaster structure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassicalLTCForecasterConfig:
    """Config for the classical LTC forecaster.

    Mirrors `PlainNeuralODEForecasterConfig` field-for-field plus the
    two τ-init scalars. Diffrax settings are intentionally identical
    so the head-to-head comparison varies only in the τ-presence.
    """

    input_dim: int
    hidden_dim: int = 4
    mlp_hidden: int = 0
    activation: str = "tanh"
    tau_init: float = 1.0
    tau_min: float = 0.1
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
        if self.tau_min <= 0:
            raise ValueError(f"tau_min must be > 0, got {self.tau_min}")
        if self.tau_init <= self.tau_min:
            raise ValueError(
                f"tau_init must be > tau_min ({self.tau_min}), "
                f"got tau_init={self.tau_init}")


def _build_solver(name: str):
    if name == "tsit5":
        return diffrax.Tsit5()
    if name == "dopri5":
        return diffrax.Dopri5()
    raise ValueError(f"unknown solver: {name}")  # pragma: no cover


class ClassicalLTCForecaster(eqx.Module):
    """Classical liquid-time-constant forecaster (non-quantum).

    Parameters:
        cell                       : ClassicalLTCCell (carries τ)
        initial_h_W                : (input_dim, hidden_dim)
        initial_h_b                : (hidden_dim,)
        delta_head_W               : (hidden_dim, input_dim)
        delta_head_b               : (input_dim,)
        delta_scale_unconstrained  : scalar (learnable; softplus + floor)
    """

    cell: ClassicalLTCCell
    initial_h_W: jnp.ndarray
    initial_h_b: jnp.ndarray
    delta_head_W: jnp.ndarray
    delta_head_b: jnp.ndarray
    delta_scale_unconstrained: jnp.ndarray

    config: ClassicalLTCForecasterConfig = eqx.field(static=True)

    def __init__(self, config: ClassicalLTCForecasterConfig,
                 *, key: jax.Array) -> None:
        self.config = config
        k_cell, k_hW, k_dW = jax.random.split(key, 3)

        cell_cfg = ClassicalLTCCellConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            mlp_hidden=config.mlp_hidden,
            activation=config.activation,
            tau_init=config.tau_init,
            tau_min=config.tau_min,
        )
        self.cell = ClassicalLTCCell(cell_cfg, key=k_cell)

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

        Mirrors `PlainNeuralODEForecaster.__call__` exactly; the only
        difference is that the cell carries learnable τ.
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
