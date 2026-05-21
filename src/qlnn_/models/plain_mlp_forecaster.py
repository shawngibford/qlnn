"""P5 commit 2 — Plain feedforward MLP forecaster (capacity-matched classical control).

Pre-reg §6 table row: "Classical | MLP | Capacity-matched classical
control". A non-ODE, non-quantum, fully-feedforward classical
baseline. The control for the H1 contrast that **isolates the
Diffrax-integrated dynamics effect** — if both the QLNN and the
Neural-ODE outperform this plain MLP, it tells us the ODE
integration is doing work (independent of the quantum question).

Architecture:

    flatten history (T, d) → (T·d,)
    →  Dense(T·d → H) + tanh
    →  Dense(H → H) + tanh         (optional second hidden layer)
    →  Dense(H → d) + delta_scale  (vector residual head)
    →  y = x[-1] + delta

NO Diffrax. NO quantum. Just MLP. The simplest possible classical
forecaster baseline. Capacity = (T·d)·H + H + H·H + H + H·d + d.

Plugs into the existing `OneStepForecaster` protocol through the
`make_vector_forecaster_adapter` adapter (P4 commit 3c) — same
(history, dt) → next_state signature.

Per pre-reg §6 binding: matched within a factor of 2 in trainable
parameter count to the QLNN/NeuralODE under test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp


def _inv_softplus(y: float) -> float:
    if y <= 0:
        raise ValueError(f"inverse softplus requires y > 0, got {y}")
    return float(math.log(math.expm1(y)))


@dataclass(frozen=True)
class PlainMLPForecasterConfig:
    """Config for the plain feedforward MLP forecaster.

    Args:
      input_dim       : d — state vector dimension (input == output dim).
      window_length   : T — history window length.
      hidden_dim      : H — hidden layer width.
      n_hidden_layers : 1 or 2 (default 2 — enough to match QLNN capacity).
      activation      : "tanh" (default) or "relu".
      delta_scale_init, delta_scale_min : softplus + floor for the
                          learnable residual scale.
      init_head_std   : initialization std-dev for all weight matrices.
    """

    input_dim: int
    window_length: int
    hidden_dim: int = 16
    n_hidden_layers: int = 2
    activation: str = "tanh"
    delta_scale_init: float = 1.0
    delta_scale_min: float = 0.01
    init_head_std: float = 0.1

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.window_length < 2:
            raise ValueError(
                f"window_length must be >= 2, got {self.window_length}")
        if self.hidden_dim < 1:
            raise ValueError(
                f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.n_hidden_layers not in (1, 2):
            raise ValueError(
                f"n_hidden_layers must be 1 or 2, "
                f"got {self.n_hidden_layers}")
        if self.activation not in ("tanh", "relu"):
            raise ValueError(
                f"activation must be 'tanh' or 'relu', "
                f"got {self.activation!r}")
        if self.delta_scale_init <= self.delta_scale_min:
            raise ValueError(
                f"delta_scale_init ({self.delta_scale_init}) must exceed "
                f"delta_scale_min ({self.delta_scale_min})")
        if self.delta_scale_min <= 0:
            raise ValueError("delta_scale_min must be > 0")


class PlainMLPForecaster(eqx.Module):
    """Feedforward MLP forecaster — capacity-matched classical control.

    Parameters (PyTree leaves):
        W1, b1 : (T·d, H), (H,)        first hidden layer
        W2, b2 : (H, H), (H,)          second hidden layer (if n_hidden=2)
        Wout, bout : (H, d), (d,)      output head
        delta_scale_unconstrained : scalar (learnable; softplus + floor)
    """

    W1: jnp.ndarray
    b1: jnp.ndarray
    W2: jnp.ndarray
    b2: jnp.ndarray
    Wout: jnp.ndarray
    bout: jnp.ndarray
    delta_scale_unconstrained: jnp.ndarray
    config: PlainMLPForecasterConfig = eqx.field(static=True)

    def __init__(self, config: PlainMLPForecasterConfig, *,
                 key: jax.Array) -> None:
        self.config = config
        d = config.input_dim
        T = config.window_length
        H = config.hidden_dim
        in_flat = T * d
        std = config.init_head_std

        k1, k2, k3 = jax.random.split(key, 3)
        # Glorot-scaled inits (divide by sqrt(fan_in)).
        self.W1 = std * jax.random.normal(k1, (in_flat, H)) / float(jnp.sqrt(in_flat))
        self.b1 = jnp.zeros((H,))
        # Always allocate W2 so the pytree shape is static; if
        # n_hidden_layers == 1, W2 stays at near-zero init and is
        # bypassed in __call__.
        self.W2 = std * jax.random.normal(k2, (H, H)) / float(jnp.sqrt(H))
        self.b2 = jnp.zeros((H,))
        self.Wout = std * jax.random.normal(k3, (H, d)) / float(jnp.sqrt(H))
        self.bout = jnp.zeros((d,))
        self.delta_scale_unconstrained = jnp.asarray(
            _inv_softplus(
                config.delta_scale_init - config.delta_scale_min),
            dtype=jnp.float32)

    def delta_scale(self) -> jnp.ndarray:
        return (jax.nn.softplus(self.delta_scale_unconstrained)
                + self.config.delta_scale_min)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Single sample. (T, d) history → (d,) next-state prediction.

        Same call signature as VectorForecaster /
        PlainNeuralODEForecaster, so the same adapter
        (`make_vector_forecaster_adapter`) works unchanged.
        """
        cfg = self.config
        if x.ndim != 2 or x.shape[-1] != cfg.input_dim:
            raise ValueError(
                f"x must have shape (T, input_dim={cfg.input_dim}), "
                f"got {tuple(x.shape)}")
        if x.shape[0] != cfg.window_length:
            raise ValueError(
                f"x must have T = {cfg.window_length} timesteps, "
                f"got T = {x.shape[0]}")

        act = jnp.tanh if cfg.activation == "tanh" else jax.nn.relu
        flat = x.reshape(-1)                         # (T·d,)
        h = act(flat @ self.W1 + self.b1)            # (H,)
        if cfg.n_hidden_layers == 2:
            h = act(h @ self.W2 + self.b2)
        delta_raw = jnp.tanh(h @ self.Wout + self.bout)
        delta = delta_raw * self.delta_scale()       # (d,)
        return x[-1] + delta

    def num_parameters(self) -> int:
        """Total trainable scalar count."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        return sum(int(jnp.size(leaf)) for leaf in leaves)
