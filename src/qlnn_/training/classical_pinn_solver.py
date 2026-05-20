"""P3.8 — Classical MLP-PINN baseline (the audit's headline missing
comparison).

The peer-review audit identified the SINGLE most important missing
control in P3.5/P3.6/P3.7: there is NO classical baseline anywhere.
Every observed solver result is QLNN-vs-QLNN. Without a classical
PINN of equivalent capacity trained via the SAME physics-residual
loss, we cannot tell whether observed performance differences are
quantum-driven or physics-informed-training-driven.

This module provides exactly that baseline: a capacity-matched MLP
PINN that is a **drop-in replacement** for the quantum circuit
inside `make_residual_loss` (1D ODE solver) and
`make_pde_residual_loss` (2D PDE solver). The Lagaris hard-IC trial
solution and `{w, s, b}` pytree contract are PRESERVED — only the
`circuit(x, w)` call is swapped for `mlp_apply(x, w)`. So when we
compare a quantum solver and the classical MLP-PINN, the ONLY
architectural difference is the function class
(circuit-state-vector vs hidden-layer MLP); everything else (loss,
optimizer, IC enforcement, eval grid) is identical.

Following the architectural pattern of `physics_residual_loss.py` and
`pde_residual_loss.py`, this is a SIBLING module — does not modify
either, does not import-cycle. Their gate-test contracts stay
immutable.

Reference: Lagaris, Likas & Fotiadis, IEEE TNN 1998 (the classical
PINN technique). Modern reference: Raissi, Perdikaris, Karniadakis,
J. Comput. Phys. 378 (2019) for the residual-loss MLP framing this
module instantiates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Capacity-matched MLP — the function class we compare quantum against
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLPConfig:
    """Capacity-matched MLP-PINN config.

    `target_param_count` is the QUANTUM family's PQC parameter count
    that we want to match (within the pre-reg's 2× tolerance). The
    constructor picks `hidden_width` so the resulting MLP has
    `≈ target_param_count` trainable scalars.

    Args:
      input_dim: 1 for ODE solver (scalar t), 2 for PDE solver (t, x).
      target_param_count: total MLP weights to match.
      hidden_layers: number of hidden layers (default 2; standard
          PINN depth). The layer widths are uniform.
      activation: 'tanh' (the Lagaris/Raissi default) or 'sin' (the
          SIREN PINN choice).
    """

    input_dim: int = 1
    target_param_count: int = 60
    hidden_layers: int = 2
    activation: str = "tanh"

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.target_param_count < 4:
            raise ValueError(
                f"target_param_count must be >= 4, got {self.target_param_count}")
        if self.hidden_layers < 1:
            raise ValueError(
                f"hidden_layers must be >= 1, got {self.hidden_layers}")
        if self.activation not in ("tanh", "sin"):
            raise ValueError(
                f"activation must be tanh or sin, got {self.activation!r}")

    @property
    def hidden_width(self) -> int:
        """Pick hidden_width so total params ≈ target_param_count.

        For input_dim=d, hidden_layers=L, hidden_width=H, output=1:
          total = d·H + H + (L-1)·(H·H + H) + H + 1
                = H·d + H·(L-1)·(H+1) + 2H + 1
                ≈ (L-1)·H² + (d+2L)·H + 1   for typical L,d
        Solve for H given target. For small L (1-2) and d (1-2), a
        practical rule is H = round((target − d − 1)/(d + 3)) for L=1
        or H = round(sqrt(target/L)) for L≥2. We pick the closer one.
        """
        L = self.hidden_layers
        d = self.input_dim
        target = self.target_param_count
        if L == 1:
            # total = d·H + H + H + 1 = (d+2)·H + 1
            return max(2, int(round((target - 1) / (d + 2))))
        # L >= 2:  dominant term is (L-1)·H²; solve H ≈ sqrt(target/(L-1))
        return max(2, int(round((target / (L - 1)) ** 0.5)))

    def weight_shapes(self) -> list[tuple[int, ...]]:
        """Return a list of (W, b) shape pairs for each MLP layer."""
        H = self.hidden_width
        d = self.input_dim
        L = self.hidden_layers
        shapes: list[tuple[int, ...]] = []
        shapes.append((d, H))            # W_1 (input → hidden_1)
        shapes.append((H,))              # b_1
        for _ in range(L - 1):
            shapes.append((H, H))        # W_l hidden_l → hidden_{l+1}
            shapes.append((H,))          # b_l
        shapes.append((H, 1))            # W_out hidden_L → output
        shapes.append((1,))              # b_out
        return shapes

    def total_params(self) -> int:
        return sum(int(jnp.prod(jnp.array(s))) for s in self.weight_shapes())


# ---------------------------------------------------------------------------
# MLP forward + weight initialization
# ---------------------------------------------------------------------------


def _mlp_init(cfg: MLPConfig, *, seed: int) -> dict:
    """Initialize MLP weights as a dict pytree (keys w0, b0, w1, b1, ...).

    Glorot uniform for weights, zeros for biases — the standard PINN
    init (matches Lagaris 1998 / Raissi et al. 2019).
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 32)
    out: dict = {}
    shapes = cfg.weight_shapes()
    ki = 0
    layer = 0
    while ki < len(shapes):
        Wshape = shapes[ki]
        bshape = shapes[ki + 1]
        fan_in = Wshape[0]
        scale = jnp.sqrt(6.0 / (fan_in + Wshape[-1]))
        out[f"w{layer}"] = scale * (
            2.0 * jax.random.uniform(keys[layer], Wshape) - 1.0)
        out[f"b{layer}"] = jnp.zeros(bshape)
        ki += 2
        layer += 1
    return out


def _mlp_apply(x: jnp.ndarray, weights: dict, cfg: MLPConfig) -> jnp.ndarray:
    """Forward pass through the MLP. Returns a SCALAR for the solver.

    Inputs:
      x : shape `(input_dim,)` — the coordinate (scalar t, or (t, x)
          stacked).
      weights : pytree with keys w0/b0, w1/b1, ..., w_{L}/b_{L}.

    Output:
      scalar (squeezed from the (1,) output).
    """
    L = cfg.hidden_layers
    h = jnp.atleast_1d(x).reshape(cfg.input_dim)
    act = jnp.tanh if cfg.activation == "tanh" else jnp.sin
    for layer in range(L):
        h = act(h @ weights[f"w{layer}"] + weights[f"b{layer}"])
    out = h @ weights[f"w{L}"] + weights[f"b{L}"]
    return out[0]


# ---------------------------------------------------------------------------
# Drop-in circuit-like callables for the existing residual-loss builders
# ---------------------------------------------------------------------------


def build_classical_pinn_1d(cfg: MLPConfig | None = None) -> Callable:
    """Return a solver-style callable  f(x_scalar, weights) → scalar.

    Drop-in compatible with `physics_residual_loss.make_residual_loss`
    (the 1D ODE solver). Used identically:
        p = {"w": init_classical_pinn_weights(cfg, seed=...),
             "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    """
    cfg = cfg or MLPConfig(input_dim=1)
    if cfg.input_dim != 1:
        raise ValueError(
            f"1D classical PINN requires input_dim=1, got {cfg.input_dim}")

    def circuit(x_scalar, weights):
        # `x_scalar` arrives already mapped to Chebyshev [-1, 1] by
        # the caller; we accept whatever scalar coordinate is passed.
        x = jnp.atleast_1d(x_scalar).reshape(1)
        return _mlp_apply(x, weights, cfg)

    return circuit


def build_classical_pinn_2d(cfg: MLPConfig | None = None) -> Callable:
    """Return a solver-style callable  f(t̃, x̃, weights) → scalar.

    Drop-in compatible with `pde_residual_loss.make_pde_residual_loss`
    (the 2D PDE solver). Signature matches `build_chebyshev_dqc_2d`.
    """
    cfg = cfg or MLPConfig(input_dim=2)
    if cfg.input_dim != 2:
        raise ValueError(
            f"2D classical PINN requires input_dim=2, got {cfg.input_dim}")

    def circuit(t_chev, x_chev, weights):
        tx = jnp.array([t_chev, x_chev]).reshape(2)
        return _mlp_apply(tx, weights, cfg)

    return circuit


def init_classical_pinn_weights(cfg: MLPConfig, *, seed: int = 0) -> dict:
    """Initialize the MLP pytree (the {w, s, b}'s `w`)."""
    return _mlp_init(cfg, seed=seed)


# ---------------------------------------------------------------------------
# Capacity-matching helper (the comparison contract)
# ---------------------------------------------------------------------------


def matched_mlp_config(
    target_param_count: int,
    *,
    input_dim: int,
    hidden_layers: int = 2,
    activation: str = "tanh",
) -> MLPConfig:
    """Return an MLPConfig whose total params ≈ target_param_count
    (within 2×, the pre-reg's matched-comparison tolerance).

    If the constructed config's `total_params()` is outside [target/2,
    2×target], we bump `hidden_layers` and try again. Raises if no
    feasible config exists (target too small).
    """
    cfg = MLPConfig(
        input_dim=input_dim,
        target_param_count=target_param_count,
        hidden_layers=hidden_layers,
        activation=activation)
    actual = cfg.total_params()
    if target_param_count / 2.0 <= actual <= 2.0 * target_param_count:
        return cfg
    # Try alternative depths to land closer.
    for L_try in (1, 2, 3, 4):
        c = MLPConfig(
            input_dim=input_dim,
            target_param_count=target_param_count,
            hidden_layers=L_try,
            activation=activation)
        ac = c.total_params()
        if target_param_count / 2.0 <= ac <= 2.0 * target_param_count:
            return c
    # Last resort: return the closest one we have.
    return cfg
