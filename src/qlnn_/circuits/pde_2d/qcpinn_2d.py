"""qcpinn_2d — DV-Circuit QCPINN, 2D (t, x) coordinate port.

The 1D QCPINN family from Farea, Khan, Celebi 2025 (arXiv:2503.16678v6),
faithfully implemented at `src/qlnn_/circuits/qcpinn.py` (P3a dual-
verified per `refs/CIRCUIT_SPECS.md` §3), is already configurable to
accept a multi-coordinate input: `QCPINNConfig.input_dim` controls the
pre-NN's input layer. The 1D builder uses `input_dim=1` (a scalar PDE
coordinate); this module wraps it at `input_dim=2` for `(t, x)` PDE
training.

**Design choice (declared, P3.9):** the pre-NN's first weight matrix
grows from `(1, pre_hidden)` to `(2, pre_hidden)`. Everything downstream
— the angle embedding, the per-topology PQC, the per-qubit ⟨Z⟩ readout,
the post-NN — is gate-by-gate identical to the 1D version. The paper's
per-topology Table 2 parameter-count formulas (4(n−1)L Alternate,
3nL Cascade, (n²+4n)L Cross-mesh, 4nL Layered) depend on n and L only,
NOT on pre-NN input dim, so they hold unchanged. The classical pre-NN
param count grows by `pre_hidden` (one extra row in pre_W1).

This is the lightest of the three P3.9 ports because Farea's
architecture already separates "data encoding via classical pre-NN" from
"quantum variational block." The 2D coordinate is absorbed entirely by
the classical pre-NN; the PQC sees the same `n` data-conditioned angles
regardless of input dim.

Compatibility contract — the returned circuit is a callable
`f(t_chev, x_chev, weights) → scalar` consumable by
`qlnn_.training.pde_residual_loss.make_pde_residual_loss`.

Note on coordinate convention: the make_pde_residual_loss closure
applies its own affine `_affine_to_chebyshev_axis` BEFORE calling the
circuit, so this circuit receives `(t_chev, x_chev) ∈ (-1, 1)²` already.
The qcpinn pre-NN does NOT need additional rescaling — it sees these
two scalars directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp

from qlnn_.circuits.qcpinn import (
    QCPINNConfig,
    _angle_embedding,
    _apply_pqc_layer,
    build_qcpinn_circuit,
    init_qcpinn_weights,
)


@dataclass(frozen=True)
class QCPINN2DConfig:
    """2D-port config for the DV-Circuit QCPINN.

    Mirrors the 1D `QCPINNConfig` but forces `input_dim=2` (the (t, x)
    coordinate pair) and `output_dim=1` (scalar field value). All other
    knobs are passed through to the underlying 1D config.

    Args:
      num_qubits   : n. Paper sweet-spot is 5 (§ Feasibility p.15).
      num_layers   : L. Paper uses single quantum layer L = 1.
      topology     : one of {Alternate, Cascade, Cross-mesh, Layered}.
      pre_hidden   : pre-NN hidden width (paper: 50, §5.2).
      post_hidden  : post-NN hidden width (paper: 50, §5.2).
      device_name  : PennyLane device.
    """

    num_qubits: int = 5
    num_layers: int = 1
    topology: str = "Cascade"
    pre_hidden: int = 50
    post_hidden: int = 50
    device_name: str = "default.qubit"

    def to_1d_config(self) -> QCPINNConfig:
        """Build the underlying 1D config with `input_dim=2`.

        Note the name: this 1D _config object_ holds the 2D _input dim_.
        The 1D config class is the same dataclass; only its `input_dim`
        field differs from a 1D-solver usage.
        """
        return QCPINNConfig(
            num_qubits=self.num_qubits,
            num_layers=self.num_layers,
            topology=self.topology,
            pre_hidden=self.pre_hidden,
            post_hidden=self.post_hidden,
            input_dim=2,
            output_dim=1,
            device_name=self.device_name,
        )


def init_qcpinn_2d_solver_params(
    cfg: QCPINN2DConfig, *, seed: int = 0,
) -> dict:
    """Build the {w, s, b} Lagaris-hard-IC outer pytree for qcpinn_2d.

    `w` carries the FULL qcpinn weights dict (pre-NN + PQC + post-NN)
    so the `make_pde_residual_loss` closure still sees a single `p["w"]`
    handle — matching the pattern used by chebyshev_dqc_2d (where `w`
    is a single HEA weight tensor).

    The output affine `(s, b)` rescales the qcpinn pipeline's scalar
    output `u_θ(t, x) ∈ ℝ` (paper §5.2) to the target field range.
    Identical role to chebyshev_dqc_2d's `(s, b)`.
    """
    qcpinn_weights = init_qcpinn_weights(cfg.to_1d_config(), seed=seed)
    return {
        "w": qcpinn_weights,
        "s": jnp.asarray(1.0),
        "b": jnp.asarray(0.0),
    }


def build_qcpinn_2d(
    cfg: QCPINN2DConfig | None = None,
) -> Callable[[jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]:
    """Return a JAX-interfaced solver circuit `f(t_chev, x_chev, weights) → scalar`.

    `weights` is the qcpinn weights dict returned by
    `init_qcpinn_2d_solver_params(...)["w"]`. The pipeline is:

      (t_chev, x_chev) → pre-NN → per-qubit angles θ_k
                       → angle embedding RX(θ_k) on each qubit
                       → per-topology PQC (L layers)
                       → ⟨Z_k⟩ for each k
                       → post-NN → scalar u_θ(t, x)

    Output is a scalar real number (the post-NN's single output).
    """
    cfg = cfg or QCPINN2DConfig()
    cfg_1d = cfg.to_1d_config()
    qnode = build_qcpinn_circuit(cfg_1d)
    n = cfg.num_qubits

    def pipeline(t_chev: jnp.ndarray, x_chev: jnp.ndarray,
                 weights: dict) -> jnp.ndarray:
        # Pack the (t, x) coordinate as the pre-NN's 2-vector input.
        xy = jnp.stack([jnp.atleast_1d(t_chev).reshape(()),
                        jnp.atleast_1d(x_chev).reshape(())])  # (2,)
        # pre-NN (Tanh activation, paper §5.2):
        # xy ∈ (2,) → h ∈ (pre_hidden,) → theta ∈ (n,)
        h = jnp.tanh(xy @ weights["pre_W1"] + weights["pre_b1"])
        theta = h @ weights["pre_W2"] + weights["pre_b2"]
        # quantum: per-qubit ⟨Z_k⟩
        z_tuple = qnode(theta, weights)
        z = jnp.stack(z_tuple) if isinstance(z_tuple, tuple) else z_tuple
        # post-NN (Tanh, paper §5.2): (n,) → (post_hidden,) → (1,)
        h2 = jnp.tanh(z @ weights["post_W1"] + weights["post_b1"])
        y = h2 @ weights["post_W2"] + weights["post_b2"]    # (1,)
        return y[0]

    return pipeline


def n_trainable_pqc_params(cfg: QCPINN2DConfig) -> int:
    """Per-topology Table 2 closed-form PQC param count.

    Same formula as the 1D case (Table 2 depends on n and L only).
    Used by tests to assert faithfulness of the port.
    """
    n, L = cfg.num_qubits, cfg.num_layers
    return {
        "Alternate":  4 * (n - 1) * L,
        "Cascade":    3 * n * L,
        "Cross-mesh": (n * n + 4 * n) * L,
        "Layered":    4 * n * L,
    }[cfg.topology]
