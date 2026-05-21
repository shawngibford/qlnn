"""te_qpinn_qnn_2d — fully-quantum trainable-embedding QPINN, 2D port.

Source: 2605.13892v1 (QPINN lid-driven cavity), corroborated by
2602.14596v1 and 2602.09291v1 (P3a dual-verified per
`refs/CIRCUIT_SPECS.md` §2). 1D faithful implementation at
`src/qlnn_/circuits/te_qpinn.py:build_te_qpinn_qnn` (Eqs. 10-18,
25-26 of the source).

1D architecture (CIRCUIT_SPECS §2):

    x  →  affine-normalize x̃ ∈ [-1, 1]
       →  U_embed(θ_Q, x̃)               — K embedding layers,
                                            each = per-qubit
                                            R_y(trainable) + R_z(θ·x̃)
                                            + nn-CNOT chain
       →  α_k = π · ⟨Z_k⟩  (per qubit)
       →  ⊗_k R_y(α_k)                   (re-encode)
       →  HEA U_var(θ_var), L layers      (Eq. 12)
       →  readout O = Σ_j ⟨Z_j⟩          (Eq. 26)

**Declared design choice (P3.9 amendment to CIRCUIT_SPECS §2):**
Split-qubit U_embed. The U_embed PQC operates on
`n_total = n_t_qubits + n_x_qubits` qubits:

  - Each embedding layer applies:
      * On t-qubits (0..n_t-1): trainable R_y(θ_emb_t,k,0)
        + input-modulated R_z(θ_emb_t,k,1 · t̃).
      * On x-qubits (n_t..n_t+n_x-1): trainable R_y(θ_emb_x,k,0)
        + input-modulated R_z(θ_emb_x,k,1 · x̃).
      * nn-CNOT chain across ALL n_total qubits (mixes t- and x-
        encodings, the "trainable embedding entangles coordinates").
  - α_k = π·⟨Z_k⟩ for all k ∈ [n_total]. Both t- and x-encoded qubits
    contribute (the re-encoding sees both halves).
  - U_var operates over n_total qubits, L layers, same {Rx,Ry,Rz} +
    nn-CNOT chain as the 1D code path.
  - Readout: Σ_j ⟨Z_j⟩ over all n_total qubits (1D paper Eq. 26,
    bounded by [-n_total, n_total] — well-conditioned for gradients).

Rationale for split-qubit U_embed (declared, P3.9):
  1. The source paper leaves U_embed's gate schedule "schematic only"
     (CIRCUIT_SPECS §2 final bullet) for the 1D case already; the
     1D code's decision was minimal-faithful + linear-in-Nq·L scaling.
     Split-qubit U_embed preserves that scaling property exactly
     (still 2·n·K embedding trained scalars in total — split is
     n_t-half + n_x-half, total = (n_t + n_x)·K·2 = 2·n·K).
  2. Symmetric with `chebyshev_dqc_2d` and `te_qpinn_fnn_2d`'s
     split-qubit layouts — direct apples-to-apples comparison.
  3. The trainable R_y's stay independent per qubit; only the
     input-modulated R_z's differ between t- and x-encoded halves
     (R_z(θ·t̃) on first half, R_z(θ·x̃) on second).

Faithfulness hooks (1D unit-test from CIRCUIT_SPECS §2) that survive:
  - Total trained-param scaling: n_total·(2K + 3L). Linear in
    n_total·(K + L) as the 1D paper requires (parameterized test).
  - HEA U_var structure verbatim from 1D (per-qubit {Rx,Ry,Rz} +
    nn-CNOT chain).
  - Σ Z readout (Eq. 26) preserved verbatim.

Coordinate convention: `make_pde_residual_loss` applies
`_affine_to_chebyshev_axis` BEFORE calling the circuit, so the
circuit receives `(t_chev, x_chev) ∈ (-1, 1)²` already mapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import pennylane as qml


@dataclass(frozen=True)
class TEQPINNQnn2DConfig:
    """2D-port config for fully-quantum trainable-embedding QPINN.

    Args:
      n_t_qubits        : number of qubits encoding the t coordinate.
      n_x_qubits        : number of qubits encoding the x coordinate.
      num_layers        : L_var (HEA variational depth).
      num_embed_layers  : K_embed (U_embed depth).
      device_name       : PennyLane device.
    """

    n_t_qubits: int = 2
    n_x_qubits: int = 2
    num_layers: int = 5
    num_embed_layers: int = 3
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.n_t_qubits < 1:
            raise ValueError(f"n_t_qubits must be >= 1, got {self.n_t_qubits}")
        if self.n_x_qubits < 1:
            raise ValueError(f"n_x_qubits must be >= 1, got {self.n_x_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.num_embed_layers < 1:
            raise ValueError(
                f"num_embed_layers must be >= 1, "
                f"got {self.num_embed_layers}")

    @property
    def num_qubits(self) -> int:
        return self.n_t_qubits + self.n_x_qubits

    @property
    def embed_t_weight_shape(self) -> tuple[int, int, int]:
        """(K, n_t, 2) — per-t-qubit (R_y trainable, R_z input scaler)."""
        return (self.num_embed_layers, self.n_t_qubits, 2)

    @property
    def embed_x_weight_shape(self) -> tuple[int, int, int]:
        """(K, n_x, 2) — per-x-qubit (R_y trainable, R_z input scaler)."""
        return (self.num_embed_layers, self.n_x_qubits, 2)

    @property
    def var_weight_shape(self) -> tuple[int, int, int]:
        """(L, n_total, 3) — HEA Rx, Ry, Rz angles across all qubits."""
        return (self.num_layers, self.num_qubits, 3)

    @property
    def n_trained_params(self) -> int:
        """Total trained scalars: 2·n_total·K + 3·n_total·L.

        Linear in n_total·(K + L) — preserves the 1D paper's scaling
        hook (CIRCUIT_SPECS §2).
        """
        n = self.num_qubits
        return 2 * n * self.num_embed_layers + 3 * n * self.num_layers


def init_te_qpinn_qnn_2d_solver_params(
    cfg: TEQPINNQnn2DConfig, *, seed: int = 0,
) -> dict:
    """Build the {w, s, b} Lagaris-hard-IC outer pytree for te_qpinn_qnn_2d.

    `w` carries: `embed_t_W`, `embed_x_W`, `var_W` — three weight
    tensors that the circuit consumes.
    """
    k = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(k, 3)
    qcirc_w = {
        "embed_t_W": 0.3 * jax.random.normal(k1, cfg.embed_t_weight_shape),
        "embed_x_W": 0.3 * jax.random.normal(k2, cfg.embed_x_weight_shape),
        "var_W":     0.1 * jax.random.normal(k3, cfg.var_weight_shape),
    }
    return {
        "w": qcirc_w,
        "s": jnp.asarray(1.0),
        "b": jnp.asarray(0.0),
    }


def build_te_qpinn_qnn_2d(
    cfg: TEQPINNQnn2DConfig | None = None,
) -> Callable[[jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]:
    """Return solver circuit `f(t_chev, x_chev, weights) → scalar`.

    Output is `Σ_j ⟨Z_j⟩` (paper Eq. 26), a scalar in
    `[-n_total, n_total]`. Drop-in compatible with
    `qlnn_.training.pde_residual_loss.make_pde_residual_loss`.
    """
    cfg = cfg or TEQPINNQnn2DConfig()
    n_t = cfg.n_t_qubits
    n_x = cfg.n_x_qubits
    n = cfg.num_qubits
    L = cfg.num_layers
    K = cfg.num_embed_layers
    dev_e = qml.device(cfg.device_name, wires=n)
    dev_v = qml.device(cfg.device_name, wires=n)

    @qml.qnode(dev_e, interface="jax")
    def embed_qnode(t_chev: jnp.ndarray, x_chev: jnp.ndarray,
                    embed_t_W: jnp.ndarray, embed_x_W: jnp.ndarray):
        # K embedding layers. Per layer:
        #   - On t-qubits (0..n_t-1): R_y(trainable) + R_z(trainable · t̃).
        #   - On x-qubits (n_t..n_t+n_x-1): R_y(trainable) + R_z(trainable · x̃).
        #   - nn-CNOT chain across ALL n qubits (mixes t- and x-
        #     encodings — entangles the coordinates).
        for layer in range(K):
            for q in range(n_t):
                qml.RY(embed_t_W[layer, q, 0], wires=q)
                qml.RZ(embed_t_W[layer, q, 1] * t_chev, wires=q)
            for q in range(n_x):
                qml.RY(embed_x_W[layer, q, 0], wires=n_t + q)
                qml.RZ(embed_x_W[layer, q, 1] * x_chev, wires=n_t + q)
            for q in range(n - 1):
                qml.CNOT(wires=[q, q + 1])
        return tuple(qml.expval(qml.PauliZ(q)) for q in range(n))

    @qml.qnode(dev_v, interface="jax")
    def var_qnode(alpha: jnp.ndarray, var_W: jnp.ndarray):
        # Re-encode α_k into HEA solver block (paper Eq. 25).
        for q in range(n):
            qml.RY(alpha[q], wires=q)
        for layer in range(L):
            for q in range(n):
                qml.RX(var_W[layer, q, 0], wires=q)
                qml.RY(var_W[layer, q, 1], wires=q)
                qml.RZ(var_W[layer, q, 2], wires=q)
            for q in range(n - 1):
                qml.CNOT(wires=[q, q + 1])
        # Readout (paper Eq. 26).
        if n == 1:
            return qml.expval(qml.PauliZ(0))
        return qml.expval(qml.sum(*(qml.PauliZ(q) for q in range(n))))

    def pipeline(t_chev: jnp.ndarray, x_chev: jnp.ndarray,
                 weights: dict) -> jnp.ndarray:
        z = embed_qnode(t_chev, x_chev,
                         weights["embed_t_W"], weights["embed_x_W"])
        z = jnp.stack(z) if isinstance(z, tuple) else z
        alpha = jnp.pi * z                          # α_k = π·⟨Z_k⟩
        return var_qnode(alpha, weights["var_W"])

    return pipeline
