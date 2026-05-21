"""te_qpinn_fnn_2d — Berger 2025 trainable-FNN-embedding, 2D port.

Source: Berger, Hosters, Möller, *"Trainable embedding quantum physics
informed neural networks for solving nonlinear PDEs,"* Sci. Rep. 2025
(`s41598-025-02959-z`). 1D faithful implementation at
`src/qlnn_/circuits/te_qpinn.py` (P3a dual-verified per
`refs/CIRCUIT_SPECS.md` §1).

1D architecture (paper Eqs. 11–13):

    x  →  FNN(x̃; θ_emb) = φ ∈ ℝ^n
       →  ⊗_k R_y( φ_k · x̃ )                    (Eq. 11)
       →  L × [ per-qubit {R_x,R_y,R_z} + nn-CNOT chain ]   (Eq. 12)
       →  O = ⟨⊗_k Z_k⟩                            (Eq. 13)

**Declared design choice (P3.9 amendment to CIRCUIT_SPECS §1):**
Split-qubit 2D port:

  - Two FNN heads (FNN_t, FNN_x), each taking the 2-vector input
    (t, x), each producing a per-qubit angle weight vector. FNN_t
    has output dim `n_t_qubits`; FNN_x has output dim `n_x_qubits`.
    The hidden layer per FNN matches the 1D paper's `fnn_hidden_dim`
    default (16).
  - Split-qubit embedding: qubits `0..n_t_qubits-1` carry
    `R_y(φ_t,k · t̃)`; qubits `n_t..n_t+n_x-1` carry
    `R_y(φ_x,k · x̃)`. This mirrors `chebyshev_dqc_2d`'s split-qubit
    feature map convention so the two families are apples-to-apples
    on PDE benchmarks.
  - HEA + readout identical to the 1D paper (Eq. 12, Eq. 13):
    L layers of {Rx, Ry, Rz} + nn-CNOT chain across all
    `n = n_t + n_x` qubits, then `⟨⊗_k Z_k⟩` as the scalar output.

Rationale for split-qubit (vs interleaved or shared-FNN):
1. Symmetric with `chebyshev_dqc_2d`'s split-qubit layout — direct
   apples-to-apples comparison.
2. Preserves the "FNN generates per-qubit angles" paper pattern
   (just doubled to handle two coordinates).
3. Per-coordinate FNN heads each see BOTH coordinates as input
   (not just their own axis) — the FNN learns coupling implicitly.

Faithfulness hooks (Berger 2025 anchors that survive the 2D port):
  - PQC rotation count N_rot = 3·n·L is unchanged (depends on
    n=n_t+n_x and L only). Asserted in test_te_qpinn_fnn_2d.py.
  - HEA structure (per-qubit {Rx,Ry,Rz} + nn-CNOT chain) is bit-
    identical to the 1D code path.
  - Tensor-product Z readout (Eq. 13) is preserved verbatim.

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
class TEQPINNFnn2DConfig:
    """2D-port config for Berger 2025 te_qpinn_fnn.

    Args:
      n_t_qubits      : number of qubits encoding the t coordinate.
      n_x_qubits      : number of qubits encoding the x coordinate.
      num_layers      : L (HEA depth) — paper default 5.
      fnn_hidden_dim  : per-FNN-head TanH hidden width — paper default 16.
      device_name     : PennyLane device.
    """

    n_t_qubits: int = 2
    n_x_qubits: int = 2
    num_layers: int = 5
    fnn_hidden_dim: int = 16
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.n_t_qubits < 1:
            raise ValueError(f"n_t_qubits must be >= 1, got {self.n_t_qubits}")
        if self.n_x_qubits < 1:
            raise ValueError(f"n_x_qubits must be >= 1, got {self.n_x_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.fnn_hidden_dim < 1:
            raise ValueError(
                f"fnn_hidden_dim must be >= 1, got {self.fnn_hidden_dim}")

    @property
    def num_qubits(self) -> int:
        return self.n_t_qubits + self.n_x_qubits

    @property
    def pqc_weight_shape(self) -> tuple[int, int, int]:
        """HEA rotation tensor: (L, n, 3) — paper Eq. 12 PQC angles.

        Faithfulness: unchanged from 1D (paper depends on n and L
        only). The 2D port preserves N_rot = 3·n·L.
        """
        return (self.num_layers, self.num_qubits, 3)

    @property
    def n_pqc_rotations(self) -> int:
        """N_rot = 3·n·L — Berger 2025 paper unit-test hook."""
        return 3 * self.num_qubits * self.num_layers


def init_te_qpinn_fnn_2d_solver_params(
    cfg: TEQPINNFnn2DConfig, *, seed: int = 0,
) -> dict:
    """Build the {w, s, b} Lagaris-hard-IC outer pytree for te_qpinn_fnn_2d.

    `w` carries the FULL te_qpinn_fnn_2d weights dict:
      - Two FNN heads (fnn_t_*, fnn_x_*), input dim 2 each.
      - One shared HEA PQC weight tensor (L, n, 3).

    Layout:
      fnn_t_W1 : (2, H)    FNN_t hidden TanH
      fnn_t_b1 : (H,)
      fnn_t_W2 : (H, n_t)  per-t-qubit angle linear head
      fnn_t_b2 : (n_t,)
      fnn_x_W1 : (2, H)    FNN_x hidden TanH (separate weights)
      fnn_x_b1 : (H,)
      fnn_x_W2 : (H, n_x)  per-x-qubit angle linear head
      fnn_x_b2 : (n_x,)
      pqc_W    : (L, n, 3) HEA Rx, Ry, Rz angles (n = n_t + n_x)
    """
    H = cfg.fnn_hidden_dim
    n_t, n_x = cfg.n_t_qubits, cfg.n_x_qubits
    k = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(k, 5)
    qcirc_w = {
        "fnn_t_W1": 0.5 * jax.random.normal(k1, (2, H)),
        "fnn_t_b1": jnp.zeros((H,)),
        "fnn_t_W2": 0.5 * jax.random.normal(k2, (H, n_t)),
        "fnn_t_b2": jnp.zeros((n_t,)),
        "fnn_x_W1": 0.5 * jax.random.normal(k3, (2, H)),
        "fnn_x_b1": jnp.zeros((H,)),
        "fnn_x_W2": 0.5 * jax.random.normal(k4, (H, n_x)),
        "fnn_x_b2": jnp.zeros((n_x,)),
        "pqc_W":    0.1 * jax.random.normal(k5, cfg.pqc_weight_shape),
    }
    return {
        "w": qcirc_w,
        "s": jnp.asarray(1.0),
        "b": jnp.asarray(0.0),
    }


def _fnn_embed_head(xy: jnp.ndarray, W1, b1, W2, b2) -> jnp.ndarray:
    """φ = FNN(xy) with a single TanH hidden layer; output dim per W2."""
    h = jnp.tanh(xy @ W1 + b1)
    return h @ W2 + b2


def build_te_qpinn_fnn_2d(
    cfg: TEQPINNFnn2DConfig | None = None,
) -> Callable[[jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]:
    """Return a JAX-interfaced solver circuit `f(t_chev, x_chev, weights) → scalar`.

    The output is `⟨⊗_k Z_k⟩` (Berger Eq. 13), a scalar in [-1, 1].
    Drop-in compatible with `qlnn_.training.pde_residual_loss
    .make_pde_residual_loss`.
    """
    cfg = cfg or TEQPINNFnn2DConfig()
    n_t = cfg.n_t_qubits
    n_x = cfg.n_x_qubits
    n = cfg.num_qubits
    L = cfg.num_layers
    dev = qml.device(cfg.device_name, wires=n)

    @qml.qnode(dev, interface="jax")
    def circuit(t_chev: jnp.ndarray, x_chev: jnp.ndarray, weights: dict):
        # Pack the (t, x) coordinate as the FNN's 2-vector input.
        xy = jnp.stack([jnp.atleast_1d(t_chev).reshape(()),
                        jnp.atleast_1d(x_chev).reshape(())])  # (2,)

        # Two FNN heads — each sees both coordinates, produces per-half
        # angle weight vector.
        phi_t = _fnn_embed_head(
            xy,
            weights["fnn_t_W1"], weights["fnn_t_b1"],
            weights["fnn_t_W2"], weights["fnn_t_b2"])           # (n_t,)
        phi_x = _fnn_embed_head(
            xy,
            weights["fnn_x_W1"], weights["fnn_x_b1"],
            weights["fnn_x_W2"], weights["fnn_x_b2"])           # (n_x,)

        # Split-qubit embedding (Berger Eq. 11 generalized):
        # qubits 0..n_t-1 carry R_y(φ_t,k · t̃);
        # qubits n_t..n_t+n_x-1 carry R_y(φ_x,k · x̃).
        for k in range(n_t):
            qml.RY(phi_t[k] * t_chev, wires=k)
        for k in range(n_x):
            qml.RY(phi_x[k] * x_chev, wires=n_t + k)

        # HEA L layers (Berger Eq. 12, Fig. 3): {Rx, Ry, Rz} + nn-CNOT chain
        # across all n = n_t + n_x qubits (mixes t- and x-encodings).
        w = weights["pqc_W"]
        for layer in range(L):
            for k in range(n):
                qml.RX(w[layer, k, 0], wires=k)
                qml.RY(w[layer, k, 1], wires=k)
                qml.RZ(w[layer, k, 2], wires=k)
            for k in range(n - 1):
                qml.CNOT(wires=[k, k + 1])

        # Readout (Berger Eq. 13): O = ⊗_k Z_k. Faithful to 1D.
        if n == 1:
            return qml.expval(qml.PauliZ(0))
        return qml.expval(qml.prod(*(qml.PauliZ(k) for k in range(n))))

    return circuit
