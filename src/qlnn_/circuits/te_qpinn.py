"""TE-QPINN — Trainable-Embedding Quantum Physics-Informed Neural Network.

Faithful implementation of the **FNN variant** (`te_qpinn_fnn`) from
Berger, Hosters, Möller, *"Trainable embedding quantum physics informed
neural networks for solving nonlinear PDEs,"* **Sci. Rep. (Nature) 2025**
(`s41598-025-02959-z`), source-grounded in `refs/CIRCUIT_SPECS.md` §1
(P3a dual-verified).

Architecture (paper Eqs. 11–13, Fig. 3):

    x  →  rescale x̃  →  FNN(x̃; θ_emb) = φ ∈ ℝ^n
       →  ⊗_k R_y( φ_k · x̃ )                       (Eq. 11)
       →  L × [ per-qubit {R_x,R_y,R_z} + nn-CNOT chain ]   (Eq. 12)
       →  O = ⊗_k Z_k                                 (Eq. 13)

The classical FNN generates the per-qubit angle weights `φ_k`; the
embedding angle on qubit k is the product `φ_k · x̃` (Eq. 11). Trainable
parameters live in TWO pytree leaves: the FNN weights (θ_emb) and the
HEA PQC weights of shape `(L, n, 3)`.

Solver task — the input is the scalar PDE/ODE coordinate. Output is a
scalar (the global tensor-Z observable's expectation, in [-1, 1]).
Plugs into `make_residual_loss` / `train_solver` from
`physics_residual_loss.py` as a drop-in alternative to the
Chebyshev-DQC circuit.

**Unit-test hook (paper anchor, CIRCUIT_SPECS §1):** PQC rotation count
`N_rot = 3·n·L`; weight shape `(L, n, 3)`. The paper states n=4 qubits,
L=5 layers ⇒ 60 PQC params (3·4·5 = 60). Both the primary extractor
and the independent dual-check confirmed this hook.

**DECLARED DESIGN CHOICES** (resolved + cited per P3a discipline):

- *Per-layer CNOT pattern.* CIRCUIT_SPECS §1 records that the paper
  explicitly says "the CNOT pattern varies in literature"; the
  defining text (Fig. 3 + Eq. 12) specifies a **nearest-neighbour
  chain**. We use the linear chain `CNOT(i, i+1)` for i=0..n−2.
- *FNN architecture.* Berger fixes the FNN as a TanH classical net but
  the specific hidden width is paper/problem-dependent. We use a
  single hidden layer of TanH activations with configurable width
  (default 16); input dim = 1 (the scalar solver coordinate).
- *Input rescale.* `x̃ = 2(t−t0)/(t1−t0) − 1 ∈ [−1, 1]`, matching the
  Chebyshev-DQC affine convention from `physics_residual_loss.py` so
  the two solver circuits are interchangeable inside `train_solver`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import pennylane as qml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TEQPINNFnnConfig:
    """`te_qpinn_fnn` configuration (Berger 2025).

    Args:
      num_qubits      : n
      num_layers      : L (the HEA depth)
      fnn_hidden_dim  : single hidden layer width of the trainable
                        embedding FNN  (TanH activation).
      device_name     : PennyLane device.
    """

    num_qubits: int = 4
    num_layers: int = 5
    fnn_hidden_dim: int = 16
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.fnn_hidden_dim < 1:
            raise ValueError(
                f"fnn_hidden_dim must be >= 1, got {self.fnn_hidden_dim}")

    # --- shapes used by both unit tests and the trainer ----------------

    @property
    def pqc_weight_shape(self) -> tuple[int, int, int]:
        """HEA rotation tensor: (L, n, 3) — the paper's PQC params."""
        return (self.num_layers, self.num_qubits, 3)

    @property
    def n_pqc_rotations(self) -> int:
        """N_rot = 3·n·L — the paper's unit-test hook."""
        return 3 * self.num_qubits * self.num_layers


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------


def init_te_qpinn_fnn_weights(cfg: TEQPINNFnnConfig, *, seed: int = 0) -> dict:
    """Return the trainable pytree: FNN (input=1 scalar coord) + PQC.

    Layout:
      fnn_W1 : (1, H)        TanH hidden layer
      fnn_b1 : (H,)
      fnn_W2 : (H, n)        linear to per-qubit φ
      fnn_b2 : (n,)
      pqc_W  : (L, n, 3)     HEA Rx,Ry,Rz angles
    """
    H, n = cfg.fnn_hidden_dim, cfg.num_qubits
    k = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(k, 5)
    return {
        "fnn_W1": 0.5 * jax.random.normal(k1, (1, H)),
        "fnn_b1": jnp.zeros((H,)),
        "fnn_W2": 0.5 * jax.random.normal(k2, (H, n)),
        "fnn_b2": jnp.zeros((n,)),
        "pqc_W":  0.1 * jax.random.normal(k3, cfg.pqc_weight_shape),
    }


def _fnn_embed(x_scalar: jnp.ndarray, weights: dict) -> jnp.ndarray:
    """φ = FNN(x̃) ∈ ℝ^n — single TanH hidden layer."""
    x = jnp.atleast_1d(x_scalar).reshape(1)             # (1,)
    h = jnp.tanh(x @ weights["fnn_W1"] + weights["fnn_b1"])
    return h @ weights["fnn_W2"] + weights["fnn_b2"]     # (n,)


def build_te_qpinn_fnn(
    cfg: TEQPINNFnnConfig | None = None,
) -> Callable[[jnp.ndarray, dict], jnp.ndarray]:
    """Return a solver-style circuit  f(x_scalar, weights) -> scalar.

    The output is `⟨⊗_k Z_k⟩` (Eq. 13), a scalar in [-1, 1]. Drop-in
    compatible with `physics_residual_loss.py:train_solver` (which
    accepts arbitrary param pytrees via optax).
    """
    cfg = cfg or TEQPINNFnnConfig()
    n = cfg.num_qubits
    L = cfg.num_layers
    dev = qml.device(cfg.device_name, wires=n)

    @qml.qnode(dev, interface="jax")
    def circuit(x_scalar: jnp.ndarray, weights: dict):
        phi = _fnn_embed(x_scalar, weights)              # (n,)
        # Embedding (Eq. 11): per-qubit R_y(φ_k · x̃)
        for k in range(n):
            qml.RY(phi[k] * x_scalar, wires=k)
        # HEA L layers (Eq. 12, Fig. 3): {Rx,Ry,Rz} + nn-CNOT chain
        w = weights["pqc_W"]
        for layer in range(L):
            for k in range(n):
                qml.RX(w[layer, k, 0], wires=k)
                qml.RY(w[layer, k, 1], wires=k)
                qml.RZ(w[layer, k, 2], wires=k)
            for k in range(n - 1):
                qml.CNOT(wires=[k, k + 1])
        # Readout (Eq. 13): O = ⊗_k Z_k, a single global observable.
        if n == 1:
            return qml.expval(qml.PauliZ(0))
        return qml.expval(qml.prod(*(qml.PauliZ(k) for k in range(n))))

    return circuit
