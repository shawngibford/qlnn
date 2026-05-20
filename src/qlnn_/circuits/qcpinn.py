"""QCPINN — Quantum-Classical Physics-Informed Neural Network.

Faithful implementation of the **DV-Circuit QCPINN** family from
Farea, Khan, Celebi, *"QCPINN: Quantum-Classical Physics-Informed
Neural Networks for Solving PDEs,"* arXiv:2503.16678v6, source-grounded
in `refs/CIRCUIT_SPECS.md` §3 (P3a dual-verified, incl. Appendix A).

Pipeline (paper §4 p.5; Fig. 1 p.6):

    x  →  pre-NN (1 hidden, 50 TanH)
       →  angle embedding on n=5 qubits          (Fig. 2)
       →  HEA  U(ψ) = ∏_k U_k(ψ_k) W_k           (Eq. 4 p.8)
                in one of 4 topologies (Table 2 p.7)
       →  per-qubit ⟨Z⟩                          (§5.2 p.9)
       →  post-NN (1 hidden, 50 TanH)
       →  scalar solution u_θ(x)

Trainable subsystems: pre-NN + per-topology PQC + post-NN. Embedding
rotations are **NOT** counted in the paper's parameter totals (only the
variational ansatz; §App.A "Open-question resolution" + p.15 worked
example double-verified by P3a).

**The unit-test hooks (CIRCUIT_SPECS §3 / §6) are the closed-form
per-topology formulas of Table 2, p. 7**, each independently verified
by the paper's own worked example at n=5, L=1 (p. 15):

  | Topology    | trainable params | 2-qubit gates | depth        |
  |-------------|------------------|---------------|--------------|
  | Alternate   | 4(n−1)L          | (n−1)L        | 6L           |
  | Cascade     | 3nL              | nL            | (n+2)L       |
  | Cross-mesh  | (n²+4n)L         | (n²−n)L       | (n²−n+4)L    |
  | Layered     | 4nL              | (n−1)L        | 6L           |

p.15 corroborates: Cascade(n=5,L=1) → (15, 5, 7); Cross-mesh → (45, 20, 24).

**DECLARED DESIGN CHOICES** (per-topology gate ordering is "figure-
derived, text-UNSPECIFIED" per spec card line 27 — Fig. 2 panels only).
Resolved as the minimal structurally-faithful schedule that reproduces
Table 2's closed-form counts exactly (params + 2q-gate counts asserted
to match in the unit tests):

  - *Cascade.*  Per layer, per qubit: RX(w), RZ(w); then a **ring of
    CRX(w)** (n controlled-rotations, each trainable). Counts:
    2n single-qubit + n CRX = 3n trained, n two-qubit. ✓ matches Table 2.
  - *Cross-mesh.*  Per layer, per qubit: 5 single-qubit rotations
    (RX,RY,RZ,RX,RZ); then **all-to-all CRX** on every ordered pair
    i ≠ j (n(n−1) controlled rotations). Counts: 5n + n(n−1) = n² + 4n
    trained, n(n−1) two-qubit. ✓.
  - *Alternate.*  Per adjacent pair (i, i+1), i = 0..n−2: 4 trained
    rotations (RX, RZ on qubit i; RX, RZ on qubit i+1) then a CNOT
    (NOT a CRX — fixed entangler per Fig. 2a). Counts: 4(n−1)
    single-qubit + (n−1) CNOT (unparameterized) = 4(n−1) trained,
    (n−1) two-qubit. ✓.
  - *Layered.*  Per layer, per qubit: 4 trained single-qubit rotations
    (RZ, RX, RZ, RX — Fig. 2d shows RZ–RX rotations then CNOT ladder
    then RX). Then a nearest-neighbour CNOT ladder of length n−1
    (unparameterized). Counts: 4n trained, n−1 two-qubit. ✓.

**Derivative method.** The paper uses PyTorch autodiff with
`shots=None` for deterministic simulation (§5.2/5.5 p.9), explicitly
avoiding parameter-shift. We do the JAX equivalent — `jax.jacrev`
through the PennyLane JAX QNode (the locked convention, gotcha #1).
Mathematically equivalent substitution, disclosed per P3a discipline.
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


_TOPOLOGIES = ("Alternate", "Cascade", "Cross-mesh", "Layered")


@dataclass(frozen=True)
class QCPINNConfig:
    """DV-Circuit QCPINN configuration.

    Args:
      num_qubits   : n. Paper sweet-spot is 5 (§ Feasibility p.15).
      num_layers   : L. Paper uses single quantum layer L = 1.
      topology     : one of {Alternate, Cascade, Cross-mesh, Layered}.
      pre_hidden   : pre-NN hidden width (paper: 50, §5.2).
      post_hidden  : post-NN hidden width (paper: 50, §5.2).
      input_dim    : scalar coordinate dim (1 for the prototype gate).
      output_dim   : solver output channel count (1 for u_θ).
      device_name  : PennyLane device.
    """

    num_qubits: int = 5
    num_layers: int = 1
    topology: str = "Cascade"
    pre_hidden: int = 50
    post_hidden: int = 50
    input_dim: int = 1
    output_dim: int = 1
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError(
                f"num_qubits must be >= 2 (entanglers need a pair), "
                f"got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.topology not in _TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {_TOPOLOGIES}, "
                f"got {self.topology!r}")
        if self.pre_hidden < 1 or self.post_hidden < 1:
            raise ValueError("pre/post_hidden must be >= 1")

    # --- closed-form counts (THE unit-test hooks, Table 2 p.7) ---------

    @property
    def n_pqc_params(self) -> int:
        n, L = self.num_qubits, self.num_layers
        return {
            "Alternate":  4 * (n - 1) * L,
            "Cascade":    3 * n * L,
            "Cross-mesh": (n * n + 4 * n) * L,
            "Layered":    4 * n * L,
        }[self.topology]

    @property
    def n_two_qubit_gates(self) -> int:
        n, L = self.num_qubits, self.num_layers
        return {
            "Alternate":  (n - 1) * L,
            "Cascade":    n * L,
            "Cross-mesh": (n * n - n) * L,
            "Layered":    (n - 1) * L,
        }[self.topology]

    @property
    def expected_depth(self) -> int:
        n, L = self.num_qubits, self.num_layers
        return {
            "Alternate":  6 * L,
            "Cascade":    (n + 2) * L,
            "Cross-mesh": (n * n - n + 4) * L,
            "Layered":    6 * L,
        }[self.topology]


# ---------------------------------------------------------------------------
# Per-topology PQC layers  (CIRCUIT_SPECS-faithful counts; see docstring)
# ---------------------------------------------------------------------------


def _cascade_layer(w_rot: jnp.ndarray, w_crx: jnp.ndarray, n: int) -> None:
    # 2n single-qubit rotations + n ring CRX = 3n trained per layer.
    for i in range(n):
        qml.RX(w_rot[i, 0], wires=i)
        qml.RZ(w_rot[i, 1], wires=i)
    for i in range(n):
        qml.CRX(w_crx[i], wires=[i, (i + 1) % n])


def _crossmesh_layer(w_rot: jnp.ndarray, w_crx: jnp.ndarray, n: int) -> None:
    # 5n single-qubit + n(n-1) all-to-all CRX = n²+4n trained per layer.
    for i in range(n):
        qml.RX(w_rot[i, 0], wires=i)
        qml.RY(w_rot[i, 1], wires=i)
        qml.RZ(w_rot[i, 2], wires=i)
        qml.RX(w_rot[i, 3], wires=i)
        qml.RZ(w_rot[i, 4], wires=i)
    idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            qml.CRX(w_crx[idx], wires=[i, j])
            idx += 1


def _alternate_layer(w_pair: jnp.ndarray, n: int) -> None:
    # Per adjacent pair: 4 trained rotations + 1 CNOT (unparameterized).
    # Total trained: 4(n-1); 2-qubit: (n-1).
    for k in range(n - 1):
        qml.RX(w_pair[k, 0], wires=k)
        qml.RZ(w_pair[k, 1], wires=k)
        qml.RX(w_pair[k, 2], wires=k + 1)
        qml.RZ(w_pair[k, 3], wires=k + 1)
        qml.CNOT(wires=[k, k + 1])


def _layered_layer(w_rot: jnp.ndarray, n: int) -> None:
    # 4n single-qubit rotations + (n-1) unparameterized CNOTs.
    for i in range(n):
        qml.RZ(w_rot[i, 0], wires=i)
        qml.RX(w_rot[i, 1], wires=i)
        qml.RZ(w_rot[i, 2], wires=i)
        qml.RX(w_rot[i, 3], wires=i)
    for k in range(n - 1):
        qml.CNOT(wires=[k, k + 1])


def _apply_pqc_layer(weights: dict, cfg: QCPINNConfig, layer: int) -> None:
    n = cfg.num_qubits
    if cfg.topology == "Cascade":
        _cascade_layer(weights["pqc_rot"][layer], weights["pqc_crx"][layer], n)
    elif cfg.topology == "Cross-mesh":
        _crossmesh_layer(
            weights["pqc_rot"][layer], weights["pqc_crx"][layer], n)
    elif cfg.topology == "Alternate":
        _alternate_layer(weights["pqc_pair"][layer], n)
    elif cfg.topology == "Layered":
        _layered_layer(weights["pqc_rot"][layer], n)
    else:
        raise AssertionError(f"unreachable: {cfg.topology}")


def _angle_embedding(theta: jnp.ndarray, n: int) -> None:
    """Angle embedding (Fig. 2 first stage): per-qubit RX(θ_k). The
    embedding ANGLES are data-dependent (NOT trained); spec §App.A
    "Open-question resolution" verified by P3a."""
    for k in range(n):
        qml.RX(theta[k], wires=k)


# ---------------------------------------------------------------------------
# Weight init  (classical pre/post + per-topology PQC)
# ---------------------------------------------------------------------------


def init_qcpinn_weights(cfg: QCPINNConfig, *, seed: int = 0) -> dict:
    n, L = cfg.num_qubits, cfg.num_layers
    k = jax.random.PRNGKey(seed)
    keys = jax.random.split(k, 10)
    out: dict = {
        # pre-NN: input_dim → pre_hidden → num_qubits
        "pre_W1": 0.3 * jax.random.normal(keys[0], (cfg.input_dim, cfg.pre_hidden)),
        "pre_b1": jnp.zeros((cfg.pre_hidden,)),
        "pre_W2": 0.3 * jax.random.normal(keys[1], (cfg.pre_hidden, n)),
        "pre_b2": jnp.zeros((n,)),
        # post-NN: num_qubits → post_hidden → output_dim
        "post_W1": 0.3 * jax.random.normal(keys[2], (n, cfg.post_hidden)),
        "post_b1": jnp.zeros((cfg.post_hidden,)),
        "post_W2": 0.3 * jax.random.normal(keys[3], (cfg.post_hidden, cfg.output_dim)),
        "post_b2": jnp.zeros((cfg.output_dim,)),
    }
    # PQC weights — shapes match the per-topology layer signatures.
    if cfg.topology == "Cascade":
        out["pqc_rot"] = 0.1 * jax.random.normal(keys[4], (L, n, 2))
        out["pqc_crx"] = 0.1 * jax.random.normal(keys[5], (L, n))
    elif cfg.topology == "Cross-mesh":
        out["pqc_rot"] = 0.1 * jax.random.normal(keys[4], (L, n, 5))
        out["pqc_crx"] = 0.1 * jax.random.normal(keys[5], (L, n * (n - 1)))
    elif cfg.topology == "Alternate":
        out["pqc_pair"] = 0.1 * jax.random.normal(keys[4], (L, n - 1, 4))
    elif cfg.topology == "Layered":
        out["pqc_rot"] = 0.1 * jax.random.normal(keys[4], (L, n, 4))
    return out


def n_trainable_pqc_params(weights: dict, cfg: QCPINNConfig) -> int:
    """Count of PQC trained scalars only (NOT the embedding, NOT the
    classical pre/post nets) — the quantity Table 2 reports."""
    total = 0
    for key in ("pqc_rot", "pqc_crx", "pqc_pair"):
        if key in weights:
            arr = jnp.asarray(weights[key])
            total += int(arr.size)
    return total


# ---------------------------------------------------------------------------
# Circuit + full pipeline
# ---------------------------------------------------------------------------


def build_qcpinn_circuit(
    cfg: QCPINNConfig | None = None,
) -> Callable[[jnp.ndarray, dict], jnp.ndarray]:
    """Return the QNode  (theta, weights) -> ⟨Z⟩ per qubit (n,).

    `theta` is the per-qubit angle-encoded data (the pre-NN's output);
    the FULL pipeline pre-NN → angles → circuit → post-NN is
    `build_qcpinn(cfg)`.
    """
    cfg = cfg or QCPINNConfig()
    n = cfg.num_qubits
    dev = qml.device(cfg.device_name, wires=n)

    @qml.qnode(dev, interface="jax")
    def circuit(theta: jnp.ndarray, weights: dict):
        _angle_embedding(theta, n)
        for layer in range(cfg.num_layers):
            _apply_pqc_layer(weights, cfg, layer)
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(n))

    return circuit


def build_qcpinn(
    cfg: QCPINNConfig | None = None,
) -> Callable[[jnp.ndarray, dict], jnp.ndarray]:
    """Return the full pipeline  (x_scalar, weights) -> scalar  u_θ(x).

    Classical pre-NN → angle embedding → DV-circuit → ⟨Z⟩ → post-NN.
    Drop-in solver circuit for `make_residual_loss` / `train_solver`.
    """
    cfg = cfg or QCPINNConfig()
    qnode = build_qcpinn_circuit(cfg)

    def pipeline(x_scalar: jnp.ndarray, weights: dict) -> jnp.ndarray:
        x = jnp.atleast_1d(x_scalar).reshape(cfg.input_dim)
        # pre-NN (Tanh activation, paper §5.2)
        h = jnp.tanh(x @ weights["pre_W1"] + weights["pre_b1"])
        theta = h @ weights["pre_W2"] + weights["pre_b2"]               # (n,)
        # quantum
        z_tuple = qnode(theta, weights)
        z = jnp.stack(z_tuple) if isinstance(z_tuple, tuple) else z_tuple
        # post-NN (Tanh, paper §5.2)
        h2 = jnp.tanh(z @ weights["post_W1"] + weights["post_b1"])
        y = h2 @ weights["post_W2"] + weights["post_b2"]                # (out,)
        # Solver scalar: take the first output (output_dim=1 is the
        # prototype-gate case; multi-output PDEs use the full vector).
        return y[0] if y.shape[0] >= 1 else y

    return pipeline
