"""Recurrence-Free Quantum Reservoir Computing (RF-QRC).

Faithful implementation of the **QRC-C4** configuration from Ahmed,
Tennie, Magri, *"Prediction of chaotic dynamics and extreme events:
a recurrence-free quantum reservoir computing approach,"* **Phys. Rev.
Research 6, 043082 (2024)**, arXiv:2405.03390v2. Source-grounded in
`refs/CIRCUIT_SPECS.md` §4 (P3a dual-verified; section/equation
citations below reference that card / the paper directly).

**RF-QRC is NOT a trainable PQC** — it does not conform to the
`AnsatzProtocol` registry interface (gradient-trained `(inputs,
weights) → (Q,) PauliZ`). The architecture is:

  - Reservoir: a **fixed/random** quantum circuit. The recurrence
    operator `P` of generic QRC is set to identity (the "RF"
    contribution, paper §III p.6); the data feature map `Φ` is
    applied **twice** in compensation (Table I QRC-C4 "(×2)", p.7);
    a fixed random `V(α)` follows (α∈ℝ^n, α∼U[0,4π], seeded once and
    frozen — paper §IV B 1, p.8).
  - Measurement: full **2^n computational-basis probability vector**
    `r̂` (paper §II C p.5, Fig. 6 p.8 `ℝ^{2^n}`), classically smoothed
    by the **leaky-integrator** `r(t+1) = (1-ε)·r(t) + ε·r̂(t+1)`
    (Eq. 2 p.3); a constant `1` bias is appended (paper §II A p.4).
  - Readout: a **classical linear map** `W_out`, the ONLY trained
    object, fit by **closed-form Tikhonov ridge**
    `(R Rᵀ + βI) W_out = R U_dᵀ` (Eq. 3 p.4); predict
    `u_p(t+1) = r(t+1)ᵀ W_out` (Eq. 4 p.4); `β ∈ {1e-6, 1e-9, 1e-12}`.

**DECLARED DESIGN CHOICES** (source-schematic items, resolved + cited
per P3a discipline):

- *Φ "fully entangled" CNOT pattern.* Fig. 23 shows the pattern only
  for `q_0..q_3` with a vertical ellipsis for `q_4..q_{n-1}` (the
  schematic-source gap recorded in CIRCUIT_SPECS §4). We resolve as
  **all-to-all upper-triangular cascade**: `CNOT(i, j)` for every pair
  `i < j` — the natural "fully connected" generalization matching the
  small-n drawing and the §III description "Fully entangling feature
  map, in which CNOT gates entangle pairs of qubits".
- *V(α) "fully entangled symmetric" structure.* Source (Fig. 24, p.16)
  is schematic; §III p.6 says it "differs from (c) by an additional
  data encoding layer following the CNOT gates." We resolve as
  per-qubit `R_Y(α_k)` → upper-triangular CNOT cascade → per-qubit
  `R_Y(α_k)` — one trained-fixed parameter per qubit (`α ∈ ℝ^n`,
  matches paper §IV B 1), symmetric R_Y wrap around the entangler.
- *Raw → [0, 2π] data rescale.* Per-channel linear:
  `θ_k(x) = 2π · (x_k − x_min_k) / (x_max_k − x_min_k)`, clipped.
  `(x_min, x_max)` are fit on the training X and frozen — standard
  for reservoir computing, consistent with paper §III "rescaled to
  the interval [0, 2π]" (Appendix A p.16).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RFQRCConfig:
    """QRC-C4 (RF-QRC) configuration.

    Args:
      num_qubits      : reservoir width n; reservoir state has dim 2^n.
      input_dim       : data-channel count; if it differs from n, the
                        rescaled input is wrapped/padded onto qubits
                        cyclically (declared design choice; the paper
                        keeps input_dim ≤ n in its experiments).
      alpha_seed      : fixed seed for the V(α) random draw (paper:
                        "predefined seed ... kept fixed").
      beta            : Tikhonov regularization. Paper sweeps
                        {1e-6, 1e-9, 1e-12}.
      leak_rate       : ε of Eq. 2; 1.0 = no leak (raw r̂ only).
      device_name     : PennyLane device.
    """

    num_qubits: int = 6
    input_dim: int = 3
    alpha_seed: int = 0
    beta: float = 1e-9
    leak_rate: float = 1.0
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if not 0.0 < self.leak_rate <= 1.0:
            raise ValueError(
                f"leak_rate must be in (0, 1], got {self.leak_rate}")
        if self.beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {self.beta}")
        if self.beta not in (1e-6, 1e-9, 1e-12):
            # paper sweeps these exact values; warn-via-allow but allow
            # anything positive for HPO.
            pass

    @property
    def reservoir_dim(self) -> int:
        return 2 ** self.num_qubits

    @property
    def feature_dim(self) -> int:
        """Reservoir vector + constant-1 bias (paper §II A p.4)."""
        return self.reservoir_dim + 1


# ---------------------------------------------------------------------------
# Fixed reservoir circuit  Φ(x) Φ(x) V(α)   (paper Eq. 9 with P = I)
# ---------------------------------------------------------------------------


def _fully_entangled_cnots(n: int) -> None:
    """Upper-triangular all-pairs CNOT cascade — resolution of the
    schematic 'fully entangled' wiring beyond Fig. 23's q_0..q_3 (the
    declared design choice; see module docstring)."""
    for i in range(n):
        for j in range(i + 1, n):
            qml.CNOT(wires=[i, j])


def _feature_map(theta: jnp.ndarray, n: int) -> None:
    """Φ — paper Fig. 23 / Appendix A p.16.

    Per qubit q_k: H, R_Y(θ_k), then a fully-entangled CNOT block, then
    R_Y(θ_k). `theta` is the rescaled data already in [0, 2π].
    """
    for k in range(n):
        qml.Hadamard(wires=k)
        qml.RY(theta[k], wires=k)
    _fully_entangled_cnots(n)
    for k in range(n):
        qml.RY(theta[k], wires=k)


def _v_alpha(alpha: jnp.ndarray, n: int) -> None:
    """V(α) — paper §III p.6 / Fig. 24, declared-design-choice form
    'fully entangled symmetric': R_Y(α_k) → CNOT cascade → R_Y(α_k)."""
    for k in range(n):
        qml.RY(alpha[k], wires=k)
    _fully_entangled_cnots(n)
    for k in range(n):
        qml.RY(alpha[k], wires=k)


def build_rf_qrc_reservoir(cfg: RFQRCConfig):
    """Return a JAX-interfaced QNode `(theta, alpha) -> probs(2^n)`.

    `theta` is the per-qubit angle-encoded data (already in [0, 2π]).
    `alpha` is the fixed V(α) draw (frozen, NOT trained).
    The output is the FULL 2^n computational-basis probability vector
    `r̂` (paper §II C p.5, Fig. 6 `ℝ^{2^n}`).
    """
    n = cfg.num_qubits
    dev = qml.device(cfg.device_name, wires=n)

    @qml.qnode(dev, interface="jax")
    def circuit(theta: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
        # P = identity (RF contribution): no previous-state encoding.
        _feature_map(theta, n)      # Φ #1
        _feature_map(theta, n)      # Φ #2   ("×2" — Table I QRC-C4)
        _v_alpha(alpha, n)
        return qml.probs(wires=range(n))

    return circuit


# ---------------------------------------------------------------------------
# Forecaster: closed-form ridge readout over the fixed reservoir
# ---------------------------------------------------------------------------


class RFQRCForecaster:
    """Fixed-reservoir + closed-form ridge readout. NOT a registry
    AnsatzProtocol — see module docstring."""

    def __init__(self, cfg: RFQRCConfig) -> None:
        self.cfg = cfg
        # Frozen parameters (NOT trained):
        # α ~ Uniform[0, 4π] from a fixed seed, frozen for the lifetime
        # of this object (paper §IV B 1, p.8 "kept fixed throughout
        # training and prediction").
        key = jax.random.PRNGKey(cfg.alpha_seed)
        self.alpha: jnp.ndarray = (4.0 * jnp.pi) * jax.random.uniform(
            key, (cfg.num_qubits,))
        self.x_min: Optional[np.ndarray] = None
        self.x_max: Optional[np.ndarray] = None
        self._range: Optional[np.ndarray] = None
        # Trained parameter (the ONLY one):
        self.W_out: Optional[np.ndarray] = None   # (2^n+1, N_u)
        # Built lazily (the fixed reservoir QNode):
        self._reservoir: Any = build_rf_qrc_reservoir(cfg)

    # --- input rescale + qubit angle mapping -------------------------------

    def _fit_rescale(self, X: np.ndarray) -> None:
        # Per-channel linear rescale into [0, 2π]; declared design
        # choice (see module docstring), fit on training X and frozen.
        self.x_min = np.asarray(X.min(axis=0), dtype=np.float64)
        self.x_max = np.asarray(X.max(axis=0), dtype=np.float64)
        # Avoid zero-range divides on constant channels.
        rng = self.x_max - self.x_min
        rng[rng < 1e-12] = 1.0
        self._range = rng                                          # (D,)

    def _theta_for_input(self, x_row: np.ndarray) -> jnp.ndarray:
        """Map one data row x ∈ ℝ^{input_dim} to per-qubit angles
        θ ∈ [0, 2π]^{num_qubits}. If input_dim != num_qubits, the
        rescaled channels are wrapped cyclically onto qubits."""
        scaled = 2.0 * np.pi * (x_row - self.x_min) / self._range
        scaled = np.clip(scaled, 0.0, 2.0 * np.pi)
        # No explicit dtype: JAX is configured float32 globally (locked
        # gotcha #2 — global x64 poisons Diffrax). Angles are O(1).
        if self.cfg.input_dim == self.cfg.num_qubits:
            return jnp.asarray(scaled)
        # Cyclic wrap (declared design choice for input_dim != n_qubits).
        idx = np.arange(self.cfg.num_qubits) % self.cfg.input_dim
        return jnp.asarray(scaled[idx])

    # --- reservoir features over a time series -----------------------------

    def reservoir_features(self, X: np.ndarray) -> np.ndarray:
        """Compute the (T, 2^n+1) reservoir-feature matrix for a series.

        Implements the leaky-integrator update Eq. 2 + the constant-1
        bias append. The QUANTUM circuit at step t depends ONLY on
        x_in(t) and the frozen α — never on r(t-1) (the recurrence is
        absent; the only temporal coupling is the classical leaky
        smoothing).
        """
        T = X.shape[0]
        eps = self.cfg.leak_rate
        out = np.empty((T, self.cfg.feature_dim), dtype=np.float64)
        r_prev = np.zeros(self.cfg.reservoir_dim, dtype=np.float64)
        for t in range(T):
            theta = self._theta_for_input(np.asarray(X[t]))
            r_hat = np.asarray(self._reservoir(theta, self.alpha))
            r = (1.0 - eps) * r_prev + eps * r_hat
            r_prev = r
            out[t, :-1] = r
            out[t, -1] = 1.0           # constant bias (§II A p.4)
        return out

    # --- ridge fit / predict ----------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Solve the closed-form Tikhonov ridge (paper Eq. 3 p.4):

            (R Rᵀ + β I) W_out = R Y

        where R = reservoir_features(X).T  shape (D, T),
              Y is the target trajectory shape (T, N_u),
              W_out ends up shape (D, N_u)  with D = 2^n + 1.
        """
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        if Y.shape[0] != X.shape[0]:
            Y = Y.T
        self._fit_rescale(np.asarray(X, dtype=np.float64))
        R = self.reservoir_features(X).T            # (D, T)
        D = R.shape[0]
        A = R @ R.T + self.cfg.beta * np.eye(D)
        B = R @ Y                                    # (D, N_u)
        self.W_out = np.linalg.solve(A, B)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W_out is None:
            raise RuntimeError("call .fit(X, Y) before .predict")
        R = self.reservoir_features(X)              # (T, D)
        return R @ self.W_out                        # (T, N_u)

    # --- introspection (used by the unit-test hooks) -----------------------

    @property
    def n_trained_params(self) -> int:
        """Trainable parameter count = size(W_out) ONLY (paper §III
        p.6; the entire quantum circuit is fixed/random)."""
        if self.W_out is None:
            return 0
        return int(np.prod(self.W_out.shape))
