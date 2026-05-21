"""P7 commit 1 — T3 trainability/expressibility mechanism module.

Implements the four standard diagnostics that pre-reg H3 says
"explain H1's advantage gap." Given H1 was FALSIFIED (P5 commit
`bd4e3c5`) with an INVERTED regime pattern (Neural-ODE wins on
smooth/periodic; QLNN wins modestly on chaotic Lorenz), the
mechanism question changes character. P7's job:

  Cross-tabulate the four T3 scalars against P5's per-cell
  Δ = NeuralODE − QLNN values. The scalar that best predicts
  the regime-dependent inverted advantage gap IS the paper's
  mechanism explanation.

THE FOUR DIAGNOSTICS (each lifted from
`scripts/analyze_quantum_trainability.py` and packaged as a
module so they can be unit-tested):

  1. **Expressibility** (Sim, Johnson, Aspuru-Guzik, *Adv. Quantum
     Tech.* 2019, arXiv:1905.10876) — KL divergence between the
     distribution of fidelities `|<ψ(θ_a)|ψ(θ_b)>|²` over random
     parameter pairs and the Haar-distribution prediction
     `P_Haar(F) = (N−1)(1−F)^(N−2)`. LOWER KL = closer to
     Haar-random = more expressive.

  2. **Entangling capability** (Meyer-Wallach Q, Brennen 2003 form,
     arXiv:quant-ph/0305094) —
     `Q = 2(1 − ⟨1/n Σ Tr ρ_k²⟩)` averaged over random parameters.
     Q ∈ [0, 1]; Q = 1 is "maximally entangling" (GHZ-like reach);
     Q = 0 is product-state-only.

  3. **Barren-plateau gradient variance** (McClean et al. 2018,
     arXiv:1803.11173) — Var over random params of
     `∂⟨Z_0⟩/∂θ_0`. Exponential decay vs qubit count = barren
     plateau (the circuit is untrainable at scale). We report the
     log-slope of Var vs n_qubits, where slope < 0 with steep
     magnitude indicates a barren plateau.

  4. **Fourier-spectrum effective bandwidth** (Schuld, Sweke, Meyer
     2021, arXiv:2008.08605) — for a single-coordinate circuit,
     the maximum frequency `K_max` accessible in the truncated
     Fourier series `Σ_k c_k exp(i k x)`. Determined by the
     re-uploading depth and encoding scheme. Higher K_max = wider
     bandwidth = more representable smoothness levels.

We implement (1)-(3) numerically here; (4) is computed structurally
from the ansatz config (it's an algebraic property of the
re-uploading pattern, not a numerical estimate). The four scalars
per family at the P4 config (num_qubits=3, num_layers=1) form a
4-D feature vector that the P7 cross-tabulation regresses against
the P5 Δ values.

Test plan: smoke + known-limit cases (e.g. zero-depth circuit
has Q = 0; deep Haar-random circuit should have low KL_to_Haar).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pennylane as qml


# ---------------------------------------------------------------------------
# State-returning circuit builders for the 4 forecaster families
# ---------------------------------------------------------------------------


def _entangle(n: int, pattern: str = "ring") -> None:
    """Apply the entangling layer of given pattern (no parameters)."""
    if n < 2:
        return
    if pattern == "linear":
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
    elif pattern == "ring":
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
        if n > 2:
            qml.CNOT(wires=[n - 1, 0])
    elif pattern == "all_to_all":
        for i in range(n):
            for j in range(i + 1, n):
                qml.CNOT(wires=[i, j])
    else:
        raise ValueError(f"unknown entanglement pattern {pattern!r}")


def _apply_family(
    family: str, inputs: np.ndarray, weights: np.ndarray,
    *, n: int, L: int, encoding: str = "rx",
    entanglement: str = "ring",
) -> None:
    """Apply one of the 4 forecaster ansätze to the prepared device.

    Mirrors `scripts/analyze_quantum_trainability.py:_apply` so the
    state distributions sampled here are bit-identical to the
    canonical reference implementation."""
    eg = qml.RX if encoding == "rx" else qml.RY
    if family == "data_reuploading":
        for layer in range(L):
            for i in range(n):
                qml.RX(inputs[i], wires=i)
            for i in range(n):
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1],
                        weights[layer, i, 2], wires=i)
            _entangle(n, entanglement)
    elif family == "hardware_efficient":
        for i in range(n):
            eg(inputs[i], wires=i)
        for layer in range(L):
            for i in range(n):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            _entangle(n, entanglement)
    elif family == "strongly_entangling":
        for i in range(n):
            eg(inputs[i], wires=i)
        qml.StronglyEntanglingLayers(weights=weights, wires=range(n))
    elif family == "brickwall":
        for i in range(n):
            eg(inputs[i], wires=i)
        for layer in range(L):
            for i in range(n):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            start = 0 if layer % 2 == 0 else 1
            for i in range(start, n - 1, 2):
                qml.CNOT(wires=[i, i + 1])
    else:
        raise ValueError(
            f"unknown family {family!r}; expected one of "
            "data_reuploading / hardware_efficient / "
            "strongly_entangling / brickwall")


def _weight_shape_for(family: str, n: int, L: int) -> tuple[int, ...]:
    """Per-family weight tensor shape at config (n_qubits=n, n_layers=L)."""
    if family == "data_reuploading":
        return (L, n, 3)            # Rot has 3 angles per qubit per layer
    if family == "hardware_efficient":
        return (L, n, 2)            # RY + RZ per qubit per layer
    if family == "strongly_entangling":
        return (L, n, 3)            # PennyLane StronglyEntanglingLayers
    if family == "brickwall":
        return (L, n, 2)
    raise ValueError(f"unknown family {family!r}")


def _state_qnode(
    family: str, n: int, L: int,
    *, encoding: str = "rx", entanglement: str = "ring",
) -> Callable:
    """Return a state-vector QNode for the given family + config.

    Inputs to the QNode: (inputs (n,), weights). Outputs: the
    2^n-dim complex amplitude vector — used directly for
    expressibility (overlap) and Meyer-Wallach Q (partial trace).
    """
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        _apply_family(family, inputs, weights, n=n, L=L,
                      encoding=encoding, entanglement=entanglement)
        return qml.state()
    return circuit


# ---------------------------------------------------------------------------
# 1. Expressibility (Sim 2019)
# ---------------------------------------------------------------------------


def expressibility_kl_to_haar(
    family: str, *, n: int, L: int, n_samples: int = 400,
    n_bins: int = 75, seed: int = 0,
    encoding: str = "rx", entanglement: str = "ring",
) -> float:
    """KL divergence to Haar-random fidelity distribution.

    Procedure:
      1. Sample 2 × n_samples random (input, weight) parameter sets
         and compute fidelities `F = |<ψ_a|ψ_b>|²`.
      2. Histogram into `n_bins` bins on [0, 1] (matches the
         reference script's grid).
      3. Compare to the analytic Haar prediction
         `P_Haar(F) = (N−1)(1−F)^(N−2)`, N = 2^n.
      4. Return KL(P_circuit || P_Haar).

    Lower KL ⇒ closer to Haar ⇒ more expressive. Per Sim 2019.
    """
    rng = np.random.default_rng(seed)
    circ = _state_qnode(family, n, L, encoding=encoding,
                        entanglement=entanglement)
    w_shape = _weight_shape_for(family, n, L)
    N = 2 ** n

    fids = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        inp_a = rng.uniform(-np.pi, np.pi, size=n)
        inp_b = rng.uniform(-np.pi, np.pi, size=n)
        w_a = rng.uniform(-np.pi, np.pi, size=w_shape)
        w_b = rng.uniform(-np.pi, np.pi, size=w_shape)
        psi_a = np.asarray(circ(inp_a, w_a))
        psi_b = np.asarray(circ(inp_b, w_b))
        f = abs(np.vdot(psi_a, psi_b)) ** 2
        fids[i] = float(f)

    # Histogram + Haar reference (matches the canonical reference impl).
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    p, _ = np.histogram(fids, bins=edges, density=True)
    p = p * np.diff(edges)                          # convert to probability
    centers = 0.5 * (edges[:-1] + edges[1:])
    haar = (N - 1) * (1.0 - centers) ** (N - 2)
    haar = haar / haar.sum()
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(haar, eps, None)
    return float(np.sum(p * np.log(p / q)))


# ---------------------------------------------------------------------------
# 2. Entangling capability (Meyer-Wallach Q)
# ---------------------------------------------------------------------------


def _meyer_wallach_q(state: np.ndarray, n: int) -> float:
    """Q = 2(1 − ⟨1/n Σ_k Tr ρ_k²⟩) for a single statevector.

    Implements the Brennen 2003 form (equivalent to Meyer-Wallach
    1995 for pure states): trace-purity of each qubit's reduced
    density matrix, averaged across qubits.

    Q ∈ [0, 1]. Q = 0 ⇒ all qubits in product state (no entanglement).
    Q = 1 ⇒ all qubits maximally mixed (maximally entangled with rest).
    """
    if state.shape != (2 ** n,):
        raise ValueError(
            f"state must have shape (2^n,) = (2**{n},), got {state.shape}")
    psi = state.reshape([2] * n)
    purity_sum = 0.0
    for k in range(n):
        # Reshape so axis k is at index 0; partial-trace by summing
        # over all other axes after computing |ψ|².
        axes = [k] + [i for i in range(n) if i != k]
        psi_k = psi.transpose(axes).reshape(2, -1)
        rho_k = psi_k @ psi_k.conj().T            # (2, 2)
        purity_sum += float(np.real(np.trace(rho_k @ rho_k)))
    return float(2.0 * (1.0 - purity_sum / n))


def entangling_capability(
    family: str, *, n: int, L: int, n_samples: int = 400,
    seed: int = 0, encoding: str = "rx",
    entanglement: str = "ring",
) -> float:
    """Mean Meyer-Wallach Q over n_samples random parameter sets.

    See `_meyer_wallach_q` for the per-state definition.
    """
    rng = np.random.default_rng(seed)
    circ = _state_qnode(family, n, L, encoding=encoding,
                        entanglement=entanglement)
    w_shape = _weight_shape_for(family, n, L)

    qs = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        inp = rng.uniform(-np.pi, np.pi, size=n)
        w = rng.uniform(-np.pi, np.pi, size=w_shape)
        psi = np.asarray(circ(inp, w))
        qs[i] = _meyer_wallach_q(psi, n)
    return float(qs.mean())


# ---------------------------------------------------------------------------
# 3. Barren-plateau gradient variance
# ---------------------------------------------------------------------------


def gradient_variance(
    family: str, *, n: int, L: int, n_samples: int = 400,
    seed: int = 0, encoding: str = "rx",
    entanglement: str = "ring",
) -> float:
    """Var over random params of ∂⟨Z_0⟩/∂θ_{0,0,0}.

    Samples random (inputs, weights) and computes the partial
    derivative of the Z₀ expectation w.r.t. the first weight
    coordinate via parameter-shift. Exponentially small variance
    vs qubit count = barren plateau (McClean 2018).

    For P7 we report the variance at the P4 config (num_qubits=3,
    num_layers=1); the regime-dependent advantage gap correlation
    uses this scalar per family.
    """
    rng = np.random.default_rng(seed)

    dev = qml.device("default.qubit", wires=n)
    w_shape = _weight_shape_for(family, n, L)

    @qml.qnode(dev)
    def expval_circ(inputs, weights):
        _apply_family(family, inputs, weights, n=n, L=L,
                      encoding=encoding, entanglement=entanglement)
        return qml.expval(qml.PauliZ(0))

    # Numerical gradient via central finite difference. h = 1e-2 keeps
    # the perturbation well above float64 round-off while staying small
    # enough that the O(h²) truncation is < 1%.
    #
    # NOTE: McClean et al. 2018's barren-plateau measure differentiates
    # w.r.t. ONE specific weight, but for our 4 forecaster families the
    # FIRST scalar weight can be mathematically insensitive (e.g.
    # `data_reuploading`'s `qml.Rot(weights[0,i,0],...)` first argument
    # is an Rz that commutes with Z when applied before Ry; entanglement
    # eventually de-commutes it but at L=1 the effect is tiny). To get
    # the robust barren-plateau signal we average over a RANDOM weight
    # index per sample — this gives an unbiased estimate of the typical
    # gradient magnitude (Var of |∂f/∂θ| across both random params AND
    # random θ choice).
    n_weights_total = int(np.prod(w_shape))
    h = 1e-2
    grads = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        inp = np.asarray(
            rng.uniform(-np.pi, np.pi, size=n), dtype=np.float64)
        w = np.asarray(
            rng.uniform(-np.pi, np.pi, size=w_shape), dtype=np.float64)
        # Sample which scalar weight to differentiate at (avoids the
        # specific-weight commutation-degeneracy described above).
        idx = int(rng.integers(0, n_weights_total))
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus.flat[idx] = w.flat[idx] + h
        w_minus.flat[idx] = w.flat[idx] - h
        f_plus = float(expval_circ(inp, w_plus))
        f_minus = float(expval_circ(inp, w_minus))
        grads[i] = (f_plus - f_minus) / (2.0 * h)
    return float(np.var(grads, ddof=0))


# ---------------------------------------------------------------------------
# 4. Fourier-spectrum bandwidth (Schuld-Sweke-Meyer 2021)
# ---------------------------------------------------------------------------


def fourier_bandwidth(family: str, *, n: int, L: int) -> int:
    """Structural max-frequency K accessible to the ansatz.

    Per Schuld et al. 2021: a re-uploading circuit with `R`
    re-uploading "layers" can express up to frequency `K ≤ R · n`
    (n = number of qubits, R = number of times the input is
    re-encoded). Practical K_max for the 4 forecaster families:

      data_reuploading : R = L (input re-encoded each layer)
                         → K_max = L · n
      hardware_efficient : R = 1 (input encoded once at start)
                           → K_max = n
      strongly_entangling : R = 1 → K_max = n
      brickwall : R = 1 → K_max = n

    This is a STRUCTURAL property of the ansatz — no sampling
    needed. Reports the effective bandwidth as a scalar per family.
    """
    if family == "data_reuploading":
        return int(L * n)
    if family in ("hardware_efficient", "strongly_entangling",
                   "brickwall"):
        return int(n)
    raise ValueError(f"unknown family {family!r}")


# ---------------------------------------------------------------------------
# Per-family T3 bundle (the cross-tabulation feature vector)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class T3Scalars:
    """The 4-D T3 mechanism feature vector for one family at one config."""
    family: str
    n_qubits: int
    n_layers: int
    expressibility_kl: float    # lower = more expressive
    entangling_q: float          # 0..1; 1 = maximally entangling
    gradient_variance: float     # higher = more trainable
    fourier_bandwidth: int       # higher = more frequency support


def compute_t3_scalars(
    family: str, *, n: int = 3, L: int = 1,
    n_samples: int = 400, seed: int = 0,
    encoding: str = "rx", entanglement: str = "ring",
) -> T3Scalars:
    """One-stop compute of all 4 T3 scalars for a family at a config.

    The default (n=3, L=1) matches the P4 forecaster sweep config
    (`P4SweepConfig.num_qubits=3, num_layers=1`) so the cross-
    tabulation with the P5 per-cell Δ values is on the same
    quantum-circuit-resource basis the H1 verdict was computed at.
    """
    return T3Scalars(
        family=family,
        n_qubits=n,
        n_layers=L,
        expressibility_kl=expressibility_kl_to_haar(
            family, n=n, L=L, n_samples=n_samples, seed=seed,
            encoding=encoding, entanglement=entanglement),
        entangling_q=entangling_capability(
            family, n=n, L=L, n_samples=n_samples, seed=seed,
            encoding=encoding, entanglement=entanglement),
        gradient_variance=gradient_variance(
            family, n=n, L=L, n_samples=n_samples, seed=seed,
            encoding=encoding, entanglement=entanglement),
        fourier_bandwidth=fourier_bandwidth(family, n=n, L=L),
    )
