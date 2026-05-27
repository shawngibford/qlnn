"""T3 — quantum-trainability / expressivity diagnostics.

Answers the standing concern "are the QLNN circuits expressive (and
trainable) enough?" with numbers instead of intuition, across the 28
distinct topologies + deliberately larger circuits. Four analyses:

  1. Expressibility (Sim et al. 2019, arXiv:1905.10876):
     KL( P_circuit(F) || P_Haar(F) ), F = |<psi(θ1)|psi(θ2)>|^2,
     P_Haar(F) = (N-1)(1-F)^(N-2), N=2^n. LOWER KL = MORE expressive
     (closer to Haar-random state coverage).
  2. Entangling capability: Meyer-Wallach Q (Brennen 2003 form),
     Q = 2(1 - mean_k Tr[ρ_k^2]), averaged over random parameters.
  3. Barren-plateau scaling: Var over random params of one gradient
     component ∂⟨Z_0⟩/∂θ, vs num_qubits × depth, per family. An
     exponential decay in qubit count = barren plateau (untrainable
     if scaled up).
  4. Fisher eigenspectrum: sorted eigenvalues of the empirical Fisher
     for the TRAINED classical-H4 vs QLNN-h3 checkpoints (reuses the
     Claim-2 machinery) — connects expressivity to the d_norm story.

GATED: sampling is moderate compute (no training, but many statevector
sims). Do NOT run while the Option-B sweep owns the machine. The
state-returning circuits below mirror the registry modules
(src/qlnn_/circuits/*.py) gate-for-gate; weight shapes are pulled from
the registry so the two stay consistent.

Outputs → results/quantum_trainability/{expressibility,entangling,
barren_plateau,fisher_spectrum}.json + summary.md

Usage:
    python scripts/analyze_quantum_trainability.py            # all
    python scripts/analyze_quantum_trainability.py --only expressibility
    python scripts/analyze_quantum_trainability.py --samples 200   # quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pennylane as qml

from qlnn_.circuits import AnsatzConfig, build

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "quantum_trainability"

FAMILIES = ["data_reuploading", "hardware_efficient",
            "strongly_entangling", "brickwall"]


# ---------------------------------------------------------------------------
# State-returning circuits — mirror src/qlnn_/circuits/*.py gate-for-gate.
# ---------------------------------------------------------------------------
def _entangle(n: int, pattern: str) -> None:
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


def _apply(family: str, inputs, weights, q: int, l: int,
           enc: str, ent: str) -> None:
    eg = qml.RX if enc == "rx" else qml.RY
    if family == "data_reuploading":
        for layer in range(l):
            for i in range(q):
                qml.RX(inputs[i], wires=i)
            for i in range(q):
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1],
                        weights[layer, i, 2], wires=i)
            _entangle(q, ent if ent != "template" else "ring")
    elif family == "hardware_efficient":
        for i in range(q):
            eg(inputs[i], wires=i)
        for layer in range(l):
            for i in range(q):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            _entangle(q, ent if ent != "template" else "ring")
    elif family == "strongly_entangling":
        for i in range(q):
            eg(inputs[i], wires=i)
        qml.StronglyEntanglingLayers(weights=weights, wires=range(q))
    elif family == "brickwall":
        for i in range(q):
            eg(inputs[i], wires=i)
        for layer in range(l):
            for i in range(q):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            start = 0 if layer % 2 == 0 else 1
            for i in range(start, q - 1, 2):
                qml.CNOT(wires=[i, i + 1])
    else:
        raise ValueError(family)


def _state_qnode(family: str, q: int, l: int, enc: str, ent: str):
    dev = qml.device("default.qubit", wires=q)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        _apply(family, inputs, weights, q, l, enc, ent)
        return qml.state()
    return circuit


def _weight_shape(family: str, q: int, l: int, enc: str, ent: str):
    params = {"encoding": enc}
    if ent != "template":
        params["entanglement"] = ent
    if family == "brickwall":
        params["reupload"] = False
    return build(AnsatzConfig(name=family, num_qubits=q, num_layers=l,
                              params=params)).weight_shape


# ---------------------------------------------------------------------------
# 1. Expressibility (KL to Haar)
# ---------------------------------------------------------------------------
def _haar_kl(fids: np.ndarray, n_qubits: int, bins: int = 75) -> float:
    N = 2 ** n_qubits
    edges = np.linspace(0.0, 1.0, bins + 1)
    p, _ = np.histogram(fids, bins=edges, density=True)
    p = p * np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    haar = (N - 1) * (1.0 - centers) ** (N - 2)
    haar = haar / haar.sum()
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(haar, eps, None)
    return float(np.sum(p * np.log(p / q)))


def expressibility(topos, samples, seed):
    rng = np.random.default_rng(seed)
    out = []
    for t in topos:
        fam, q, l, enc, ent = t
        qn = _state_qnode(fam, q, l, enc, ent)
        ws = _weight_shape(fam, q, l, enc, ent)
        fids = []
        for _ in range(samples):
            x = rng.uniform(-np.pi, np.pi, size=q)
            w1 = rng.uniform(-np.pi, np.pi, size=ws)
            w2 = rng.uniform(-np.pi, np.pi, size=ws)
            s1 = np.asarray(qn(x, w1)); s2 = np.asarray(qn(x, w2))
            fids.append(abs(np.vdot(s1, s2)) ** 2)
        kl = _haar_kl(np.asarray(fids), q)
        out.append({"family": fam, "num_qubits": q, "num_layers": l,
                    "encoding": enc, "entanglement": ent,
                    "expressibility_kl_to_haar": kl,
                    "interpretation": "lower = more expressive"})
        print(f"  expressibility {fam} {q}q/{l}L {enc}/{ent}: KL={kl:.4f}")
    return out


# ---------------------------------------------------------------------------
# 2. Entangling capability (Meyer-Wallach Q)
# ---------------------------------------------------------------------------
def _meyer_wallach(state: np.ndarray, q: int) -> float:
    psi = state.reshape([2] * q)
    purities = []
    for k in range(q):
        axes = [a for a in range(q) if a != k]
        # reduced density matrix on qubit k
        rho_k = np.tensordot(psi, psi.conj(), axes=(axes, axes))
        purities.append(np.real(np.trace(rho_k @ rho_k)))
    return float(2.0 * (1.0 - np.mean(purities)))


def entangling(topos, samples, seed):
    rng = np.random.default_rng(seed + 1)
    out = []
    for t in topos:
        fam, q, l, enc, ent = t
        qn = _state_qnode(fam, q, l, enc, ent)
        ws = _weight_shape(fam, q, l, enc, ent)
        qs = []
        for _ in range(samples):
            x = rng.uniform(-np.pi, np.pi, size=q)
            w = rng.uniform(-np.pi, np.pi, size=ws)
            qs.append(_meyer_wallach(np.asarray(qn(x, w)), q))
        out.append({"family": fam, "num_qubits": q, "num_layers": l,
                    "encoding": enc, "entanglement": ent,
                    "meyer_wallach_Q_mean": float(np.mean(qs)),
                    "meyer_wallach_Q_std": float(np.std(qs))})
        print(f"  MW-Q {fam} {q}q/{l}L {enc}/{ent}: "
              f"{np.mean(qs):.4f}±{np.std(qs):.4f}")
    return out


# ---------------------------------------------------------------------------
# 3. Barren-plateau gradient-variance scaling (reuses the registry)
# ---------------------------------------------------------------------------
def barren_plateau(samples, seed, qubits, depths):
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(seed + 2)
    out = []
    for fam in FAMILIES:
        for q in qubits:
            for l in depths:
                params = {"encoding": "rx"}
                if fam in ("data_reuploading", "hardware_efficient"):
                    params["entanglement"] = "ring"
                if fam == "brickwall":
                    params["reupload"] = False
                circ = build(AnsatzConfig(name=fam, num_qubits=q,
                                          num_layers=l, params=params))
                ws = circ.weight_shape

                def cost(w, x):
                    return circ(x, w)[0]   # <Z_0>

                g = jax.jit(jax.grad(lambda w, x: cost(w, x)))
                grads = []
                for _ in range(samples):
                    x = jnp.asarray(rng.uniform(-np.pi, np.pi, size=q))
                    w = jnp.asarray(rng.uniform(-np.pi, np.pi, size=ws))
                    grads.append(float(np.asarray(g(w, x)).flatten()[0]))
                var = float(np.var(grads))
                out.append({"family": fam, "num_qubits": q,
                            "num_layers": l, "grad_var": var})
                print(f"  barren {fam} {q}q/{l}L: Var[∂Z0]={var:.3e}")
    return out


# ---------------------------------------------------------------------------
# 4. Fisher eigenspectrum (reuse trained checkpoints + Claim-2 machinery)
# ---------------------------------------------------------------------------
def fisher_spectrum():
    """Sorted empirical-Fisher eigenvalues for the trained classical-H4
    vs QLNN-h3 checkpoints. Reuses results/effective_dimension if its
    Fisher matrices were cached; else reports the d_norm summary only.
    """
    ed = REPO_ROOT / "results" / "effective_dimension" / "effective_dimension.json"
    if not ed.exists():
        print("  fisher_spectrum: effective_dimension.json absent — skip")
        return None
    d = json.loads(ed.read_text())
    # The committed artifact stores per-seed d_norm + D; the raw Fisher
    # eigenspectrum is recomputed by run_effective_dimension.py with
    # --emit-spectrum (added separately). Here we surface the d_norm
    # distribution as the spectrum proxy + flag for the full recompute.
    return {
        "classical_H4": d.get("classical_H4", {}).get("aggregate"),
        "qlnn_h3": d.get("qlnn_h3", {}).get("aggregate"),
        "note": ("full eigenspectrum requires "
                 "run_effective_dimension.py --emit-spectrum (gated); "
                 "d_norm aggregates surfaced here as the proxy"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["expressibility", "entangling",
                    "barren_plateau", "fisher_spectrum"], default=None)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    topo_path = (REPO_ROOT / "results" / "circuit_search_space"
                 / "topologies.json")
    if not topo_path.exists():
        raise SystemExit("run scripts/build_circuit_search_space.py first")
    topos = [(t["family"], t["num_qubits"], t["num_layers"],
              t["encoding"], t["entanglement"])
             for t in json.loads(topo_path.read_text())["topologies"]]

    OUT.mkdir(parents=True, exist_ok=True)
    run = args.only

    if run in (None, "expressibility"):
        print("=== expressibility (KL to Haar) ===")
        (OUT / "expressibility.json").write_text(json.dumps(
            expressibility(topos, args.samples, args.seed), indent=2) + "\n")
    if run in (None, "entangling"):
        print("=== entangling capability (Meyer-Wallach Q) ===")
        (OUT / "entangling.json").write_text(json.dumps(
            entangling(topos, args.samples, args.seed), indent=2) + "\n")
    if run in (None, "barren_plateau"):
        print("=== barren-plateau gradient-variance scaling ===")
        (OUT / "barren_plateau.json").write_text(json.dumps(
            barren_plateau(args.samples, args.seed,
                           qubits=[2, 4, 6, 8], depths=[1, 3, 5]),
            indent=2) + "\n")
    if run in (None, "fisher_spectrum"):
        print("=== Fisher spectrum ===")
        fs = fisher_spectrum()
        if fs is not None:
            (OUT / "fisher_spectrum.json").write_text(
                json.dumps(fs, indent=2) + "\n")

    print(f"\nwrote → {OUT}/")


if __name__ == "__main__":
    main()
