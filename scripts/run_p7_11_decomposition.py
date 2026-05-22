"""P7.11 commit 3 — Forecaster H1 with complete 2×2 mechanism decomposition.

Extends P7.10's three-verdict decomposition with the fourth corner:
the non-liquid quantum forecaster. With all four corners of the
fairness 2-by-2 filled, FOUR pairwise contrasts are computed:

|                  | non-liquid (no τ)         | liquid (learnable τ)       |
|------------------|---------------------------|----------------------------|
| **Classical**    | plain_neuralode           | classical_ltc              |
| **Quantum**      | non_liquid_<ansatz>       | <ansatz> (existing P4)     |

Per-cell `delta` values (best-QLNN per cell across the 4 liquid +
4 non-liquid + rf_qrc = 9 quantum families):

  Δ_combined           = liquid_qlnn − plain_neuralode
                         (the pre-reg-mandated H1 contrast)
  Δ_quantum_via_LTC    = liquid_qlnn − classical_ltc
                         (isolates the quantum-circuit contribution
                          via the classical-side τ control; from P7.10)
  Δ_liquid_via_classical = classical_ltc − plain_neuralode
                         (isolates the liquid-τ contribution via the
                          classical-side; from P7.10)
  Δ_liquid_via_quantum = liquid_qlnn − non_liquid_qlnn
                         (NEW: isolates the liquid-τ contribution on
                          the QUANTUM side — same encoder + A + integration,
                          just adds the 1/τ leak)
  Δ_quantum_via_nonliquid = non_liquid_qlnn − plain_neuralode
                         (NEW: isolates the quantum-circuit contribution
                          via the non-liquid pair on the bottom row of the 2×2)

The per-cell algebraic identities should hold exactly:
  Δ_combined = Δ_quantum_via_LTC + Δ_liquid_via_classical    (P7.10 identity)
  Δ_combined = Δ_liquid_via_quantum + Δ_quantum_via_nonliquid (P7.11 identity)

The CROSS-CHECK across the two τ-isolation paths:
  Δ_liquid_via_classical ?= Δ_liquid_via_quantum
If they agree (same sign + similar magnitude), the τ-attribution is
robust to which side of the 2×2 we isolate it from. If they
disagree, that's itself a paper-worthy finding (the liquid-τ
contribution interacts differently with the quantum vs classical
hidden state).

Inputs (READ-ONLY):
  results/p4_forecaster_rollout/{system}_{family}/seed_N/metrics.json
    — 5 quantum + 4 non-liquid quantum forecaster families
       × 3 ODE × 3 seeds = 81 cells (45 liquid + 36 non-liquid)
  results/p5_matched_baselines/{system}_{baseline}/seed_N/metrics.json
    — 4 classical baselines × 3 ODE × 3 seeds = 36 classical cells

Outputs:
  results/p7_11_decomposition/
    h1_combined.json
    h1_quantum_via_ltc.json
    h1_liquid_via_classical.json
    h1_liquid_via_quantum.json       (NEW)
    h1_quantum_via_nonliquid.json     (NEW)
    per_cell_records.json
    README.md
    provenance.json
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path

from quantum_liquid_neuralode.evaluation.h1_verdict import (
    CellRecord, h1_verdict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p7_11_decomposition"
P4_PATH = REPO_ROOT / "results" / "p4_forecaster_rollout"
P5_PATH = REPO_ROOT / "results" / "p5_matched_baselines"

SYSTEMS = ("lotka_volterra", "van_der_pol", "lorenz")
REGIME = {
    "lotka_volterra": "smooth_periodic",
    "van_der_pol":    "smooth_periodic",
    "lorenz":         "broadband_multiscale",
}
# Liquid quantum forecaster families (includes rf_qrc per P7.10 fix).
QLNN_FAMILIES = (
    "data_reuploading", "hardware_efficient",
    "strongly_entangling", "brickwall",
    "rf_qrc",
)
# Non-liquid quantum forecaster families (P7.11; rf_qrc excluded
# because it is already non-liquid by construction).
NON_LIQUID_QLNN_FAMILIES = (
    "non_liquid_data_reuploading",
    "non_liquid_hardware_efficient",
    "non_liquid_strongly_entangling",
    "non_liquid_brickwall",
)


def _best_relL2(system: str, seed: int,
                families: tuple) -> tuple[float | None, str | None]:
    best, best_fam = None, None
    for fam in families:
        p = P4_PATH / f"{system}_{fam}" / f"seed_{seed}" / "metrics.json"
        if not p.exists():
            continue
        v = float(json.loads(p.read_text())["relative_l2"])
        if best is None or v < best:
            best, best_fam = v, fam
    return best, best_fam


def _baseline_relL2(system: str, baseline: str, seed: int) -> float | None:
    p = P5_PATH / f"{system}_{baseline}" / f"seed_{seed}" / "metrics.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["relative_l2"])


def _git_prov() -> dict:
    try:
        c = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        b = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
        d = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip())
    except Exception:
        c, b, d = "unknown", "unknown", True
    return {"git_commit": c, "git_branch": b, "git_dirty": d}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P7.11 forecaster 2×2 decomposition — start {start}", flush=True)

    per_cell = []
    print("\nPer-cell data (LQ=liquid quantum best, NLQ=non-liquid quantum best):",
          flush=True)
    for system in SYSTEMS:
        for seed in (0, 1, 2):
            lq_v, lq_fam = _best_relL2(system, seed, QLNN_FAMILIES)
            nlq_v, nlq_fam = _best_relL2(system, seed, NON_LIQUID_QLNN_FAMILIES)
            ne_v = _baseline_relL2(system, "plain_neuralode", seed)
            ltc_v = _baseline_relL2(system, "classical_ltc", seed)
            sk_v = _baseline_relL2(system, "skyline", seed)
            if any(v is None for v in (lq_v, nlq_v, ne_v, ltc_v)):
                print(f"  [{system:<15} seed={seed}] MISSING — SKIP",
                      flush=True)
                continue
            # Per-cell deltas (algebraic identities documented above).
            d_combined = ne_v - lq_v
            d_quantum_via_ltc = ltc_v - lq_v
            d_liquid_via_classical = ne_v - ltc_v
            d_liquid_via_quantum = nlq_v - lq_v
            d_quantum_via_nonliquid = ne_v - nlq_v
            per_cell.append({
                "system": system, "seed": seed,
                "regime": REGIME[system],
                "liquid_qlnn_best_family": lq_fam,
                "liquid_qlnn_relL2": lq_v,
                "non_liquid_qlnn_best_family": nlq_fam,
                "non_liquid_qlnn_relL2": nlq_v,
                "plain_neuralode_relL2": ne_v,
                "classical_ltc_relL2": ltc_v,
                "skyline_relL2": sk_v,
                "delta_combined": d_combined,
                "delta_quantum_via_ltc": d_quantum_via_ltc,
                "delta_liquid_via_classical": d_liquid_via_classical,
                "delta_liquid_via_quantum": d_liquid_via_quantum,
                "delta_quantum_via_nonliquid": d_quantum_via_nonliquid,
            })
            print(f"  [{system:<15} s{seed}] lq={lq_fam:<24}({lq_v:.3f}) "
                  f"nlq={nlq_fam:<32}({nlq_v:.3f}) "
                  f"ne={ne_v:.3f} ltc={ltc_v:.3f}",
                  flush=True)
            print(f"        Δ_combined={d_combined:+.3f}  "
                  f"Δ_q_via_ltc={d_quantum_via_ltc:+.3f}  "
                  f"Δ_τ_via_cls={d_liquid_via_classical:+.3f}",
                  flush=True)
            print(f"        Δ_τ_via_q={d_liquid_via_quantum:+.3f}  "
                  f"Δ_q_via_nlq={d_quantum_via_nonliquid:+.3f}",
                  flush=True)

    if len(per_cell) < 3:
        print(f"\n  TOO FEW CELLS ({len(per_cell)}) — verdict skipped",
              flush=True)
        return

    print(f"\nRunning FIVE paired-bootstrap H1 verdicts on n={len(per_cell)} "
          f"cells ...", flush=True)

    def _verdict(records, name):
        v = h1_verdict(records, n_iter=10000,
                       skyline_threshold=10.0, seed=0)
        out_path = OUT / f"h1_{name}.json"
        out_path.write_text(json.dumps(v, indent=2) + "\n")
        b = v.get("bootstrap")
        if b is not None:
            print(f"  {name:<28}: outcome={v['outcome']:<12} "
                  f"Δ={b['delta_diff_mean']:+.4f} "
                  f"CI=[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]",
                  flush=True)
        else:
            print(f"  {name:<28}: outcome={v['outcome']} (no bootstrap)",
                  flush=True)
        return v

    # Build five sets of CellRecords with appropriate (qlnn, baseline)
    # slot assignments. h1_verdict treats `qlnn_relL2 - neuralode_relL2`
    # per cell as the Δ; we re-label each verdict's "QLNN"-slot and
    # "Neural-ODE"-slot accordingly.
    def _records(qlnn_key: str, neuralode_key: str) -> list[CellRecord]:
        return [
            CellRecord(system=r["system"], seed=r["seed"],
                       qlnn_relL2=r[qlnn_key],
                       neuralode_relL2=r[neuralode_key],
                       skyline_relL2=r["skyline_relL2"])
            for r in per_cell
        ]

    v_comb = _verdict(_records("liquid_qlnn_relL2", "plain_neuralode_relL2"),
                      "combined")
    v_q_ltc = _verdict(_records("liquid_qlnn_relL2", "classical_ltc_relL2"),
                       "quantum_via_ltc")
    v_l_cls = _verdict(_records("classical_ltc_relL2", "plain_neuralode_relL2"),
                       "liquid_via_classical")
    v_l_q = _verdict(_records("liquid_qlnn_relL2", "non_liquid_qlnn_relL2"),
                     "liquid_via_quantum")
    v_q_nlq = _verdict(_records("non_liquid_qlnn_relL2", "plain_neuralode_relL2"),
                       "quantum_via_nonliquid")

    (OUT / "per_cell_records.json").write_text(
        json.dumps(per_cell, indent=2) + "\n")

    # Algebraic identity checks.
    print(f"\nAlgebraic identity checks (per-cell, should be ~0 exact):",
          flush=True)
    if all(v.get("bootstrap") for v in (v_comb, v_q_ltc, v_l_cls)):
        a = v_comb["bootstrap"]["delta_diff_mean"]
        b1 = v_q_ltc["bootstrap"]["delta_diff_mean"]
        b2 = v_l_cls["bootstrap"]["delta_diff_mean"]
        print(f"  Δ_combined = Δ_q_via_ltc + Δ_τ_via_cls?",
              flush=True)
        print(f"    {a:+.4f} ≈ {b1:+.4f} + {b2:+.4f} = {b1 + b2:+.4f}  "
              f"(|diff|={abs(a - (b1 + b2)):.4f})", flush=True)
    if all(v.get("bootstrap") for v in (v_comb, v_l_q, v_q_nlq)):
        a = v_comb["bootstrap"]["delta_diff_mean"]
        b1 = v_l_q["bootstrap"]["delta_diff_mean"]
        b2 = v_q_nlq["bootstrap"]["delta_diff_mean"]
        print(f"  Δ_combined = Δ_τ_via_q + Δ_q_via_nlq?",
              flush=True)
        print(f"    {a:+.4f} ≈ {b1:+.4f} + {b2:+.4f} = {b1 + b2:+.4f}  "
              f"(|diff|={abs(a - (b1 + b2)):.4f})", flush=True)

    # τ-cross-check: do the two τ-isolation paths AGREE?
    if all(v.get("bootstrap") for v in (v_l_cls, v_l_q)):
        d_cls = v_l_cls["bootstrap"]["delta_diff_mean"]
        d_q = v_l_q["bootstrap"]["delta_diff_mean"]
        print(f"\nτ-isolation cross-check (do the two paths AGREE?):",
              flush=True)
        print(f"  Δ_τ_via_classical = {d_cls:+.4f}", flush=True)
        print(f"  Δ_τ_via_quantum   = {d_q:+.4f}", flush=True)
        print(f"  Sign agreement: "
              f"{'YES' if d_cls * d_q > 0 else 'NO'}", flush=True)

    prov = {**_git_prov(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "wall_clock_start_utc": start,
            "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z"}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    (OUT / "README.md").write_text(
        "# results/p7_11_decomposition/\n\n"
        "P7.11 commit 3 — forecaster H1 with COMPLETE 2×2 mechanism decomposition.\n\n"
        "Five paired-bootstrap H1 verdicts at n=9:\n"
        "  - h1_combined.json              : QLNN − Neural-ODE  (pre-reg)\n"
        "  - h1_quantum_via_ltc.json       : QLNN − classical_LTC\n"
        "  - h1_liquid_via_classical.json  : classical_LTC − Neural-ODE\n"
        "  - h1_liquid_via_quantum.json    : QLNN − non_liquid_QLNN  (NEW)\n"
        "  - h1_quantum_via_nonliquid.json : non_liquid_QLNN − Neural-ODE (NEW)\n\n"
        "Two algebraic identities (per-cell, exact):\n"
        "  Δ_combined = Δ_quantum_via_ltc + Δ_liquid_via_classical\n"
        "  Δ_combined = Δ_liquid_via_quantum + Δ_quantum_via_nonliquid\n\n"
        "τ-isolation cross-check: do Δ_liquid_via_classical and\n"
        "Δ_liquid_via_quantum agree in sign and magnitude? If yes, the\n"
        "liquid-τ attribution is robust to which side of the 2×2 we\n"
        "isolate it from. If not, that's itself a paper-worthy finding.\n")

    print(f"\nWritten: {OUT}", flush=True)


if __name__ == "__main__":
    main()
