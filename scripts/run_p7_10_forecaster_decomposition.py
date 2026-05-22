"""P7.10 commit 3 — Forecaster H1 with LTC decomposition.

Aggregates the P4 quantum forecaster best-cell results with the P5
classical baselines (now including the P7.10 classical LTC) and
computes THREE paired-bootstrap H1 verdicts:

  Δ_combined = QLNN − Neural-ODE          (pre-reg-mandated)
  Δ_quantum  = QLNN − classical_LTC       (isolated quantum contribution)
  Δ_liquid   = classical_LTC − Neural-ODE (isolated liquid-τ contribution)

with Δ_combined ≈ Δ_quantum + Δ_liquid as a sanity check on the
decomposition.

Per pre-reg amendment A12, this is the headline FORECASTER-task
verdict for the paper: a clean attribution of the forecaster
underperformance (which is FALSIFIED at the original Δ_combined,
CI [-0.79, -0.05] negative) to the quantum vs liquid-τ components.

Inputs (READ-ONLY):
  results/p4_forecaster_rollout/{system}_{family}/seed_N/metrics.json
    — 5 quantum forecaster families × 3 ODE × 3 seeds = 45 cells
  results/p5_matched_baselines/{system}_{baseline}/seed_N/metrics.json
    — 4 classical baselines × 3 ODE × 3 seeds, post-P7.10:
        plain_neuralode, plain_mlp, skyline, classical_ltc

Outputs:
  results/p7_10_forecaster_decomposition/h1_combined.json
  results/p7_10_forecaster_decomposition/h1_quantum_isolated.json
  results/p7_10_forecaster_decomposition/h1_liquid_isolated.json
  results/p7_10_forecaster_decomposition/per_cell_records.json
  results/p7_10_forecaster_decomposition/README.md
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
OUT = REPO_ROOT / "results" / "p7_10_forecaster_decomposition"
P4_PATH = REPO_ROOT / "results" / "p4_forecaster_rollout"
P5_PATH = REPO_ROOT / "results" / "p5_matched_baselines"

SYSTEMS = ("lotka_volterra", "van_der_pol", "lorenz")
REGIME = {
    "lotka_volterra": "smooth_periodic",
    "van_der_pol":    "smooth_periodic",
    "lorenz":         "broadband_multiscale",
}
QLNN_FAMILIES = (
    "data_reuploading", "hardware_efficient",
    "strongly_entangling", "brickwall",
)


def _best_qlnn_forecaster(system: str, seed: int) -> tuple[float | None, str | None]:
    best, best_fam = None, None
    for fam in QLNN_FAMILIES:
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
    print(f"P7.10 forecaster decomposition — start {start}", flush=True)

    per_cell = []
    print("\nPer-cell data:", flush=True)
    for system in SYSTEMS:
        for seed in (0, 1, 2):
            qlnn_v, qlnn_fam = _best_qlnn_forecaster(system, seed)
            ne_v = _baseline_relL2(system, "plain_neuralode", seed)
            ltc_v = _baseline_relL2(system, "classical_ltc", seed)
            sk_v = _baseline_relL2(system, "skyline", seed)
            if any(v is None for v in (qlnn_v, ne_v, ltc_v)):
                print(f"  [{system:<15} seed={seed}] MISSING "
                      f"(qlnn={qlnn_v}, ne={ne_v}, ltc={ltc_v}) — SKIP",
                      flush=True)
                continue
            d_combined = ne_v - qlnn_v
            d_quantum = ltc_v - qlnn_v
            d_liquid = ne_v - ltc_v
            per_cell.append({
                "system": system, "seed": seed,
                "regime": REGIME[system],
                "qlnn_best_family": qlnn_fam,
                "qlnn_relL2": qlnn_v,
                "neuralode_relL2": ne_v,
                "classical_ltc_relL2": ltc_v,
                "skyline_relL2": sk_v,
                "delta_combined": d_combined,
                "delta_quantum_isolated": d_quantum,
                "delta_liquid_isolated": d_liquid,
            })
            print(f"  [{system:<15} seed={seed}] qlnn={qlnn_fam:<22} "
                  f"({qlnn_v:.4f}) ne={ne_v:.4f} ltc={ltc_v:.4f}  "
                  f"Δ_comb={d_combined:+.4f}  Δ_q={d_quantum:+.4f}  "
                  f"Δ_τ={d_liquid:+.4f}", flush=True)

    if len(per_cell) < 3:
        print(f"\n  TOO FEW CELLS ({len(per_cell)}) — verdict skipped",
              flush=True)
        return

    # Three verdicts.
    print(f"\nRunning three paired-bootstrap H1 verdicts on n={len(per_cell)} "
          f"cells ...", flush=True)

    def _verdict(records, name):
        v = h1_verdict(records, n_iter=10000,
                       skyline_threshold=10.0, seed=0)
        out_path = OUT / f"h1_{name}.json"
        out_path.write_text(json.dumps(v, indent=2) + "\n")
        b = v.get("bootstrap")
        if b is not None:
            print(f"  {name:<22}: outcome={v['outcome']:<12} "
                  f"Δ_diff={b['delta_diff_mean']:+.4f} "
                  f"CI=[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]",
                  flush=True)
        else:
            print(f"  {name:<22}: outcome={v['outcome']} (no bootstrap)",
                  flush=True)
        return v

    # Build CellRecord lists. h1_verdict reads (qlnn_relL2,
    # neuralode_relL2) per record; we re-instantiate for each verdict
    # with the appropriate "QLNN" and "NeuralODE" semantics:
    #  combined:  QLNN vs Neural-ODE
    #  quantum:   QLNN vs classical_LTC (relabel ltc → "neuralode" slot)
    #  liquid:    classical_LTC vs Neural-ODE (relabel ltc → "qlnn" slot)
    combined_recs = [
        CellRecord(system=r["system"], seed=r["seed"],
                   qlnn_relL2=r["qlnn_relL2"],
                   neuralode_relL2=r["neuralode_relL2"],
                   skyline_relL2=r["skyline_relL2"])
        for r in per_cell
    ]
    quantum_recs = [
        CellRecord(system=r["system"], seed=r["seed"],
                   qlnn_relL2=r["qlnn_relL2"],
                   neuralode_relL2=r["classical_ltc_relL2"],
                   skyline_relL2=r["skyline_relL2"])
        for r in per_cell
    ]
    liquid_recs = [
        CellRecord(system=r["system"], seed=r["seed"],
                   qlnn_relL2=r["classical_ltc_relL2"],
                   neuralode_relL2=r["neuralode_relL2"],
                   skyline_relL2=r["skyline_relL2"])
        for r in per_cell
    ]

    v_combined = _verdict(combined_recs, "combined")
    v_quantum = _verdict(quantum_recs, "quantum_isolated")
    v_liquid = _verdict(liquid_recs, "liquid_isolated")

    (OUT / "per_cell_records.json").write_text(
        json.dumps(per_cell, indent=2) + "\n")

    # Decomposition sanity check.
    if all(v.get("bootstrap") for v in (v_combined, v_quantum, v_liquid)):
        d_c = v_combined["bootstrap"]["delta_diff_mean"]
        d_q = v_quantum["bootstrap"]["delta_diff_mean"]
        d_l = v_liquid["bootstrap"]["delta_diff_mean"]
        print(f"\nDecomposition sanity check:", flush=True)
        print(f"  Δ_combined           = {d_c:+.4f}", flush=True)
        print(f"  Δ_quantum + Δ_liquid = {d_q:+.4f} + {d_l:+.4f} = "
              f"{d_q + d_l:+.4f}", flush=True)
        print(f"  difference           = {abs(d_c - (d_q + d_l)):.4f} "
              f"(should be ~0 exactly: per-cell identity)", flush=True)

    prov = {**_git_prov(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "wall_clock_start_utc": start,
            "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z"}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    (OUT / "README.md").write_text(
        "# results/p7_10_forecaster_decomposition/\n\n"
        "P7.10 commit 3 — forecaster H1 with LTC decomposition.\n\n"
        "Three paired-bootstrap H1 verdicts at n=9:\n"
        "  - h1_combined.json          : QLNN − Neural-ODE  (original)\n"
        "  - h1_quantum_isolated.json  : QLNN − classical_LTC\n"
        "  - h1_liquid_isolated.json   : classical_LTC − Neural-ODE\n\n"
        "Per-cell records have all three Δ values plus per-baseline\n"
        "relative-L2 values for the all-to-all forecaster table in\n"
        "paper/sections/04_forecaster_results.tex.\n")

    print(f"\nWritten: {OUT}", flush=True)


if __name__ == "__main__":
    main()
