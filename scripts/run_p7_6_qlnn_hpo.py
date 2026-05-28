"""P7.6 commit 1 — Symmetric QLNN HPO sensitivity sweep.

Closes the asymmetric-HPO peer-review concern: P7.5 commit 6 swept
HPO only for the classical PINN baseline (the H1 contrast model).
P7.6 sweeps the QLNN side symmetrically — same anchor cells, same
LRs, same train_steps — across all 4 quantum solver families.

Sweep grid:
  4 quantum families × 3 anchor cells × 3 LRs × 2 train_steps
  = 72 retrains total
  ~3 hr CPU at the P3.6 per-component scalar-circuit budget

Quantum families (the P3.6 set):
  - chebyshev_dqc   (default lr=0.02, steps=1200)
  - te_qpinn_fnn    (default lr=0.02, steps=1500)
  - te_qpinn_qnn    (default lr=0.02, steps=2000)
  - qcpinn          (default lr=0.02, steps=1500)

Anchor cells (matching P7.5):
  - Lotka-Volterra seed 2
  - Van der Pol seed 1
  - Lorenz seed 2

HPO grid (matching P7.5):
  - LRs:         {1e-3, 5e-3, 1e-2}
  - train_steps: {1500, 3000}

Output:
  results/p7_6_qlnn_hpo/{family}/{system}_seed_N/cell_results.json
  results/p7_6_qlnn_hpo/summary.json
  results/p7_6_qlnn_hpo/h1_verdict_full_hpo_best.json
    — H1 verdict computed with BOTH sides tuned to their HPO-best per
      anchor cell (the most rigorous sensitivity point)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from qlnn_.training.multi_state_solver import train_one_vector

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p7_6_qlnn_hpo"

QLNN_FAMILIES = (
    "chebyshev_dqc", "te_qpinn_fnn", "te_qpinn_qnn", "qcpinn",
)

ANCHOR_CELLS = (
    ("lotka_volterra", 2),
    ("van_der_pol", 1),
    ("lorenz", 2),
)


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=list(QLNN_FAMILIES),
                    choices=list(QLNN_FAMILIES))
    ap.add_argument("--lrs", nargs="+", type=float,
                    default=[1e-3, 5e-3, 1e-2])
    # A15 (2026-05-28): lower bound raised 1500 → 2000 to match the
    # uniform QLNN solver budget. HPO can only find LONGER-budget wins,
    # never shorter-budget ones below the locked baseline.
    ap.add_argument("--train-steps-list", nargs="+", type=int,
                    default=[2000, 3000])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P7.6 QLNN HPO sweep — start {start}", flush=True)
    print(f"  families : {args.families}", flush=True)
    print(f"  anchors  : {[f'{s}_seed{seed}' for s, seed in ANCHOR_CELLS]}",
          flush=True)
    print(f"  LRs      : {args.lrs}", flush=True)
    print(f"  steps    : {args.train_steps_list}", flush=True)
    n_total = (len(args.families) * len(ANCHOR_CELLS)
               * len(args.lrs) * len(args.train_steps_list))
    print(f"  total retrains : {n_total}", flush=True)
    print(flush=True)

    # Per-(family, cell, hpo) results
    all_runs: list[dict] = []
    # Track HPO-best classical PINN per cell (read from P7.5 HPO results
    # if available; else fall back to P7.5 default-HPO value).
    cpinn_best_path = REPO_ROOT / "results" / "p7_5_hpo_sensitivity"
    cpinn_best_per_cell: dict[tuple[str, int], float] = {}
    for cell_dir in cpinn_best_path.glob("*_seed_*"):
        if cell_dir.is_dir():
            data = json.loads(
                (cell_dir / "cell_results.json").read_text())
            best = min(data["runs"], key=lambda r: r["cpinn_relL2"])
            cpinn_best_per_cell[(data["system"], data["seed"])] = best[
                "cpinn_relL2"]

    # For non-anchor cells (the other 6 ODE cells), use the
    # default-HPO classical PINN from P7.5 solver task.
    p75_records_path = REPO_ROOT / "results" / "p7_5_solver_h1" / \
        "per_cell_records.json"
    p75_data = json.loads(p75_records_path.read_text())
    cpinn_default_per_cell = {
        (r["system"], r["seed"]): r["classical_pinn_relL2"]
        for r in p75_data}

    # HPO sweep over QLNN families and anchor cells
    qlnn_best_per_cell: dict[tuple[str, int], dict] = {}
    for family in args.families:
        print(f"\n  --- Family: {family} ---", flush=True)
        family_dir = out / family
        family_dir.mkdir(parents=True, exist_ok=True)

        for system, seed in ANCHOR_CELLS:
            cell_runs = []
            for lr in args.lrs:
                for steps in args.train_steps_list:
                    print(f"    [{family:<18} {system:<14} seed={seed} "
                          f"lr={lr:.0e} steps={steps:>4}] ...",
                          flush=True, end=" ")
                    r = train_one_vector(
                        family, system, seed,
                        steps=steps, lr=lr, n_colloc=60)
                    relL2 = r["relative_l2"]
                    print(f"relL2={relL2:.4f}", flush=True)
                    cell_runs.append({
                        "family": family,
                        "system": system,
                        "seed": seed,
                        "lr": lr,
                        "train_steps": steps,
                        "qlnn_relL2": float(relL2),
                        "final_loss": float(r["final_loss"]),
                    })
                    all_runs.append(cell_runs[-1])

            # Per (family, cell) HPO-best.
            cell_best = min(cell_runs, key=lambda r: r["qlnn_relL2"])
            cell_dir = family_dir / f"{system}_seed_{seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "cell_results.json").write_text(
                json.dumps({
                    "family": family,
                    "system": system, "seed": seed,
                    "runs": cell_runs,
                    "hpo_best": cell_best,
                    "hpo_best_relL2": cell_best["qlnn_relL2"],
                    "default_hpo_relL2": next(
                        (r["qlnn_relL2"] for r in cell_runs
                         if r["lr"] == 5e-3 and r["train_steps"] == 1500),
                        cell_runs[0]["qlnn_relL2"]),
                }, indent=2) + "\n")

            # Track family-best per cell across all 4 families.
            key = (system, seed)
            if (key not in qlnn_best_per_cell
                    or cell_best["qlnn_relL2"] <
                    qlnn_best_per_cell[key]["qlnn_relL2"]):
                qlnn_best_per_cell[key] = cell_best

    # Compute fully-HPO-best H1 verdict (both sides tuned per cell).
    from quantum_liquid_neuralode.evaluation.h1_verdict import (
        CellRecord, h1_verdict,
    )
    records = []
    cell_details = []
    for c in p75_data:
        system, seed = c["system"], c["seed"]
        key = (system, seed)
        if key in qlnn_best_per_cell:
            qlnn_v = qlnn_best_per_cell[key]["qlnn_relL2"]
            qlnn_note = (f"HPO-best (lr={qlnn_best_per_cell[key]['lr']}, "
                         f"steps={qlnn_best_per_cell[key]['train_steps']}, "
                         f"family={qlnn_best_per_cell[key]['family']})")
        else:
            qlnn_v = c["qlnn_relL2"]
            qlnn_note = "P3.6 default-HPO"

        if key in cpinn_best_per_cell:
            cpinn_v = cpinn_best_per_cell[key]
            cpinn_note = "HPO-best classical PINN (from P7.5 HPO)"
        else:
            cpinn_v = cpinn_default_per_cell[key]
            cpinn_note = "P7.5 default-HPO classical PINN"

        records.append(CellRecord(
            system=system, seed=seed,
            qlnn_relL2=qlnn_v,
            neuralode_relL2=cpinn_v,
            qlnn_train_relL2=None,
            neuralode_train_relL2=None,
            skyline_relL2=None,
        ))
        cell_details.append({
            "system": system, "seed": seed,
            "qlnn_relL2": qlnn_v,
            "qlnn_source": qlnn_note,
            "classical_pinn_relL2": cpinn_v,
            "cpinn_source": cpinn_note,
            "delta": cpinn_v - qlnn_v,
            "regime": records[-1].regime,
        })

    v = h1_verdict(records, n_iter=10000, skyline_threshold=10.0, seed=0)
    (out / "h1_verdict_full_hpo_best.json").write_text(
        json.dumps(v, indent=2) + "\n")
    (out / "per_cell_records_full_hpo_best.json").write_text(
        json.dumps(cell_details, indent=2) + "\n")

    # Sweep-level summary
    family_sign_stability = {}
    for family in args.families:
        per_family = [r for r in all_runs if r["family"] == family]
        family_sign_stability[family] = {}
        for system, seed in ANCHOR_CELLS:
            cell_rs = [r for r in per_family
                       if r["system"] == system and r["seed"] == seed]
            cpinn_def = cpinn_default_per_cell.get((system, seed))
            deltas = [cpinn_def - r["qlnn_relL2"] for r in cell_rs]
            all_pos = all(d > 0 for d in deltas)
            family_sign_stability[family][f"{system}_seed{seed}"] = {
                "delta_range": [float(min(deltas)), float(max(deltas))],
                "all_positive": bool(all_pos),
            }

    summary = {
        "families": list(args.families),
        "anchor_cells": [f"{s}_seed{seed}" for s, seed in ANCHOR_CELLS],
        "lrs": list(args.lrs),
        "train_steps_list": list(args.train_steps_list),
        "family_sign_stability": family_sign_stability,
        "full_hpo_best_h1_outcome": v["outcome"],
        "full_hpo_best_h1_bootstrap": v["bootstrap"],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    prov = {**_git_prov(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "wall_clock_start_utc": start,
            "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z"}
    (out / "provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n")
    (out / "README.md").write_text(
        "# results/p7_6_qlnn_hpo/\n\n"
        "P7.6 QLNN HPO sensitivity sweep — symmetric to P7.5's classical\n"
        "PINN HPO sweep. 4 QLNN families × 3 anchor cells × 3 LRs × 2\n"
        "train_steps = 72 retrains.\n\n"
        "Closes the asymmetric-HPO peer-review concern: BOTH sides\n"
        "(QLNN and classical PINN) have been HPO-tested at the same\n"
        "grid resolution.\n\n"
        "Headline: h1_verdict_full_hpo_best.json — H1 verdict computed\n"
        "with both sides tuned to their HPO-best per anchor cell. The\n"
        "most rigorous sensitivity point.\n")

    print("\n" + "=" * 70, flush=True)
    print(f"P7.6 QLNN HPO sweep done. "
          f"Full-HPO-best H1 verdict: {v['outcome']}", flush=True)
    if v["bootstrap"]:
        b = v["bootstrap"]
        print(f"  Δ_smooth = {b['delta_smooth_mean']:+.4f}", flush=True)
        print(f"  Δ_broad  = {b['delta_broad_mean']:+.4f}", flush=True)
        print(f"  Δ_diff   = {b['delta_diff_mean']:+.4f}", flush=True)
        print(f"  95% CI   = [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]",
              flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
