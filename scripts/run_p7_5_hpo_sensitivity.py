"""P7.5 commit 5 — HPO sensitivity sweep.

Closes audit concern Y3 (HPO budget unfixed). Re-trains the classical
PINN baseline at varied hyperparameters on 3 anchor cells to confirm
the SOLVER-task H1 CONFIRMED outcome is HPO-invariant.

Sweep:
  3 anchor cells × 3 learning rates × 2 train_steps = 18 retrains

Anchor cells (per the plan):
  - Lotka-Volterra seed 2 (QLNN ties classical on smooth)
  - Van der Pol seed 1 (both fail, Δ=+0.181 solver task)
  - Lorenz seed 2 (QLNN narrowly wins on broad, Δ=+0.018 solver task)

Learning rates: 1e-3, 5e-3, 1e-2
Train steps: 1500 (default), 3000 (2× extended)

For each combination, compute classical_PINN_relL2 and Δ vs the
fixed P3.6 QLNN best-ansatz value. Output:
  results/p7_5_hpo_sensitivity/{system}_seed_N/cell_results.json
  results/p7_5_hpo_sensitivity/summary.json (per-cell sign-stability)
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

from qlnn_.training.p7_5_solver_h1 import (
    load_p36_qlnn_best,
    train_classical_pinn_solver_one_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p7_5_hpo_sensitivity"

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
    ap.add_argument("--lrs", nargs="+", type=float,
                    default=[1e-3, 5e-3, 1e-2])
    # A15 (2026-05-28): lower bound raised 1500 → 2000 to match the
    # uniform QLNN solver budget. Sensitivity sweep now brackets the
    # locked default (2000) on its upper side only — [2000, 3000] —
    # so HPO can only find LONGER-budget wins, never shorter-budget
    # ones below the locked baseline.
    ap.add_argument("--train-steps-list", nargs="+", type=int,
                    default=[2000, 3000])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P7.5 HPO sensitivity sweep — start {start}", flush=True)
    print(f"  anchor cells: {[f'{s}_seed{seed}' for s, seed in ANCHOR_CELLS]}",
          flush=True)
    print(f"  LRs        : {args.lrs}", flush=True)
    print(f"  steps_list : {args.train_steps_list}", flush=True)
    print(flush=True)

    n_combos = len(args.lrs) * len(args.train_steps_list)
    print(f"  Total retrains: {len(ANCHOR_CELLS) * n_combos}", flush=True)
    print(flush=True)

    cell_summaries = []
    for system, seed in ANCHOR_CELLS:
        # Pin QLNN value (no HPO variation on QLNN side — its data is
        # committed in P3.6; only re-tune the classical PINN baseline).
        qlnn_v, qlnn_fam = load_p36_qlnn_best(system, seed)
        print(f"  --- Cell {system}_seed{seed}: "
              f"QLNN_best (fixed at P3.6 config) = {qlnn_v:.4f} "
              f"({qlnn_fam})", flush=True)

        cell_runs = []
        for lr in args.lrs:
            for steps in args.train_steps_list:
                r = train_classical_pinn_solver_one_cell(
                    system, seed, n_colloc=60,
                    steps=steps, lr=lr, target_param_count=60)
                cpinn_v = r["relative_l2"]
                cpinn_train = r["train_relative_l2"]
                delta = cpinn_v - qlnn_v
                cell_runs.append({
                    "system": system, "seed": seed,
                    "qlnn_best_relL2": qlnn_v,
                    "qlnn_best_family": qlnn_fam,
                    "lr": lr, "train_steps": steps,
                    "cpinn_relL2": cpinn_v,
                    "cpinn_train_relL2": cpinn_train,
                    "delta": float(delta),
                })
                print(f"    lr={lr:.0e}  steps={steps:>4}  "
                      f"cpinn={cpinn_v:.4f}  train={cpinn_train:.4f}  "
                      f"Δ={delta:+.4f}", flush=True)

        # Per-cell summary: sign-stability of Δ across HPO.
        deltas = [r["delta"] for r in cell_runs]
        all_positive = all(d > 0 for d in deltas)
        all_negative = all(d < 0 for d in deltas)
        sign_str = ("ALL POSITIVE (QLNN wins at every HPO)"
                    if all_positive
                    else ("ALL NEGATIVE (classical wins everywhere)"
                          if all_negative
                          else "MIXED (sign flips across HPO)"))
        print(f"    → cell summary: Δ range [{min(deltas):+.4f}, "
              f"{max(deltas):+.4f}]  ({sign_str})", flush=True)
        print(flush=True)

        cell_dir = out / f"{system}_seed_{seed}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "cell_results.json").write_text(
            json.dumps({
                "system": system, "seed": seed,
                "qlnn_best_relL2": qlnn_v,
                "qlnn_best_family": qlnn_fam,
                "runs": cell_runs,
                "delta_min": float(min(deltas)),
                "delta_max": float(max(deltas)),
                "delta_mean": float(np.mean(deltas)),
                "sign_stability": (
                    "all_positive" if all_positive
                    else ("all_negative" if all_negative else "mixed")),
            }, indent=2) + "\n")
        cell_summaries.append({
            "system": system, "seed": seed,
            "qlnn_relL2": qlnn_v,
            "delta_min": float(min(deltas)),
            "delta_max": float(max(deltas)),
            "sign_stability": (
                "all_positive" if all_positive
                else ("all_negative" if all_negative else "mixed")),
        })

    # Sweep-level summary
    summary = {
        "anchor_cells": [f"{s}_seed{seed}" for s, seed in ANCHOR_CELLS],
        "lrs": list(args.lrs),
        "train_steps_list": list(args.train_steps_list),
        "per_cell": cell_summaries,
        "overall_sign_stability": (
            "all_positive_across_all_cells"
            if all(c["sign_stability"] == "all_positive"
                   for c in cell_summaries)
            else "mixed_or_negative_in_some_cell"),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    prov = {**_git_prov(), "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "wall_clock_start_utc": start,
            "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z"}
    (out / "provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n")
    (out / "README.md").write_text(
        "# results/p7_5_hpo_sensitivity/\n\n"
        "P7.5 HPO sensitivity sweep. Closes audit concern Y3 (HPO\n"
        "budget unfixed). Classical PINN baseline retrained at 3 LRs ×\n"
        "2 train_steps at 3 anchor cells (LV s2, VdP s1, Lorenz s2),\n"
        "while QLNN side stays at the fixed P3.6 multi_state config.\n\n"
        "Verdict: if `overall_sign_stability == all_positive_across_all_cells`,\n"
        "the SOLVER-task H1 CONFIRMED outcome is HPO-invariant — the\n"
        "QLNN advantage holds at every (LR, steps) combination tested.\n")

    print("=" * 70, flush=True)
    print(f"HPO sensitivity overall: {summary['overall_sign_stability']}",
          flush=True)
    print("=" * 70, flush=True)
    print(f"\nResults: {out}/", flush=True)


if __name__ == "__main__":
    main()
