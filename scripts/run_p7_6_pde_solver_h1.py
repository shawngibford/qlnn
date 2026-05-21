"""P7.6 commit 2 — PDE solver-task H1 verdict (zero new compute).

Combines existing-on-disk P3.7-P3.9 PDE quantum data with P3.8
classical PINN PDE data to compute the SOLVER-task H1 verdict on
the PDE side. Then combines with P7.5 ODE solver-task data to
produce the n=18 combined verdict.

Data sources (all already on disk):
  - QLNN best-ansatz on PDEs:
      results/p3_8_review/{pde}_chebyshev_dqc_2d/seed_N/metrics.json
      results/p3_9_pde_matrix/{pde}_{qcpinn_2d, te_qpinn_fnn_2d,
                                     te_qpinn_qnn_2d}/seed_N/metrics.json
  - Classical PINN PDE baseline:
      results/p3_8_review/{pde}_classical_pinn/seed_N/metrics.json

PDE × seed cells: 3 PDEs × 3 seeds = 9 PDE cells.
Regime tags (per pre-reg §2 / pde_systems regime annotation):
  - heat            : smooth_periodic
  - burgers_smooth  : smooth_periodic
  - allen_cahn      : broadband_multiscale

Combined ODE + PDE: 9 + 9 = **n=18 cells** for the bootstrap
(vs n=9 in P7.5).

Outputs:
  results/p7_6_pde_solver_h1/h1_analysis_pde_solver.json
    — PDE-only verdict
  results/p7_6_pde_solver_h1/h1_analysis_combined_solver.json
    — ODE + PDE combined (n=18) — THE PAPER'S MOST RIGOROUS HEADLINE
  results/p7_6_pde_solver_h1/per_cell_records.json
    — all 18 cells
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

from quantum_liquid_neuralode.evaluation.h1_verdict import (
    CellRecord, h1_verdict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p7_6_pde_solver_h1"
P38_PATH = REPO_ROOT / "results" / "p3_8_review"
P39_PATH = REPO_ROOT / "results" / "p3_9_pde_matrix"
ODE_RECORDS = REPO_ROOT / "results" / "p7_5_solver_h1" / "per_cell_records.json"

PDE_SYSTEMS = ("heat", "burgers_smooth", "allen_cahn")
PDE_REGIME = {
    "heat": "smooth_periodic",
    "burgers_smooth": "smooth_periodic",
    "allen_cahn": "broadband_multiscale",
}

PDE_QLNN_FAMILIES = {
    # family-name → (path-root, family-subpath-name)
    "chebyshev_dqc_2d": (P38_PATH, "chebyshev_dqc_2d"),
    "qcpinn_2d":        (P39_PATH, "qcpinn_2d"),
    "te_qpinn_fnn_2d":  (P39_PATH, "te_qpinn_fnn_2d"),
    "te_qpinn_qnn_2d":  (P39_PATH, "te_qpinn_qnn_2d"),
}


def load_pde_qlnn_best(pde: str, seed: int) -> tuple[float | None, str | None]:
    """Best QLNN PDE-solver relL² across 4 P3.7-3.9 families."""
    best = None
    best_fam = None
    for family, (root, sub) in PDE_QLNN_FAMILIES.items():
        p = root / f"{pde}_{sub}" / f"seed_{seed}" / "metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        v = float(m["relative_l2"])
        if best is None or v < best:
            best = v
            best_fam = family
    return best, best_fam


def load_pde_classical_pinn(pde: str, seed: int) -> float | None:
    p = P38_PATH / f"{pde}_classical_pinn" / f"seed_{seed}" / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return float(m["relative_l2"])


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
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P7.6 PDE solver-task H1 verdict — start {start}", flush=True)
    print(flush=True)

    # ---- Build PDE CellRecords from on-disk data ------------------------
    pde_records: list[CellRecord] = []
    pde_details: list[dict] = []
    print("Building PDE CellRecords (zero new compute) ...", flush=True)
    for pde in PDE_SYSTEMS:
        for seed in (0, 1, 2):
            qlnn_v, qlnn_fam = load_pde_qlnn_best(pde, seed)
            cpinn_v = load_pde_classical_pinn(pde, seed)
            if qlnn_v is None or cpinn_v is None:
                print(f"  [{pde:<14} seed={seed}] MISSING — skipping",
                      flush=True)
                continue
            # Build CellRecord — override regime via system name.
            # We need to use a SYSTEM name that h1_verdict.regime_for_system
            # recognizes. Pre-reg lists heat/burgers_smooth in
            # SMOOTH_PERIODIC_SYSTEMS and allen_cahn in
            # BROADBAND_MULTISCALE_SYSTEMS. Confirmed in
            # quantum_liquid_neuralode.evaluation.h1_verdict.
            rec = CellRecord(
                system=pde, seed=seed,
                qlnn_relL2=qlnn_v,
                neuralode_relL2=cpinn_v,
                qlnn_train_relL2=None,
                neuralode_train_relL2=None,
                skyline_relL2=None,
            )
            pde_records.append(rec)
            pde_details.append({
                "system": pde, "seed": seed,
                "qlnn_best_family": qlnn_fam,
                "qlnn_relL2": qlnn_v,
                "classical_pinn_relL2": cpinn_v,
                "delta": rec.delta,
                "regime": rec.regime,
            })
            print(f"  [{pde:<14} seed={seed}]  "
                  f"QLNN_best={qlnn_fam:<20}({qlnn_v:.4f})  "
                  f"classical_PINN={cpinn_v:.4f}  "
                  f"Δ={rec.delta:+.4f}  ({rec.regime})",
                  flush=True)

    # ---- PDE-only H1 verdict ---------------------------------------------
    print(f"\nPDE-only H1 verdict (n={len(pde_records)} cells):", flush=True)
    v_pde = h1_verdict(pde_records, n_iter=10000,
                       skyline_threshold=10.0, seed=0)
    (out / "h1_analysis_pde_solver.json").write_text(
        json.dumps(v_pde, indent=2) + "\n")
    print(f"  outcome: {v_pde['outcome']}", flush=True)
    if v_pde["bootstrap"]:
        b = v_pde["bootstrap"]
        print(f"  Δ_smooth = {b['delta_smooth_mean']:+.4f}", flush=True)
        print(f"  Δ_broad  = {b['delta_broad_mean']:+.4f}", flush=True)
        print(f"  Δ_diff   = {b['delta_diff_mean']:+.4f}", flush=True)
        print(f"  95% CI   = [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]",
              flush=True)

    # ---- Combined ODE + PDE H1 verdict (n=18) ---------------------------
    print(f"\nCombined ODE + PDE H1 verdict:", flush=True)
    ode_data = json.loads(ODE_RECORDS.read_text())
    combined_records = list(pde_records)
    combined_details = list(pde_details)
    for c in ode_data:
        combined_records.append(CellRecord(
            system=c["system"], seed=c["seed"],
            qlnn_relL2=c["qlnn_relL2"],
            neuralode_relL2=c["classical_pinn_relL2"],
            qlnn_train_relL2=None,
            neuralode_train_relL2=None,
            skyline_relL2=None,
        ))
        combined_details.append({
            "system": c["system"], "seed": c["seed"],
            "qlnn_best_family": c.get("qlnn_best_family"),
            "qlnn_relL2": c["qlnn_relL2"],
            "classical_pinn_relL2": c["classical_pinn_relL2"],
            "delta": c["delta"],
            "regime": c["regime"],
            "task": "ode_solver",
        })
    # Tag PDE records with task type
    for d in pde_details:
        d["task"] = "pde_solver"

    v_combined = h1_verdict(combined_records, n_iter=10000,
                            skyline_threshold=10.0, seed=0)
    (out / "h1_analysis_combined_solver.json").write_text(
        json.dumps(v_combined, indent=2) + "\n")
    print(f"  outcome (n={len(combined_records)} cells): "
          f"{v_combined['outcome']}",
          flush=True)
    if v_combined["bootstrap"]:
        b = v_combined["bootstrap"]
        print(f"  Δ_smooth = {b['delta_smooth_mean']:+.4f}", flush=True)
        print(f"  Δ_broad  = {b['delta_broad_mean']:+.4f}", flush=True)
        print(f"  Δ_diff   = {b['delta_diff_mean']:+.4f}", flush=True)
        print(f"  95% CI   = [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]",
              flush=True)
        print(f"  n_smooth = {b['n_smooth']}  n_broad = {b['n_broad']}",
              flush=True)

    (out / "per_cell_records.json").write_text(
        json.dumps(combined_details, indent=2) + "\n")

    # ---- Provenance + README -------------------------------------------
    prov = {**_git_prov(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "wall_clock_start_utc": start,
            "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z"}
    (out / "provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n")
    (out / "README.md").write_text(
        "# results/p7_6_pde_solver_h1/\n\n"
        "P7.6 PDE solver-task H1 verdict (zero new compute — all data\n"
        "already on disk from P3.7-3.9 + P3.8). Plus the **combined\n"
        "ODE + PDE** H1 verdict at n=18 cells.\n\n"
        "Data sources (all read-only):\n"
        "  - QLNN: results/p3_9_pde_matrix/ + results/p3_8_review/\n"
        "  - classical PINN: results/p3_8_review/\n"
        "  - ODE side: results/p7_5_solver_h1/per_cell_records.json\n\n"
        "Outputs:\n"
        "  - h1_analysis_pde_solver.json     (PDE-only, n=9)\n"
        "  - h1_analysis_combined_solver.json (ODE+PDE, n=18)\n"
        "  - per_cell_records.json           (all 18 cells)\n")
    print("=" * 70, flush=True)
    print("P7.6 PDE solver-task H1 verdict done.", flush=True)


if __name__ == "__main__":
    main()
