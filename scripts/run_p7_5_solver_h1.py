"""P7.5 sweep CLI — solver-task H1 verdict (PRIMARY per pre-reg §7).

Closes audit concern R1 (solver-task gating violation).

Pipeline:
  1. Train classical-PINN-as-solver on 3 ODE systems × 3 seeds
     (the H1 contrast — pre-reg §6 "Classical | classical PINN |
     Solver-task classical control").
  2. Load existing P3.6 QLNN best-ansatz solver relL² per cell
     (already on disk, no recompute).
  3. Build CellRecords for h1_verdict() at BOTH skyline thresholds
     (0.5 conservative, 0.75 sensitivity).
  4. Write the verdict to `results/p7_5_solver_h1/
     h1_analysis_solver_task.json` — the PAPER'S PRIMARY HEADLINE.

Computes ~30 min CPU (9 classical-PINN solver trainings × ~3 min each).

The forecaster-task H1 from P5 becomes corroborating evidence per
pre-reg §7's gating rule.
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
from qlnn_.training.p7_5_solver_h1 import (
    load_p36_qlnn_best,
    train_classical_pinn_solver_one_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p7_5_solver_h1"
VERDICT_DIR = REPO_ROOT / "results" / "p7_5_solver_h1"

P36_SYSTEMS = ("lotka_volterra", "van_der_pol", "lorenz")
# P7.8 expansion: FHN added to VECTOR_ODES. When invoked with
# --systems fitzhugh_nagumo, this script trains the classical-PINN
# solver baseline on FHN and writes to results/p7_5_solver_h1/
# fitzhugh_nagumo_classical_pinn/seed_N/, matching the existing
# (LV/VdP/Lorenz) layout. The H1 verdict aggregation reads them
# uniformly.
P7_8_EXTRA_SYSTEMS = ("fitzhugh_nagumo",)
ALL_SYSTEMS = P36_SYSTEMS + P7_8_EXTRA_SYSTEMS


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


def _pkg() -> dict:
    import jax, optax, pennylane, equinox, diffrax, numpy
    return {"jax": jax.__version__, "optax": optax.__version__,
            "pennylane": pennylane.__version__,
            "equinox": equinox.__version__, "diffrax": diffrax.__version__,
            "numpy": numpy.__version__}


def _write_seed(r: dict, base: Path) -> None:
    d = base / f"{r['system']}_classical_pinn" / f"seed_{r['seed']}"
    d.mkdir(parents=True, exist_ok=True)
    scalars = {k: r[k] for k in (
        "family", "system", "seed", "dim", "regime",
        "steps", "lr", "n_colloc", "trainable_params",
        "config_str", "final_loss", "mae", "mae_per_component",
        "relative_l2", "train_relative_l2")}
    scalars["loss_history"] = r["loss_history"]
    (d / "metrics.json").write_text(json.dumps(scalars, indent=2) + "\n")
    np.savez_compressed(
        d / "field.npz",
        t_eval=r["t_eval"], u_pred=r["u_pred"], u_ref=r["u_ref"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="+", default=list(P36_SYSTEMS),
                    choices=list(ALL_SYSTEMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    # A15 (2026-05-28): default raised 1500 → 2000 to match the uniform
    # QLNN solver budget (solver_demo._UNIFORM_SOLVER_STEPS). Cross-side
    # parity: same step count on quantum and classical PINN sides.
    ap.add_argument("--steps", type=int, default=2000,
                    help="Optax training steps for classical PINN solver")
    ap.add_argument("--n-colloc", type=int, default=60,
                    help="Interior collocation count")
    ap.add_argument("--target-params", type=int, default=60,
                    help="Capacity-matched MLP target param count")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P7.5 solver-task H1 sweep — start {start}", flush=True)
    print(f"  systems   : {args.systems}", flush=True)
    print(f"  seeds     : {args.seeds}", flush=True)
    print(f"  steps     : {args.steps}", flush=True)
    print(f"  n_colloc  : {args.n_colloc}", flush=True)
    print(f"  target_p  : {args.target_params}", flush=True)
    print(flush=True)

    # ---- Train classical PINN solver per (system, seed) -----------------
    classical_records: dict[tuple[str, int], dict] = {}
    for system in args.systems:
        for seed in args.seeds:
            r = train_classical_pinn_solver_one_cell(
                system, seed,
                n_colloc=args.n_colloc, steps=args.steps,
                target_param_count=args.target_params)
            _write_seed(r, out)
            classical_records[(system, seed)] = r
            print(
                f"  [{system:<14} classical_pinn   ] seed={seed}  "
                f"relL2={r['relative_l2']:.4f}  "
                f"train_relL2={r['train_relative_l2']:.4f}  "
                f"params={r['trainable_params']}",
                flush=True)

    # ---- Build CellRecords (combine with P3.6 QLNN best) ----------------
    print(f"\nBuilding CellRecords for H1 verdict ...", flush=True)
    records: list[CellRecord] = []
    cell_details: list[dict] = []
    for system in args.systems:
        for seed in args.seeds:
            qlnn_v, qlnn_fam = load_p36_qlnn_best(system, seed)
            cp = classical_records.get((system, seed))
            if qlnn_v is None or cp is None:
                print(f"  [{system} seed={seed}] MISSING data — skipping",
                      flush=True)
                continue
            rec = CellRecord(
                system=system, seed=seed,
                qlnn_relL2=qlnn_v,
                neuralode_relL2=cp["relative_l2"],
                qlnn_train_relL2=None,    # P3.6 didn't record train relL2;
                                          # P7.5 commit 3 backfills this.
                neuralode_train_relL2=cp["train_relative_l2"],
                skyline_relL2=None,       # Skyline lives in P5 sweep
                                          # (forecaster); add separately below.
            )
            records.append(rec)
            cell_details.append({
                "system": system, "seed": seed,
                "qlnn_best_family": qlnn_fam,
                "qlnn_relL2": qlnn_v,
                "classical_pinn_relL2": cp["relative_l2"],
                "classical_pinn_train_relL2": cp["train_relative_l2"],
                "delta": rec.delta,
                "regime": rec.regime,
            })
            print(f"  [{system:<14} seed={seed}]  "
                  f"QLNN_best={qlnn_fam:<18}({qlnn_v:.3f})  "
                  f"classical_PINN={cp['relative_l2']:.3f}  "
                  f"Δ={rec.delta:+.4f}  ({rec.regime})",
                  flush=True)

    # Augment with skyline values from P5 (if present) for the guard.
    p5_records = REPO_ROOT / "results" / "p5_h1_verdict" / "per_cell_records.json"
    if p5_records.exists():
        p5_data = json.loads(p5_records.read_text())
        skyline_by_cell = {(r["system"], r["seed"]): r.get("skyline_relL2")
                           for r in p5_data}
        records = [
            CellRecord(
                system=r.system, seed=r.seed,
                qlnn_relL2=r.qlnn_relL2,
                neuralode_relL2=r.neuralode_relL2,
                qlnn_train_relL2=r.qlnn_train_relL2,
                neuralode_train_relL2=r.neuralode_train_relL2,
                skyline_relL2=skyline_by_cell.get((r.system, r.seed)),
            )
            for r in records
        ]
        print(f"  augmented with skyline values from P5 sweep",
              flush=True)

    # ---- H1 verdict at both pre-reg-amendment thresholds ----------------
    print(f"\nComputing H1 verdict (solver task) ...", flush=True)
    v_strict = h1_verdict(
        records, n_iter=10000, skyline_threshold=0.5, seed=0)
    v_sensitivity = h1_verdict(
        records, n_iter=10000, skyline_threshold=0.75, seed=0)

    (out / "h1_analysis_solver_task.json").write_text(
        json.dumps(v_strict, indent=2) + "\n")
    (out / "h1_analysis_solver_task_sensitivity.json").write_text(
        json.dumps(v_sensitivity, indent=2) + "\n")
    (out / "per_cell_records.json").write_text(
        json.dumps(cell_details, indent=2) + "\n")

    # ---- Provenance + config + README -----------------------------------
    cfg_record = {
        "systems": list(args.systems),
        "seeds": list(args.seeds),
        "steps": args.steps,
        "n_colloc": args.n_colloc,
        "target_params": args.target_params,
        "scope_note": ("P7.5 SOLVER-TASK H1 verdict. THE PRIMARY "
                       "headline per pre-reg §7 GATING rule "
                       "'CONFIRMED iff ... it holds on the SOLVER "
                       "task'. Forecaster-task verdict from P5 is "
                       "corroborating evidence."),
    }
    (out / "config.json").write_text(
        json.dumps(cfg_record, indent=2) + "\n")
    prov = {**_git_prov(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "package_versions": _pkg(),
            "wall_clock_start_utc": start,
            "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z"}
    (out / "provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n")
    (out / "README.md").write_text(
        "# results/p7_5_solver_h1/\n\n"
        "**P7.5 PRIMARY H1 VERDICT** — pre-reg §7 gating task.\n\n"
        "Pipeline:\n"
        "  - QLNN best-ansatz solver relL²: read from results/p3_6_multi_state\n"
        "    (4 families × 3 systems × 3 seeds = 36 cells already on disk)\n"
        "  - Classical PINN solver baseline: trained here (physics-residual\n"
        "    Lagaris hard-IC, capacity-matched MLP)\n"
        "  - H1 verdict: paired-bootstrap CI of (Δ_smooth − Δ_broad) per\n"
        "    pre-reg §7\n\n"
        "Outputs:\n"
        "  - h1_analysis_solver_task.json (PRIMARY at skyline_threshold=0.5)\n"
        "  - h1_analysis_solver_task_sensitivity.json (at 0.75)\n"
        "  - per_cell_records.json (9 (system, seed) Δ records)\n"
        "  - {system}_classical_pinn/seed_N/ — per-cell baseline metrics\n\n"
        "The forecaster-task H1 from results/p5_h1_verdict is now\n"
        "corroborating evidence per pre-reg §7 gating rule.\n")

    print("=" * 70, flush=True)
    print(f"PRIMARY (SOLVER-TASK) H1 VERDICT: {v_strict['outcome']}",
          flush=True)
    print(f"  threshold=0.5: {v_strict['outcome']}", flush=True)
    if v_strict["bootstrap"]:
        b = v_strict["bootstrap"]
        print(f"    Δ_diff = {b['delta_diff_mean']:+.4f}  "
              f"95% CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]",
              flush=True)
    print(f"  threshold=0.75: {v_sensitivity['outcome']}", flush=True)
    if v_sensitivity["bootstrap"]:
        b = v_sensitivity["bootstrap"]
        print(f"    Δ_diff = {b['delta_diff_mean']:+.4f}  "
              f"95% CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]",
              flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
