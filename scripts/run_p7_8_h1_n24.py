"""P7.8 commit — Combined ODE+PDE solver-task H1 verdict at n=24.

Extends the P7.6 commit-2 n=18 verdict by adding 2 new pre-reg §4
hardness-ladder systems:
  - fitzhugh_nagumo (ODE, BROADBAND/MULTISCALE) — added P7.8 c1
  - burgers_shock   (PDE, BROADBAND/MULTISCALE) — added P7.8 c2

Result coverage:
  ODE solver: LV (3) + VdP (3) + Lorenz (3) + FHN (3) = 12 cells
    smooth: LV (3), VdP (3) → 6 cells
    broad:  Lorenz (3), FHN (3) → 6 cells
  PDE solver: heat (3) + burgers_smooth (3) + burgers_shock (3)
              + allen_cahn (3) = 12 cells
    smooth: heat (3), burgers_smooth (3) → 6 cells
    broad:  burgers_shock (3), allen_cahn (3) → 6 cells

Combined: **n=24** (12 smooth + 12 broad) — symmetrical bins, 1.33×
the P7.6 n=18 verdict's sample size.

What's NOT here (documented as P7.8 c3 / PRE_REG_AMENDMENT A11):
  - kuramoto: 12D per-component scalar circuits → ~7 hr per cell,
    deferred to follow-up.
  - kdv: mechanism-gate PASS (jacrev³ works), but integrated train
    cost is ~8 hr/seed at canonical 32×32 colloc × 2400 steps →
    deferred to follow-up.

Inputs (READ-ONLY):
  results/p3_6_multi_state/{family}_{ode}/seed_N/metrics.json
    — QLNN solver per-cell relL² (best of 4 families per cell)
  results/p7_5_solver_h1/{ode}_classical_pinn/seed_N/metrics.json
    — classical PINN solver per-cell relL² (matched H1 contrast)
  results/p3_8_review/{pde}_chebyshev_dqc_2d/seed_N/metrics.json
  results/p3_9_pde_matrix/{pde}_{family}/seed_N/metrics.json
    — QLNN PDE solver per-cell relL² (best of 4 families per cell)
  results/p3_8_review/{pde}_classical_pinn/seed_N/metrics.json
    — classical PINN PDE solver per-cell relL² (matched contrast)

Outputs:
  results/p7_8_solver_h1_n24/h1_analysis_combined_n24.json
    — THE PAPER'S PRIMARY HEADLINE post-P7.8
  results/p7_8_solver_h1_n24/per_cell_records.json (n=24)
  results/p7_8_solver_h1_n24/provenance.json
"""
from __future__ import annotations

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
OUT = REPO_ROOT / "results" / "p7_8_solver_h1_n24"

# --- ODE side -----------------------------------------------------------
P36_PATH = REPO_ROOT / "results" / "p3_6_multi_state"
P75_PATH = REPO_ROOT / "results" / "p7_5_solver_h1"
ODE_SYSTEMS = ("lotka_volterra", "van_der_pol", "lorenz", "fitzhugh_nagumo")
ODE_REGIME = {
    "lotka_volterra":  "smooth_periodic",
    "van_der_pol":     "smooth_periodic",
    "lorenz":          "broadband_multiscale",
    "fitzhugh_nagumo": "broadband_multiscale",
}
ODE_QLNN_FAMILIES = ("chebyshev_dqc", "te_qpinn_fnn", "te_qpinn_qnn", "qcpinn")

# --- PDE side -----------------------------------------------------------
P38_PATH = REPO_ROOT / "results" / "p3_8_review"
P39_PATH = REPO_ROOT / "results" / "p3_9_pde_matrix"
PDE_SYSTEMS = ("heat", "burgers_smooth", "allen_cahn", "burgers_shock")
PDE_REGIME = {
    "heat":           "smooth_periodic",
    "burgers_smooth": "smooth_periodic",
    "allen_cahn":     "broadband_multiscale",
    "burgers_shock":  "broadband_multiscale",
}
# 4 quantum families on each PDE: chebyshev_dqc_2d lives in P3.8,
# the other 3 live in P3.9.
PDE_QLNN_LOC = {
    "chebyshev_dqc_2d": (P38_PATH, "chebyshev_dqc_2d"),
    "qcpinn_2d":        (P39_PATH, "qcpinn_2d"),
    "te_qpinn_fnn_2d":  (P39_PATH, "te_qpinn_fnn_2d"),
    "te_qpinn_qnn_2d":  (P39_PATH, "te_qpinn_qnn_2d"),
}


def _best_ode_qlnn(system: str, seed: int) -> tuple[float | None, str | None]:
    best, best_fam = None, None
    for fam in ODE_QLNN_FAMILIES:
        p = P36_PATH / f"{fam}_{system}" / f"seed_{seed}" / "metrics.json"
        if not p.exists():
            continue
        v = float(json.loads(p.read_text())["relative_l2"])
        if best is None or v < best:
            best, best_fam = v, fam
    return best, best_fam


def _ode_classical_pinn(system: str, seed: int) -> float | None:
    p = P75_PATH / f"{system}_classical_pinn" / f"seed_{seed}" / "metrics.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["relative_l2"])


def _best_pde_qlnn(pde: str, seed: int) -> tuple[float | None, str | None]:
    best, best_fam = None, None
    for fam, (root, sub) in PDE_QLNN_LOC.items():
        p = root / f"{pde}_{sub}" / f"seed_{seed}" / "metrics.json"
        if not p.exists():
            continue
        v = float(json.loads(p.read_text())["relative_l2"])
        if best is None or v < best:
            best, best_fam = v, fam
    return best, best_fam


def _pde_classical_pinn(pde: str, seed: int) -> float | None:
    p = P38_PATH / f"{pde}_classical_pinn" / f"seed_{seed}" / "metrics.json"
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
    print(f"P7.8 combined ODE+PDE H1 verdict @ n=24 — start {start}",
          flush=True)

    records: list[CellRecord] = []
    details: list[dict] = []

    # ---- ODE side: 12 cells -------------------------------------------
    print("\nODE solver cells:", flush=True)
    for ode in ODE_SYSTEMS:
        for seed in (0, 1, 2):
            qv, qfam = _best_ode_qlnn(ode, seed)
            cv = _ode_classical_pinn(ode, seed)
            if qv is None or cv is None:
                print(f"  [{ode:<16} seed={seed}] MISSING (qlnn={qv}, "
                      f"cpinn={cv}) — skip", flush=True)
                continue
            rec = CellRecord(
                system=ode, seed=seed,
                qlnn_relL2=qv, neuralode_relL2=cv,
                qlnn_train_relL2=None, neuralode_train_relL2=None,
                skyline_relL2=None,
            )
            records.append(rec)
            details.append({
                "task": "ode_solver",
                "system": ode, "seed": seed,
                "qlnn_best_family": qfam,
                "qlnn_relL2": qv,
                "classical_pinn_relL2": cv,
                "delta": rec.delta,
                "regime": rec.regime,
            })
            print(f"  [{ode:<16} seed={seed}]  qlnn={qfam:<16}({qv:.4f})  "
                  f"cpinn={cv:.4f}  Δ={rec.delta:+.4f}  ({rec.regime})",
                  flush=True)

    # ---- PDE side: 12 cells -------------------------------------------
    print("\nPDE solver cells:", flush=True)
    for pde in PDE_SYSTEMS:
        for seed in (0, 1, 2):
            qv, qfam = _best_pde_qlnn(pde, seed)
            cv = _pde_classical_pinn(pde, seed)
            if qv is None or cv is None:
                print(f"  [{pde:<16} seed={seed}] MISSING (qlnn={qv}, "
                      f"cpinn={cv}) — skip", flush=True)
                continue
            rec = CellRecord(
                system=pde, seed=seed,
                qlnn_relL2=qv, neuralode_relL2=cv,
                qlnn_train_relL2=None, neuralode_train_relL2=None,
                skyline_relL2=None,
            )
            records.append(rec)
            details.append({
                "task": "pde_solver",
                "system": pde, "seed": seed,
                "qlnn_best_family": qfam,
                "qlnn_relL2": qv,
                "classical_pinn_relL2": cv,
                "delta": rec.delta,
                "regime": rec.regime,
            })
            print(f"  [{pde:<16} seed={seed}]  qlnn={qfam:<18}({qv:.4f})  "
                  f"cpinn={cv:.4f}  Δ={rec.delta:+.4f}  ({rec.regime})",
                  flush=True)

    # ---- Run verdict ---------------------------------------------------
    print(f"\nRunning paired-bootstrap H1 verdict on n={len(records)} cells "
          f"...", flush=True)
    verdict = h1_verdict(records, n_iter=10000,
                         skyline_threshold=10.0, seed=0)
    (OUT / "h1_analysis_combined_n24.json").write_text(
        json.dumps(verdict, indent=2) + "\n")
    (OUT / "per_cell_records.json").write_text(
        json.dumps(details, indent=2) + "\n")

    prov = {**_git_prov(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "wall_clock_start_utc": start,
            "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z"}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    # README
    (OUT / "README.md").write_text(
        "# results/p7_8_solver_h1_n24/\n\n"
        "P7.8 commit — combined ODE+PDE solver-task H1 verdict at\n"
        "**n=24** (12 smooth + 12 broad), 1.33× the P7.6 n=18 verdict.\n\n"
        "Adds two pre-reg §4 systems:\n"
        "  - fitzhugh_nagumo (ODE, BROADBAND/MULTISCALE)\n"
        "  - burgers_shock   (PDE, BROADBAND/MULTISCALE)\n\n"
        "Skips (deferred to follow-up paper, see PRE_REG_AMENDMENT A11):\n"
        "  - kuramoto (12D high-dim, ~7 hr/cell)\n"
        "  - kdv (jacrev³ mechanism gate PASS but integrated cost ~8 hr/seed)\n"
    )

    print("=" * 70, flush=True)
    print(f"P7.8 combined H1 verdict: {verdict['outcome']}", flush=True)
    if verdict["bootstrap"]:
        b = verdict["bootstrap"]
        print(f"  Δ_smooth = {b['delta_smooth_mean']:+.4f}", flush=True)
        print(f"  Δ_broad  = {b['delta_broad_mean']:+.4f}", flush=True)
        print(f"  Δ_diff   = {b['delta_diff_mean']:+.4f}", flush=True)
        print(f"  95% CI   = [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]",
              flush=True)
        print(f"  n_smooth = {b['n_smooth']}, n_broad = {b['n_broad']}",
              flush=True)
    print(f"\nWritten: {OUT}", flush=True)


if __name__ == "__main__":
    main()
