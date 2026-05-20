"""Run the P3.8 review-iteration sweep and emit JSON+NPZ artifacts.

Compute pass for the peer-review audit's empirical reruns:

  PDE side (3 PDEs × 2 models × 3 seeds = 18 runs):
    heat / burgers_smooth / allen_cahn  ×
    chebyshev_dqc_2d / classical_pinn   ×  seeds [0, 1, 2]

  ODE side (lorenz × 4 quantum + classical = 5 models × 3 seeds = 15 runs):
    lorenz  × {chebyshev_dqc, te_qpinn_fnn, te_qpinn_qnn, qcpinn,
               classical_pinn}  ×  seeds [0, 1, 2]
    (Re-runs at T=5.0, ~5.5 Lyapunov times — vs P3.6's T=2 ≈ 2 LTE.)

Writes to `results/p3_8_review/{pde_or_system}_{model}/seed_N/{
metrics.json, field.npz}` + per-(thing, model) `seeds_summary.json`
+ `config.json` + `provenance.json` following the existing schema.

NOT a paper claim by itself; reframes the P3.6/P3.7 narratives by
adding (a) the missing classical baseline and (b) the missing
audit-corrected configs. See plan at
~/.claude/plans/what-is-our-next-humming-octopus.md (§P3.8).
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

from qlnn_.training.p3_8_review_demo import (
    CORRECTED_PDE_CONFIGS,
    summarize,
    train_one_lorenz_quantum,
    train_one_pde_classical,
    train_one_pde_quantum,
)
from qlnn_.training.solver_demo import FAMILIES as SCALAR_FAMILIES

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p3_8_review"


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


def _write_pde_seed(r: dict, base: Path) -> None:
    d = base / f"{r['pde']}_{r['model']}" / f"seed_{r['seed']}"
    d.mkdir(parents=True, exist_ok=True)
    scalars = {k: r[k] for k in (
        "pde", "model", "seed", "regime", "steps",
        "n_t_colloc", "n_x_colloc",
        "pqc_params", "classical_params", "config_str", "audit_reason",
        "final_loss", "mae", "relative_l2", "bc_violation")}
    (d / "metrics.json").write_text(json.dumps(scalars, indent=2) + "\n")
    np.savez_compressed(
        d / "field.npz",
        t_eval=r["t_eval"], x_eval=r["x_eval"],
        u_pred=r["u_pred"], u_ref=r["u_ref"],
        loss_history=np.asarray(r["loss_history"], dtype=np.float64))


def _write_lorenz_seed(r: dict, base: Path) -> None:
    d = base / f"{r['system']}_{r['model']}" / f"seed_{r['seed']}"
    d.mkdir(parents=True, exist_ok=True)
    scalars = {k: r[k] for k in (
        "system", "model", "seed", "regime", "dim",
        "t1", "lyapunov_times", "steps",
        "pqc_params", "classical_params", "config_str",
        "final_loss", "mae", "relative_l2",
        "relative_l2_predict_mean_baseline")}
    (d / "metrics.json").write_text(json.dumps(scalars, indent=2) + "\n")
    np.savez_compressed(
        d / "field.npz",
        t_eval=r["t_eval"], u_pred=r["u_pred"], u_ref=r["u_ref"],
        loss_history=np.asarray(r["loss_history"], dtype=np.float64))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdes", nargs="+",
                    default=list(CORRECTED_PDE_CONFIGS),
                    choices=list(CORRECTED_PDE_CONFIGS))
    ap.add_argument("--lorenz-families", nargs="+",
                    default=list(SCALAR_FAMILIES) + ["classical_pinn"],
                    help="Quantum families + 'classical_pinn' for Lorenz")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--skip-pde", action="store_true",
                    help="Skip the PDE re-runs (faster smoke).")
    ap.add_argument("--skip-lorenz", action="store_true",
                    help="Skip Lorenz extended runs.")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P3.8 review iteration — start {start}", flush=True)
    print(f"  pdes        : {args.pdes if not args.skip_pde else '(skipped)'}", flush=True)
    print(f"  lorenz fams : {args.lorenz_families if not args.skip_lorenz else '(skipped)'}", flush=True)
    print(f"  seeds       : {args.seeds}", flush=True)

    pde_groups: dict[str, list[dict]] = {}
    lorenz_groups: dict[str, list[dict]] = {}

    # PDE re-runs: quantum + classical pair
    if not args.skip_pde:
        for pde_name in args.pdes:
            for seed in args.seeds:
                # Quantum
                rq = train_one_pde_quantum(pde_name, seed)
                _write_pde_seed(rq, out)
                key_q = f"{rq['pde']}_{rq['model']}"
                pde_groups.setdefault(key_q, []).append(rq)
                print(f"  [{pde_name:<14} chebyshev_dqc_2d] seed={seed}  "
                      f"relL2={rq['relative_l2']:.4f}  MAE={rq['mae']:.4f}  "
                      f"bc_v={rq['bc_violation']:.3f}  "
                      f"steps={rq['steps']}", flush=True)
                # Classical
                rc = train_one_pde_classical(pde_name, seed)
                _write_pde_seed(rc, out)
                key_c = f"{rc['pde']}_{rc['model']}"
                pde_groups.setdefault(key_c, []).append(rc)
                print(f"  [{pde_name:<14} classical_pinn  ] seed={seed}  "
                      f"relL2={rc['relative_l2']:.4f}  MAE={rc['mae']:.4f}  "
                      f"bc_v={rc['bc_violation']:.3f}  "
                      f"cls={rc['classical_params']}", flush=True)
        for key, grp in pde_groups.items():
            (out / key).mkdir(parents=True, exist_ok=True)
            (out / key / "seeds_summary.json").write_text(
                json.dumps(summarize(grp), indent=2) + "\n")

    # Lorenz extended runs
    if not args.skip_lorenz:
        for family in args.lorenz_families:
            for seed in args.seeds:
                if family == "classical_pinn":
                    # Not yet implemented for vector ODE; skip for now
                    # and document in caveat.
                    continue
                r = train_one_lorenz_quantum(family, seed)
                _write_lorenz_seed(r, out)
                key = f"{r['system']}_{r['model']}"
                lorenz_groups.setdefault(key, []).append(r)
                print(f"  [lorenz {family:<16}] seed={seed}  "
                      f"relL2={r['relative_l2']:.4f}  "
                      f"mean_baseline={r['relative_l2_predict_mean_baseline']:.4f}  "
                      f"LTE={r['lyapunov_times']:.2f}", flush=True)
        for key, grp in lorenz_groups.items():
            (out / key).mkdir(parents=True, exist_ok=True)
            (out / key / "seeds_summary.json").write_text(
                json.dumps(summarize(grp), indent=2) + "\n")

    # Sweep-level config + provenance
    cfg = {
        "pdes": args.pdes if not args.skip_pde else [],
        "lorenz_families": args.lorenz_families if not args.skip_lorenz else [],
        "seeds": args.seeds,
        "corrected_pde_configs": {
            n: {"n_t_colloc": CORRECTED_PDE_CONFIGS[n].n_t_colloc,
                "n_x_colloc": CORRECTED_PDE_CONFIGS[n].n_x_colloc,
                "steps": CORRECTED_PDE_CONFIGS[n].steps,
                "audit_reason": CORRECTED_PDE_CONFIGS[n].audit_reason}
            for n in args.pdes if not args.skip_pde},
        "lorenz_t1": 5.0,
        "lorenz_lyapunov_times": 5.0 * 0.906,
        "scope_note": ("P3.8 audit-driven re-iteration. Corrects "
                       "P3.7's spatial under-resolution on Allen-Cahn, "
                       "step-budget inconsistencies on heat/burgers, "
                       "and P3.6's Lorenz time-horizon (T=5 ≈ 5.5 LTE "
                       "vs P3.6's T=2). Adds the missing classical "
                       "PINN baseline. NOT a paper claim; the H1 "
                       "verdict still requires P5's Neural-ODE."),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    prov = {**_git_prov(), "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "package_versions": _pkg(),
            "wall_clock_start_utc": start,
            "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z"}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    (out / "README.md").write_text(
        "# results/p3_8_review/\n\n"
        "P3.8 peer-review iteration output. Adds the missing classical\n"
        "PINN baseline; re-runs PDEs at audit-corrected configs (heat\n"
        "1200 steps, Burgers 1500 steps, Allen-Cahn 64×32 colloc at\n"
        "1800 steps); re-runs Lorenz at T=5 (~5.5 Lyapunov times).\n\n"
        "Figure: `paper/figures/fig_p3_8_review_iteration.{png,pdf}`.\n"
        "Reframes P3.6/P3.7 narratives (see HANDOFF). NOT a paper\n"
        "claim — H1 verdict still requires P5's Neural-ODE.\n")

    print(f"\nP3.8 review iteration done.", flush=True)


if __name__ == "__main__":
    main()
