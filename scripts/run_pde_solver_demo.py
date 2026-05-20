"""Run the P3.7 PDE solver demo and emit JSON+NPZ artifacts.

Thin CLI on `src/qlnn_/training/pde_demo.py`. Solves 3 PDEs (heat /
burgers_smooth / allen_cahn) via the Chebyshev-DQC 2D solver across
3 seeds and writes per-(pde, seed) artifacts + per-pde summaries
following the project's standard schema.

Usage:
    python scripts/run_pde_solver_demo.py                    # full sweep
    python scripts/run_pde_solver_demo.py --steps 300         # smoke
    python scripts/run_pde_solver_demo.py --pdes heat         # subset

NOT a paper claim; numbers are seed-dependent CPU JAX runs; not
pinned by verify_paper_integrity.py.
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

from qlnn_.training.pde_demo import (
    PDE_BENCH,
    summarize_pde_seeds,
    train_one_pde,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p3_7_pde_solver"


def _git_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip())
    except Exception:
        commit, branch, dirty = "unknown", "unknown", True
    return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}


def _package_versions() -> dict:
    import jax, optax, pennylane, equinox, diffrax, numpy
    return {
        "jax": jax.__version__, "optax": optax.__version__,
        "pennylane": pennylane.__version__,
        "equinox": equinox.__version__, "diffrax": diffrax.__version__,
        "numpy": numpy.__version__,
    }


def _write_per_seed(r: dict, base: Path) -> None:
    d = base / r["pde"] / f"seed_{r['seed']}"
    d.mkdir(parents=True, exist_ok=True)
    metrics = {k: r[k] for k in (
        "pde", "seed", "regime", "steps", "lr",
        "n_t_qubits", "n_x_qubits", "num_layers",
        "n_t_colloc", "n_x_colloc", "pqc_params",
        "final_loss", "mae", "rmse", "relative_l2")}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        d / "field.npz",
        t_eval=r["t_eval"], x_eval=r["x_eval"],
        u_pred=r["u_pred"], u_ref=r["u_ref"],
        loss_history=np.asarray(r["loss_history"], dtype=np.float64))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdes", nargs="+", default=list(PDE_BENCH),
                    choices=list(PDE_BENCH))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=None,
                    help="Override per-PDE default step counts (smoke).")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P3.7 PDE solver demo — start {start}")
    print(f"  pdes  : {args.pdes}")
    print(f"  seeds : {args.seeds}")
    print(f"  out   : {out}")

    # Loop directly so each completed run is WRITTEN + PRINTED before
    # moving to the next — avoids ~20+ min of silent accumulation that
    # would happen if we batched all runs first then wrote/printed at
    # the end. With PYTHONUNBUFFERED=1 the stdout flushes per-run.
    results: list[dict] = []
    groups: dict[str, list[dict]] = {}
    for pde_name in args.pdes:
        for seed in args.seeds:
            r = train_one_pde(pde_name, seed, steps_override=args.steps)
            _write_per_seed(r, out)
            results.append(r)
            groups.setdefault(r["pde"], []).append(r)
            print(f"  [{r['pde']:<16}] seed={r['seed']}  "
                  f"relL2={r['relative_l2']:.4f}  "
                  f"MAE={r['mae']:.4f}  "
                  f"loss={r['final_loss']:.2e}  "
                  f"pqc={r['pqc_params']:>3}",
                  flush=True)

    for name, grp in groups.items():
        d = out / name
        (d / "seeds_summary.json").write_text(
            json.dumps(summarize_pde_seeds(grp), indent=2) + "\n")

    cfg = {
        "pdes": args.pdes,
        "seeds": args.seeds,
        "steps_override": args.steps,
        "per_pde_defaults": {
            n: {"steps": PDE_BENCH[n].steps,
                "n_t_colloc": PDE_BENCH[n].n_t_colloc,
                "n_x_colloc": PDE_BENCH[n].n_x_colloc,
                "regime": PDE_BENCH[n].regime,
                "description": PDE_BENCH[n].description}
            for n in args.pdes},
        "scope_note": ("P3.7 PDE solver demo (heat + Burgers + Allen-Cahn). "
                       "Single family (chebyshev_dqc_2d); cross-family on "
                       "PDEs is P6 territory. Demo output, NOT a paper "
                       "claim; not in verify_paper_integrity contract."),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    prov = {
        **_git_provenance(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "wall_clock_start_utc": start,
        "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z",
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    (out / "README.md").write_text(
        "# results/p3_7_pde_solver/\n\n"
        "P3.7 PDE solver demo output. Three PDEs (heat / burgers_smooth /\n"
        "allen_cahn) via the Chebyshev-DQC 2D solver. NOT a paper claim;\n"
        "seed-dependent CPU JAX runs; not pinned by\n"
        "`scripts/verify_paper_integrity.py`. Figure:\n"
        "`paper/figures/fig_p3_7_pde_solver.{png,pdf}` via\n"
        "`scripts/make_pde_solver_figure.py`.\n")

    print(f"\nP3.7 sweep done. {len(results)} runs across "
          f"{len(args.pdes)} PDEs × {len(args.seeds)} seeds.")


if __name__ == "__main__":
    main()
