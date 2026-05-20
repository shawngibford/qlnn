"""Run the P3.5 solver-comparison sweep and emit JSON+NPZ artifacts.

Thin CLI on top of `src/qlnn_/training/solver_demo.py`. Trains the
4 P3 solver families on the 2 ODE benchmarks across N seeds, then
writes one directory per (family × ode) under
`results/p3_solver_demo/` following the project's seeds-summary
schema (mean ± 95% t-CI, mirroring results/*/seeds_summary.json).

Usage:
    python scripts/run_solver_demo.py                       # full sweep
    python scripts/run_solver_demo.py --steps 200           # quick smoke
    python scripts/run_solver_demo.py --seeds 0 1           # subset seeds
    python scripts/run_solver_demo.py --families chebyshev_dqc qcpinn

See ODE_PDE_PRE_REG.md §3 / the plan at
~/.claude/plans/what-is-our-next-humming-octopus.md for the role of
this demo (visible first-results sprint between P3 and P4 — NOT a
paper claim; numbers are seed-dependent CPU JAX, not pinned by
verify_paper_integrity.py).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from qlnn_.training.solver_demo import (
    FAMILIES,
    ODES,
    run_sweep,
    summarize_seeds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p3_solver_demo"


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
        "jax": jax.__version__,
        "optax": optax.__version__,
        "pennylane": pennylane.__version__,
        "equinox": equinox.__version__,
        "diffrax": diffrax.__version__,
        "numpy": numpy.__version__,
    }


def _write_per_seed(result: dict, base: Path) -> None:
    d = base / f"{result['family']}_{result['ode']}" / f"seed_{result['seed']}"
    d.mkdir(parents=True, exist_ok=True)
    # metrics.json — scalar fields only, mirrors results/*/seed_*/metrics.json
    metrics = {
        k: result[k] for k in (
            "family", "ode", "seed", "steps", "lr", "n_colloc",
            "pqc_params", "classical_params", "config_str",
            "final_loss", "mae", "rmse")
    }
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # curves.npz — full predicted trajectory + exact + loss history
    np.savez_compressed(
        d / "curves.npz",
        t_eval=result["t_eval"],
        u_pred=result["u_pred"],
        exact=result["exact"],
        loss_history=np.asarray(result["loss_history"], dtype=np.float64),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES))
    ap.add_argument("--odes", nargs="+", default=list(ODES),
                    choices=list(ODES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=None,
                    help="Override per-family default step counts (use for "
                         "quick smoke runs; default uses each family's "
                         "tuned step budget).")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="Output directory.")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P3.5 solver demo — start {start}")
    print(f"  families : {args.families}")
    print(f"  odes     : {args.odes}")
    print(f"  seeds    : {args.seeds}")
    print(f"  out      : {out}")

    # Run sweep
    results = run_sweep(
        families=args.families,
        odes=args.odes,
        seeds=args.seeds,
        steps_override=args.steps)

    # Per-seed artifacts + (family, ode)-level summaries
    seen_groups: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        _write_per_seed(r, out)
        seen_groups.setdefault((r["family"], r["ode"]), []).append(r)
        print(f"  [{r['family']:<14}] {r['ode']:<8} seed={r['seed']}  "
              f"MAE={r['mae']:.4f}  loss={r['final_loss']:.2e}  "
              f"pqc={r['pqc_params']:>3}  cls={r['classical_params']:>3}")

    for (fam, ode), grp in seen_groups.items():
        d = out / f"{fam}_{ode}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "seeds_summary.json").write_text(
            json.dumps(summarize_seeds(grp), indent=2) + "\n")

    # Sweep-level config + provenance
    cfg = {
        "families": args.families,
        "odes": args.odes,
        "seeds": args.seeds,
        "steps_override": args.steps,
        "per_family_default_steps": {f: FAMILIES[f][1] for f in args.families},
        "ode_descriptions": {o: ODES[o].description for o in args.odes},
        "scope_note": ("P3.5 visible-first-results sprint; NOT a paper "
                       "claim, NOT in verify_paper_integrity's contract; "
                       "see plan at ~/.claude/plans/what-is-our-next-"
                       "humming-octopus.md."),
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

    # README (small) — explains why these numbers aren't pinned.
    (out / "README.md").write_text(
        "# results/p3_solver_demo/\n\n"
        "P3.5 visible-first-results sprint output. NOT a paper claim.\n"
        "Numbers are seed-dependent CPU JAX runs; not pinned by\n"
        "`scripts/verify_paper_integrity.py`. The figure is rendered by\n"
        "`scripts/make_solver_demo_figure.py` and writes to\n"
        "`paper/figures/fig_p3_solver_demo.{png,pdf}`. Per-(family, ode)\n"
        "`seeds_summary.json` mirrors the project's standard schema.\n"
        "Per-seed `curves.npz` holds `t_eval`, `u_pred`, `exact`, and\n"
        "`loss_history` for plotting.\n")

    print(f"\nP3.5 sweep done. Wrote:")
    for path in sorted(out.rglob("*.json")):
        rel = path.relative_to(REPO_ROOT)
        print(f"    {rel}")


if __name__ == "__main__":
    main()
