"""Run the P3.6 multi-state ODE sweep and emit JSON+NPZ artifacts.

Thin CLI on top of `src/qlnn_/training/multi_state_solver.py`. Trains
the 4 P3 solver families on the 3 H1-relevant vector ODE systems
across N seeds. Writes per-(family × system × seed) artifacts plus
(family × system)-level seeds_summary.json, following the same schema
as the P3.5 demo and the project's standard `results/*/seeds_summary.json`.

Usage:
    python scripts/run_multi_state_demo.py                  # full sweep
    python scripts/run_multi_state_demo.py --steps 200      # smoke
    python scripts/run_multi_state_demo.py --systems lorenz # subset

See the plan at ~/.claude/plans/what-is-our-next-humming-octopus.md
for the role of this demo (extends P3.5 to vector-state ODEs;
exploratory output, NOT pinned by verify_paper_integrity.py).
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

from qlnn_.training.multi_state_solver import (
    VECTOR_ODES,
    run_vector_sweep,
    summarize_vector_seeds,
)
from qlnn_.training.solver_demo import FAMILIES as SCALAR_FAMILIES

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p3_6_multi_state"


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


def _write_per_seed(result: dict, base: Path) -> None:
    d = base / f"{result['family']}_{result['system']}" / f"seed_{result['seed']}"
    d.mkdir(parents=True, exist_ok=True)
    metrics = {
        k: result[k] for k in (
            "family", "system", "seed", "dim", "regime", "steps", "lr",
            "n_colloc", "pqc_params", "classical_params",
            "per_component_pqc_params", "config_str",
            "final_loss", "mae", "mae_per_component", "relative_l2")
    }
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        d / "curves.npz",
        t_eval=result["t_eval"],
        u_pred=result["u_pred"],
        u_ref=result["u_ref"],
        loss_history=np.asarray(result["loss_history"], dtype=np.float64),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+",
                    default=list(SCALAR_FAMILIES),
                    choices=list(SCALAR_FAMILIES))
    ap.add_argument("--systems", nargs="+",
                    default=list(VECTOR_ODES),
                    choices=list(VECTOR_ODES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=None,
                    help="Override per-family default step counts.")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P3.6 multi-state ODE demo — start {start}")
    print(f"  families : {args.families}")
    print(f"  systems  : {args.systems}")
    print(f"  seeds    : {args.seeds}")
    print(f"  out      : {out}")

    results = run_vector_sweep(
        families=args.families, systems=args.systems, seeds=args.seeds,
        steps_override=args.steps)

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        _write_per_seed(r, out)
        groups.setdefault((r["family"], r["system"]), []).append(r)
        print(f"  [{r['family']:<14}] {r['system']:<14} seed={r['seed']}  "
              f"dim={r['dim']}  MAE={r['mae']:.4f}  "
              f"relL2={r['relative_l2']:.4f}  "
              f"loss={r['final_loss']:.2e}  "
              f"pqc={r['pqc_params']:>3}  cls={r['classical_params']:>3}")

    for (fam, sysname), grp in groups.items():
        d = out / f"{fam}_{sysname}"
        (d / "seeds_summary.json").write_text(
            json.dumps(summarize_vector_seeds(grp), indent=2) + "\n")

    cfg = {
        "families": args.families,
        "systems": args.systems,
        "seeds": args.seeds,
        "steps_override": args.steps,
        "per_family_default_steps": {f: SCALAR_FAMILIES[f][1]
                                      for f in args.families},
        "system_descriptions": {s: VECTOR_ODES[s].description
                                for s in args.systems},
        "system_regimes": {s: VECTOR_ODES[s].regime for s in args.systems},
        "system_dims": {s: VECTOR_ODES[s].dim for s in args.systems},
        "scope_note": ("P3.6 multi-state ODE solver demo (extends P3.5 "
                       "to vector-state). NOT a paper claim; not in "
                       "verify_paper_integrity contract."),
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
        "# results/p3_6_multi_state/\n\n"
        "P3.6 multi-state ODE solver demo output (commits in HANDOFF.md).\n"
        "Extends P3.5's scalar demo to vector-state ODEs via\n"
        "per-component scalar circuits. NOT a paper claim; numbers are\n"
        "seed-dependent CPU JAX runs, not pinned by\n"
        "`scripts/verify_paper_integrity.py`. Figure rendered by\n"
        "`scripts/make_multi_state_figure.py` →\n"
        "`paper/figures/fig_p3_6_multi_state.{png,pdf}`.\n")

    print(f"\nP3.6 sweep done. {len(results)} runs across "
          f"{len(args.families)} families × {len(args.systems)} systems × "
          f"{len(args.seeds)} seeds.")


if __name__ == "__main__":
    main()
