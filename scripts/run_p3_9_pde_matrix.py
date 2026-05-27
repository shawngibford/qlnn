"""Run the P3.9 PDE multi-family matrix sweep.

Sweep: {qcpinn_2d, te_qpinn_fnn_2d, te_qpinn_qnn_2d} × {heat,
burgers_smooth, allen_cahn} × seeds [0, 1, 2] = 27 runs at the
audit-corrected configs (heat 1200, Burgers 1500, AC 64×32×1800).

Writes to `results/p3_9_pde_matrix/{pde}_{family}/seed_N/{
metrics.json, field.npz}` + per-({pde}, {family}) `seeds_summary.json`
+ `config.json` + `provenance.json` following the existing schema.

`chebyshev_dqc_2d` is OUT of this sweep — its data is already in
`results/p3_8_review/{pde}_chebyshev_dqc_2d/seed_*` at the same
configs. The P3.9 figure script reads both directories so the bars
appear together. Pass `--include-chebyshev` to re-run it for parity
sanity-check (adds 9 runs).

By default streams per-run output to stdout BEFORE the next run
starts (PYTHONUNBUFFERED=1 recommended). NOT a paper claim — see
HANDOFF for the P3.9 framing (multi-family PDE coverage but H1
verdict still awaits P5).
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

from qlnn_.training.p3_8_review_demo import CORRECTED_PDE_CONFIGS
from qlnn_.training.p3_9_pde_matrix import (
    QUANTUM_FAMILIES,
    summarize,
    train_one_cell,
)
from qlnn_.training.pde_demo import PDE_BENCH
from quantum_liquid_neuralode.data_processing.pde_systems import (
    assert_dataset_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p3_9_pde_matrix"

# Default families exclude chebyshev_dqc_2d (already in p3_8_review).
DEFAULT_NEW_FAMILIES = ("qcpinn_2d", "te_qpinn_fnn_2d", "te_qpinn_qnn_2d")


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdes", nargs="+",
                    default=list(CORRECTED_PDE_CONFIGS),
                    choices=list(CORRECTED_PDE_CONFIGS))
    ap.add_argument("--families", nargs="+",
                    default=list(DEFAULT_NEW_FAMILIES),
                    choices=list(QUANTUM_FAMILIES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--include-chebyshev", action="store_true",
                    help="Also re-run chebyshev_dqc_2d (already in "
                          "p3_8_review at the same configs)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    families = list(args.families)
    if args.include_chebyshev and "chebyshev_dqc_2d" not in families:
        families = ["chebyshev_dqc_2d"] + families

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P3.9 PDE multi-family matrix — start {start}", flush=True)
    print(f"  pdes     : {args.pdes}", flush=True)
    print(f"  families : {families}", flush=True)
    print(f"  seeds    : {args.seeds}", flush=True)

    # P6 G4: gate on per-system PDE field hashes BEFORE any training
    # consumes the .npz references. Analytic-only PDEs (heat) have
    # npz_basename=None in PDE_BENCH and are skipped.
    for _pde_name in args.pdes:
        if PDE_BENCH[_pde_name].npz_basename is not None:
            assert_dataset_hash(_pde_name)

    groups: dict[str, list[dict]] = {}
    for pde_name in args.pdes:
        for family in families:
            for seed in args.seeds:
                r = train_one_cell(pde_name, family, seed)
                _write_seed(r, out)
                key = f"{r['pde']}_{r['model']}"
                groups.setdefault(key, []).append(r)
                print(
                    f"  [{pde_name:<14} {family:<18}] seed={seed}  "
                    f"relL2={r['relative_l2']:.4f}  "
                    f"MAE={r['mae']:.4f}  "
                    f"bc_v={r['bc_violation']:.3f}  "
                    f"final_loss={r['final_loss']:.3e}",
                    flush=True)

    for key, grp in groups.items():
        (out / key).mkdir(parents=True, exist_ok=True)
        (out / key / "seeds_summary.json").write_text(
            json.dumps(summarize(grp), indent=2) + "\n")

    cfg = {
        "pdes": list(args.pdes),
        "families": list(families),
        "seeds": list(args.seeds),
        "corrected_pde_configs": {
            n: {"n_t_colloc": CORRECTED_PDE_CONFIGS[n].n_t_colloc,
                "n_x_colloc": CORRECTED_PDE_CONFIGS[n].n_x_colloc,
                "steps": CORRECTED_PDE_CONFIGS[n].steps,
                "audit_reason": CORRECTED_PDE_CONFIGS[n].audit_reason}
            for n in args.pdes},
        "scope_note": ("P3.9 PDE multi-family matrix. Adds the 3 PINN-style "
                        "quantum-family 2D ports to the PDE side so the "
                        "matrix matches the ODE matrix shape (4 quantum × "
                        "3 PDEs × 3 seeds). NOT a paper claim; the H1 "
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
        "# results/p3_9_pde_matrix/\n\n"
        "P3.9 PDE multi-family matrix output. Adds qcpinn_2d,\n"
        "te_qpinn_fnn_2d, te_qpinn_qnn_2d (the 3 PINN-style 2D ports)\n"
        "alongside chebyshev_dqc_2d (already in results/p3_8_review/).\n"
        "All at the audit-corrected configs (heat 1200, Burgers 1500,\n"
        "AC 64×32×1800).\n\n"
        "Figure: `paper/figures/fig_p3_9_pde_matrix.{png,pdf}`.\n"
        "Closes the P3.8 audit coverage gap (PDE side was single-family).\n"
        "NOT a paper claim — H1 verdict still requires P5's Neural-ODE.\n")

    print(f"\nP3.9 PDE matrix done.", flush=True)


if __name__ == "__main__":
    main()
