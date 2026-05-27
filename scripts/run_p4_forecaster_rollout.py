"""P4 sweep CLI — forecaster autoregressive rollout on 3 ODE systems.

Runs the 5-family × 3-system × 3-seed = 45-cell matrix:

  Forecaster families  ×  ODE systems  ×  seeds [0,1,2]
  ─────────────────────────────────────────────────────
  data_reuploading     |  lotka_volterra
  hardware_efficient   |  van_der_pol
  strongly_entangling  |  lorenz
  brickwall            |
  rf_qrc               |

Per-cell pipeline (see `qlnn_.training.p4_forecaster_demo`):
  - Integrate canonical reference (synthetic_ode.simulate).
  - Train/test chronological split.
  - Train forecaster (4 quantum families via VectorForecaster +
    optax-adam; rf_qrc via closed-form Tikhonov ridge).
  - Autoregressive rollout from the test trajectory's initial window.
  - Compute the pre-reg §5 metric suite (relative-L2, VPT,
    spectral_error, invariant_drift).
  - Persistence floor relative-L2 (NOT reported as a win; plotted
    for context).

Writes:
  results/p4_forecaster_rollout/{system}_{family}/seed_N/
    metrics.json   (per-seed scalar metrics)
    field.npz      (predicted + reference + rel-L2 curve)
  results/p4_forecaster_rollout/{system}_{family}/seeds_summary.json
  results/p4_forecaster_rollout/config.json
  results/p4_forecaster_rollout/provenance.json
  results/p4_forecaster_rollout/README.md

Stream per-run output as it goes (PYTHONUNBUFFERED=1 recommended).

Compute budget: ~70 min CPU on default.qubit JAX for the full
matrix. The first cell of each quantum family pays JIT compile;
subsequent same-family cells reuse cache.
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

from qlnn_.training.p4_forecaster_demo import (
    ALL_FAMILIES_P4,
    P4SweepConfig,
    SYSTEMS_P4,
    summarize_p4,
    train_and_rollout_one_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p4_forecaster_rollout"


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
    d = base / f"{r['system']}_{r['family']}" / f"seed_{r['seed']}"
    d.mkdir(parents=True, exist_ok=True)
    scalars = {k: r[k] for k in (
        "system", "family", "seed",
        "n_train_points", "n_test_points",
        "trainable_params", "sampled_dt", "rollout_steps",
        "dt_step", "relative_l2", "train_relative_l2",
        "vpt_step", "vpt_time", "vpt_lyapunov",
        "spectral_error", "invariant_drift_final",
        "persistence_floor_relative_l2")}
    scalars["train_loss_history"] = r["train_loss_history"]
    (d / "metrics.json").write_text(json.dumps(scalars, indent=2) + "\n")
    np.savez_compressed(
        d / "field.npz",
        u_pred=r["u_pred"], u_ref=r["u_ref"],
        rel_l2_curve=r["rel_l2_curve"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS_P4),
                    choices=list(SYSTEMS_P4))
    ap.add_argument("--families", nargs="+", default=list(ALL_FAMILIES_P4),
                    choices=list(ALL_FAMILIES_P4))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-points", type=int, default=800,
                    help="Trajectory length (default 800)")
    ap.add_argument("--train-steps", type=int, default=200,
                    help="Optax steps for VectorForecaster (default 200)")
    ap.add_argument("--rollout-steps", type=int, default=200,
                    help="Autoregressive rollout horizon (default 200)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    cfg = P4SweepConfig(
        n_points=args.n_points,
        train_steps=args.train_steps,
        rollout_steps=args.rollout_steps,
    )

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P4 forecaster rollout sweep — start {start}", flush=True)
    print(f"  systems  : {args.systems}", flush=True)
    print(f"  families : {args.families}", flush=True)
    print(f"  seeds    : {args.seeds}", flush=True)
    print(f"  n_points : {cfg.n_points}, "
          f"train_steps={cfg.train_steps}, "
          f"rollout_steps={cfg.rollout_steps}", flush=True)

    groups: dict[str, list[dict]] = {}
    for system in args.systems:
        for family in args.families:
            for seed in args.seeds:
                r = train_and_rollout_one_cell(
                    system, family, seed, cfg=cfg)
                _write_seed(r, out)
                key = f"{r['system']}_{r['family']}"
                groups.setdefault(key, []).append(r)
                vpt_lyap = ("-" if r["vpt_lyapunov"] is None
                            else f"{r['vpt_lyapunov']:.2f}")
                train_rl2 = ("-" if r.get("train_relative_l2") is None
                             else f"{r['train_relative_l2']:.4f}")
                print(
                    f"  [{system:<14} {family:<20}] seed={seed}  "
                    f"relL2={r['relative_l2']:.4f}  "
                    f"train_relL2={train_rl2}  "
                    f"vpt={r['vpt_step']}step  "
                    f"vpt_lyap={vpt_lyap}  "
                    f"spec={r['spectral_error']:.4f}  "
                    f"pers={r['persistence_floor_relative_l2']:.4f}",
                    flush=True)

    for key, grp in groups.items():
        (out / key).mkdir(parents=True, exist_ok=True)
        (out / key / "seeds_summary.json").write_text(
            json.dumps(summarize_p4(grp), indent=2) + "\n")

    config_record = {
        "systems": list(args.systems),
        "families": list(args.families),
        "seeds": list(args.seeds),
        "n_points": cfg.n_points,
        "train_frac": cfg.train_frac,
        "window_length": cfg.window_length,
        "num_qubits": cfg.num_qubits,
        "num_layers": cfg.num_layers,
        "train_steps": cfg.train_steps,
        "learning_rate": cfg.learning_rate,
        "rollout_steps": cfg.rollout_steps,
        "vpt_threshold": cfg.vpt_threshold,
        "rfqrc_num_qubits": cfg.rfqrc_num_qubits,
        "rfqrc_leak_rate": cfg.rfqrc_leak_rate,
        "rfqrc_beta": cfg.rfqrc_beta,
        "scope_note": ("P4 forecaster autoregressive rollout. "
                       "Adds the data-driven track that the pre-reg "
                       "§3.2 requires. NOT yet H1 evidence — H1 is "
                       "defined as the QLNN−NeuralODE gap; the "
                       "mandatory baseline awaits P5."),
    }
    (out / "config.json").write_text(
        json.dumps(config_record, indent=2) + "\n")
    prov = {**_git_prov(), "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "package_versions": _pkg(),
            "wall_clock_start_utc": start,
            "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z"}
    (out / "provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n")
    (out / "README.md").write_text(
        "# results/p4_forecaster_rollout/\n\n"
        "P4 forecaster autoregressive rollout output. 5 forecaster\n"
        "families × 3 ODE systems × 3 seeds = 45 cells.\n\n"
        "Per-cell metrics (pre-reg §5):\n"
        "  - relative_l2 over rollout horizon (PRIMARY endpoint)\n"
        "  - VPT (in Lyapunov times for Lorenz; physical-time for\n"
        "    LV/VdP)\n"
        "  - spectral_error (FFT PSD L2)\n"
        "  - invariant_drift (LV only; others have no invariant)\n"
        "  - persistence_floor_relative_l2 (context, NOT a win)\n\n"
        "Figure: `paper/figures/fig_p4_forecaster_rollout.{png,pdf}`.\n"
        "NOT yet H1 evidence — H1 is defined as the QLNN−NeuralODE\n"
        "advantage gap (pre-reg §2 / §7); the mandatory Neural-ODE\n"
        "baseline awaits P5.\n")
    print(f"\nP4 forecaster rollout done.", flush=True)


if __name__ == "__main__":
    main()
