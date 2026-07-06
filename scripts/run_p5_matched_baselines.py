"""P5 sweep CLI — matched baselines + H1 verdict.

Runs the 3 P5 baseline families on the 3 P4 ODE systems × 3 seeds:
  - plain_neuralode (MANDATORY H1 contrast)
  - plain_mlp       (capacity-matched classical control)
  - skyline         (structural upper bound, no training)

Total: 3 × 3 × 3 = 27 baseline cells. ~30-45 min wall-clock.

Then computes the H1 verdict per pre-reg §7 by combining these
baseline results with the P4 QLNN forecaster results
(`results/p4_forecaster_rollout/`).

Output:
  results/p5_matched_baselines/{system}_{family}/seed_N/
    metrics.json + field.npz
  results/p5_matched_baselines/{system}_{family}/seeds_summary.json
  results/p5_h1_verdict/h1_analysis.json  — THE headline artifact
  results/p5_h1_verdict/per_cell_records.json
  results/p5_matched_baselines/config.json + provenance.json + README.md

The H1 verdict computes:
  Δ_cell = NeuralODE_relL2(cell) − QLNN_best_relL2(cell)
  Δ_smooth = mean Δ over SMOOTH/PERIODIC systems × seeds
  Δ_broad = mean Δ over BROADBAND/MULTISCALE/CHAOTIC systems × seeds
  H1 = paired-bootstrap CI of (Δ_smooth − Δ_broad)
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
from qlnn_.training.p4_forecaster_demo import P4SweepConfig, SYSTEMS_P4
from qlnn_.training.p5_matched_baselines import (
    P5_BASELINE_FAMILIES, summarize_p5,
    train_and_rollout_baseline_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p5_matched_baselines"
VERDICT_DIR = REPO_ROOT / "results" / "p5_h1_verdict"
P4_RESULTS = REPO_ROOT / "results" / "p4_forecaster_rollout"


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
        "dt_step", "relative_l2",
        "vpt_step", "vpt_time", "vpt_lyapunov",
        "spectral_error", "invariant_drift_final",
        "persistence_floor_relative_l2")}
    scalars["train_loss_history"] = r["train_loss_history"]
    (d / "metrics.json").write_text(json.dumps(scalars, indent=2) + "\n")
    np.savez_compressed(
        d / "field.npz",
        u_pred=r["u_pred"], u_ref=r["u_ref"],
        rel_l2_curve=r["rel_l2_curve"])


def _load_p4_qlnn_best_relL2(system: str, seed: int) -> float | None:
    """For a (system, seed) cell, return the BEST QLNN relative-L2
    across the 4 vector-QLNN forecaster families from P4. Returns
    None if P4 data is missing."""
    best = None
    for family in ("data_reuploading", "hardware_efficient",
                   "strongly_entangling", "brickwall"):
        p = (P4_RESULTS / f"{system}_{family}" /
             f"seed_{seed}" / "metrics.json")
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        v = float(m["relative_l2"])
        if best is None or v < best:
            best = v
    return best


def _load_baseline_relL2(
    base: Path, system: str, family: str, seed: int,
) -> float | None:
    p = base / f"{system}_{family}" / f"seed_{seed}" / "metrics.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["relative_l2"])


def _build_cell_records(
    out_p5: Path, seeds: list[int],
) -> list[CellRecord]:
    """Combine P4 QLNN best-ansatz + P5 baseline results into the
    CellRecord list consumed by h1_verdict."""
    records: list[CellRecord] = []
    for system in SYSTEMS_P4:
        for seed in seeds:
            qlnn = _load_p4_qlnn_best_relL2(system, seed)
            neuralode = _load_baseline_relL2(
                out_p5, system, "plain_neuralode", seed)
            skyline = _load_baseline_relL2(
                out_p5, system, "skyline", seed)
            if qlnn is None or neuralode is None:
                continue
            records.append(CellRecord(
                system=system, seed=seed,
                qlnn_relL2=qlnn, neuralode_relL2=neuralode,
                skyline_relL2=skyline))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS_P4),
                    choices=list(SYSTEMS_P4))
    ap.add_argument("--families", nargs="+",
                    default=list(P5_BASELINE_FAMILIES),
                    choices=list(P5_BASELINE_FAMILIES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-points", type=int, default=800)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--rollout-steps", type=int, default=200)
    ap.add_argument("--bootstrap-iter", type=int, default=10000,
                    help="H1 paired-bootstrap iterations "
                         "(pre-reg §5: ≥10000)")
    ap.add_argument("--skyline-threshold", type=float, default=0.5,
                    help="Pre-reg §7 skyline-out-of-reach threshold")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--verdict-out", type=Path, default=VERDICT_DIR)
    ap.add_argument("--no-verdict", action="store_true",
                    help="Skip the post-sweep H1 verdict aggregation. "
                         "Used by the Anvil SLURM array (slurm/"
                         "05_a19_baselines.sbatch) where each task "
                         "trains ONE cell; the verdict runs once in "
                         "Phase D instead of per-task.")
    args = ap.parse_args()

    cfg = P4SweepConfig(
        n_points=args.n_points,
        train_steps=args.train_steps,
        rollout_steps=args.rollout_steps,
    )

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P5 matched baselines sweep — start {start}", flush=True)
    print(f"  systems  : {args.systems}", flush=True)
    print(f"  families : {args.families}", flush=True)
    print(f"  seeds    : {args.seeds}", flush=True)

    groups: dict[str, list[dict]] = {}
    for system in args.systems:
        for family in args.families:
            for seed in args.seeds:
                r = train_and_rollout_baseline_cell(
                    system, family, seed, cfg=cfg)
                _write_seed(r, out)
                key = f"{r['system']}_{r['family']}"
                groups.setdefault(key, []).append(r)
                print(
                    f"  [{system:<14} {family:<16}] seed={seed}  "
                    f"relL2={r['relative_l2']:.4f}  "
                    f"vpt={r['vpt_step']}step  "
                    f"spec={r['spectral_error']:.4f}  "
                    f"params={r['trainable_params']}",
                    flush=True)

    for key, grp in groups.items():
        (out / key).mkdir(parents=True, exist_ok=True)
        (out / key / "seeds_summary.json").write_text(
            json.dumps(summarize_p5(grp), indent=2) + "\n")

    # Sweep config + provenance
    cfg_record = {
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
        "scope_note": ("P5 matched baselines. Adds the MANDATORY "
                       "Neural-ODE H1 contrast (pre-reg §6) + capacity-"
                       "matched MLP control + known-structure skyline. "
                       "Combined with P4 QLNN forecaster data, the H1 "
                       "verdict module computes Δ_smooth−Δ_broad CI."),
    }
    (out / "config.json").write_text(
        json.dumps(cfg_record, indent=2) + "\n")
    prov = {**_git_prov(), "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "package_versions": _pkg(),
            "wall_clock_start_utc": start,
            "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z"}
    (out / "provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n")
    (out / "README.md").write_text(
        "# results/p5_matched_baselines/\n\n"
        "P5 mandatory baselines per pre-reg §6:\n"
        "  - plain_neuralode (MANDATORY H1 contrast)\n"
        "  - plain_mlp (capacity-matched classical control)\n"
        "  - skyline (structural upper bound)\n\n"
        "Combined with `results/p4_forecaster_rollout/` (QLNN data),\n"
        "the H1 verdict module computes Δ_smooth − Δ_broad CI per\n"
        "pre-reg §7. Output: `results/p5_h1_verdict/h1_analysis.json`.\n")

    if args.no_verdict:
        print("\nP5 baseline sweep done — --no-verdict set, skipping "
              "H1 aggregation (runs once in Phase D).", flush=True)
        return

    print(f"\nP5 baseline sweep done — computing H1 verdict ...", flush=True)

    # ---- Compute H1 verdict ----------------------------------------------
    cells = _build_cell_records(out, args.seeds)
    if not cells:
        print("  ERROR: no cells matched (missing P4 QLNN data?)",
              flush=True)
        return

    verdict = h1_verdict(
        cells,
        n_iter=args.bootstrap_iter,
        skyline_threshold=args.skyline_threshold,
        alpha=0.05,
    )

    verdict_out: Path = args.verdict_out
    verdict_out.mkdir(parents=True, exist_ok=True)
    (verdict_out / "h1_analysis.json").write_text(
        json.dumps(verdict, indent=2) + "\n")
    (verdict_out / "per_cell_records.json").write_text(json.dumps([
        {
            "system": c.system, "seed": c.seed,
            "regime": c.regime,
            "qlnn_relL2": c.qlnn_relL2,
            "neuralode_relL2": c.neuralode_relL2,
            "skyline_relL2": c.skyline_relL2,
            "delta": c.delta,
        }
        for c in cells
    ], indent=2) + "\n")

    print("=" * 70, flush=True)
    print(f"H1 VERDICT: {verdict['outcome']}", flush=True)
    print("=" * 70, flush=True)
    print(verdict["reasoning"], flush=True)
    if verdict["bootstrap"] is not None:
        b = verdict["bootstrap"]
        print(f"  Δ_smooth = {b['delta_smooth_mean']:.4f}", flush=True)
        print(f"  Δ_broad  = {b['delta_broad_mean']:.4f}", flush=True)
        print(f"  Δ_diff   = {b['delta_diff_mean']:.4f}", flush=True)
        print(f"  95% CI   = [{b['ci_low']:.4f}, {b['ci_high']:.4f}]",
              flush=True)
    print(f"\nResults: {verdict_out}/h1_analysis.json", flush=True)


if __name__ == "__main__":
    main()
