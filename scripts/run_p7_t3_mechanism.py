"""P7 T3 mechanism sweep — compute the 4 T3 scalars per family.

Computes `T3Scalars` (expressibility KL, entangling Q, gradient
variance, Fourier bandwidth) for each of the 4 forecaster families
at the P4 sweep config (num_qubits=3, num_layers=1):

  - data_reuploading
  - hardware_efficient
  - strongly_entangling
  - brickwall

Plus a small qubit-scaling study for the barren-plateau scalar
(McClean 2018) so the per-family signature isn't a single-point
estimate — gradient variance is computed at n ∈ {2, 3, 4, 5}
to detect exponential decay (the barren-plateau signature) per
family at the smallest non-trivial scales we can simulate
cheaply.

Output:
  results/p7_t3_mechanism/t3_scalars.json   (per-family bundle)
  results/p7_t3_mechanism/gradient_scaling.json (BP curve per family)
  results/p7_t3_mechanism/config.json + provenance.json + README.md

Wall-clock: ~2-3 min on default.qubit (mostly expressibility, which
samples 2× n_samples random pairs per family).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from qlnn_.diagnostics.t3_mechanism import (
    compute_t3_scalars, gradient_variance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p7_t3_mechanism"

FORECASTER_FAMILIES = (
    "data_reuploading",
    "hardware_efficient",
    "strongly_entangling",
    "brickwall",
)


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+",
                    default=list(FORECASTER_FAMILIES),
                    choices=list(FORECASTER_FAMILIES))
    ap.add_argument("--n-qubits", type=int, default=3,
                    help="Match P4 SweepConfig.num_qubits (default 3)")
    ap.add_argument("--n-layers", type=int, default=1,
                    help="Match P4 SweepConfig.num_layers (default 1)")
    ap.add_argument("--n-samples", type=int, default=400,
                    help="Random samples for each diagnostic (default 400)")
    ap.add_argument("--bp-qubit-scaling", nargs="+", type=int,
                    default=[2, 3, 4, 5],
                    help="Qubit counts for the barren-plateau scaling curve")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P7 T3 mechanism sweep — start {start}", flush=True)
    print(f"  families   : {args.families}", flush=True)
    print(f"  P4 config  : num_qubits={args.n_qubits}, "
          f"num_layers={args.n_layers}", flush=True)
    print(f"  n_samples  : {args.n_samples}", flush=True)
    print(f"  BP scaling : qubits ∈ {args.bp_qubit_scaling}", flush=True)

    # ---- T3 scalars per family at the P4 config -------------------------
    per_family: dict[str, dict] = {}
    for family in args.families:
        print(f"  [{family:<20}] computing 4 T3 scalars ...", flush=True)
        scalars = compute_t3_scalars(
            family, n=args.n_qubits, L=args.n_layers,
            n_samples=args.n_samples, seed=args.seed)
        per_family[family] = dataclasses.asdict(scalars)
        print(f"    KL_to_Haar={scalars.expressibility_kl:.4f}  "
              f"Q={scalars.entangling_q:.4f}  "
              f"Var(grad)={scalars.gradient_variance:.4e}  "
              f"K_max={scalars.fourier_bandwidth}",
              flush=True)

    (out / "t3_scalars.json").write_text(
        json.dumps(per_family, indent=2) + "\n")

    # ---- Barren-plateau scaling curve (gradient variance vs n_qubits) ----
    print(f"\nBarren-plateau scaling study (gradient variance vs n_qubits):",
          flush=True)
    bp_scaling: dict[str, dict[int, float]] = {}
    for family in args.families:
        bp_scaling[family] = {}
        for n in args.bp_qubit_scaling:
            v = gradient_variance(
                family, n=n, L=args.n_layers,
                n_samples=args.n_samples, seed=args.seed)
            bp_scaling[family][n] = v
            print(f"  [{family:<20}] n={n}  Var(grad)={v:.4e}",
                  flush=True)
    (out / "gradient_scaling.json").write_text(
        json.dumps({fam: {str(n): v for n, v in d.items()}
                    for fam, d in bp_scaling.items()}, indent=2) + "\n")

    # ---- Provenance + config + README ----------------------------------
    cfg_record = {
        "families": list(args.families),
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "n_samples": args.n_samples,
        "bp_qubit_scaling": list(args.bp_qubit_scaling),
        "seed": args.seed,
        "scope_note": ("P7 H3 mechanism: 4 T3 scalars per forecaster "
                       "family at the P4 config. Cross-tabulated "
                       "against P5's per-cell Δ values to identify "
                       "the circuit property that best predicts the "
                       "inverted regime advantage gap. NOT yet H3 "
                       "evidence; the cross-tab + correlation analysis "
                       "lands in the next P7 sub-commit."),
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
        "# results/p7_t3_mechanism/\n\n"
        "P7 H3 mechanism: 4 T3 diagnostics per forecaster family at\n"
        "the P4 sweep config (num_qubits=3, num_layers=1). Headline\n"
        "output:\n\n"
        "  - `t3_scalars.json`: per-family\n"
        "      {expressibility_kl, entangling_q, gradient_variance,\n"
        "       fourier_bandwidth}\n"
        "  - `gradient_scaling.json`: BP scaling vs n_qubits per family\n\n"
        "These scalars are cross-tabulated against P5's per-cell Δ\n"
        "values (results/p5_h1_verdict/per_cell_records.json) by\n"
        "scripts/make_p7_mechanism_figure.py to surface the property\n"
        "that best predicts the inverted regime-dependent advantage.\n")
    print(f"\nP7 T3 mechanism sweep done — wrote → {out}/", flush=True)


if __name__ == "__main__":
    main()
