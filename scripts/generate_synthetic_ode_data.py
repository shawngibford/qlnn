"""Emit the canonical-5 synthetic ODE benchmark CSVs into data/synthetic/.

Each CSV is qZETA-schema (DATE + state columns) so the existing
train_baseline.py / train_qlnn.py pipeline + ansatz registry + Option-B
gate machinery consume it with zero trainer changes.

The CSVs are reproducible artifacts (this script is committed; the data
dir is gitignored, same convention as the qZETA raw data).

Usage:
    python scripts/generate_synthetic_ode_data.py
    python scripts/generate_synthetic_ode_data.py --n-points 4000 --noise 0.02
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_liquid_neuralode.data_processing.synthetic_ode import (
    SYSTEMS,
    make_ode_dataframe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "synthetic"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=4000,
                    help="trajectory length after burn-in (≈5x the 778-row "
                         "qZETA run so windowing has comparable sample count)")
    ap.add_argument("--noise", type=float, default=0.02,
                    help="Gaussian observation-noise std in state units "
                         "(0 = clean dynamics)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in SYSTEMS:
        df, target = make_ode_dataframe(
            name, n_points=args.n_points, noise_std=args.noise,
            seed=args.seed)
        csv_path = OUT / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        feature_cols = [c for c in df.columns if c != "DATE"]
        manifest[name] = {
            "csv": str(csv_path.relative_to(REPO_ROOT)),
            "rows": len(df),
            "feature_cols": feature_cols,
            "target_col": target,
            "noise_std": args.noise,
            "n_points": args.n_points,
            "seed": args.seed,
        }
        print(f"wrote {csv_path}  ({len(df)} rows, "
              f"{len(feature_cols)} state cols, target={target})")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest → {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
