"""Emit the PDE hardness-ladder field artifacts into data/pde/.

Each artifact is an .npz FIELD (u[t,x] + grids + IC + periodic-BC tag +
conserved invariants + params + a sha256 provenance hash) — NOT the
qZETA-CSV scalar schema. The scalar seam is deliberately blocked for
PDEs (see ODE_PDE_PRE_REG.md §4 / pde_systems.py docstring); downstream
solver/forecaster code consumes the field directly.

Reproducible artifact convention (same as generate_synthetic_ode_data):
this script is committed; data/ is gitignored. The npz `sha256` is the
per-dataset lock — record it before any P6 run so the field a model
trained on is provably the field this script produced.

Usage:
    python scripts/generate_pde_data.py
    python scripts/generate_pde_data.py --systems kdv burgers_shock
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_liquid_neuralode.data_processing.pde_systems import (
    PDE_SYSTEMS,
    make_pde_npz,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "pde"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(PDE_SYSTEMS),
                    choices=list(PDE_SYSTEMS),
                    help="which PDE systems to emit (default: all 4)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in args.systems:
        npz_path = OUT / f"{name}.npz"
        meta = make_pde_npz(name, str(npz_path))
        meta["npz"] = str(npz_path.relative_to(REPO_ROOT))
        manifest[name] = meta
        print(f"wrote {npz_path}  regime={meta['regime']}  "
              f"grid={meta['n_x']}×{meta['n_steps'] // meta['sample_every'] + 1}"
              f"  sha256={meta['sha256'][:12]}…")

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"\nmanifest → {OUT / 'manifest.json'}  "
          f"({len(manifest)} PDE field artifacts)")


if __name__ == "__main__":
    main()
