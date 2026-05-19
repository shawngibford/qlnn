"""Per-dataset baseline locks — so the Option-B gates mean the same
thing on every dataset in the unified matrix.

The original results/baseline_lock.json hard-codes bioreactor numbers
(G1 MAE < 0.2594, G2 σ ≤ 0.00831). For a cross-dataset comparison each
dataset needs its OWN classical reference, derived identically:

  G1 accuracy bar  = classical matched-param (H=4) test MAE on that dataset
  G2 σ gate        = 0.5 · σ(classical H=4 test MAE)  (Claim-1 ≥2× rule)
  best classical   = min classical MAE across the H-sweep (hardest bar)

Scans results/unified_matrix/<dataset>__classical_H*/seeds_summary.json
and emits results/unified_matrix/baseline_lock__<dataset>.json with the
same schema as the canonical lock, so scripts/check_circuit_regression.py
generalizes per dataset (pass --lock <path>).

GATED: run only after the classical models for a dataset have trained.

Usage:
    python scripts/build_dataset_baseline_locks.py            # all datasets
    python scripts/build_dataset_baseline_locks.py --dataset lorenz_m472
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "unified_matrix"
MANIFEST = REPO_ROOT / "configs" / "unified_matrix" / "matrix_manifest.json"
CLASSICAL_H = [2, 4, 8, 16, 32]


def _summary(stem: str) -> dict | None:
    p = RESULTS / stem / "seeds_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _lock_for(dataset: str) -> dict | None:
    cells = {}
    for h in CLASSICAL_H:
        s = _summary(f"{dataset}__classical_H{h}")
        if s is None:
            continue
        m = s["test"]["mae_raw"]
        cells[h] = {"mean": m["mean"], "std": m["std"],
                    "n_seeds": s.get("n_seeds")}
    if 4 not in cells:
        print(f"  [skip] {dataset}: classical_H4 not trained yet "
              f"({len(cells)}/{len(CLASSICAL_H)} classical cells present)")
        return None
    h4 = cells[4]
    sigma_cl = h4["std"]
    best_h = min(cells, key=lambda k: cells[k]["mean"])
    return {
        "dataset": dataset,
        "eval_protocol": "locked h=3, train-only MinMax, window 24",
        "classical": {
            "matched_param_H4": {"params": 90, "test_mae": h4},
            "best_cell": {"hidden_size": best_h,
                          "test_mae": cells[best_h]},
            "all_cells": cells,
        },
        "claim1": {
            "sigma_classical_H4": sigma_cl,
            "g2_sigma_gate": 0.5 * sigma_cl,
        },
        "option_b_gates": {
            "G1_accuracy": f"candidate test MAE < {h4['mean']:.6f} "
                           f"(classical H=4 on {dataset})",
            "G2_reproducibility": f"candidate σ ≤ {0.5 * sigma_cl:.6f}",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="single dataset key; default = all in the manifest")
    args = ap.parse_args()
    if not MANIFEST.exists():
        raise SystemExit("run scripts/generate_unified_matrix.py first")
    datasets = (json.loads(MANIFEST.read_text())["datasets"]
                if args.dataset is None else [args.dataset])
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 0
    for ds in datasets:
        lock = _lock_for(ds)
        if lock is None:
            continue
        out = RESULTS / f"baseline_lock__{ds}.json"
        out.write_text(json.dumps(lock, indent=2) + "\n")
        g1 = lock["classical"]["matched_param_H4"]["test_mae"]["mean"]
        g2 = lock["claim1"]["g2_sigma_gate"]
        print(f"  wrote {out.name}  G1<{g1:.4f}  G2≤{g2:.5f}")
        n += 1
    print(f"\n{n} per-dataset baseline lock(s) written.")


if __name__ == "__main__":
    main()
