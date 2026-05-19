"""Pick the top-K circuits from the combined Phase-2 grid + Phase-3 Optuna
results and emit full-protocol promotion YAMLs into
`configs/circuit_search_promoted/`.

Selection: best (lowest) **test MAE** at h=3 — strictly proxy-budget numbers,
since Phase 4 promotion is exactly the step where we *verify* the proxy
ranking with the full 5-seed locked protocol.

For each promoted config, this script writes a YAML that:
- copies the winning circuit's ansatz config verbatim
- sets `seeds: [0, 1, 2, 3, 4]` (locked protocol)
- carries the `circuit_search.is_promoted: true` meta-flag so
  `summarize_circuit_search.py` can tag the row

Skips circuits that are already in the locked headline:
- `data_reuploading 4q/3L ring rx` is `results/qlnn_hybrid_h3/` and doesn't
  need re-promoting.

Usage:
    python scripts/promote_top_circuits.py --top 3
    python scripts/promote_top_circuits.py --top 5 --include-optuna
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE2_TABLE = REPO_ROOT / "results" / "circuit_search" / "circuit_search_table.json"
OPTUNA_DIR = REPO_ROOT / "results" / "circuit_search_optuna"
OUT_CONFIGS = REPO_ROOT / "configs" / "circuit_search_promoted"
OUT_CONFIGS.mkdir(parents=True, exist_ok=True)


def _load_phase2() -> list[dict[str, Any]]:
    if not PHASE2_TABLE.exists():
        return []
    rows = json.loads(PHASE2_TABLE.read_text())
    # Normalize the row shape with an explicit `source` tag.
    for r in rows:
        r["source"] = "phase2"
    return rows


def _load_optuna_trials() -> list[dict[str, Any]]:
    if not OPTUNA_DIR.exists():
        return []
    rows = []
    for trial_dir in sorted(OPTUNA_DIR.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
            continue
        summary_path = trial_dir / "seeds_summary.json"
        if not summary_path.exists():
            continue
        s = json.loads(summary_path.read_text())
        # The matching `circuit_search` YAML was written to a tempfile so it
        # isn't on disk anymore. The optuna study DB has the spec — but
        # we also persisted it via Optuna's user_attrs on each trial. To
        # avoid coupling to the SQLite DB here, infer from the run's saved
        # config.json (train_qlnn.py snapshots it on every run).
        cfg_path = trial_dir / "config.json"
        ansatz = {"name": "data_reuploading", "params": {}}
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            ansatz_block = cfg.get("model", {}).get("ansatz") or ansatz
            ansatz = {
                "name": ansatz_block.get("name", "data_reuploading"),
                "params": ansatz_block.get("params") or {},
            }
            num_qubits = int(cfg.get("model", {}).get("num_qubits", 4))
            num_layers = int(cfg.get("model", {}).get("num_layers", 3))
        else:
            num_qubits, num_layers = 4, 3
        rows.append({
            "run": trial_dir.name,
            "source": "optuna",
            "axis": "optuna",
            "level": trial_dir.name,
            "is_reference": False,
            "ansatz_name": ansatz["name"],
            "ansatz_params": ansatz["params"],
            "num_qubits": num_qubits,
            "num_layers": num_layers,
            "test_mae_raw": {"mean": s["test"]["mae_raw"]["mean"]},
            "test_r2_raw": {"mean": s["test"]["r2_raw"]["mean"]},
        })
    return rows


def _phase2_ansatz_params(stem: str) -> dict[str, Any]:
    """Read the source YAML for a Phase-2 row to recover its full ansatz
    params (the JSON table only has the `name`)."""
    yp = REPO_ROOT / "configs" / "circuit_search" / f"{stem}.yaml"
    if not yp.exists():
        return {}
    y = yaml.safe_load(yp.read_text())
    return (y.get("model", {}).get("ansatz") or {}).get("params") or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--include-optuna", action="store_true",
                        help="Also consider Optuna trials, not just the Phase 2 grid.")
    parser.add_argument("--include-reference", action="store_true",
                        help="Don't skip the reference cell (default: skipped because "
                             "results/qlnn_hybrid_h3/ already has it with 5 seeds).")
    args = parser.parse_args()

    rows = _load_phase2()
    if args.include_optuna:
        rows += _load_optuna_trials()
    if not rows:
        raise SystemExit("no results found in either results/circuit_search/ or "
                         "results/circuit_search_optuna/")

    # Hydrate phase-2 rows with their ansatz params (they're only in the YAML).
    for r in rows:
        if r["source"] == "phase2" and "ansatz_params" not in r:
            r["ansatz_params"] = _phase2_ansatz_params(r["run"])

    if not args.include_reference:
        rows = [r for r in rows if not r.get("is_reference", False)]

    # Sort by lowest test MAE.
    rows.sort(key=lambda r: r["test_mae_raw"]["mean"])

    print(f"=== top {args.top} circuits by test MAE (proxy budget) ===")
    template_yaml = REPO_ROOT / "configs" / "circuit_search" / "reference.yaml"
    template = yaml.safe_load(template_yaml.read_text())

    for rank, r in enumerate(rows[:args.top], 1):
        cfg = json.loads(json.dumps(template))  # deep copy via JSON
        cfg["circuit_search"] = {
            "axis": "promoted",
            "level": f"top{rank}_{r['ansatz_name']}_Q{r['num_qubits']}_L{r['num_layers']}",
            "is_reference": False,
            "is_promoted": True,
            "source_run": r["run"],
            "proxy_test_mae": r["test_mae_raw"]["mean"],
        }
        cfg["model"]["num_qubits"] = r["num_qubits"]
        cfg["model"]["num_layers"] = r["num_layers"]
        cfg["model"]["ansatz"] = {
            "name": r["ansatz_name"],
            "params": r.get("ansatz_params") or {},
        }
        cfg["seeds"] = [0, 1, 2, 3, 4]  # locked protocol

        stem = f"top{rank}_{r['ansatz_name']}_Q{r['num_qubits']}_L{r['num_layers']}"
        out = OUT_CONFIGS / f"{stem}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"  rank {rank}: {r['ansatz_name']} (Q={r['num_qubits']}, L={r['num_layers']}) "
              f"proxy_test_mae={r['test_mae_raw']['mean']:.4f}  -> {out}")


if __name__ == "__main__":
    main()
