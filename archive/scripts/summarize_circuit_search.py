"""Aggregate per-axis circuit-search results into a paper-style table + JSON.

Discovers every run directory under `results/circuit_search/*/seeds_summary.json`,
reads the matching YAML config to recover the (axis, level, ansatz_name)
membership, and emits:

    results/circuit_search/circuit_search_table.md       — paper-ready table
    results/circuit_search/circuit_search_table.json     — machine-readable
    results/circuit_search/circuit_search_table.csv      — long-form

The figure side (per-axis effect plots, Pareto) is in
`scripts/make_paper_figures.py`.

Usage:
    python scripts/summarize_circuit_search.py
    python scripts/summarize_circuit_search.py \\
        --extra-results results/circuit_search_promoted \\
        --extra-configs configs/circuit_search_promoted
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "circuit_search"
CONFIGS = REPO_ROOT / "configs" / "circuit_search"
_EXTRA_CONFIG_DIRS: list[Path] = []


def _safe_load(path: Path) -> Any:
    with path.open() as f:
        if path.suffix == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def _resolve_membership(stem: str) -> dict[str, Any]:
    """Read the matching YAML in `configs/circuit_search/` (or any
    `--extra-configs` dir) and pull the `circuit_search` meta-block plus the
    ansatz selection so the row can be plotted on the right axis.
    """
    candidates = [CONFIGS / f"{stem}.yaml"] + [d / f"{stem}.yaml" for d in _EXTRA_CONFIG_DIRS]
    yaml_path = next((p for p in candidates if p.exists()), None)
    if yaml_path is None:
        # Optuna trials or hand-added runs may not have a config — degrade
        # gracefully so they still show up in the long-form CSV.
        return {
            "axis": "unknown",
            "level": stem,
            "is_reference": False,
            "ansatz_name": "unknown",
            "num_qubits": -1,
            "num_layers": -1,
        }
    y = _safe_load(yaml_path)
    meta = y.get("circuit_search", {}) or {}
    model = y.get("model", {}) or {}
    ansatz = model.get("ansatz") or {}
    return {
        "axis": meta.get("axis", "unknown"),
        "level": meta.get("level", stem),
        "is_reference": bool(meta.get("is_reference", False)),
        "ansatz_name": ansatz.get("name", "data_reuploading"),
        "num_qubits": int(model.get("num_qubits", -1)),
        "num_layers": int(model.get("num_layers", -1)),
    }


def _collect_row(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "seeds_summary.json"
    if not summary_path.exists():
        return None
    summary = _safe_load(summary_path)
    test = summary.get("test", {})
    val = summary.get("val", {})

    def _get(d: dict, key: str) -> dict[str, float | None]:
        v = d.get(key, {}) or {}
        return {
            "mean": v.get("mean"),
            "std": v.get("std"),
            "ci95_half_width": v.get("ci95_half_width"),
        }

    info = _resolve_membership(run_dir.name)
    return {
        "run": run_dir.name,
        "axis": info["axis"],
        "level": info["level"],
        "is_reference": info["is_reference"],
        "ansatz_name": info["ansatz_name"],
        "num_qubits": info["num_qubits"],
        "num_layers": info["num_layers"],
        "n_seeds": int(summary.get("n_seeds", 0)),
        "test_mae_raw": _get(test, "mae_raw"),
        "test_rmse_raw": _get(test, "rmse_raw"),
        "test_r2_raw": _get(test, "r2_raw"),
        "test_delta_r2_raw": _get(test, "delta_r2_raw"),
        "val_mse_norm": _get(val, "mse_norm"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-results", type=Path, action="append", default=[],
                        help="Extra results-dir to scan (e.g. results/circuit_search_promoted). "
                             "Repeatable.")
    parser.add_argument("--extra-configs", type=Path, action="append", default=[],
                        help="Extra configs-dir to scan when resolving the YAML for a "
                             "matching run. Repeatable; pair with --extra-results.")
    args = parser.parse_args()
    _EXTRA_CONFIG_DIRS.extend(d if d.is_absolute() else REPO_ROOT / d
                              for d in args.extra_configs)

    if not RESULTS.exists():
        print(f"no results in {RESULTS}/ — run scripts/run_circuit_search.sh first")
        return
    run_dirs = sorted(p for p in RESULTS.iterdir() if p.is_dir())
    for extra in args.extra_results:
        extra_path = extra if extra.is_absolute() else REPO_ROOT / extra
        if extra_path.exists():
            run_dirs += sorted(p for p in extra_path.iterdir() if p.is_dir())
    rows = [r for r in (_collect_row(rd) for rd in run_dirs) if r is not None]
    if not rows:
        print(f"no seeds_summary.json under {RESULTS}/*")
        return

    # 1. JSON
    out_json = RESULTS / "circuit_search_table.json"
    out_json.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out_json}")

    # 2. Markdown
    md = ["# Circuit search — per-axis ablation (proxy budget: single seed, h=3)\n"]
    md.append("Each row is a 1-seed run at the locked h=3 evaluation. The "
              "*reference* row is the historical 4q/3L data-reuploading/ring/RX "
              "circuit (matches `results/qlnn_hybrid_h3/seed_0/`).\n")
    md.append("| Axis | Level | Ansatz | Q | L | Test MAE | Test RMSE | Test R² | Test ΔOD R² |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    # Sort: reference first, then by axis, then by level.
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            0 if r["is_reference"] else 1,
            r["axis"],
            r["level"],
        ),
    )
    for r in rows_sorted:
        def _fmt(d):
            mu = d["mean"]
            return "—" if mu is None else f"{mu:.4f}"
        md.append(
            f"| {r['axis']} | {r['level']}{' ★' if r['is_reference'] else ''} | "
            f"{r['ansatz_name']} | {r['num_qubits']} | {r['num_layers']} | "
            f"{_fmt(r['test_mae_raw'])} | {_fmt(r['test_rmse_raw'])} | "
            f"{_fmt(r['test_r2_raw'])} | {_fmt(r['test_delta_r2_raw'])} |"
        )
    out_md = RESULTS / "circuit_search_table.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")

    # 3. CSV
    out_csv = RESULTS / "circuit_search_table.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run", "axis", "level", "is_reference", "ansatz_name",
            "num_qubits", "num_layers", "n_seeds",
            "test_mae_raw_mean", "test_mae_raw_std",
            "test_rmse_raw_mean", "test_r2_raw_mean", "test_delta_r2_raw_mean",
            "val_mse_norm_mean",
        ])
        for r in rows_sorted:
            w.writerow([
                r["run"], r["axis"], r["level"], r["is_reference"], r["ansatz_name"],
                r["num_qubits"], r["num_layers"], r["n_seeds"],
                r["test_mae_raw"]["mean"], r["test_mae_raw"]["std"],
                r["test_rmse_raw"]["mean"], r["test_r2_raw"]["mean"],
                r["test_delta_r2_raw"]["mean"], r["val_mse_norm"]["mean"],
            ])
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
