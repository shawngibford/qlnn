"""Generate a paper-ready comparison table across baseline runs.

Reads `seeds_summary.json` from one or more multi-seed baseline directories and
emits a single markdown table plus a JSON blob suitable for the paper's
"classical baseline" section.

Usage:
    python scripts/summarize_baselines.py \\
        --runs results/baseline_classical_euler results/baseline_classical_dopri5 \\
        --runs results/baseline_classical_physics \\
        --labels "Liquid-ODE (Euler)" "Liquid-ODE (dopri5)" "+physics" \\
        --output results/baseline_classical_table

It always pulls baselines.json from the FIRST run, since the persistence /
linear baselines are deterministic and identical across all runs (same data).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(p: Path) -> dict:
    with p.open("r") as f:
        return json.load(f)


def _fmt(mean: float, std: float, *, sig: int = 4) -> str:
    return f"{mean:.{sig}f} ± {std:.{sig}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, type=Path,
                        help="Directories produced by train_baseline.py")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="One label per run (same order).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for the markdown table + JSON blob.")
    args = parser.parse_args()

    if len(args.runs) != len(args.labels):
        raise SystemExit("--runs and --labels must have matching lengths")

    args.output.mkdir(parents=True, exist_ok=True)

    base = _read_json(args.runs[0] / "baselines.json")

    rows: list[dict] = []
    rows.append({
        "model": "Persistence (OD(t+h)=OD(t))",
        "val_mae": f"{base['persistence']['val']['mae_raw']:.4f}",
        "val_rmse": f"{base['persistence']['val']['rmse_raw']:.4f}",
        "val_r2": f"{base['persistence']['val']['r2_raw']:.4f}",
        "test_mae": f"{base['persistence']['test']['mae_raw']:.4f}",
        "test_rmse": f"{base['persistence']['test']['rmse_raw']:.4f}",
        "test_r2": f"{base['persistence']['test']['r2_raw']:.4f}",
        "n_seeds": "n/a",
    })
    rows.append({
        "model": "Linear extrapolation",
        "val_mae": f"{base['linear']['val']['mae_raw']:.4f}",
        "val_rmse": f"{base['linear']['val']['rmse_raw']:.4f}",
        "val_r2": f"{base['linear']['val']['r2_raw']:.4f}",
        "test_mae": f"{base['linear']['test']['mae_raw']:.4f}",
        "test_rmse": f"{base['linear']['test']['rmse_raw']:.4f}",
        "test_r2": f"{base['linear']['test']['r2_raw']:.4f}",
        "n_seeds": "n/a",
    })

    for run_dir, label in zip(args.runs, args.labels):
        s = _read_json(run_dir / "seeds_summary.json")
        rows.append({
            "model": label,
            "val_mae": _fmt(s["val"]["mae_raw"]["mean"], s["val"]["mae_raw"]["std"]),
            "val_rmse": _fmt(s["val"]["rmse_raw"]["mean"], s["val"]["rmse_raw"]["std"]),
            "val_r2": _fmt(s["val"]["r2_raw"]["mean"], s["val"]["r2_raw"]["std"]),
            "test_mae": _fmt(s["test"]["mae_raw"]["mean"], s["test"]["mae_raw"]["std"]),
            "test_rmse": _fmt(s["test"]["rmse_raw"]["mean"], s["test"]["rmse_raw"]["std"]),
            "test_r2": _fmt(s["test"]["r2_raw"]["mean"], s["test"]["r2_raw"]["std"]),
            "n_seeds": str(s["n_seeds"]),
        })

    # Markdown table
    md_lines: list[str] = []
    md_lines.append("# Classical baseline — 1h-ahead OD forecast")
    md_lines.append("")
    md_lines.append("Dataset: qZETA bioreactor (778 rows). Splits: train=70%, val=15%, test=15% (chronological).")
    md_lines.append("Window: 24 steps, horizon: 1 h. OD scaling: fixed MinMax [0.0, 3.8].")
    md_lines.append("Metrics in raw OD units (MAE/RMSE) or unitless (R²). Mean ± std across seeds.")
    md_lines.append("")
    md_lines.append("| Model | val MAE | val RMSE | val R² | test MAE | test RMSE | test R² | seeds |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['val_mae']} | {r['val_rmse']} | {r['val_r2']} | "
            f"{r['test_mae']} | {r['test_rmse']} | {r['test_r2']} | {r['n_seeds']} |"
        )

    md_path = args.output / "baseline_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    # JSON blob (paper-import-friendly)
    blob = {
        "rows": rows,
        "runs": [str(p) for p in args.runs],
        "labels": list(args.labels),
    }
    (args.output / "baseline_table.json").write_text(json.dumps(blob, indent=2) + "\n")

    print("\n".join(md_lines))
    print(f"\nwrote {md_path}")


if __name__ == "__main__":
    main()
