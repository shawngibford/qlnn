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
import csv
import json
import re
from pathlib import Path


def _read_json(p: Path) -> dict:
    with p.open("r") as f:
        return json.load(f)


def _fmt(mean: float, std: float, *, sig: int = 4) -> str:
    return f"{mean:.{sig}f} ± {std:.{sig}f}"


def _fmt_ci(field: dict, *, sig: int = 4) -> str:
    """Format ``mean ± ci95_half_width`` (preferred), falling back to ``mean ± std``.

    seeds_summary.json gained ``ci95_half_width`` once aggregate_seed_metrics
    started emitting the t-distribution CI alongside std (R3 Tier 2.2). Older
    JSONs lack that field — fall back to std so this script keeps reading
    legacy runs.
    """
    mean = field["mean"]
    hw = field.get("ci95_half_width")
    if hw is None or (isinstance(hw, float) and (hw != hw)):  # None or NaN
        return f"{mean:.{sig}f} ± {field['std']:.{sig}f}"
    return f"{mean:.{sig}f} ± {hw:.{sig}f}"


_SEED_DIR_RE = re.compile(r"^seed_(-?\d+)$")


def _seed_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(seed_int, seed_dir), ...] sorted by seed_int.

    Numeric sort so seed_10 doesn't precede seed_2 lexicographically.
    """
    out: list[tuple[int, Path]] = []
    if not run_dir.exists():
        return out
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = _SEED_DIR_RE.match(child.name)
        if not m:
            continue
        if not (child / "metrics.json").exists():
            continue
        out.append((int(m.group(1)), child))
    out.sort(key=lambda pair: pair[0])
    return out


# Long-form per-seed table columns, in emission order.
_PER_SEED_COLUMNS: tuple[str, ...] = (
    "model",
    "seed",
    "best_epoch",
    "val_mse_norm",
    "val_mae_raw",
    "val_rmse_raw",
    "val_r2_raw",
    "test_mse_norm",
    "test_mae_raw",
    "test_rmse_raw",
    "test_r2_raw",
)


def _collect_per_seed_rows(run_dir: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    for seed, seed_dir in _seed_dirs(run_dir):
        m = _read_json(seed_dir / "metrics.json")
        val = m.get("val", {})
        test = m.get("test", {})
        rows.append({
            "model": label,
            "seed": seed,
            "best_epoch": m.get("best_epoch", ""),
            "val_mse_norm": val.get("mse_norm", ""),
            "val_mae_raw": val.get("mae_raw", ""),
            "val_rmse_raw": val.get("rmse_raw", ""),
            "val_r2_raw": val.get("r2_raw", ""),
            "test_mse_norm": test.get("mse_norm", ""),
            "test_mae_raw": test.get("mae_raw", ""),
            "test_rmse_raw": test.get("rmse_raw", ""),
            "test_r2_raw": test.get("r2_raw", ""),
        })
    return rows


def _format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_per_seed_tables(output_dir: Path, per_seed_rows: list[dict]) -> None:
    csv_path = output_dir / "per_seed_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_PER_SEED_COLUMNS))
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow({col: row.get(col, "") for col in _PER_SEED_COLUMNS})

    md_lines: list[str] = []
    md_lines.append("# Per-seed results (supplementary)")
    md_lines.append("")
    md_lines.append(
        "Full per-seed metrics for every run aggregated by `baseline_table.md`. "
        "Use this for reviewer-side CIs and paired analyses."
    )
    md_lines.append("")
    md_lines.append("| " + " | ".join(_PER_SEED_COLUMNS) + " |")
    md_lines.append("|" + "|".join(["---"] * len(_PER_SEED_COLUMNS)) + "|")
    for row in per_seed_rows:
        md_lines.append(
            "| " + " | ".join(_format_cell(row.get(col, "")) for col in _PER_SEED_COLUMNS) + " |"
        )
    (output_dir / "per_seed_table.md").write_text("\n".join(md_lines) + "\n")


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

    per_seed_rows: list[dict] = []
    for run_dir, label in zip(args.runs, args.labels):
        s = _read_json(run_dir / "seeds_summary.json")
        per_seed_rows.extend(_collect_per_seed_rows(run_dir, label))
        rows.append({
            "model": label,
            # Prefer 95% t-CI half-width (R3 Tier 2.2). Falls back to bare std
            # silently for legacy seeds_summary.json files that pre-date the
            # CI-emitting aggregator.
            "val_mae": _fmt_ci(s["val"]["mae_raw"]),
            "val_rmse": _fmt_ci(s["val"]["rmse_raw"]),
            "val_r2": _fmt_ci(s["val"]["r2_raw"]),
            "test_mae": _fmt_ci(s["test"]["mae_raw"]),
            "test_rmse": _fmt_ci(s["test"]["rmse_raw"]),
            "test_r2": _fmt_ci(s["test"]["r2_raw"]),
            "n_seeds": str(s["n_seeds"]),
        })

    # Markdown table
    md_lines: list[str] = []
    md_lines.append("# Classical baseline — 1h-ahead OD forecast")
    md_lines.append("")
    md_lines.append("Dataset: qZETA bioreactor (778 rows). Splits: train=70%, val=15%, test=15% (chronological).")
    md_lines.append("Window: 24 steps, horizon: 1 h. OD scaling: fixed MinMax [0.0, 3.8].")
    md_lines.append(
        "Metrics in raw OD units (MAE/RMSE) or unitless (R²). "
        "Mean ± 95% t-CI half-width across seeds (legacy runs without CI metadata "
        "fall back to mean ± std). For head-to-head significance, see the "
        "paired-bootstrap reports under `results/paired_comparison_*`."
    )
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

    _write_per_seed_tables(args.output, per_seed_rows)

    print("\n".join(md_lines))
    print(f"\nwrote {md_path}")
    print(f"wrote {args.output / 'per_seed_table.csv'} ({len(per_seed_rows)} rows)")
    print(f"wrote {args.output / 'per_seed_table.md'}")


if __name__ == "__main__":
    main()
