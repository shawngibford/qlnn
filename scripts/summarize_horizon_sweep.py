"""Paper-style horizon-sweep table generator.

Reads multiple multi-seed run directories produced by `train_baseline.py`
(one per forecast horizon) and emits per-horizon comparison tables. The key
observation the paper wants to surface is that persistence's R² collapses
as horizon grows; the Liquid-ODE row that keeps a meaningful R² over the
same horizons is the paper claim.

Each run directory must contain:
    protocol.json        -- supplies horizon_hours (and window counts)
    baselines.json       -- persistence + linear-extrapolation metrics
    seeds_summary.json   -- mean/std across seeds for the trained model

Usage:
    python scripts/summarize_horizon_sweep.py \
        --runs results/horizon_sweep/euler_h{1,3,6,12} \
        --label "Liquid-ODE (Euler)" \
        --output results/horizon_sweep

For multiple model variants (e.g., Euler + QLNN), call with parallel run-set
arguments:
    python scripts/summarize_horizon_sweep.py \
        --runs results/horizon_sweep/euler_h1 \
               results/horizon_sweep/euler_h3 \
               results/horizon_sweep/euler_h6 \
               results/horizon_sweep/euler_h12 \
        --label "Liquid-ODE (Euler)" \
        --runs results/horizon_sweep/qlnn_h1 \
               results/horizon_sweep/qlnn_h3 \
               results/horizon_sweep/qlnn_h6 \
        --label "QLNN (hybrid)" \
        --output results/horizon_sweep

Output:
    <output>/horizon_sweep_table.md     -- paper-style markdown (R² + MAE)
    <output>/horizon_sweep_table.json   -- machine-readable blob
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(p: Path) -> dict:
    with p.open("r") as f:
        return json.load(f)


def _fmt_pm(mean: float, std: float, sig: int = 4) -> str:
    return f"{mean:.{sig}f} ± {std:.{sig}f}"


def _fmt_plain(v: float, sig: int = 4) -> str:
    return f"{v:.{sig}f}"


def _horizon_label(h: float) -> str:
    if abs(h - round(h)) < 1e-6:
        return f"h={int(round(h))}"
    return f"h={h:g}"


def _load_horizon_row(run_dir: Path) -> dict[str, Any]:
    """Pull the per-horizon record (baselines + trained model + counts)."""
    protocol = _read_json(run_dir / "protocol.json")
    baselines = _read_json(run_dir / "baselines.json")
    summary = _read_json(run_dir / "seeds_summary.json")
    return {
        "run_dir": str(run_dir),
        "horizon_hours": float(protocol["horizon_hours"]),
        "n_train": int(protocol["n_train_windows"]),
        "n_val": int(protocol["n_val_windows"]),
        "n_test": int(protocol["n_test_windows"]),
        "persistence": baselines["persistence"],
        "linear": baselines["linear"],
        "model": summary,
        "n_seeds": int(summary["n_seeds"]),
    }


def _sort_by_horizon(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: r["horizon_hours"])


def _build_metric_table(
    *,
    metric_key: str,           # "r2_raw" | "mae_raw" | "rmse_raw" | "mse_norm"
    split: str,                # "val" | "test"
    horizons: list[float],
    model_rows: list[tuple[str, list[dict]]],
    persistence_by_h: dict[float, dict],
    linear_by_h: dict[float, dict],
) -> list[str]:
    """One markdown table for a single (metric, split) pair."""
    headers = ["Model"] + [_horizon_label(h) for h in horizons]
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    def _row_for(label: str, cells: list[str]) -> str:
        return "| " + " | ".join([label] + cells) + " |"

    # Persistence
    cells = []
    for h in horizons:
        rec = persistence_by_h.get(h)
        cells.append(_fmt_plain(rec[split][metric_key]) if rec else "—")
    lines.append(_row_for("Persistence", cells))

    # Linear extrapolation
    cells = []
    for h in horizons:
        rec = linear_by_h.get(h)
        cells.append(_fmt_plain(rec[split][metric_key]) if rec else "—")
    lines.append(_row_for("Linear extrap.", cells))

    # Trained models
    for label, rows in model_rows:
        by_h = {r["horizon_hours"]: r for r in rows}
        cells = []
        for h in horizons:
            r = by_h.get(h)
            if r is None:
                cells.append("—")
                continue
            agg = r["model"][split][metric_key]
            cells.append(_fmt_pm(agg["mean"], agg["std"]))
        lines.append(_row_for(label, cells))

    return lines


def _build_window_count_table(horizons: list[float], persistence_by_h: dict[float, dict],
                              all_rows_by_h: dict[float, dict]) -> list[str]:
    headers = ["Split"] + [_horizon_label(h) for h in horizons]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for split_name, key in [("train", "n_train"), ("val", "n_val"), ("test", "n_test")]:
        cells = [str(all_rows_by_h[h][key]) if h in all_rows_by_h else "—" for h in horizons]
        lines.append("| " + " | ".join([split_name] + cells) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        action="append",
        required=True,
        help="One --runs group per model. Each group lists run dirs across horizons.",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="One label per --runs group (same order).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.runs) != len(args.label):
        raise SystemExit("--runs groups and --label entries must be 1:1 in count and order")

    args.output.mkdir(parents=True, exist_ok=True)

    # Load every (model, horizon) row.
    model_rows: list[tuple[str, list[dict]]] = []
    horizons_set: set[float] = set()
    persistence_by_h: dict[float, dict] = {}
    linear_by_h: dict[float, dict] = {}
    counts_by_h: dict[float, dict] = {}

    for run_group, label in zip(args.runs, args.label):
        rows = [_load_horizon_row(p) for p in run_group]
        rows = _sort_by_horizon(rows)
        for r in rows:
            h = r["horizon_hours"]
            horizons_set.add(h)
            # Persistence + linear are deterministic per horizon — first writer
            # wins, and subsequent rows at the same horizon must match.
            if h not in persistence_by_h:
                persistence_by_h[h] = r["persistence"]
                linear_by_h[h] = r["linear"]
                counts_by_h[h] = r
        model_rows.append((label, rows))

    horizons = sorted(horizons_set)

    # ---- Markdown ----
    md: list[str] = []
    md.append("# Horizon ablation — Liquid-ODE and friends")
    md.append("")
    md.append(
        "Per-horizon comparison of trained models against persistence and linear "
        "extrapolation. Dataset: qZETA bioreactor (778 rows); splits 70/15/15 "
        "(chronological); window_size=24, stride=1."
    )
    md.append("")
    md.append(
        "Persistence and linear-extrapolation rows are deterministic (no seeds). "
        "Model rows report mean ± std across the seeds listed in each "
        "`seeds_summary.json`. Cells marked `—` mean that horizon has no run "
        "for that model variant."
    )
    md.append("")

    md.append("## Window counts per split")
    md.append("")
    md.extend(_build_window_count_table(horizons, persistence_by_h, counts_by_h))
    md.append("")
    md.append(
        "Splits with fewer than 30 test windows yield unstable metrics — treat "
        "those columns as supplementary, not headline."
    )
    md.append("")

    md.append("## Test R² (raw OD units)")
    md.append("")
    md.extend(
        _build_metric_table(
            metric_key="r2_raw",
            split="test",
            horizons=horizons,
            model_rows=model_rows,
            persistence_by_h=persistence_by_h,
            linear_by_h=linear_by_h,
        )
    )
    md.append("")
    md.append(
        "*Key observation:* persistence R² collapses as horizon grows. "
        "Trained-model rows that hold a meaningful R² where persistence falls "
        "off are the paper's discriminating claim."
    )
    md.append("")

    md.append("## Test MAE (raw OD units)")
    md.append("")
    md.extend(
        _build_metric_table(
            metric_key="mae_raw",
            split="test",
            horizons=horizons,
            model_rows=model_rows,
            persistence_by_h=persistence_by_h,
            linear_by_h=linear_by_h,
        )
    )
    md.append("")

    md.append("## Validation R² (raw OD units)")
    md.append("")
    md.extend(
        _build_metric_table(
            metric_key="r2_raw",
            split="val",
            horizons=horizons,
            model_rows=model_rows,
            persistence_by_h=persistence_by_h,
            linear_by_h=linear_by_h,
        )
    )
    md.append("")

    md.append("## Validation MAE (raw OD units)")
    md.append("")
    md.extend(
        _build_metric_table(
            metric_key="mae_raw",
            split="val",
            horizons=horizons,
            model_rows=model_rows,
            persistence_by_h=persistence_by_h,
            linear_by_h=linear_by_h,
        )
    )
    md.append("")

    md_path = args.output / "horizon_sweep_table.md"
    md_path.write_text("\n".join(md) + "\n")

    # ---- JSON blob ----
    blob = {
        "horizons": horizons,
        "window_counts": {
            str(h): {
                "n_train": counts_by_h[h]["n_train"],
                "n_val": counts_by_h[h]["n_val"],
                "n_test": counts_by_h[h]["n_test"],
            }
            for h in horizons
        },
        "persistence": {str(h): persistence_by_h[h] for h in horizons},
        "linear": {str(h): linear_by_h[h] for h in horizons},
        "models": [
            {
                "label": label,
                "rows": [
                    {
                        "horizon_hours": r["horizon_hours"],
                        "n_seeds": r["n_seeds"],
                        "val": r["model"]["val"],
                        "test": r["model"]["test"],
                    }
                    for r in rows
                ],
            }
            for label, rows in model_rows
        ],
    }
    (args.output / "horizon_sweep_table.json").write_text(json.dumps(blob, indent=2) + "\n")

    print("\n".join(md))
    print(f"\nwrote {md_path}")
    print(f"wrote {args.output / 'horizon_sweep_table.json'}")


if __name__ == "__main__":
    main()
