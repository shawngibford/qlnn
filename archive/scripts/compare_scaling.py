"""Side-by-side comparison of two scaling-mode runs.

Used to sanity-check the R3 finding 6 fix: did switching from the fixed
``od_max=3.8`` scaler to a train-only OD scaler materially change reported
metrics? If deltas are within seed noise, the original (leaky) numbers can
be cited with a footnote confirming the result is robust. If deltas are
large, the corrected numbers must replace the originals in the paper.

Usage::

    python scripts/compare_scaling.py \\
        --train-only-dir /tmp/qlnn_trainonly_smoke \\
        --fixed-dir       /tmp/qlnn_fixed_smoke \\
        --output          /tmp/scaling_compare.md

Prints a markdown table (also writable to a file via ``--output``).

The two run directories must have the canonical layout produced by
``scripts/train_baseline.py`` / ``scripts/train_qlnn.py`` — at minimum:

    <dir>/seeds_summary.json
    <dir>/protocol.json
    <dir>/baselines.json

If either file is missing, the script bails with a clear error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Metrics we report deltas for. Keep aligned with ForecastMetrics.
METRIC_FIELDS = ("mse_norm", "mae_raw", "rmse_raw", "r2_raw")
SPLITS = ("val", "test")


def _load_json(p: Path) -> dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(f"required file missing: {p}")
    return json.loads(p.read_text())


def _fmt(x: float | None, *, prec: int = 6) -> str:
    if x is None:
        return "—"
    if x != x:  # NaN
        return "NaN"
    return f"{x:.{prec}f}"


def _gather_run(run_dir: Path) -> dict[str, Any]:
    summary = _load_json(run_dir / "seeds_summary.json")
    protocol = _load_json(run_dir / "protocol.json")
    baselines = _load_json(run_dir / "baselines.json")
    return {
        "dir": run_dir,
        "summary": summary,
        "protocol": protocol,
        "baselines": baselines,
    }


def _metric_mean(summary: dict[str, Any], split: str, field: str) -> float | None:
    block = summary.get(split, {}).get(field)
    if block is None:
        return None
    return float(block["mean"])


def _metric_std(summary: dict[str, Any], split: str, field: str) -> float | None:
    block = summary.get(split, {}).get(field)
    if block is None:
        return None
    s = block.get("std")
    return None if s is None else float(s)


def _table_models(run_a: dict[str, Any], run_b: dict[str, Any]) -> str:
    """Markdown table comparing the *model* (seeds_summary) rows.

    Columns: metric | split | train_only mean±std | fixed mean±std | Δ | Δ/σ_pooled
    """
    lines = [
        "## Model metrics (mean ± unbiased std across seeds)",
        "",
        "| Metric | Split | Train-only (A) | Fixed [0, od_max] (B) | Δ = A − B | Δ / σ̂_pooled |",
        "|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        for field in METRIC_FIELDS:
            a_mean = _metric_mean(run_a["summary"], split, field)
            b_mean = _metric_mean(run_b["summary"], split, field)
            a_std = _metric_std(run_a["summary"], split, field)
            b_std = _metric_std(run_b["summary"], split, field)

            a_cell = "—" if a_mean is None else f"{_fmt(a_mean)} ± {_fmt(a_std)}"
            b_cell = "—" if b_mean is None else f"{_fmt(b_mean)} ± {_fmt(b_std)}"

            if a_mean is None or b_mean is None:
                delta = "—"
                z = "—"
            else:
                d = a_mean - b_mean
                delta = _fmt(d)
                # Pooled std (assume independent runs, equal-ish seed counts).
                if a_std is not None and b_std is not None and (a_std > 0 or b_std > 0):
                    import math
                    pooled = math.sqrt((a_std ** 2 + b_std ** 2) / 2.0)
                    z = _fmt(d / pooled, prec=2) if pooled > 0 else "inf"
                else:
                    z = "—"

            lines.append(
                f"| {field} | {split} | {a_cell} | {b_cell} | {delta} | {z} |"
            )
    return "\n".join(lines)


def _table_baselines(run_a: dict[str, Any], run_b: dict[str, Any]) -> str:
    lines = [
        "## Deterministic baselines (persistence, linear)",
        "",
        "| Model | Metric | Split | Train-only (A) | Fixed (B) | Δ = A − B |",
        "|---|---|---|---|---|---|",
    ]
    for base in ("persistence", "linear"):
        for split in SPLITS:
            for field in METRIC_FIELDS:
                a = run_a["baselines"].get(base, {}).get(split, {}).get(field)
                b = run_b["baselines"].get(base, {}).get(split, {}).get(field)
                delta = None if (a is None or b is None) else (float(a) - float(b))
                lines.append(
                    f"| {base} | {field} | {split} | {_fmt(a)} | {_fmt(b)} | {_fmt(delta)} |"
                )
    return "\n".join(lines)


def _protocol_summary(run_a: dict[str, Any], run_b: dict[str, Any]) -> str:
    pa = run_a["protocol"]
    pb = run_b["protocol"]
    return "\n".join(
        [
            "## Protocol",
            "",
            f"- Run A (train-only scaler): `{run_a['dir']}`",
            f"  - od_scaler_mode = `{pa.get('od_scaler_mode')}`, "
            f"data_min = {pa.get('od_data_min')}, data_max = {pa.get('od_data_max')}, "
            f"phys_max = {pa.get('od_phys_max')}",
            f"  - n_train_windows = {pa.get('n_train_windows')}, "
            f"n_val_windows = {pa.get('n_val_windows')}, "
            f"n_test_windows = {pa.get('n_test_windows')}",
            "",
            f"- Run B (fixed-bounds scaler): `{run_b['dir']}`",
            f"  - od_scaler_mode = `{pb.get('od_scaler_mode')}`, "
            f"data_min = {pb.get('od_data_min')}, data_max = {pb.get('od_data_max')}, "
            f"phys_max = {pb.get('od_phys_max')}",
            f"  - n_train_windows = {pb.get('n_train_windows')}, "
            f"n_val_windows = {pb.get('n_val_windows')}, "
            f"n_test_windows = {pb.get('n_test_windows')}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two training runs that differ only in OD scaler mode "
            "(train-only vs fixed [0, od_max])."
        )
    )
    parser.add_argument(
        "--train-only-dir",
        type=Path,
        required=True,
        help="Run dir where od_max=null (train-only OD scaler).",
    )
    parser.add_argument(
        "--fixed-dir",
        type=Path,
        required=True,
        help="Run dir where od_max is a fixed number (legacy / sensitivity).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, also write the markdown report to this path.",
    )
    args = parser.parse_args()

    run_a = _gather_run(args.train_only_dir)
    run_b = _gather_run(args.fixed_dir)

    parts = [
        "# OD-scaler sensitivity analysis",
        "",
        "Run A = train-only OD scaler (closes R3 finding 6 soft-leakage).",
        "Run B = fixed [0, od_max] OD scaler (legacy / domain-prior comparator).",
        "",
        _protocol_summary(run_a, run_b),
        "",
        _table_models(run_a, run_b),
        "",
        _table_baselines(run_a, run_b),
        "",
    ]
    report = "\n".join(parts)
    print(report)
    if args.output is not None:
        args.output.write_text(report)


if __name__ == "__main__":
    main()
