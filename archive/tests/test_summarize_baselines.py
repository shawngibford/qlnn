"""Smoke test for the per-seed-table emission added to summarize_baselines.py."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "summarize_baselines.py"


def _write_run(run_dir: Path, *, seeds: list[int]) -> None:
    """Create a minimal multi-seed run directory with the files the summarizer reads."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # baselines.json — only the first run is read, but every run path must exist.
    baselines = {
        "persistence": {
            "val": {"mse_norm": 0.001, "mae_raw": 0.05, "rmse_raw": 0.07, "r2_raw": 0.9},
            "test": {"mse_norm": 0.002, "mae_raw": 0.06, "rmse_raw": 0.08, "r2_raw": 0.91},
        },
        "linear": {
            "val": {"mse_norm": 0.0015, "mae_raw": 0.055, "rmse_raw": 0.075, "r2_raw": 0.88},
            "test": {"mse_norm": 0.0025, "mae_raw": 0.065, "rmse_raw": 0.085, "r2_raw": 0.89},
        },
    }
    (run_dir / "baselines.json").write_text(json.dumps(baselines))

    # seeds_summary.json — only mean/std are read for the headline table.
    def _stat(mean: float, std: float = 0.001) -> dict:
        return {"mean": mean, "std": std, "min": mean - std, "max": mean + std}

    summary = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "val": {
            "mse_norm": _stat(0.001), "mae_raw": _stat(0.05),
            "rmse_raw": _stat(0.07), "r2_raw": _stat(0.93),
        },
        "test": {
            "mse_norm": _stat(0.002), "mae_raw": _stat(0.06),
            "rmse_raw": _stat(0.08), "r2_raw": _stat(0.92),
        },
    }
    (run_dir / "seeds_summary.json").write_text(json.dumps(summary))

    for s in seeds:
        seed_dir = run_dir / f"seed_{s}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "best_epoch": s + 1,
            "val": {
                "mse_norm": 0.001 + 0.0001 * s,
                "mae_raw": 0.05 + 0.001 * s,
                "rmse_raw": 0.07 + 0.001 * s,
                "r2_raw": 0.93 - 0.001 * s,
            },
            "test": {
                "mse_norm": 0.002 + 0.0001 * s,
                "mae_raw": 0.06 + 0.001 * s,
                "rmse_raw": 0.08 + 0.001 * s,
                "r2_raw": 0.92 - 0.001 * s,
            },
        }
        (seed_dir / "metrics.json").write_text(json.dumps(metrics))


def test_summarize_baselines_emits_per_seed_table(tmp_path: Path) -> None:
    run_a = tmp_path / "run_A"
    run_b = tmp_path / "run_B"
    _write_run(run_a, seeds=[0, 1])
    _write_run(run_b, seeds=[0, 1])

    out_dir = tmp_path / "out"

    cmd = [
        sys.executable, str(SCRIPT),
        "--runs", str(run_a), str(run_b),
        "--labels", "Alpha", "Beta",
        "--output", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"summarizer failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"

    # Original artifacts still produced.
    assert (out_dir / "baseline_table.md").exists()
    assert (out_dir / "baseline_table.json").exists()

    # New artifacts.
    csv_path = out_dir / "per_seed_table.csv"
    md_path = out_dir / "per_seed_table.md"
    assert csv_path.exists()
    assert md_path.exists()

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Exactly 2 models * 2 seeds = 4 rows.
    assert len(rows) == 4

    expected_columns = {
        "model", "seed", "best_epoch",
        "val_mse_norm", "val_mae_raw", "val_rmse_raw", "val_r2_raw",
        "test_mse_norm", "test_mae_raw", "test_rmse_raw", "test_r2_raw",
    }
    assert expected_columns.issubset(set(fieldnames))

    # Both labels appear; seeds are integers; numerics are non-empty.
    labels_seen = {r["model"] for r in rows}
    assert labels_seen == {"Alpha", "Beta"}
    for r in rows:
        assert r["seed"] in {"0", "1"}
        assert r["val_mae_raw"] != ""
        assert r["test_r2_raw"] != ""
