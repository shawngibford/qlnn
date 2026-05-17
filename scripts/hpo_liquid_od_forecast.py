from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_liquid_od_baseline.py"


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError("lo/hi must be positive and hi>lo")
    return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))


def _sample_config(rng: np.random.Generator) -> dict[str, object]:
    # Keep the search space modest and defensible.
    # NOTE: window_size/stride are fixed via CLI args to ensure trials are comparable
    # (changing them changes which windows exist in val/test).

    hidden_size = int(rng.choice([16, 32, 64, 128]))
    tau_min = float(rng.choice([0.01, 0.05, 0.1, 0.2]))

    forecast_steps = int(rng.choice([1, 2, 4, 6, 12]))
    delta_scale = float(rng.choice([0.02, 0.05, 0.08, 0.1]))

    lr = _loguniform(rng, 1e-4, 3e-3)
    weight_decay = _loguniform(rng, 1e-8, 1e-3)

    return {
        "hidden_size": hidden_size,
        "tau_min": tau_min,
        "forecast_steps": forecast_steps,
        "delta_scale": delta_scale,
        "lr": lr,
        "weight_decay": weight_decay,
    }


def _run_trial(*, trial_dir: Path, cfg: dict[str, object], common: dict[str, object]) -> dict:
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(trial_dir),
        "--metrics-only",
        "--no-plots",
        "--no-perm",
        "--epochs",
        str(common["epochs"]),
        "--eval-every",
        str(common["eval_every"]),
        "--patience",
        str(common["patience"]),
        "--seed",
        str(common["seed"]),
        "--horizon-hours",
        str(common["horizon_hours"]),
        "--horizon-tol-hours",
        str(common["horizon_tol_hours"]),
        "--od-min",
        str(common["od_min"]),
        "--od-max",
        str(common["od_max"]),
        "--train-ratio",
        str(common["train_ratio"]),
        "--val-ratio",
        str(common["val_ratio"]),
        "--batch-size",
        str(common["batch_size"]),
        "--feature-cols",
        *common["feature_cols"],
        "--target-col",
        str(common["target_col"]),
        "--window-size",
        str(common["window_size"]),
        "--stride",
        str(common["stride"]),
    ]

    # Hyperparameters
    cmd += [
        "--hidden-size",
        str(cfg["hidden_size"]),
        "--tau-min",
        str(cfg["tau_min"]),
        "--forecast-steps",
        str(cfg["forecast_steps"]),
        "--delta-scale",
        str(cfg["delta_scale"]),
        "--lr",
        str(cfg["lr"]),
        "--weight-decay",
        str(cfg["weight_decay"]),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    (trial_dir / "stdout.txt").write_text(proc.stdout)
    (trial_dir / "stderr.txt").write_text(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"Trial failed (exit={proc.returncode}). See {trial_dir}/stderr.txt")

    metrics_path = trial_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())

    return {
        "runtime_sec": dt,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Random-search HPO for 1h-ahead OD forecasting. Selects by validation MSE_norm; test is held out. "
            "Window size/stride are fixed (CLI args) so trials are comparable."
        )
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--horizon-hours", type=float, default=1.0)
    parser.add_argument("--horizon-tol-hours", type=float, default=1e-3)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--od-min", type=float, default=0.0)
    parser.add_argument("--od-max", type=float, default=3.8)

    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=["OD", "PRE", "TEMP_EXT", "TEMP_CULTURE", "PAR_LIGHT", "PH", "DO"],
    )
    parser.add_argument("--target-col", type=str, default="OD")

    # IMPORTANT: keep fixed across trials for fair comparisons.
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "hpo_od_forecast_1h",
    )

    parser.add_argument(
        "--run-best-full",
        action="store_true",
        help="After HPO, rerun the best config with full artifacts/plots in output-dir/best_full_run.",
    )

    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("trials must be positive")

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    common = {
        "epochs": int(args.epochs),
        "eval_every": int(args.eval_every),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "horizon_hours": float(args.horizon_hours),
        "horizon_tol_hours": float(args.horizon_tol_hours),
        "od_min": float(args.od_min),
        "od_max": float(args.od_max),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "batch_size": int(args.batch_size),
        "feature_cols": list(args.feature_cols),
        "target_col": str(args.target_col),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
    }

    (out_dir / "common.json").write_text(json.dumps(common, indent=2) + "\n")

    rows: list[dict[str, object]] = []

    best_val = float("inf")
    best_cfg: dict[str, object] | None = None

    for i in range(args.trials):
        cfg = _sample_config(rng)

        trial_dir = out_dir / f"trial_{i:04d}"
        # Make runs deterministic across trials by using a fixed seed.
        # (If you want stochastic robustness, run HPO multiple times or add repeats.)

        try:
            result = _run_trial(trial_dir=trial_dir, cfg=cfg, common=common)
        except Exception as e:
            rows.append({"trial": i, **cfg, "status": "failed", "error": str(e)})
            continue

        metrics = result["metrics"]
        val_mse = float(metrics["model"]["val"]["mse_norm"])
        val_r2 = float(metrics["model"]["val"]["r2_raw"])
        test_mse = float(metrics["model"]["test"]["mse_norm"])
        test_r2 = float(metrics["model"]["test"]["r2_raw"])

        persist_val_mse = float(metrics["baselines"]["persistence"]["val"]["mse_norm"])
        persist_test_mse = float(metrics["baselines"]["persistence"]["test"]["mse_norm"])

        rows.append(
            {
                "trial": i,
                "window_size": int(common["window_size"]),
                "stride": int(common["stride"]),
                **cfg,
                "status": "ok",
                "runtime_sec": float(result["runtime_sec"]),
                "val_mse_norm": val_mse,
                "val_r2_raw": val_r2,
                "test_mse_norm": test_mse,
                "test_r2_raw": test_r2,
                "persist_val_mse_norm": persist_val_mse,
                "persist_test_mse_norm": persist_test_mse,
                "improve_val_vs_persist_mse": persist_val_mse - val_mse,
                "improve_test_vs_persist_mse": persist_test_mse - test_mse,
            }
        )

        if val_mse < best_val:
            best_val = val_mse
            best_cfg = cfg

        print(
            f"trial {i:04d}/{args.trials-1:04d} | val_mse={val_mse:.6f} | "
            f"persist_val_mse={persist_val_mse:.6f} | best_val_mse={best_val:.6f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)

    if best_cfg is None:
        raise RuntimeError("No successful trials")

    best_cfg_full = {
        "window_size": int(common["window_size"]),
        "stride": int(common["stride"]),
        **best_cfg,
    }
    (out_dir / "best_config.json").write_text(json.dumps(best_cfg_full, indent=2) + "\n")

    print(f"\nBest config (by val_mse_norm={best_val:.6f}) saved to: {out_dir / 'best_config.json'}")

    if args.run_best_full:
        best_dir = out_dir / "best_full_run"
        best_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--output-dir",
            str(best_dir),
            "--epochs",
            str(args.epochs),
            "--eval-every",
            str(args.eval_every),
            "--patience",
            str(args.patience),
            "--seed",
            str(args.seed),
            "--horizon-hours",
            str(args.horizon_hours),
            "--horizon-tol-hours",
            str(args.horizon_tol_hours),
            "--od-min",
            str(args.od_min),
            "--od-max",
            str(args.od_max),
            "--train-ratio",
            str(args.train_ratio),
            "--val-ratio",
            str(args.val_ratio),
            "--batch-size",
            str(args.batch_size),
            "--feature-cols",
            *args.feature_cols,
            "--target-col",
            args.target_col,
            "--window-size",
            str(common["window_size"]),
            "--stride",
            str(common["stride"]),
            "--hidden-size",
            str(best_cfg["hidden_size"]),
            "--tau-min",
            str(best_cfg["tau_min"]),
            "--forecast-steps",
            str(best_cfg["forecast_steps"]),
            "--delta-scale",
            str(best_cfg["delta_scale"]),
            "--lr",
            str(best_cfg["lr"]),
            "--weight-decay",
            str(best_cfg["weight_decay"]),
        ]

        print(f"\nRunning best config with full plots into: {best_dir}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
