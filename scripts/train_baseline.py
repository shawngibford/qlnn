"""Train the classical Liquid-ODE baseline across multiple seeds.

Reads a YAML config, runs N seeds with the same hyperparameters, aggregates
metrics (mean ± std), and writes a single canonical artifact directory:

    <output_dir>/
        config.json                 # frozen copy of the config that ran
        protocol.json               # locked data/split/window protocol
        baselines.json              # persistence + linear baseline metrics (no training)
        seed_<k>/
            metrics.json
            history.csv
            best_state.pt
        seeds_summary.json          # mean/std/min/max across seeds (the paper-table row)

This is the artifact the paper cites for "classical Liquid-ODE baseline."

Usage:
    python scripts/train_baseline.py --config configs/baseline.yaml \\
        --output-dir results/baseline_classical

    # Smoke test on 1 seed, 5 epochs:
    python scripts/train_baseline.py --config configs/baseline.yaml \\
        --output-dir results/_smoke --seeds 0 --epochs 5 --eval-every 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from quantum_liquid_neuralode.data_processing import (
    HorizonWindows,
    apply_minmax,
    fit_minmax,
    load_qzeta,
    make_horizon_windows,
    split_indices,
    time_hours_from_date,
)
from quantum_liquid_neuralode.evaluation import (
    compute_metrics,
    linear_extrapolation_forecast,
    persistence_forecast,
)
from quantum_liquid_neuralode.evaluation.metrics import aggregate_seed_metrics
from quantum_liquid_neuralode.models import LiquidODForecaster
from quantum_liquid_neuralode.training import (
    PhysicsLossConfig,
    TrainerConfig,
    history_to_dicts,
    train_one,
)
from quantum_liquid_neuralode.utils import select_device


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _segment_windows(
    df_n: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    time_hours: np.ndarray,
    start: int,
    end: int,
    window_size: int,
    stride: int,
    horizon_hours: float,
    horizon_tol_hours: float,
) -> HorizonWindows:
    feat = df_n[feature_cols].iloc[start:end].to_numpy(dtype=np.float32)
    od = df_n[target_col].iloc[start:end].to_numpy(dtype=np.float32)
    t = time_hours[start:end].astype(np.float64)
    return make_horizon_windows(
        features=feat,
        od=od,
        time_hours=t,
        window_size=window_size,
        stride=stride,
        horizon_hours=horizon_hours,
        horizon_tolerance_hours=horizon_tol_hours,
        index_offset=start,
    )


def _resolve_csv_path(csv_path_str: str) -> Path:
    p = Path(csv_path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed classical Liquid-ODE baseline.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write artifacts.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override the seeds in the config (space-separated integers).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs (useful for smoke tests).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Override training.eval_every.",
    )
    parser.add_argument("--device", type=str, default=None, choices=["mps", "cuda", "cpu"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.seeds is not None:
        cfg["seeds"] = list(args.seeds)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.eval_every is not None:
        cfg.setdefault("training", {})["eval_every"] = int(args.eval_every)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    # ---- Data ----
    csv_path = _resolve_csv_path(cfg["data"]["csv_path"])
    feature_cols: list[str] = list(cfg["data"]["feature_cols"])
    target_col: str = cfg["data"]["target_col"]
    if target_col not in feature_cols:
        raise ValueError("target_col must be included in feature_cols (residual forecast needs OD(t)).")

    df = load_qzeta(csv_path)
    n = len(df)
    split = split_indices(n, train_ratio=cfg["split"]["train_ratio"], val_ratio=cfg["split"]["val_ratio"])
    time_hours = time_hours_from_date(df)

    cols_to_scale = list(dict.fromkeys(feature_cols + [target_col]))
    fixed_bounds: dict[str, tuple[float, float]] = {}
    if cfg["data"].get("od_max") is not None:
        fixed_bounds[target_col] = (float(cfg["data"]["od_min"]), float(cfg["data"]["od_max"]))

    scalers = fit_minmax(df, cols_to_scale, fit_end=split.train_end, fixed_bounds=fixed_bounds)
    df_n = apply_minmax(df, cols_to_scale, scalers)

    win = cfg["windows"]
    common_kwargs = dict(
        feature_cols=feature_cols,
        target_col=target_col,
        time_hours=time_hours,
        window_size=int(win["window_size"]),
        stride=int(win["stride"]),
        horizon_hours=float(win["horizon_hours"]),
        horizon_tol_hours=float(win["horizon_tol_hours"]),
    )
    w_train = _segment_windows(df_n, start=0, end=split.train_end, **common_kwargs)
    w_val = _segment_windows(df_n, start=split.train_end, end=split.val_end, **common_kwargs)
    w_test = _segment_windows(df_n, start=split.val_end, end=n, **common_kwargs)

    horizon_hours = float(win["horizon_hours"])

    # ---- Protocol record ----
    protocol = {
        "n_rows": int(n),
        "train_end": int(split.train_end),
        "val_end": int(split.val_end),
        "n_train_windows": len(w_train),
        "n_val_windows": len(w_val),
        "n_test_windows": len(w_test),
        "horizon_hours": horizon_hours,
        "window_size": int(win["window_size"]),
        "stride": int(win["stride"]),
        "od_min": float(cfg["data"]["od_min"]),
        "od_max": float(cfg["data"]["od_max"]),
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    # ---- Baselines (deterministic, no training) ----
    od_scaler = scalers[target_col]
    persist_val_pred = persistence_forecast(w_val.od_last)
    persist_test_pred = persistence_forecast(w_test.od_last)
    linear_val_pred = linear_extrapolation_forecast(
        od_last=w_val.od_last, od_prev=w_val.od_prev, dt_last_hours=w_val.dt_last, horizon_hours=horizon_hours
    )
    linear_test_pred = linear_extrapolation_forecast(
        od_last=w_test.od_last, od_prev=w_test.od_prev, dt_last_hours=w_test.dt_last, horizon_hours=horizon_hours
    )

    baselines_record = {
        "persistence": {
            "val": compute_metrics(y_true_norm=w_val.y, y_pred_norm=persist_val_pred, od_scaler=od_scaler).to_dict(),
            "test": compute_metrics(y_true_norm=w_test.y, y_pred_norm=persist_test_pred, od_scaler=od_scaler).to_dict(),
        },
        "linear": {
            "val": compute_metrics(y_true_norm=w_val.y, y_pred_norm=linear_val_pred, od_scaler=od_scaler).to_dict(),
            "test": compute_metrics(y_true_norm=w_test.y, y_pred_norm=linear_test_pred, od_scaler=od_scaler).to_dict(),
        },
    }
    (output_dir / "baselines.json").write_text(json.dumps(baselines_record, indent=2) + "\n")

    # ---- Device ----
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = select_device(prefer_mps=True)

    log = (lambda s: None) if args.quiet else (lambda s: print(s))
    log(f"device: {device}")
    log(
        f"rows: {n} | train_end={split.train_end} val_end={split.val_end} | "
        f"windows: train={len(w_train)} val={len(w_val)} test={len(w_test)}"
    )
    log(
        f"baselines | persist val MSE_norm={baselines_record['persistence']['val']['mse_norm']:.6f}, "
        f"R2_raw={baselines_record['persistence']['val']['r2_raw']:.4f}"
    )

    # ---- Per-seed training ----
    seeds: list[int] = list(cfg["seeds"])
    od_index = feature_cols.index(target_col)

    physics_cfg = PhysicsLossConfig(
        lambda_logistic=float(cfg.get("physics", {}).get("lambda_logistic", 0.0)),
        lambda_smooth=float(cfg.get("physics", {}).get("lambda_smooth", 0.0)),
        mu_norm=float(cfg.get("physics", {}).get("mu_norm", 0.4)),
        K_norm=float(cfg.get("physics", {}).get("K_norm", 1.0)),
    )
    trainer_cfg = TrainerConfig(
        epochs=int(cfg["training"]["epochs"]),
        batch_size=int(cfg["training"]["batch_size"]),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        eval_every=int(cfg["training"]["eval_every"]),
        patience=int(cfg["training"]["patience"]),
        grad_clip_norm=float(cfg["training"]["grad_clip_norm"]),
        physics=physics_cfg,
    )

    model_cfg = cfg["model"]

    val_metrics_all = []
    test_metrics_all = []

    for seed in seeds:
        log(f"\n=== seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = LiquidODForecaster(
            input_size=len(feature_cols),
            hidden_size=int(model_cfg["hidden_size"]),
            horizon_hours=horizon_hours,
            forecast_steps=int(model_cfg["forecast_steps"]),
            od_index=od_index,
            delta_scale=float(model_cfg["delta_scale"]),
            tau_min=float(model_cfg["tau_min"]),
            ode_method=str(model_cfg.get("ode_method", "euler")),
            rtol=float(model_cfg.get("rtol", 1e-3)),
            atol=float(model_cfg.get("atol", 1e-4)),
        )

        result = train_one(
            model=model,
            x_train=w_train.x, t_train=w_train.t, y_train=w_train.y, od_last_train=w_train.od_last,
            x_val=w_val.x, t_val=w_val.t, y_val=w_val.y,
            x_test=w_test.x, t_test=w_test.t, y_test=w_test.y,
            od_scaler=od_scaler,
            device=device,
            cfg=trainer_cfg,
            horizon_hours=horizon_hours,
            od_index=od_index,
            seed=seed,
            log_fn=log,
        )

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "best_epoch": int(result.best_epoch),
                    "val": result.val_metrics.to_dict(),
                    "test": result.test_metrics.to_dict(),
                },
                indent=2,
            )
            + "\n"
        )
        pd.DataFrame(history_to_dicts(result.history)).to_csv(seed_dir / "history.csv", index=False)
        torch.save(result.model_state, seed_dir / "best_state.pt")

        val_metrics_all.append(result.val_metrics)
        test_metrics_all.append(result.test_metrics)

        log(
            f"seed {seed} | val MAE_raw={result.val_metrics.mae_raw:.4f} R2={result.val_metrics.r2_raw:.4f} | "
            f"test MAE_raw={result.test_metrics.mae_raw:.4f} R2={result.test_metrics.r2_raw:.4f}"
        )

    # ---- Aggregate ----
    seeds_summary = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "val": aggregate_seed_metrics(val_metrics_all),
        "test": aggregate_seed_metrics(test_metrics_all),
    }
    (output_dir / "seeds_summary.json").write_text(json.dumps(seeds_summary, indent=2) + "\n")

    log("\n=== summary across seeds (test) ===")
    for k, v in seeds_summary["test"].items():
        log(f"  {k:12s}  mean={v['mean']:.6f}  std={v['std']:.6f}  min={v['min']:.6f}  max={v['max']:.6f}")
    log(f"\nartifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
