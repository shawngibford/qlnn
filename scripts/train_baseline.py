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
from quantum_liquid_neuralode.utils import select_device, write_provenance


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Prediction clipping (R3 finding 6 follow-up).
#
# The OD scaler is now fit on the training slice only — closing the
# soft-leakage of the test-set max (~3.8) into the scaler. With train-only
# fitting the model can technically emit normalized OD > 1 when test OD
# exceeds the training-segment max. Predictions are still allowed in the
# whole real line and inverse-transform faithfully, but for evaluation we
# clip to the strain's physical OD range [0, od_phys_max] (a legitimate
# domain prior, separate from the scaler-fit data). Clipping is applied
# in *raw* space and mapped back to normalized space so it composes
# cleanly with the existing normalized-space metric (mse_norm) and the
# raw-space metrics (mae_raw, rmse_raw, r2_raw).
#
# Inlined here rather than living in a shared utility because two
# concurrent training scripts (train_baseline.py, train_qlnn.py) each
# own their own copy until a common helper module is introduced. TODO:
# factor into quantum_liquid_neuralode.evaluation once the metrics-module
# coordination settles.
# ---------------------------------------------------------------------------
def clip_predictions_norm(
    y_pred_norm: np.ndarray,
    od_scaler,
    *,
    clip_raw_max: float | None,
    clip_raw_min: float = 0.0,
) -> np.ndarray:
    """Clip normalized predictions to a physically reasonable raw-OD range.

    Args:
        y_pred_norm: array of normalized predictions (the model's raw output).
        od_scaler: the (already-fit) MinMaxScaler used for OD.
        clip_raw_max: upper bound in raw OD units (e.g. 3.8 from strain spec).
            If None, no clipping is applied (returns y_pred_norm unchanged).
        clip_raw_min: lower bound in raw OD units (default 0.0 — OD is
            non-negative).

    Returns:
        Array same shape as y_pred_norm, clipped to the normalized image of
        [clip_raw_min, clip_raw_max].
    """
    if clip_raw_max is None:
        return y_pred_norm
    bounds_raw = np.array([[float(clip_raw_min)], [float(clip_raw_max)]], dtype=np.float64)
    bounds_norm = od_scaler.transform(bounds_raw).reshape(-1)
    lo, hi = float(bounds_norm[0]), float(bounds_norm[1])
    return np.clip(y_pred_norm, lo, hi)


def _predict_norm_with_clip(
    model: torch.nn.Module,
    x: np.ndarray,
    t: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    od_scaler,
    clip_raw_max: float | None,
) -> np.ndarray:
    """Run the model in eval mode and return clipped normalized predictions."""
    model.eval()
    preds: list[np.ndarray] = []
    n = x.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            tb = torch.from_numpy(t[i : i + batch_size].astype(np.float32)).to(device)
            yp = model(xb, tb).detach().cpu().numpy()
            preds.append(yp)
    y_pred_norm = np.concatenate(preds, axis=0)
    return clip_predictions_norm(
        y_pred_norm, od_scaler, clip_raw_max=clip_raw_max
    )


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
    write_provenance(output_dir, csv_path, REPO_ROOT)
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
        # Legacy fixed-bounds mode (sensitivity comparator). With od_max=null
        # in the YAML, this branch is skipped and fit_minmax falls back to
        # train-only fitting for OD (R3 finding 6: close test-set leakage).
        od_min_cfg = cfg["data"].get("od_min", 0.0)
        fixed_bounds[target_col] = (float(od_min_cfg), float(cfg["data"]["od_max"]))

    scalers = fit_minmax(df, cols_to_scale, fit_end=split.train_end, fixed_bounds=fixed_bounds)
    df_n = apply_minmax(df, cols_to_scale, scalers)
    od_scaler = scalers[target_col]
    # Record the actual fitted scaler bounds (post-fit), independent of the
    # YAML — this is what's in effect for the run.
    od_data_min = float(od_scaler.data_min_[0])
    od_data_max = float(od_scaler.data_max_[0])
    od_scaler_mode = "fixed" if cfg["data"].get("od_max") is not None else "train_only"
    clip_raw_max = cfg["data"].get("od_phys_max", None)
    if clip_raw_max is not None:
        clip_raw_max = float(clip_raw_max)

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
        # OD scaler bookkeeping. `od_scaler_mode`:
        #   - "train_only": OD MinMax fit on the training slice (R3 fix);
        #     `od_data_min`/`od_data_max` are the actual fitted bounds.
        #   - "fixed": OD MinMax pinned to the YAML's [od_min, od_max] (legacy
        #     domain-prior comparator); `od_data_min`/`od_data_max` echo those.
        "od_scaler_mode": od_scaler_mode,
        "od_data_min": od_data_min,
        "od_data_max": od_data_max,
        # Domain prior used to clip predictions in raw OD space at eval time.
        # Independent of the scaler fit — captures the strain's physical OD
        # max. None disables clipping.
        "od_phys_max": clip_raw_max,
        # Legacy fields (kept for back-compat with downstream summaries).
        "od_min": od_data_min,
        "od_max": od_data_max,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    # ---- Baselines (deterministic, no training) ----
    persist_val_pred = persistence_forecast(w_val.od_last)
    persist_test_pred = persistence_forecast(w_test.od_last)
    linear_val_pred = linear_extrapolation_forecast(
        od_last=w_val.od_last, od_prev=w_val.od_prev, dt_last_hours=w_val.dt_last, horizon_hours=horizon_hours
    )
    linear_test_pred = linear_extrapolation_forecast(
        od_last=w_test.od_last, od_prev=w_test.od_prev, dt_last_hours=w_test.dt_last, horizon_hours=horizon_hours
    )

    # Clip baseline predictions with the same domain-prior the trained
    # models will be clipped against — apples-to-apples comparison.
    persist_val_pred_c = clip_predictions_norm(persist_val_pred, od_scaler, clip_raw_max=clip_raw_max)
    persist_test_pred_c = clip_predictions_norm(persist_test_pred, od_scaler, clip_raw_max=clip_raw_max)
    linear_val_pred_c = clip_predictions_norm(linear_val_pred, od_scaler, clip_raw_max=clip_raw_max)
    linear_test_pred_c = clip_predictions_norm(linear_test_pred, od_scaler, clip_raw_max=clip_raw_max)

    baselines_record = {
        "persistence": {
            "val": compute_metrics(y_true_norm=w_val.y, y_pred_norm=persist_val_pred_c, od_scaler=od_scaler, od_last_norm=w_val.od_last).to_dict(),
            "test": compute_metrics(y_true_norm=w_test.y, y_pred_norm=persist_test_pred_c, od_scaler=od_scaler, od_last_norm=w_test.od_last).to_dict(),
        },
        "linear": {
            "val": compute_metrics(y_true_norm=w_val.y, y_pred_norm=linear_val_pred_c, od_scaler=od_scaler, od_last_norm=w_val.od_last).to_dict(),
            "test": compute_metrics(y_true_norm=w_test.y, y_pred_norm=linear_test_pred_c, od_scaler=od_scaler, od_last_norm=w_test.od_last).to_dict(),
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

    # NOTE: `lambda_smooth` was removed from PhysicsLossConfig (R1 BLOCKER B2;
    # see trainer.py). Any `lambda_smooth` key still present in legacy YAML
    # configs is intentionally ignored here.
    physics_cfg = PhysicsLossConfig(
        lambda_logistic=float(cfg.get("physics", {}).get("lambda_logistic", 0.0)),
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

        # Re-evaluate the best model with raw-space clipping of predictions to
        # the strain's physical OD range (R3 finding 6 follow-up). Done here in
        # the script — rather than inside train_one — so that the trainer's
        # selection-time metrics (which drive early stopping) are unchanged
        # and only the reported / aggregated numbers reflect the clip.
        model.load_state_dict(result.model_state)
        model.to(device)
        val_pred_clipped = _predict_norm_with_clip(
            model, w_val.x, w_val.t,
            device=device, batch_size=trainer_cfg.batch_size,
            od_scaler=od_scaler, clip_raw_max=clip_raw_max,
        )
        test_pred_clipped = _predict_norm_with_clip(
            model, w_test.x, w_test.t,
            device=device, batch_size=trainer_cfg.batch_size,
            od_scaler=od_scaler, clip_raw_max=clip_raw_max,
        )
        # Pass od_last_norm so compute_metrics also reports delta_* fields
        # (R3 ΔOD-reporting fix). Persistence has delta_r2_raw < 0 by
        # construction; trained models should show delta_r2_raw > 0 if they
        # actually capture OD-change signal.
        result.val_metrics = compute_metrics(
            y_true_norm=w_val.y, y_pred_norm=val_pred_clipped, od_scaler=od_scaler,
            od_last_norm=w_val.od_last,
        )
        result.test_metrics = compute_metrics(
            y_true_norm=w_test.y, y_pred_norm=test_pred_clipped, od_scaler=od_scaler,
            od_last_norm=w_test.od_last,
        )

        # `delta_scale` is now a learnable scalar (softplus + floor). Log the
        # value the best checkpoint converged to so the paper / sweeps can see
        # whether the model grew or shrank the delta-headroom relative to its
        # init (Phase-B finding: the old fixed delta_scale=0.1 was a cap).
        delta_scale_learned = float(model.delta_scale().detach().cpu().item())

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "best_epoch": int(result.best_epoch),
                    "delta_scale_learned": delta_scale_learned,
                    "val": result.val_metrics.to_dict(),
                    "test": result.test_metrics.to_dict(),
                },
                indent=2,
            )
            + "\n"
        )
        pd.DataFrame(history_to_dicts(result.history)).to_csv(seed_dir / "history.csv", index=False)
        torch.save(result.model_state, seed_dir / "best_state.pt")

        # Per-window predictions (val + test) for downstream paired-bootstrap
        # analyses (Phase-C statistical-rigor work). `od_last_norm` lets the
        # bootstrap script reconstruct the persistence baseline without
        # rerunning anything. (R3 Tier 2.2.)
        np.savez(
            seed_dir / "predictions.npz",
            val_y_true_norm=w_val.y.astype(np.float32),
            val_y_pred_norm=val_pred_clipped.astype(np.float32),
            val_od_last_norm=w_val.od_last.astype(np.float32),
            test_y_true_norm=w_test.y.astype(np.float32),
            test_y_pred_norm=test_pred_clipped.astype(np.float32),
            test_od_last_norm=w_test.od_last.astype(np.float32),
        )

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
