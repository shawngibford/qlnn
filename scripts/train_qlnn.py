"""Multi-seed trainer for the hybrid QLNN forecaster.

Writes results in the SAME shape as scripts/train_baseline.py so
scripts/summarize_baselines.py can place QLNN rows alongside the classical
ones in the paper table without any reshaping.

Usage:
    python scripts/train_qlnn.py --config configs/qlnn_hybrid.yaml \\
        --output-dir results/qlnn_hybrid

    # Smoke test (1 seed, few epochs):
    python scripts/train_qlnn.py --config configs/qlnn_hybrid.yaml \\
        --output-dir results/_qlnn_smoke --seeds 0 --epochs 3 --eval-every 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from qlnn_ import (
    QLNNForecaster,
    QLNNForecasterConfig,
    QLNNPhysicsLossConfig,
    QLNNTrainerConfig,
    history_to_dicts,
    train_one_qlnn,
)
from quantum_liquid_neuralode.data_processing import (
    apply_minmax,
    fit_minmax,
    HorizonWindows,
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
from quantum_liquid_neuralode.utils import write_provenance


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Prediction clipping (R3 finding 6 follow-up). See train_baseline.py for the
# motivation; this is the JAX/Equinox-side mirror.
# ---------------------------------------------------------------------------
def clip_predictions_norm(
    y_pred_norm: np.ndarray,
    od_scaler,
    *,
    clip_raw_max: float | None,
    clip_raw_min: float = 0.0,
) -> np.ndarray:
    """Clip normalized predictions to a physically-reasonable raw-OD range.

    Mirrors the helper in scripts/train_baseline.py — duplicated here rather
    than imported because that script is not on this script's import path.
    TODO: lift into quantum_liquid_neuralode.evaluation once the metrics
    module coordination settles.
    """
    if clip_raw_max is None:
        return y_pred_norm
    bounds_raw = np.array([[float(clip_raw_min)], [float(clip_raw_max)]], dtype=np.float64)
    bounds_norm = od_scaler.transform(bounds_raw).reshape(-1)
    lo, hi = float(bounds_norm[0]), float(bounds_norm[1])
    return np.clip(y_pred_norm, lo, hi)


def _qlnn_predict_norm_clipped(
    model: eqx.Module,
    x: np.ndarray,
    t: np.ndarray,
    *,
    batch_size: int,
    od_scaler,
    clip_raw_max: float | None,
) -> np.ndarray:
    """Run the QLNN model in eval mode and return clipped predictions."""
    x_j = jnp.asarray(x)
    t_j = jnp.asarray(t.astype(np.float32))
    n = x_j.shape[0]

    @eqx.filter_jit
    def _batch(m: eqx.Module, xb: jnp.ndarray, tb: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(m)(xb, tb)

    preds: list[np.ndarray] = []
    for i in range(0, n, batch_size):
        yp = _batch(model, x_j[i : i + batch_size], t_j[i : i + batch_size])
        preds.append(np.asarray(yp))
    y_pred_norm = np.concatenate(preds, axis=0)
    return clip_predictions_norm(y_pred_norm, od_scaler, clip_raw_max=clip_raw_max)


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
    parser = argparse.ArgumentParser(description="Multi-seed hybrid QLNN forecaster trainer.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
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
        # in the YAML, fit_minmax falls back to train-only fitting for OD
        # (R3 finding 6 fix: close test-set leakage).
        od_min_cfg = cfg["data"].get("od_min", 0.0)
        fixed_bounds[target_col] = (float(od_min_cfg), float(cfg["data"]["od_max"]))

    scalers = fit_minmax(df, cols_to_scale, fit_end=split.train_end, fixed_bounds=fixed_bounds)
    df_n = apply_minmax(df, cols_to_scale, scalers)
    od_scaler = scalers[target_col]
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
        # OD scaler bookkeeping (see scripts/train_baseline.py for semantics).
        "od_scaler_mode": od_scaler_mode,
        "od_data_min": od_data_min,
        "od_data_max": od_data_max,
        "od_phys_max": clip_raw_max,
        "od_min": od_data_min,
        "od_max": od_data_max,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "stack": "jax+pennylane",
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    # ---- Baselines ----
    persist_val_pred = persistence_forecast(w_val.od_last)
    persist_test_pred = persistence_forecast(w_test.od_last)
    linear_val_pred = linear_extrapolation_forecast(
        od_last=w_val.od_last, od_prev=w_val.od_prev, dt_last_hours=w_val.dt_last, horizon_hours=horizon_hours
    )
    linear_test_pred = linear_extrapolation_forecast(
        od_last=w_test.od_last, od_prev=w_test.od_prev, dt_last_hours=w_test.dt_last, horizon_hours=horizon_hours
    )
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

    log = (lambda s: None) if args.quiet else (lambda s: print(s, flush=True))
    log(f"jax default backend: {jax.default_backend()}")
    log(
        f"rows: {n} | train_end={split.train_end} val_end={split.val_end} | "
        f"windows: train={len(w_train)} val={len(w_val)} test={len(w_test)}"
    )
    log(
        f"baselines | persist val MAE={baselines_record['persistence']['val']['mae_raw']:.4f}, "
        f"R2={baselines_record['persistence']['val']['r2_raw']:.4f}"
    )

    # ---- Per-seed training ----
    seeds: list[int] = list(cfg["seeds"])
    od_index = feature_cols.index(target_col)

    model_cfg = cfg["model"]
    physics_cfg_raw = cfg.get("physics", {}) or {}
    physics_cfg = QLNNPhysicsLossConfig(
        lambda_logistic=float(physics_cfg_raw.get("lambda_logistic", 0.0)),
        mu_norm=float(physics_cfg_raw.get("mu_norm", 0.4)),
        K_norm=float(physics_cfg_raw.get("K_norm", 1.0)),
    )
    trainer_cfg = QLNNTrainerConfig(
        epochs=int(cfg["training"]["epochs"]),
        batch_size=int(cfg["training"]["batch_size"]),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        eval_every=int(cfg["training"]["eval_every"]),
        patience=int(cfg["training"]["patience"]),
        grad_clip_norm=float(cfg["training"]["grad_clip_norm"]),
        physics=physics_cfg,
    )
    if physics_cfg.lambda_logistic > 0:
        log(
            f"physics ON | lambda_logistic={physics_cfg.lambda_logistic} "
            f"mu={physics_cfg.mu_norm} K={physics_cfg.K_norm}"
        )

    val_metrics_all = []
    test_metrics_all = []

    for seed in seeds:
        log(f"\n=== seed {seed} ===")

        forecaster_cfg = QLNNForecasterConfig(
            input_dim=len(feature_cols),
            num_qubits=int(model_cfg["num_qubits"]),
            num_layers=int(model_cfg["num_layers"]),
            horizon_hours=horizon_hours,
            od_index=od_index,
            delta_scale=float(model_cfg["delta_scale"]),
            tau_min=float(model_cfg["tau_min"]),
            tau_init=float(model_cfg["tau_init"]),
            solver=str(model_cfg["solver"]),
            rtol=float(model_cfg["rtol"]),
            atol=float(model_cfg["atol"]),
            dt0=float(model_cfg["dt0"]),
            max_steps=int(model_cfg["max_steps"]),
            init_head_std=float(model_cfg["init_head_std"]),
        )

        model = QLNNForecaster(forecaster_cfg, key=jax.random.PRNGKey(seed))

        result = train_one_qlnn(
            model=model,
            x_train=w_train.x, t_train=w_train.t, y_train=w_train.y,
            x_val=w_val.x, t_val=w_val.t, y_val=w_val.y,
            x_test=w_test.x, t_test=w_test.t, y_test=w_test.y,
            od_scaler=od_scaler,
            cfg=trainer_cfg,
            seed=seed,
            log_fn=log,
            od_index=od_index,
            horizon_hours=horizon_hours,
        )

        # Re-evaluate the best model with raw-space clipping of predictions
        # to the strain's physical OD range (R3 finding 6 follow-up). The
        # trainer's selection-time metrics drove early stopping unchanged;
        # only the reported / aggregated numbers reflect the clip.
        val_pred_clipped = _qlnn_predict_norm_clipped(
            result.model, w_val.x, w_val.t,
            batch_size=trainer_cfg.batch_size,
            od_scaler=od_scaler, clip_raw_max=clip_raw_max,
        )
        test_pred_clipped = _qlnn_predict_norm_clipped(
            result.model, w_test.x, w_test.t,
            batch_size=trainer_cfg.batch_size,
            od_scaler=od_scaler, clip_raw_max=clip_raw_max,
        )
        # Pass od_last_norm so ΔOD metrics also populate in the report.
        result.val_metrics = compute_metrics(
            y_true_norm=w_val.y, y_pred_norm=val_pred_clipped, od_scaler=od_scaler,
            od_last_norm=w_val.od_last,
        )
        result.test_metrics = compute_metrics(
            y_true_norm=w_test.y, y_pred_norm=test_pred_clipped, od_scaler=od_scaler,
            od_last_norm=w_test.od_last,
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
        eqx.tree_serialise_leaves(seed_dir / "best_model.eqx", result.model)

        # Per-window predictions (val + test) for downstream paired-bootstrap
        # analyses (Phase-C statistical-rigor work). Same schema as
        # train_baseline.py so scripts/run_paired_comparison.py can pair
        # QLNN vs classical without special-casing the stack.
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
            f"seed {seed} | val MAE={result.val_metrics.mae_raw:.4f} R2={result.val_metrics.r2_raw:.4f} | "
            f"test MAE={result.test_metrics.mae_raw:.4f} R2={result.test_metrics.r2_raw:.4f}"
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
