"""Compute and report the empirical-Fisher effective dimension for the
trained classical Liquid-ODE (H=4) and the trained QLNN at h=3.

Implements Step 5 / Claim 2 of `hypothesis.md`:

    d_norm(QLNN) > d_norm(classical_H=4) + 1.0   (acceptance threshold)

Both models are scored on the same 500 training windows of the locked
evaluation protocol, using the same trace-normalized empirical Fisher and
the same Abbas-2021 trained-theta formula. The script also produces a
monotonicity sanity-check curve (d_norm at n ∈ {100, 200, 350, 500}) and
the seed-aggregated headline number (mean ± std across seeds {0, 1, 2, 3, 4}).

Artifacts written to ``results/effective_dimension/``:
  - effective_dimension.json   : raw numbers, all seeds, both models
  - effective_dimension.md     : paper-style table
  - monotonicity_check.csv     : per-(model, seed, n) d_norm

Usage:
    .venv/bin/python scripts/run_effective_dimension.py
    .venv/bin/python scripts/run_effective_dimension.py --seeds 0
    .venv/bin/python scripts/run_effective_dimension.py --skip-curve
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch

from qlnn_ import QLNNForecaster, QLNNForecasterConfig
from qlnn_.circuits import AnsatzConfig
from qlnn_.diagnostics import effective_dimension as ed_jax
from quantum_liquid_neuralode.data_processing import (
    apply_minmax,
    fit_minmax,
    load_qzeta,
    make_horizon_windows,
    split_indices,
    time_hours_from_date,
)
from quantum_liquid_neuralode.diagnostics import effective_dimension as ed_torch
from quantum_liquid_neuralode.models import LiquidODForecaster


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSICAL_DIR = REPO_ROOT / "results" / "param_sweep" / "euler_h3_hidden4"
QLNN_DIR = REPO_ROOT / "results" / "qlnn_hybrid_h3"
OUTPUT_DIR = REPO_ROOT / "results" / "effective_dimension"


def _resolve_csv_path(csv_path_str: str) -> Path:
    p = Path(csv_path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


# ---------------------------------------------------------------------------
# Data reconstruction (the locked protocol). Identical across the two model
# stacks; the configs differ only in the model section.
# ---------------------------------------------------------------------------
def _build_training_windows(cfg: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Re-run the locked protocol to recover the training-window (x, t) arrays.

    Returns (x_train, t_train, od_index).
    """
    csv_path = _resolve_csv_path(cfg["data"]["csv_path"])
    feature_cols: list[str] = list(cfg["data"]["feature_cols"])
    target_col: str = cfg["data"]["target_col"]

    df = load_qzeta(csv_path)
    n = len(df)
    split = split_indices(
        n,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
    )
    time_hours = time_hours_from_date(df)

    cols_to_scale = list(dict.fromkeys(feature_cols + [target_col]))
    fixed_bounds: dict[str, tuple[float, float]] = {}
    if cfg["data"].get("od_max") is not None:
        od_min_cfg = cfg["data"].get("od_min", 0.0)
        fixed_bounds[target_col] = (float(od_min_cfg), float(cfg["data"]["od_max"]))
    scalers = fit_minmax(df, cols_to_scale, fit_end=split.train_end, fixed_bounds=fixed_bounds)
    df_n = apply_minmax(df, cols_to_scale, scalers)

    win = cfg["windows"]
    feat = df_n[feature_cols].iloc[: split.train_end].to_numpy(dtype=np.float32)
    od = df_n[target_col].iloc[: split.train_end].to_numpy(dtype=np.float32)
    t = time_hours[: split.train_end].astype(np.float64)
    w_train = make_horizon_windows(
        features=feat,
        od=od,
        time_hours=t,
        window_size=int(win["window_size"]),
        stride=int(win["stride"]),
        horizon_hours=float(win["horizon_hours"]),
        horizon_tolerance_hours=float(win["horizon_tol_hours"]),
        index_offset=0,
    )
    od_index = feature_cols.index(target_col)
    return w_train.x, w_train.t, od_index


# ---------------------------------------------------------------------------
# Model rebuilding.
# ---------------------------------------------------------------------------
def _rebuild_classical(cfg: dict, ckpt: Path, od_index: int) -> torch.nn.Module:
    model_cfg = cfg["model"]
    win = cfg["windows"]
    feature_cols = list(cfg["data"]["feature_cols"])
    model = LiquidODForecaster(
        input_size=len(feature_cols),
        hidden_size=int(model_cfg["hidden_size"]),
        horizon_hours=float(win["horizon_hours"]),
        forecast_steps=int(model_cfg["forecast_steps"]),
        od_index=od_index,
        delta_scale=float(model_cfg["delta_scale"]),
        tau_min=float(model_cfg["tau_min"]),
        ode_method=str(model_cfg.get("ode_method", "euler")),
        rtol=float(model_cfg.get("rtol", 1e-3)),
        atol=float(model_cfg.get("atol", 1e-4)),
    )
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _rebuild_qlnn(cfg: dict, ckpt: Path, od_index: int, seed: int) -> eqx.Module:
    model_cfg = cfg["model"]
    win = cfg["windows"]
    feature_cols = list(cfg["data"]["feature_cols"])

    # Read the ansatz spec if present so the rebuilt skeleton matches the
    # circuit the checkpoint was trained with. Absent = legacy
    # data_reuploading default.
    ansatz_block = model_cfg.get("ansatz")
    ansatz_cfg: AnsatzConfig | None
    if ansatz_block is None:
        ansatz_cfg = None
    else:
        ansatz_cfg = AnsatzConfig(
            name=str(ansatz_block["name"]),
            num_qubits=int(model_cfg["num_qubits"]),
            num_layers=int(model_cfg["num_layers"]),
            params=dict(ansatz_block.get("params") or {}),
        )

    fcfg = QLNNForecasterConfig(
        input_dim=len(feature_cols),
        num_qubits=int(model_cfg["num_qubits"]),
        num_layers=int(model_cfg["num_layers"]),
        horizon_hours=float(win["horizon_hours"]),
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
        ansatz=ansatz_cfg,
    )
    # The Equinox checkpoint is just the leaves; we rebuild a skeleton with
    # the same seed (matches the trainer's init) and deserialize into it.
    # Note: the diagnostics module enables float64 globally, but the
    # checkpoint was saved with float32 leaves; cast the skeleton to f32
    # to match the on-disk dtype before deserialization.
    skeleton = QLNNForecaster(fcfg, key=jax.random.PRNGKey(seed))
    skeleton = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.float32) if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
        skeleton,
    )
    return eqx.tree_deserialise_leaves(str(ckpt), skeleton)


# ---------------------------------------------------------------------------
# Per-model Fisher computation.
# ---------------------------------------------------------------------------
def _fisher_classical(
    model: torch.nn.Module,
    x_all: np.ndarray,
    t_all: np.ndarray,
    sample_indices: list[int],
    log,
) -> np.ndarray:
    theta_flat, forward_scalar = ed_torch.classical_forward_from_flat(model, x_all, t_all)
    log(f"  classical |theta|={int(theta_flat.shape[0])}, n={len(sample_indices)}")
    F = ed_torch.empirical_fisher(forward_scalar, theta_flat, sample_indices)
    return F.numpy()


def _fisher_qlnn(
    model: eqx.Module,
    x_all: np.ndarray,
    t_all: np.ndarray,
    sample_indices: list[int],
    log,
) -> np.ndarray:
    x_j = jnp.asarray(x_all)
    t_j = jnp.asarray(t_all.astype(np.float32))
    theta_flat, forward_scalar = ed_jax.qlnn_forward_from_flat(model, x_j, t_j)
    log(f"  QLNN     |theta|={int(theta_flat.shape[0])}, n={len(sample_indices)}")

    # jit-compile the per-sample gradient once. `jacrev` (reverse-mode) is
    # is fine; for D ≈ 100 it's actually D forward-mode JVPs but tiny vs the
    # quantum forward cost so it doesn't matter.
    @jax.jit
    def grad_one(theta: jnp.ndarray, idx: int) -> jnp.ndarray:
        # Diffrax uses custom_vjp internally — must use jacrev (reverse-mode).
        return jax.jacrev(lambda th: forward_scalar(th, idx))(theta)

    D = int(theta_flat.shape[0])
    fisher = np.zeros((D, D), dtype=np.float64)
    for k, i in enumerate(sample_indices):
        g = np.asarray(grad_one(theta_flat, i), dtype=np.float64)
        fisher += np.outer(g, g)
        if (k + 1) % 50 == 0 or k == len(sample_indices) - 1:
            log(f"    QLNN gradient {k + 1}/{len(sample_indices)}")
    return fisher / float(len(sample_indices))


# ---------------------------------------------------------------------------
# Top-level pipeline.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Seeds to evaluate (default all 5).",
    )
    parser.add_argument(
        "--n-full",
        type=int,
        default=500,
        help="Full sample size for the headline number (default 500).",
    )
    parser.add_argument(
        "--curve-ns",
        nargs="+",
        type=int,
        default=[100, 200, 350, 500],
        help="Sample sizes for the monotonicity sanity-check curve.",
    )
    parser.add_argument(
        "--skip-curve",
        action="store_true",
        help="Skip the monotonicity curve (just compute the headline n).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = (lambda s: None) if args.quiet else (lambda s: print(s, flush=True))

    # Configs.
    cls_cfg = json.loads((CLASSICAL_DIR / "config.json").read_text())
    qlnn_cfg = json.loads((QLNN_DIR / "config.json").read_text())

    # Both configs should describe the same protocol; use the classical one
    # to materialize training windows.
    x_train, t_train, od_index = _build_training_windows(cls_cfg)
    n_total = int(x_train.shape[0])
    log(f"reconstructed {n_total} training windows (cfg target = 500)")

    n_full = min(int(args.n_full), n_total)
    # Use the *same* sample-index sequence across models/seeds so any
    # difference in d_norm is purely model-driven.
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_total)

    def take(n: int) -> list[int]:
        return perm[:n].tolist()

    curve_ns = sorted({int(v) for v in args.curve_ns if int(v) <= n_full})
    if n_full not in curve_ns and not args.skip_curve:
        curve_ns = sorted(set(curve_ns) | {n_full})

    # ---- Per-seed loop ----
    headline: dict[str, dict[str, Any]] = {"classical": {"seeds": {}}, "qlnn": {"seeds": {}}}
    curve_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        log(f"\n=== seed {seed} ===")
        cls_ckpt = CLASSICAL_DIR / f"seed_{seed}" / "best_state.pt"
        qlnn_ckpt = QLNN_DIR / f"seed_{seed}" / "best_model.eqx"
        if not cls_ckpt.exists():
            log(f"  skipping seed {seed}: missing classical ckpt {cls_ckpt}")
            continue
        if not qlnn_ckpt.exists():
            log(f"  skipping seed {seed}: missing QLNN ckpt {qlnn_ckpt}")
            continue

        # Classical
        cls_model = _rebuild_classical(cls_cfg, cls_ckpt, od_index)
        F_cls_full = _fisher_classical(cls_model, x_train, t_train, take(n_full), log=log)
        d_cls_full = ed_torch.normalized_effective_dimension(
            torch.as_tensor(F_cls_full), n=n_full
        )
        D_cls = int(F_cls_full.shape[0])
        log(f"  classical d_norm @ n={n_full}: {d_cls_full:.4f}  (D={D_cls})")
        headline["classical"]["seeds"][str(seed)] = {
            "d_norm": d_cls_full,
            "n": n_full,
            "D": D_cls,
        }

        # QLNN
        qlnn_model = _rebuild_qlnn(qlnn_cfg, qlnn_ckpt, od_index, seed)
        F_q_full = _fisher_qlnn(qlnn_model, x_train, t_train, take(n_full), log=log)
        d_q_full = ed_jax.normalized_effective_dimension(jnp.asarray(F_q_full), n=n_full)
        D_q = int(F_q_full.shape[0])
        log(f"  QLNN      d_norm @ n={n_full}: {d_q_full:.4f}  (D={D_q})")
        headline["qlnn"]["seeds"][str(seed)] = {
            "d_norm": d_q_full,
            "n": n_full,
            "D": D_q,
        }

        # ---- Monotonicity curve (compute on a single seed each — seed 0 by
        # default — to keep wall-clock manageable). Doing it for every seed
        # is also fine; the gradients are reusable so it's nearly free for
        # the classical side, and even for QLNN it's just gradient
        # accumulation reordering. We re-run from scratch per n for
        # simplicity / readability rather than micro-optimizing.
        if not args.skip_curve:
            for n_k in curve_ns:
                if n_k == n_full:
                    d_cls_k = d_cls_full
                    d_q_k = d_q_full
                else:
                    idxs = take(n_k)
                    F_cls_k = _fisher_classical(cls_model, x_train, t_train, idxs, log=log)
                    d_cls_k = ed_torch.normalized_effective_dimension(
                        torch.as_tensor(F_cls_k), n=n_k
                    )
                    F_q_k = _fisher_qlnn(qlnn_model, x_train, t_train, idxs, log=log)
                    d_q_k = ed_jax.normalized_effective_dimension(jnp.asarray(F_q_k), n=n_k)
                curve_rows.append(
                    {"model": "classical_H4", "seed": seed, "n": n_k, "d_norm": d_cls_k}
                )
                curve_rows.append(
                    {"model": "qlnn_h3", "seed": seed, "n": n_k, "d_norm": d_q_k}
                )

    # ---- Aggregate ----
    def _agg(seeds_dict: dict[str, dict[str, Any]]) -> dict[str, float]:
        vals = [v["d_norm"] for v in seeds_dict.values()]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n_seeds": 0}
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return {
            "mean": mean,
            "std": std,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_seeds": len(vals),
        }

    headline["classical"]["aggregate"] = _agg(headline["classical"]["seeds"])
    headline["qlnn"]["aggregate"] = _agg(headline["qlnn"]["seeds"])

    # Monotonicity check (per-model, per-seed).
    monotone_summary: dict[str, dict[str, bool]] = {"classical_H4": {}, "qlnn_h3": {}}
    if not args.skip_curve:
        df_curve = pd.DataFrame(curve_rows)
        for (model_name, seed), grp in df_curve.groupby(["model", "seed"]):
            grp_sorted = grp.sort_values("n")
            d = grp_sorted["d_norm"].to_numpy()
            mono = bool(np.all(np.diff(d) > -1e-6))
            monotone_summary[model_name][str(seed)] = mono

    # ---- Write artifacts ----
    threshold_met = (
        headline["qlnn"]["aggregate"].get("mean", float("nan"))
        - headline["classical"]["aggregate"].get("mean", float("nan"))
    )
    out_json: dict[str, Any] = {
        "protocol": {
            "n_headline": n_full,
            "curve_ns": curve_ns,
            "seeds": args.seeds,
        },
        "classical_H4": headline["classical"],
        "qlnn_h3": headline["qlnn"],
        "delta_d_norm_qlnn_minus_classical": float(threshold_met),
        "threshold": 1.0,
        "pre_registered_hypothesis_met": bool(threshold_met > 1.0),
        "monotonicity_check": monotone_summary,
    }
    (OUTPUT_DIR / "effective_dimension.json").write_text(json.dumps(out_json, indent=2) + "\n")

    # Markdown table.
    md_lines: list[str] = []
    md_lines.append("# Empirical-Fisher effective dimension (Abbas et al. 2021)\n")
    md_lines.append(f"Sample size n = {n_full}. Trained-theta single-θ specialization.\n")
    md_lines.append("| Model | D | mean d_norm | std | min | max | n_seeds |")
    md_lines.append("|---|---|---|---|---|---|---|")
    for tag, label in (("classical", "Classical Liquid-ODE (H=4)"), ("qlnn", "QLNN (h=3)")):
        agg = headline[tag]["aggregate"]
        any_seed = next(iter(headline[tag]["seeds"].values()), None)
        D = any_seed["D"] if any_seed else "—"
        md_lines.append(
            f"| {label} | {D} | {agg['mean']:.4f} | {agg.get('std', 0.0):.4f} | "
            f"{agg.get('min', float('nan')):.4f} | {agg.get('max', float('nan')):.4f} | "
            f"{agg['n_seeds']} |"
        )
    md_lines.append("")
    md_lines.append(f"**Δd_norm = d(QLNN) − d(classical) = {threshold_met:+.4f}**")
    md_lines.append(
        f"**Pre-registered acceptance threshold (Claim 2): Δd_norm > 1.0 — "
        f"{'MET' if threshold_met > 1.0 else 'NOT MET'}.**"
    )
    md_lines.append("")
    md_lines.append("## Per-seed numbers")
    md_lines.append("")
    md_lines.append("| Seed | classical d_norm | QLNN d_norm | Δ |")
    md_lines.append("|---|---|---|---|")
    for s in args.seeds:
        c = headline["classical"]["seeds"].get(str(s))
        q = headline["qlnn"]["seeds"].get(str(s))
        if c is None or q is None:
            continue
        md_lines.append(
            f"| {s} | {c['d_norm']:.4f} | {q['d_norm']:.4f} | "
            f"{(q['d_norm'] - c['d_norm']):+.4f} |"
        )
    if not args.skip_curve:
        md_lines.append("")
        md_lines.append("## Monotonicity sanity check (d_norm vs n)")
        md_lines.append("")
        df_curve = pd.DataFrame(curve_rows)
        for model_name, grp in df_curve.groupby("model"):
            pivot = grp.pivot_table(index="seed", columns="n", values="d_norm")
            md_lines.append(f"### {model_name}")
            md_lines.append("")
            md_lines.append("| seed | " + " | ".join(f"n={c}" for c in pivot.columns) + " |")
            md_lines.append("|" + "---|" * (len(pivot.columns) + 1))
            for seed_idx, row in pivot.iterrows():
                md_lines.append(
                    f"| {seed_idx} | " + " | ".join(f"{v:.4f}" for v in row.values) + " |"
                )
            mono_seeds = monotone_summary.get(model_name, {})
            all_mono = all(mono_seeds.values()) if mono_seeds else False
            md_lines.append("")
            md_lines.append(
                f"Monotonic increasing across n for every seed: "
                f"**{'YES' if all_mono else 'NO'}** "
                f"(per-seed: {mono_seeds})"
            )
            md_lines.append("")
    (OUTPUT_DIR / "effective_dimension.md").write_text("\n".join(md_lines) + "\n")

    # Monotonicity CSV.
    if not args.skip_curve:
        with (OUTPUT_DIR / "monotonicity_check.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "seed", "n", "d_norm"])
            w.writeheader()
            for r in curve_rows:
                w.writerow(r)

    log(f"\nartifacts written to: {OUTPUT_DIR}")
    log(
        f"headline: classical={headline['classical']['aggregate']['mean']:.4f} "
        f"vs QLNN={headline['qlnn']['aggregate']['mean']:.4f} "
        f"(Δ={threshold_met:+.4f}, threshold>1.0: "
        f"{'MET' if threshold_met > 1.0 else 'NOT MET'})"
    )


if __name__ == "__main__":
    main()
