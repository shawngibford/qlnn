"""Unified model × dataset matrix — the SAME model suite evaluated
identically on every dataset so the ODE battery and the bioreactor OD
forecasting are directly comparable.

Model suite (21, dataset-agnostic — defined once, applied everywhere):
  classical (5): param-sweep capacity axis hidden_size ∈ {2,4,8,16,32}
                 (train_baseline.py). The classical "regime" axis IS
                 capacity — lr_schedule / init_circuit_std are QLNN-only
                 knobs, so adding R1-R3 to classical would be ill-defined.
  qlnn     (16): 4 ansatz families {data_reuploading, hardware_efficient,
                 strongly_entangling, brickwall} × 4 regularization
                 regimes {R0_control, R1_weight_decay, R2_physics_prior,
                 R3_smooth_convergence} (train_qlnn.py). Generalizes the
                 Option-B 3×4=12 to the full 4×4.

Datasets (11): qzeta_od + 5 ODE systems × {m472, full}
  - m472 (~778 rows → ~472 train windows) = EXACT qZETA parity, the
    head-to-head comparison with the data-volume confound removed.
  - full (4000 rows) = data-scaling ablation.

= 231 configs. LOCKED protocol everywhere (window=24, h=3, 70/15/15,
3-seed PROXY budget — promotion to 5 seeds is a separate gated step,
same funnel as Option-B). Emitted to configs/unified_matrix/.

Prereq: scripts/generate_synthetic_ode_data.py (writes the manifest).

Usage:
    python scripts/generate_unified_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "synthetic" / "manifest.json"
OUT = REPO_ROOT / "configs" / "unified_matrix"

PROXY_SEEDS = [0, 1, 2]

# --- Regularization regimes (identical knob semantics to Option-B) -------
REGIMES = {
    "R0_control": dict(weight_decay=0.0, lr=0.002, lr_schedule="constant",
                       grad_clip_norm=1.0, init_circuit_std=0.05,
                       lambda_logistic=0.0),
    "R1_weight_decay": dict(weight_decay=0.001, lr=0.002,
                            lr_schedule="constant", grad_clip_norm=1.0,
                            init_circuit_std=0.05, lambda_logistic=0.0),
    "R2_physics_prior": dict(weight_decay=0.0, lr=0.002,
                             lr_schedule="constant", grad_clip_norm=1.0,
                             init_circuit_std=0.05, lambda_logistic=0.1),
    "R3_smooth_convergence": dict(weight_decay=0.0, lr=0.002,
                                  lr_schedule="cosine", grad_clip_norm=0.5,
                                  init_circuit_std=0.02, lambda_logistic=0.0),
}
ANSATZ_FAMILIES = {
    "data_reuploading":   {"entanglement": "ring", "encoding": "rx"},
    "hardware_efficient": {"entanglement": "ring", "encoding": "rx"},
    "strongly_entangling": {"encoding": "rx"},
    "brickwall":          {"encoding": "rx", "reupload": False},
}
CLASSICAL_HIDDEN = [2, 4, 8, 16, 32]


def _windows():
    return {"window_size": 24, "stride": 1,
            "horizon_hours": 3.0, "horizon_tol_hours": 0.0835}


def _split():
    return {"train_ratio": 0.70, "val_ratio": 0.15}


def _datasets() -> dict[str, dict]:
    """Return {dataset_key: data_block}. qZETA + all ODE variants."""
    ds = {
        "qzeta_od": {
            "csv_path": "data/raw/qZETA_data_copy.csv",
            "feature_cols": ["OD", "PRE", "TEMP_EXT", "TEMP_CULTURE",
                             "PAR_LIGHT", "PH", "DO"],
            "target_col": "OD",
            "od_min": 0.0, "od_max": None, "od_phys_max": 3.8,
        }
    }
    man = json.loads(MANIFEST.read_text())
    for key, info in man.items():
        ds[key] = {
            "csv_path": info["csv"],
            "feature_cols": info["feature_cols"],
            "target_col": info["target_col"],
            "od_min": None, "od_max": None,
            "od_phys_max": None,   # ODE states are signed — no clip
        }
    return ds


def _classical_cfg(data: dict, hidden: int) -> dict:
    return {
        "data": data, "windows": _windows(), "split": _split(),
        "model": {"hidden_size": hidden, "tau_min": 0.1,
                  "forecast_steps": 4, "delta_scale": 0.1,
                  "ode_method": "euler"},
        "training": {"epochs": 300, "batch_size": 64, "lr": 0.002,
                     "weight_decay": 0.0, "eval_every": 10,
                     "patience": 10, "grad_clip_norm": 1.0},
        "physics": {"lambda_logistic": 0.0},
        "seeds": list(PROXY_SEEDS),
    }


def _qlnn_cfg(data: dict, family: str, regime: str) -> dict:
    r = REGIMES[regime]
    return {
        "data": data, "windows": _windows(), "split": _split(),
        "model": {
            "ansatz": {"name": family, "params": ANSATZ_FAMILIES[family]},
            "num_qubits": 4, "num_layers": 3, "delta_scale": 0.1,
            "tau_min": 0.1, "tau_init": 1.0, "init_head_std": 0.1,
            "init_circuit_std": r["init_circuit_std"],
            "solver": "tsit5", "rtol": 0.001, "atol": 0.0001,
            "dt0": 0.05, "max_steps": 12288,
        },
        "training": {"epochs": 60, "batch_size": 32, "lr": r["lr"],
                     "weight_decay": r["weight_decay"], "eval_every": 5,
                     "patience": 6, "grad_clip_norm": r["grad_clip_norm"],
                     "lr_schedule": r["lr_schedule"]},
        "physics": {"lambda_logistic": r["lambda_logistic"],
                    "mu_norm": 0.4, "K_norm": 1.0},
        "seeds": list(PROXY_SEEDS),
    }


HEADER = (
    "# Auto-generated by scripts/generate_unified_matrix.py.\n"
    "# Unified model×dataset matrix — SAME model suite, every dataset,\n"
    "# LOCKED protocol (window=24,h=3,70/15/15,3-seed proxy).\n"
)


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("run scripts/generate_synthetic_ode_data.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = _datasets()
    manifest = {"datasets": list(datasets), "models": [], "configs": []}

    # Model suite (dataset-agnostic identity).
    models = []
    for h in CLASSICAL_HIDDEN:
        models.append(("classical", f"classical_H{h}",
                        lambda d, h=h: _classical_cfg(d, h)))
    for fam in ANSATZ_FAMILIES:
        for reg in REGIMES:
            models.append(("qlnn", f"qlnn_{fam}__{reg}",
                           lambda d, f=fam, r=reg: _qlnn_cfg(d, f, r)))
    manifest["models"] = [m[1] for m in models]

    n = 0
    for ds_key, data in datasets.items():
        for stack, model_key, builder in models:
            cfg = builder(data)
            cfg["unified_matrix"] = {
                "dataset": ds_key, "model": model_key, "stack": stack,
                "proxy_seeds": len(PROXY_SEEDS),
            }
            stem = f"{ds_key}__{model_key}"
            (OUT / f"{stem}.yaml").write_text(
                HEADER + yaml.safe_dump(cfg, sort_keys=False))
            manifest["configs"].append({"stem": stem, "dataset": ds_key,
                                        "model": model_key, "stack": stack})
            n += 1
    (OUT / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {n} configs ({len(datasets)} datasets × "
          f"{len(models)} models) → {OUT}/")
    print(f"  classical models: {len(CLASSICAL_HIDDEN)}  "
          f"qlnn models: {len(ANSATZ_FAMILIES) * len(REGIMES)}")


if __name__ == "__main__":
    main()
