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

# --- Prior-topology additions (fair-comparison expansion) ----------------
# Distinct circuit topologies from the prior axis-ablation grid + the
# dedup'd unique Optuna specs + promoted runs. Each is a (family,
# num_qubits, num_layers, encoding, entanglement) tuple. Run at NATIVE
# regime (R0_control) only — the 4-regime study stays scoped to the core
# 4 ansatz families already in the matrix. Topologies equal to a core
# family baseline (4q/3L/ring/rx, i.e. the R0 cells) are auto-dropped by
# the dedup in main(). entanglement "-" = ansatz template controls it
# internally (strongly_entangling); brickwall ignores it too.
PRIOR_TOPOLOGIES = [
    # axis-ablation grid (entanglement / encoding / depth / qubits axes)
    ("data_reuploading", 4, 3, "rx", "linear"),
    ("data_reuploading", 4, 3, "rx", "all_to_all"),
    ("data_reuploading", 4, 3, "ry", "ring"),
    ("data_reuploading", 4, 1, "rx", "ring"),
    ("data_reuploading", 4, 2, "rx", "ring"),
    ("data_reuploading", 4, 5, "rx", "ring"),
    ("data_reuploading", 2, 3, "rx", "ring"),
    ("data_reuploading", 6, 3, "rx", "ring"),
    # dedup'd unique Optuna specs (20, from results/circuit_search_optuna)
    ("hardware_efficient", 4, 2, "ry", "linear"),
    ("brickwall", 2, 1, "rx", "all_to_all"),
    ("brickwall", 6, 5, "ry", "linear"),
    ("strongly_entangling", 2, 2, "rx", "-"),
    ("hardware_efficient", 4, 3, "rx", "linear"),
    ("brickwall", 6, 1, "ry", "ring"),
    ("data_reuploading", 4, 5, "ry", "linear"),
    ("data_reuploading", 4, 3, "rx", "ring"),    # == dr baseline → dropped
    ("brickwall", 4, 5, "ry", "all_to_all"),
    ("strongly_entangling", 6, 2, "ry", "-"),
    ("hardware_efficient", 2, 3, "rx", "linear"),
    ("strongly_entangling", 4, 1, "rx", "-"),
    ("data_reuploading", 6, 1, "rx", "linear"),
    ("strongly_entangling", 6, 3, "rx", "-"),    # == promoted top3
    ("strongly_entangling", 4, 3, "ry", "-"),
    ("data_reuploading", 4, 2, "rx", "linear"),
    ("hardware_efficient", 4, 5, "rx", "linear"),
    ("brickwall", 4, 3, "rx", "all_to_all"),
    ("brickwall", 4, 3, "rx", "-"),
]
# Core-family R0 baselines (already in the 16 ansatz×regime cells) — any
# PRIOR_TOPOLOGIES entry equal to one of these is redundant and dropped.
_CORE_BASELINES = {
    ("data_reuploading", 4, 3, "rx", "ring"),
    ("hardware_efficient", 4, 3, "rx", "ring"),
    ("strongly_entangling", 4, 3, "rx", "-"),
    ("brickwall", 4, 3, "rx", "-"),
}


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


def _qlnn_topology_cfg(data: dict, fam: str, q: int, l: int,
                       enc: str, ent: str) -> dict:
    """A prior-topology QLNN at NATIVE regime (R0_control)."""
    params: dict = {"encoding": enc}
    if fam in ("data_reuploading", "hardware_efficient") and ent != "-":
        params["entanglement"] = ent
    if fam == "brickwall":
        params["reupload"] = False
    r = REGIMES["R0_control"]
    return {
        "data": data, "windows": _windows(), "split": _split(),
        "model": {
            "ansatz": {"name": fam, "params": params},
            "num_qubits": q, "num_layers": l, "delta_scale": 0.1,
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


def _classical_ablation_cfg(data: dict, kind: str) -> dict:
    """Bioreactor-origin classical ablations, generalized as model
    variants at the matched H=4 capacity.
      dopri5  — adaptive ODE solver instead of fixed-step euler
      physics — logistic-growth loss regularizer (lambda_logistic>0)
    """
    cfg = _classical_cfg(data, 4)
    if kind == "dopri5":
        cfg["model"]["ode_method"] = "dopri5"
    elif kind == "physics":
        cfg["physics"]["lambda_logistic"] = 0.1
    else:
        raise ValueError(kind)
    return cfg


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

    # --- Fair-comparison expansion -----------------------------------
    # Classical bioreactor-origin ablations as model variants (H=4).
    for kind in ("dopri5", "physics"):
        models.append(("classical", f"classical_H4_{kind}",
                       lambda d, k=kind: _classical_ablation_cfg(d, k)))
    # Prior QLNN topologies at NATIVE regime, dedup'd against the core
    # family R0 baselines and against each other.
    seen: set = set(_CORE_BASELINES)
    n_prior = 0
    for (fam, q, l, enc, ent) in PRIOR_TOPOLOGIES:
        key = (fam, q, l, enc, ent)
        if key in seen:
            continue
        seen.add(key)
        n_prior += 1
        mk = f"qlnn_prior_{fam}_q{q}l{l}_{enc}_{ent.replace('-', 'tmpl')}"
        models.append(("qlnn", mk,
                       lambda d, f=fam, q=q, l=l, e=enc, t=ent:
                       _qlnn_topology_cfg(d, f, q, l, e, t)))
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

    # fixed-OD-clip: a qZETA-ONLY data-preprocessing ablation (NOT a
    # model — a fixed [0,od_max] clip is undefined for signed-state
    # ODEs). One extra classical config on qZETA, explicitly flagged.
    fc = _classical_cfg(datasets["qzeta_od"], 4)
    fc["data"] = dict(fc["data"])
    fc["data"]["od_max"] = 3.8     # legacy fixed-bounds mode (vs train-MinMax)
    fc["unified_matrix"] = {"dataset": "qzeta_od",
                            "model": "classical_H4_fixed_od_clip",
                            "stack": "classical", "proxy_seeds": len(PROXY_SEEDS),
                            "qzeta_only": True,
                            "note": "data-preprocessing ablation, not a model"}
    (OUT / "qzeta_od__classical_H4_fixed_od_clip.yaml").write_text(
        HEADER + yaml.safe_dump(fc, sort_keys=False))
    manifest["configs"].append({"stem": "qzeta_od__classical_H4_fixed_od_clip",
                                "dataset": "qzeta_od",
                                "model": "classical_H4_fixed_od_clip",
                                "stack": "classical", "qzeta_only": True})
    n += 1

    (OUT / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    n_classical = len(CLASSICAL_HIDDEN) + 2
    n_qlnn = len(ANSATZ_FAMILIES) * len(REGIMES) + n_prior
    print(f"wrote {n} configs ({len(datasets)} datasets × "
          f"{len(models)} models + 1 qZETA-only fixed-clip) → {OUT}/")
    print(f"  classical: {n_classical} (5 capacity + dopri5 + physics)")
    print(f"  qlnn: {n_qlnn} (16 family×regime + {n_prior} prior topologies)")
    print(f"  + qZETA-only: fixed_od_clip (flagged, not a model)")


if __name__ == "__main__":
    main()
