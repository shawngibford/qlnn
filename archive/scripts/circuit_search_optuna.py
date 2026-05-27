"""Optuna-driven Bayesian search over QLNN ansatz topology.

Phase 3 of the circuit search plan. Runs a discrete TPE search over

    ansatz family   ∈ {data_reuploading, hardware_efficient, strongly_entangling, brickwall}
    entanglement    ∈ {linear, ring, all_to_all}        (skipped for strongly_entangling)
    encoding        ∈ {rx, ry}
    num_layers      ∈ {1, 2, 3, 5}
    num_qubits      ∈ {2, 4, 6}

Each trial trains a single-seed QLNN with the locked h=3 evaluation protocol
and full epoch budget; the objective is the best validation MSE_norm
(min — lower is better). Per-trial artifacts are saved to
`results/circuit_search_optuna/trial_NNNN/` (same shape as a `train_qlnn.py`
run output so `summarize_circuit_search.py` can pick them up too).

Study state is persisted in a SQLite DB so the search is **resumable**
across sessions — just re-run with the same `--study-name`.

Usage:
    # Full search (~50 trials, ~4 hours)
    python scripts/circuit_search_optuna.py --n-trials 50

    # Resume an existing study
    python scripts/circuit_search_optuna.py --n-trials 20 \\
        --study-name qlnn_circuit_search_v1

    # CI smoke (3 trials, 5 epochs each)
    python scripts/circuit_search_optuna.py --n-trials 3 --epochs 5 \\
        --study-name SMOKE
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import optuna
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = "qlnn_circuit_search_v1"
OUT_BASE = REPO_ROOT / "results" / "circuit_search_optuna"
OUT_BASE.mkdir(parents=True, exist_ok=True)


# Discrete search-space definitions (kept here so they're easy to grep / edit).
ANSATZ_FAMILIES = (
    "data_reuploading",
    "hardware_efficient",
    "strongly_entangling",
    "brickwall",
)
ENTANGLEMENTS = ("linear", "ring", "all_to_all")
ENCODINGS = ("rx", "ry")
NUM_LAYERS = (1, 2, 3, 5)
NUM_QUBITS = (2, 4, 6)


def _suggest_ansatz(trial: optuna.Trial) -> dict[str, Any]:
    family = trial.suggest_categorical("ansatz_family", ANSATZ_FAMILIES)
    num_qubits = int(trial.suggest_categorical("num_qubits", NUM_QUBITS))
    num_layers = int(trial.suggest_categorical("num_layers", NUM_LAYERS))
    encoding = trial.suggest_categorical("encoding", ENCODINGS)

    params: dict[str, Any] = {"encoding": encoding}
    # Ansätze whose entanglement topology is FIXED by the template (not
    # parameterized): suggesting an entanglement knob for them would silently
    # do nothing and TPE would overfit to a phantom hyperparameter, wasting
    # trials on apparent duplicates.
    #
    # `strongly_entangling` uses qml.StronglyEntanglingLayers internally.
    # `brickwall` uses alternating even-odd CNOT pairs — also fixed.
    FIXED_TOPOLOGY = {"strongly_entangling", "brickwall"}
    if family not in FIXED_TOPOLOGY:
        params["entanglement"] = trial.suggest_categorical(
            "entanglement", ENTANGLEMENTS
        )

    return {
        "family": family,
        "num_qubits": num_qubits,
        "num_layers": num_layers,
        "params": params,
    }


def _write_trial_config(spec: dict[str, Any], path: Path) -> None:
    """Compose a complete train_qlnn YAML by layering the trial's ansatz
    spec on top of the reference template.
    """
    ref_yaml = REPO_ROOT / "configs" / "circuit_search" / "reference.yaml"
    cfg = yaml.safe_load(ref_yaml.read_text())
    cfg["circuit_search"] = {
        "axis": "optuna",
        "level": f"{spec['family']}_Q{spec['num_qubits']}_L{spec['num_layers']}_{spec['params'].get('encoding','rx')}",
        "is_reference": False,
    }
    cfg["model"]["num_qubits"] = spec["num_qubits"]
    cfg["model"]["num_layers"] = spec["num_layers"]
    cfg["model"]["ansatz"] = {"name": spec["family"], "params": spec["params"]}
    cfg["seeds"] = [0]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _objective(args: argparse.Namespace) -> "callable":
    """Return the Optuna objective closure (closes over CLI args)."""
    def objective(trial: optuna.Trial) -> float:
        spec = _suggest_ansatz(trial)
        trial.set_user_attr("ansatz_spec", spec)

        run_dir = OUT_BASE / f"trial_{trial.number:04d}"
        if run_dir.exists():
            shutil.rmtree(run_dir)

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "trial.yaml"
            _write_trial_config(spec, cfg_path)

            cmd = [
                args.python, "scripts/train_qlnn.py",
                "--config", str(cfg_path),
                "--output-dir", str(run_dir),
                "--seeds", "0",
                "--quiet",
            ]
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]

            logging.info(f"trial {trial.number}: spec={spec}")
            # If invoked from a git worktree where the editable install points
            # at the main repo, prepend the worktree's src/ to PYTHONPATH so
            # the new ansatz modules are visible to the subprocess.
            env = os.environ.copy()
            worktree_src = REPO_ROOT / "src"
            if (worktree_src / "qlnn_" / "circuits" / "protocol.py").exists():
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{worktree_src}:{existing}" if existing else str(worktree_src)
            proc = subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env,
            )
            if proc.returncode != 0:
                logging.warning(
                    f"trial {trial.number} FAILED — stderr tail:\n"
                    + "\n".join(proc.stderr.splitlines()[-15:])
                )
                # Tell Optuna to skip this point instead of poisoning the study
                # with a bogus 0.0 minimum.
                raise optuna.exceptions.TrialPruned()

        summary_path = run_dir / "seeds_summary.json"
        if not summary_path.exists():
            raise optuna.exceptions.TrialPruned()
        summary = json.loads(summary_path.read_text())
        val_mse = summary["val"]["mse_norm"]["mean"]
        trial.set_user_attr("test_mae_raw_mean", summary["test"]["mae_raw"]["mean"])
        trial.set_user_attr("test_r2_raw_mean", summary["test"]["r2_raw"]["mean"])
        return float(val_mse)

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna circuit search for the QLNN.")
    parser.add_argument("--study-name", default=DEFAULT_STUDY)
    parser.add_argument("--storage", default=str(OUT_BASE / "study.db"),
                        help="SQLite path; pass an explicit URL for other backends.")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs per trial (default = YAML value).")
    parser.add_argument("--python", default=".venv/bin/python",
                        help="Python interpreter to call scripts/train_qlnn.py with.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    storage_url = (
        args.storage if "://" in args.storage else f"sqlite:///{args.storage}"
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    study.optimize(_objective(args), n_trials=args.n_trials, gc_after_trial=True)

    print(f"\n=== Study {args.study_name} — {len(study.trials)} total trials ===")
    print(f"best val MSE_norm = {study.best_value:.6f}")
    print(f"best params       = {study.best_params}")
    print(f"best ansatz spec  = {study.best_trial.user_attrs.get('ansatz_spec')}")

    # Persist a JSON summary of the top-10 for human review.
    rows = []
    for t in sorted(study.trials, key=lambda t: (t.value if t.value is not None else float('inf'))):
        if t.value is None:
            continue
        rows.append({
            "number": t.number,
            "val_mse_norm": t.value,
            "test_mae_raw_mean": t.user_attrs.get("test_mae_raw_mean"),
            "test_r2_raw_mean": t.user_attrs.get("test_r2_raw_mean"),
            "ansatz_spec": t.user_attrs.get("ansatz_spec"),
        })
    out = OUT_BASE / f"{args.study_name}_top.json"
    out.write_text(json.dumps(rows[:25], indent=2) + "\n")
    print(f"\ntop-25 trials written to {out}")


if __name__ == "__main__":
    sys.exit(main())
