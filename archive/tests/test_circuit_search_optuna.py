"""Smoke test for the Optuna circuit search driver.

Validates the *plumbing* without actually training a QLNN — we monkey-patch
the subprocess call to write a fake `seeds_summary.json` so the test runs
in seconds and stays offline.

A real end-to-end run (TPESampler picks a circuit, train_qlnn runs, the JSON
is consumed) is exercised by hand from the command line; that path is too
slow for CI.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna")

import scripts.circuit_search_optuna as cso


class _FakeProc:
    returncode = 0
    stderr = ""
    stdout = ""


def _fake_subprocess_run(cmd, *, cwd, capture_output, text, env=None):
    """Fake `train_qlnn.py` by writing the bare-minimum seeds_summary.json
    the optuna objective reads. The `env` kwarg is accepted because the
    real driver passes a PYTHONPATH-injected env dict for worktree support.
    """
    out_dir = Path(cmd[cmd.index("--output-dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    # Derive a deterministic but non-trivial val MSE from the trial number
    # encoded in the dir name so the TPE sampler actually has signal.
    trial_n = int(out_dir.name.split("_")[-1])
    val_mse = 0.08 + 0.001 * (trial_n % 5)
    (out_dir / "seeds_summary.json").write_text(json.dumps({
        "val": {"mse_norm": {"mean": val_mse}},
        "test": {
            "mae_raw": {"mean": 0.25 + 0.002 * trial_n},
            "r2_raw": {"mean": 0.05 - 0.003 * trial_n},
        },
    }))
    return _FakeProc()


def test_optuna_search_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(cso.subprocess, "run", _fake_subprocess_run)
    # Redirect output to a tmp dir so the test never pollutes the repo.
    monkeypatch.setattr(cso, "OUT_BASE", tmp_path)

    import argparse
    args = argparse.Namespace(
        python="python",
        epochs=2,
        n_trials=3,
        study_name="UNIT_TEST",
        storage=str(tmp_path / "study.db"),
    )
    storage_url = f"sqlite:///{args.storage}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(cso._objective(args), n_trials=args.n_trials)

    assert len(study.trials) == 3
    for t in study.trials:
        assert t.value is not None and t.value > 0
        assert "ansatz_spec" in t.user_attrs
        assert t.user_attrs["ansatz_spec"]["family"] in cso.ANSATZ_FAMILIES
        # Each trial wrote its summary into tmp_path/trial_NNNN/
        run_dir = tmp_path / f"trial_{t.number:04d}"
        assert (run_dir / "seeds_summary.json").exists()

    # Cleanup belt-and-suspenders (tmp_path auto-cleans, but be explicit).
    shutil.rmtree(tmp_path / "trial_0000", ignore_errors=True)
