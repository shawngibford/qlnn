"""Smoke tests for scripts/make_diagnostic_figures.py.

Fast + offline: verifies (1) the T1 registry is well-formed, (2) every
T1 figure degrades gracefully (prints SKIP, does not raise) when its
on-disk inputs are absent — the same graceful-skip contract the headline
figure module honors.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

mod = importlib.import_module("scripts.make_diagnostic_figures")


def test_t1_registry_is_seven_callables():
    assert len(mod.T1) == 7
    assert all(callable(fn) for fn in mod.T1)
    names = {fn.__name__ for fn in mod.T1}
    assert names == {
        "fig_learning_curves", "fig_forecast_trajectory",
        "fig_pred_vs_actual", "fig_residual_analysis",
        "fig_paired_bootstrap", "fig_seed_strip",
        "fig_all_circuit_diagrams",
    }


def test_supp_registry_and_gallery_skips_without_table(
        tmp_path, monkeypatch, capsys):
    assert [fn.__name__ for fn in mod.SUPP] == ["fig_circuit_gallery"]
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    mod.fig_circuit_gallery()
    assert "SKIP fig_circuit_gallery" in capsys.readouterr().out


def test_t2_registry_is_four_callables():
    assert len(mod.T2) == 4
    assert all(callable(fn) for fn in mod.T2)
    assert {fn.__name__ for fn in mod.T2} == {
        "fig_accuracy_variance_frontier", "fig_regularization_arrows",
        "fig_circuit_regime_heatmap", "fig_master_comparison",
    }


def test_t2_skips_when_option_b_results_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for fn in mod.T2:
        fn()
    assert capsys.readouterr().out.count("SKIP") == 4


def test_all_t1_skip_gracefully_when_data_absent(tmp_path, monkeypatch, capsys):
    """Point ROOT at an empty dir — every figure must SKIP, not raise."""
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "OUT", tmp_path / "figures")
    (tmp_path / "figures").mkdir()
    for fn in mod.T1:
        if fn.__name__ == "fig_all_circuit_diagrams":
            # This one depends only on the ansatz registry, not on-disk
            # data — it should actually succeed. Just ensure no raise.
            fn()
            continue
        fn()
    out = capsys.readouterr().out
    assert out.count("SKIP") >= 6  # the 6 data-dependent figures skipped


def test_to_raw_inverts_minmax(monkeypatch, tmp_path):
    """raw = norm·(hi−lo)+lo using the per-run protocol bounds."""
    run = "results/_unit"
    proto_dir = tmp_path / run
    proto_dir.mkdir(parents=True)
    (proto_dir / "protocol.json").write_text(
        '{"od_data_min": 0.5, "od_data_max": 2.5}')
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    import numpy as np
    got = mod._to_raw(np.array([0.0, 0.5, 1.0]), run)
    assert np.allclose(got, [0.5, 1.5, 2.5])
