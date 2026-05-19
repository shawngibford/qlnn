"""Invariants for the unified model×dataset matrix generator.

The scientific value of the cross-task comparison rests entirely on the
SAME model suite being applied identically to every dataset, with only
the data block differing. These tests pin that contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "unified_matrix"
MAN = CFG / "matrix_manifest.json"

pytestmark = pytest.mark.skipif(
    not MAN.exists(),
    reason="run scripts/generate_unified_matrix.py first")


def _man():
    return json.loads(MAN.read_text())


def test_matrix_dimensions():
    m = _man()
    assert len(m["models"]) == 21          # 5 classical + 16 qlnn
    assert len(m["datasets"]) == 11        # qzeta + 5 ODE × {m472,full}
    assert len(m["configs"]) == 231


def test_model_suite_identity_is_dataset_agnostic():
    """Every dataset must be paired with the EXACT same 21 model keys."""
    m = _man()
    by_ds: dict[str, set] = {}
    for c in m["configs"]:
        by_ds.setdefault(c["dataset"], set()).add(c["model"])
    suites = list(by_ds.values())
    assert all(s == suites[0] for s in suites), \
        "model suite differs across datasets — comparison would be invalid"
    assert len(suites[0]) == 21


def test_qzeta_keeps_physical_clip_ode_disables_it():
    qz = yaml.safe_load((CFG / "qzeta_od__classical_H4.yaml").read_text())
    assert qz["data"]["od_phys_max"] == 3.8
    lz = yaml.safe_load((CFG / "lorenz_m472__classical_H4.yaml").read_text())
    assert lz["data"]["od_phys_max"] is None   # signed states ⇒ no clip


@pytest.mark.parametrize("stem", [
    "qzeta_od__qlnn_brickwall__R3_smooth_convergence",
    "kuramoto_full__qlnn_strongly_entangling__R2_physics_prior",
    "van_der_pol_m472__classical_H16",
])
def test_configs_well_formed(stem):
    y = yaml.safe_load((CFG / f"{stem}.yaml").read_text())
    d = y["data"]
    assert d["target_col"] in d["feature_cols"]
    assert y["seeds"] == [0, 1, 2]                     # proxy budget
    assert y["windows"]["horizon_hours"] == 3.0        # locked protocol
    assert y["unified_matrix"]["dataset"] in stem
    if y["unified_matrix"]["stack"] == "qlnn":
        assert "ansatz" in y["model"]
        assert "lr_schedule" in y["training"]
    else:
        assert "hidden_size" in y["model"]


def test_regimes_actually_differ():
    """R1/R2/R3 must encode distinct regularization vs R0 control."""
    base = yaml.safe_load(
        (CFG / "qzeta_od__qlnn_data_reuploading__R0_control.yaml").read_text())
    wd = yaml.safe_load(
        (CFG / "qzeta_od__qlnn_data_reuploading__R1_weight_decay.yaml"
         ).read_text())
    phys = yaml.safe_load(
        (CFG / "qzeta_od__qlnn_data_reuploading__R2_physics_prior.yaml"
         ).read_text())
    smooth = yaml.safe_load(
        (CFG / "qzeta_od__qlnn_data_reuploading__R3_smooth_convergence.yaml"
         ).read_text())
    assert base["training"]["weight_decay"] == 0.0
    assert wd["training"]["weight_decay"] > 0.0
    assert phys["physics"]["lambda_logistic"] > 0.0
    assert smooth["training"]["lr_schedule"] == "cosine"
    assert smooth["model"]["init_circuit_std"] < base["model"]["init_circuit_std"]
