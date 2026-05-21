"""P5 commit 5 — H1 verdict module tests.

The H1 verdict is the publication-critical headline number; the
mechanical decision rule MUST be implemented exactly per pre-reg §7.
Tests cover:
  - Regime partition (pre-reg locked)
  - CellRecord delta computation
  - Underfit + skyline guards
  - Paired bootstrap statistical properties (CI matches expected
    distribution properties)
  - Mechanical decision: CONFIRMED on clean smooth-favored data
  - Mechanical decision: FALSIFIED on broad-favored data
  - Mechanical decision: FALSIFIED on zero-effect data
  - Mechanical decision: INCONCLUSIVE when a regime is excluded
"""
from __future__ import annotations

import numpy as np
import pytest

from quantum_liquid_neuralode.evaluation.h1_verdict import (
    BROADBAND_MULTISCALE_SYSTEMS,
    CellRecord,
    SMOOTH_PERIODIC_SYSTEMS,
    apply_guards,
    h1_bootstrap,
    h1_verdict,
    regime_for_system,
)


# ---------- regime partition (pre-reg locked) -----------------------------

def test_regime_partition_lock():
    """The pre-reg's regime partition (locked at commit 2646d74) MUST
    NOT be changed without a pre-reg amendment + commit."""
    assert "lotka_volterra" in SMOOTH_PERIODIC_SYSTEMS
    assert "van_der_pol" in SMOOTH_PERIODIC_SYSTEMS
    assert "kuramoto" in SMOOTH_PERIODIC_SYSTEMS
    assert "burgers_smooth" in SMOOTH_PERIODIC_SYSTEMS
    assert "heat" in SMOOTH_PERIODIC_SYSTEMS

    assert "lorenz" in BROADBAND_MULTISCALE_SYSTEMS
    assert "fitzhugh_nagumo" in BROADBAND_MULTISCALE_SYSTEMS
    assert "kdv" in BROADBAND_MULTISCALE_SYSTEMS
    assert "allen_cahn" in BROADBAND_MULTISCALE_SYSTEMS

    # No system is in BOTH regimes.
    overlap = set(SMOOTH_PERIODIC_SYSTEMS) & set(BROADBAND_MULTISCALE_SYSTEMS)
    assert overlap == set(), f"regime overlap: {overlap}"


def test_regime_for_system_routes_correctly():
    assert regime_for_system("lotka_volterra") == "smooth_periodic"
    assert regime_for_system("lorenz") == "broadband_multiscale"


def test_regime_for_system_rejects_unknown():
    with pytest.raises(ValueError, match="unknown system"):
        regime_for_system("nonsense")


# ---------- CellRecord ----------------------------------------------------

def test_cell_record_delta_sign_convention():
    """Δ = NeuralODE − QLNN; positive when QLNN is better
    (improvement-direction convention from pre-reg §7)."""
    c = CellRecord(
        system="lotka_volterra", seed=0,
        qlnn_relL2=0.2, neuralode_relL2=0.5)
    assert c.delta == pytest.approx(0.3)
    # Now QLNN is worse:
    c2 = CellRecord(
        system="lotka_volterra", seed=0,
        qlnn_relL2=0.5, neuralode_relL2=0.2)
    assert c2.delta == pytest.approx(-0.3)


def test_cell_record_regime_property():
    c = CellRecord(system="lorenz", seed=0,
                   qlnn_relL2=0.5, neuralode_relL2=0.6)
    assert c.regime == "broadband_multiscale"


# ---------- guards --------------------------------------------------------

def test_apply_guards_keeps_clean_cells():
    cells = [
        CellRecord(system="lotka_volterra", seed=s,
                   qlnn_relL2=0.1, neuralode_relL2=0.2,
                   qlnn_train_relL2=0.05, neuralode_train_relL2=0.05,
                   skyline_relL2=0.01)
        for s in range(3)
    ] + [
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.5, neuralode_relL2=0.6,
                   qlnn_train_relL2=0.05, neuralode_train_relL2=0.05,
                   skyline_relL2=0.05)
        for s in range(3)
    ]
    kept, ex = apply_guards(cells)
    assert len(kept) == 6
    assert ex["underfit"] == []
    assert ex["skyline_out_of_reach"] == []


def test_apply_guards_excludes_underfit_cells():
    cells = [
        CellRecord(system="lotka_volterra", seed=0,
                   qlnn_relL2=0.1, neuralode_relL2=0.2,
                   qlnn_train_relL2=0.8),   # >> threshold
    ]
    kept, ex = apply_guards(cells, underfit_threshold=0.5)
    assert kept == []
    assert "lotka_volterra_seed0_qlnn" in ex["underfit"]


def test_apply_guards_excludes_skyline_out_of_reach():
    """A system whose skyline relL2 (mean) > threshold is excluded —
    ALL cells of that system are dropped per pre-reg §7."""
    cells = [
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.9, neuralode_relL2=0.95,
                   skyline_relL2=0.8)   # skyline can't reach this system
        for s in range(3)
    ]
    kept, ex = apply_guards(cells, skyline_threshold=0.5)
    assert kept == []
    assert len(ex["skyline_out_of_reach"]) == 3
    assert all("lorenz" in tag for tag in ex["skyline_out_of_reach"])


# ---------- bootstrap statistical properties ------------------------------

def test_bootstrap_point_estimate_matches_means():
    """The point estimate (delta_diff_mean) should equal the sample
    means' difference, not the bootstrap-averaged value."""
    cells = []
    # Smooth cells: QLNN beats NeuralODE by 0.3 on average.
    for s in range(5):
        cells.append(CellRecord(
            system="lotka_volterra", seed=s,
            qlnn_relL2=0.1, neuralode_relL2=0.4))
    # Broad cells: tie.
    for s in range(5):
        cells.append(CellRecord(
            system="lorenz", seed=s,
            qlnn_relL2=0.5, neuralode_relL2=0.5))

    result = h1_bootstrap(cells, n_iter=500, seed=0)
    assert result["delta_smooth_mean"] == pytest.approx(0.3)
    assert result["delta_broad_mean"] == pytest.approx(0.0)
    assert result["delta_diff_mean"] == pytest.approx(0.3)


def test_bootstrap_ci_excludes_zero_for_strong_effect():
    """With a strong, deterministic effect (Δ_smooth = 0.3, Δ_broad = 0),
    the bootstrap 95% CI should exclude 0."""
    cells = [
        CellRecord(system="lotka_volterra", seed=s,
                   qlnn_relL2=0.1, neuralode_relL2=0.4)
        for s in range(10)
    ] + [
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.5, neuralode_relL2=0.5)
        for s in range(10)
    ]
    result = h1_bootstrap(cells, n_iter=2000, seed=0)
    assert result["ci_low"] > 0, (
        f"strong-effect CI should be > 0; got [{result['ci_low']}, "
        f"{result['ci_high']}]")
    # Note: with zero-variance (deterministic) per-regime samples,
    # bootstrap CI collapses to a single point. >= captures that.
    assert result["ci_high"] >= result["ci_low"]


def test_bootstrap_ci_includes_zero_for_null_effect():
    """If Δ_smooth ≈ Δ_broad, the CI should include 0."""
    cells = [
        CellRecord(system="lotka_volterra", seed=s,
                   qlnn_relL2=0.3, neuralode_relL2=0.3)
        for s in range(5)
    ] + [
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.5, neuralode_relL2=0.5)
        for s in range(5)
    ]
    result = h1_bootstrap(cells, n_iter=2000, seed=0)
    assert result["ci_low"] <= 0 <= result["ci_high"], (
        f"null-effect CI should include 0; got [{result['ci_low']:.4f}, "
        f"{result['ci_high']:.4f}]")


def test_bootstrap_rejects_empty_regime():
    cells = [
        CellRecord(system="lotka_volterra", seed=0,
                   qlnn_relL2=0.1, neuralode_relL2=0.2),
    ]
    with pytest.raises(ValueError, match="BROADBAND"):
        h1_bootstrap(cells, n_iter=100)


# ---------- mechanical decision: CONFIRMED -------------------------------

def test_h1_verdict_confirmed_on_strong_smooth_favored_effect():
    """Clean smooth-favored effect → CONFIRMED."""
    cells = [
        CellRecord(system="lotka_volterra", seed=s,
                   qlnn_relL2=0.1, neuralode_relL2=0.4)
        for s in range(10)
    ] + [
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.5, neuralode_relL2=0.5)
        for s in range(10)
    ]
    result = h1_verdict(cells, n_iter=2000, seed=0)
    assert result["outcome"] == "CONFIRMED", (
        f"got {result['outcome']}; reasoning: {result['reasoning']}")
    assert result["bootstrap"]["delta_diff_mean"] > 0.2


# ---------- mechanical decision: FALSIFIED -------------------------------

def test_h1_verdict_falsified_on_broad_favored_effect():
    """If QLNN beats NeuralODE more on BROAD than SMOOTH, that's the
    inverse of H1 → FALSIFIED (CI is negative)."""
    cells = [
        CellRecord(system="lotka_volterra", seed=s,
                   qlnn_relL2=0.5, neuralode_relL2=0.5)
        for s in range(10)
    ] + [
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.1, neuralode_relL2=0.4)
        for s in range(10)
    ]
    result = h1_verdict(cells, n_iter=2000, seed=0)
    assert result["outcome"] == "FALSIFIED"
    assert result["bootstrap"]["delta_diff_mean"] < -0.2


def test_h1_verdict_falsified_on_null_effect():
    """No effect → FALSIFIED (CI includes 0)."""
    cells = [
        CellRecord(system="lotka_volterra", seed=s,
                   qlnn_relL2=0.3, neuralode_relL2=0.3)
        for s in range(5)
    ] + [
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.5, neuralode_relL2=0.5)
        for s in range(5)
    ]
    result = h1_verdict(cells, n_iter=2000, seed=0)
    assert result["outcome"] == "FALSIFIED"


# ---------- mechanical decision: INCONCLUSIVE -----------------------------

def test_h1_verdict_inconclusive_when_broad_regime_excluded():
    """If skyline-out-of-reach removes all BROADBAND cells, the
    verdict is INCONCLUSIVE for that regime."""
    cells = [
        CellRecord(system="lotka_volterra", seed=s,
                   qlnn_relL2=0.1, neuralode_relL2=0.4,
                   skyline_relL2=0.05)
        for s in range(3)
    ] + [
        # Lorenz: all skyline relL2 > threshold → out-of-reach
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.9, neuralode_relL2=0.95,
                   skyline_relL2=0.9)
        for s in range(3)
    ]
    result = h1_verdict(cells, skyline_threshold=0.5)
    assert result["outcome"] == "INCONCLUSIVE"
    assert "broad" in result["reasoning"].lower() or \
           "BROADBAND" in result["reasoning"]


def test_h1_verdict_outputs_required_keys():
    """The output dict must be JSON-serializable and have the
    expected keys for h1_analysis.json."""
    import json
    cells = [
        CellRecord(system="lotka_volterra", seed=s,
                   qlnn_relL2=0.1, neuralode_relL2=0.4)
        for s in range(3)
    ] + [
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.5, neuralode_relL2=0.5)
        for s in range(3)
    ]
    result = h1_verdict(cells, n_iter=500, seed=0)
    # Must serialize cleanly to JSON.
    json_str = json.dumps(result)
    parsed = json.loads(json_str)
    # Required top-level keys per spec:
    for k in ("outcome", "reasoning", "bootstrap", "guards", "thresholds"):
        assert k in parsed, f"missing required key '{k}' in h1_verdict output"
    assert parsed["outcome"] in ("CONFIRMED", "FALSIFIED", "INCONCLUSIVE")
