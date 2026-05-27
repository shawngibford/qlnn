"""P6 Gap G6 — forecaster underfit-guard exclusion tests.

The pre-reg A6 amendment mandates an underfit-guard exclusion for
the forecaster H1 aggregator: any cell whose train-side relative-L2
exceeds the A1 threshold (0.5) is excluded from H1 aggregation, with
the inclusion treated as INCONCLUSIVE for that regime rather than
silently FALSIFIED/CONFIRMED.

This test exercises the existing `apply_guards` / `h1_verdict` code
path through synthetic per-cell fixtures with a mix of below- and
above-threshold `qlnn_train_relL2` values, and confirms:

  1. Cells with train_relL2 > 0.5 are excluded.
  2. The verdict JSON exposes the A6 alias `excluded_cells_a6`.
  3. When exclusion drops a regime's support to zero, the verdict
     outcome flips to INCONCLUSIVE (not CONFIRMED / FALSIFIED).
  4. Legacy cells (train_relL2 is None) are silently skipped by the
     guard — the WARN logging lives in the runner scripts, the
     aggregator just treats None as "guard inactive for this cell".
"""
from __future__ import annotations

from quantum_liquid_neuralode.evaluation.h1_verdict import (
    CellRecord, apply_guards, h1_verdict,
)


# A1 underfit threshold — pre-reg amendment A1 + A6.
A1_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# 1. Direct exclusion: cells above threshold are dropped.
# ---------------------------------------------------------------------------


def test_underfit_guard_excludes_above_threshold_cells():
    cells = [
        # Smooth: one clean, one underfit (qlnn above threshold).
        CellRecord(system="lotka_volterra", seed=0,
                   qlnn_relL2=0.1, neuralode_relL2=0.2,
                   qlnn_train_relL2=0.05, neuralode_train_relL2=0.05),
        CellRecord(system="van_der_pol", seed=0,
                   qlnn_relL2=0.2, neuralode_relL2=0.3,
                   qlnn_train_relL2=0.8,  # > 0.5 — UNDERFIT
                   neuralode_train_relL2=0.05),
        # Broad: one clean, one underfit (neuralode side).
        CellRecord(system="lorenz", seed=0,
                   qlnn_relL2=0.4, neuralode_relL2=0.5,
                   qlnn_train_relL2=0.1,
                   neuralode_train_relL2=0.9),  # > 0.5 — UNDERFIT
        CellRecord(system="lorenz", seed=1,
                   qlnn_relL2=0.45, neuralode_relL2=0.55,
                   qlnn_train_relL2=0.1, neuralode_train_relL2=0.1),
    ]
    kept, ex = apply_guards(cells, underfit_threshold=A1_THRESHOLD)
    assert len(kept) == 2
    kept_tags = {f"{c.system}_seed{c.seed}" for c in kept}
    assert kept_tags == {"lotka_volterra_seed0", "lorenz_seed1"}
    assert "van_der_pol_seed0_qlnn" in ex["underfit"]
    assert "lorenz_seed0_neuralode" in ex["underfit"]


# ---------------------------------------------------------------------------
# 2. Verdict exposes the A6 alias key.
# ---------------------------------------------------------------------------


def test_verdict_exposes_excluded_cells_a6_alias():
    """`excluded_cells_a6` is the pre-reg A6 alias for the underfit list."""
    cells = [
        CellRecord(system="lotka_volterra", seed=s,
                   qlnn_relL2=0.1, neuralode_relL2=0.4,
                   qlnn_train_relL2=0.05, neuralode_train_relL2=0.05)
        for s in range(3)
    ] + [
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.5, neuralode_relL2=0.5,
                   qlnn_train_relL2=0.05, neuralode_train_relL2=0.05)
        for s in range(3)
    ] + [
        CellRecord(system="van_der_pol", seed=0,
                   qlnn_relL2=0.2, neuralode_relL2=0.3,
                   qlnn_train_relL2=0.95,  # underfit
                   neuralode_train_relL2=0.05),
    ]
    v = h1_verdict(cells, n_iter=500,
                   underfit_threshold=A1_THRESHOLD,
                   skyline_threshold=10.0, seed=0)
    guards = v["guards"]
    assert "excluded_cells_a6" in guards
    assert guards["excluded_cells_a6"] == guards["excluded_underfit"]
    assert "van_der_pol_seed0_qlnn" in guards["excluded_cells_a6"]


# ---------------------------------------------------------------------------
# 3. Verdict flips to INCONCLUSIVE when a regime's support drops to 0.
# ---------------------------------------------------------------------------


def test_verdict_inconclusive_when_regime_emptied_by_underfit():
    """Every smooth cell underfit → no smooth survivors → INCONCLUSIVE."""
    cells = [
        # All smooth cells are underfit on the QLNN side.
        CellRecord(system="lotka_volterra", seed=s,
                   qlnn_relL2=0.1, neuralode_relL2=0.4,
                   qlnn_train_relL2=0.9,  # > 0.5
                   neuralode_train_relL2=0.05)
        for s in range(3)
    ] + [
        # Broad cells are clean.
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.5, neuralode_relL2=0.5,
                   qlnn_train_relL2=0.05, neuralode_train_relL2=0.05)
        for s in range(3)
    ]
    v = h1_verdict(cells, n_iter=500,
                   underfit_threshold=A1_THRESHOLD,
                   skyline_threshold=10.0, seed=0)
    assert v["outcome"] == "INCONCLUSIVE", (
        f"got {v['outcome']}; reasoning: {v['reasoning']}")
    assert v["bootstrap"] is None
    # All three smooth cells should be in the A6 exclusion list.
    a6 = v["guards"]["excluded_cells_a6"]
    assert len(a6) == 3
    assert all("lotka_volterra" in tag for tag in a6)


def test_verdict_inconclusive_when_both_regimes_emptied():
    """Both regimes underfit → INCONCLUSIVE (mentions smooth first)."""
    cells = [
        CellRecord(system="lotka_volterra", seed=s,
                   qlnn_relL2=0.1, neuralode_relL2=0.4,
                   qlnn_train_relL2=0.9, neuralode_train_relL2=0.05)
        for s in range(2)
    ] + [
        CellRecord(system="lorenz", seed=s,
                   qlnn_relL2=0.5, neuralode_relL2=0.5,
                   qlnn_train_relL2=0.9, neuralode_train_relL2=0.05)
        for s in range(2)
    ]
    v = h1_verdict(cells, n_iter=500,
                   underfit_threshold=A1_THRESHOLD,
                   skyline_threshold=10.0, seed=0)
    assert v["outcome"] == "INCONCLUSIVE"
    assert v["guards"]["kept_n"] == 0


# ---------------------------------------------------------------------------
# 4. Legacy cells (train_relL2 = None) are silently skipped by the guard.
# ---------------------------------------------------------------------------


def test_legacy_cells_with_none_train_relL2_are_kept():
    """A cell whose train_relative_l2 is None (legacy pre-G6) must NOT
    be excluded — the guard simply does not run on that cell. The
    WARN-skip notification is the runner script's responsibility."""
    cells = [
        # Legacy cell: train_relL2 unrecorded.
        CellRecord(system="lotka_volterra", seed=0,
                   qlnn_relL2=0.1, neuralode_relL2=0.2,
                   qlnn_train_relL2=None, neuralode_train_relL2=None),
        # Modern cell: train_relL2 within threshold.
        CellRecord(system="lorenz", seed=0,
                   qlnn_relL2=0.4, neuralode_relL2=0.5,
                   qlnn_train_relL2=0.1, neuralode_train_relL2=0.1),
    ]
    kept, ex = apply_guards(cells, underfit_threshold=A1_THRESHOLD)
    assert len(kept) == 2, (
        "legacy cells (train_relL2=None) should pass the guard untouched")
    assert ex["underfit"] == []


def test_legacy_and_modern_mixed_excludes_only_modern_underfit():
    """In a mix of legacy (None) and modern (above-threshold) cells,
    only the modern underfit cells are excluded. The legacy ones are
    silently kept (the WARN lives in the runner)."""
    cells = [
        # Legacy modern-pass on smooth side.
        CellRecord(system="lotka_volterra", seed=0,
                   qlnn_relL2=0.1, neuralode_relL2=0.4,
                   qlnn_train_relL2=None),
        CellRecord(system="van_der_pol", seed=0,
                   qlnn_relL2=0.2, neuralode_relL2=0.5,
                   qlnn_train_relL2=0.1, neuralode_train_relL2=0.1),
        # Modern underfit on broad side → excluded.
        CellRecord(system="lorenz", seed=0,
                   qlnn_relL2=0.4, neuralode_relL2=0.5,
                   qlnn_train_relL2=0.99, neuralode_train_relL2=0.1),
        # Modern clean broad cell.
        CellRecord(system="lorenz", seed=1,
                   qlnn_relL2=0.4, neuralode_relL2=0.5,
                   qlnn_train_relL2=0.1, neuralode_train_relL2=0.1),
    ]
    kept, ex = apply_guards(cells, underfit_threshold=A1_THRESHOLD)
    kept_tags = {f"{c.system}_seed{c.seed}" for c in kept}
    assert kept_tags == {
        "lotka_volterra_seed0", "van_der_pol_seed0", "lorenz_seed1"}
    assert ex["underfit"] == ["lorenz_seed0_qlnn"]
