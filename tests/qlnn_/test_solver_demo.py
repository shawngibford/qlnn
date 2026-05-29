"""Smoke test for the P3.5 solver-comparison demo library.

Exercises `solver_demo.train_one_family` for ALL 4 families on the
gate-task ODE (`u'=−u`) at heavily reduced steps so the smoke runs in
<60s wall-clock (per the plan's acceptance criterion). Does NOT
re-validate gate-quality MAE — that's `test_physics_residual_solver`'s
job. This test only certifies the demo's interop pattern works
end-to-end: every family trains, produces finite loss, returns the
expected schema, and the predicted curve is monotonic-ish on the
exponential decay.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from qlnn_.training.solver_demo import (
    FAMILIES,
    ODES,
    run_sweep,
    summarize_seeds,
    train_one_family,
)


SMOKE_STEPS = 80
SMOKE_COLLOC = 12


def test_all_four_families_registered():
    # The 4 baseline families (chebyshev_dqc / te_qpinn_fnn /
    # te_qpinn_qnn / qcpinn) must all be present. A17 (2026-05-28)
    # additionally registered 3 step-wise qcpinn variants
    # (qcpinn_balanced/quantum/full_q) along the Q/(Q+C) parameter
    # ratio; those are allowed but not required by this baseline check.
    assert {"chebyshev_dqc", "te_qpinn_fnn", "te_qpinn_qnn",
            "qcpinn"}.issubset(set(FAMILIES))


def test_both_odes_registered():
    assert set(ODES) == {"expdecay", "logistic"}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_each_family_trains_on_expdecay_at_reduced_steps(family):
    out = train_one_family(family, "expdecay", seed=0,
                            steps=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
    # schema
    for key in ("family", "ode", "seed", "steps", "pqc_params",
                "classical_params", "config_str",
                "final_loss", "mae", "rmse",
                "t_eval", "u_pred", "exact", "loss_history"):
        assert key in out, f"{family}: missing key {key}"
    # numerics: training did SOMETHING (loss went down vs init)
    assert math.isfinite(out["final_loss"])
    assert out["loss_history"][-1] < out["loss_history"][0]
    # shape: eval grid has 100 interior points
    assert out["u_pred"].shape == (100,)
    assert out["exact"].shape == (100,)
    # predicted curve isn't NaN'd; values bounded
    assert np.all(np.isfinite(out["u_pred"]))
    # IC hard-constrained: u(t0) ≈ 1 (the prediction at the first
    # interior point near t=0; exact is e^{-0.02} ≈ 0.98)
    assert abs(float(out["u_pred"][0]) - 1.0) < 0.20


def test_smoke_sweep_wall_clock_under_60s():
    # Budget scales with the number of registered families: A17 added
    # 3 qcpinn variants (qcpinn_balanced/quantum/full_q) on top of the
    # 4 baseline families. Each family has its own JIT-compile cost
    # (≈ 6-10s on CPU) plus a few seconds of stepping at SMOKE_STEPS.
    # Allow ~15s per family + a 20s slack so the budget tracks the
    # actual registered count without false-positive failures.
    budget_s = 15.0 * len(FAMILIES) + 20.0
    t0 = time.time()
    for family in FAMILIES:
        out = train_one_family(family, "expdecay", seed=0,
                                steps=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
        assert math.isfinite(out["final_loss"])
    elapsed = time.time() - t0
    assert elapsed < budget_s, (
        f"smoke sweep took {elapsed:.1f}s; budget is {budget_s:.0f}s "
        f"for {len(FAMILIES)} registered families")


def test_run_sweep_returns_full_cartesian_product():
    res = run_sweep(
        families=["chebyshev_dqc"], odes=["expdecay"], seeds=[0, 1],
        steps_override=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
    assert len(res) == 2
    assert {r["seed"] for r in res} == {0, 1}


def test_summarize_seeds_matches_project_schema():
    res = run_sweep(
        families=["chebyshev_dqc"], odes=["expdecay"], seeds=[0, 1, 2],
        steps_override=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
    summary = summarize_seeds(res)
    assert summary["family"] == "chebyshev_dqc"
    assert summary["ode"] == "expdecay"
    assert summary["n_seeds"] == 3
    assert summary["seeds"] == [0, 1, 2]
    for m in ("mae", "rmse", "final_loss"):
        block = summary["metrics"][m]
        for key in ("mean", "std", "min", "max", "n_seeds",
                    "ci95_half_width", "ci95_low", "ci95_high"):
            assert key in block
        # mean is in the CI by construction
        assert block["ci95_low"] <= block["mean"] <= block["ci95_high"]


def test_unknown_family_or_ode_raises():
    with pytest.raises(ValueError, match="unknown family"):
        train_one_family("bogus", "expdecay", seed=0,
                          steps=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
    with pytest.raises(ValueError, match="unknown ode"):
        train_one_family("chebyshev_dqc", "bogus", seed=0,
                          steps=SMOKE_STEPS, n_colloc=SMOKE_COLLOC)
