"""P4 evaluation subpackage — rollout + pre-registered metric suite.

The autoregressive rollout path required by `ODE_PDE_PRE_REG.md` §3.2
+ §5. Two modules:

  - `rollout.py` — generic `autoregressive_rollout(...)` adapter
    that wraps any one-step forecaster (QLNN, classical MLP,
    Neural-ODE baseline, rf_qrc) under a single callable protocol.
  - `rollout_metrics.py` — the pre-reg §5 locked metric suite:
    `relative_l2_error` (primary endpoint), `valid_prediction_time`
    (with optional Lyapunov normalization), `spectral_error` (FFT
    PSD L2), `invariant_drift` (with KdV mass/energy + LV invariant
    built in), and the `LYAPUNOV_EXPONENT` per-system table.

NOTE: 1-step / h-step MAE is BANNED as a headline per the pre-reg.
Permitted only as a sanity diagnostic, never as a paper claim.
"""

from qlnn_.evaluation.rollout import (
    OneStepForecaster,
    autoregressive_rollout,
    autoregressive_rollout_python_loop,
    autoregressive_rollout_with_history,
    make_history_slider,
)
from qlnn_.evaluation.rollout_metrics import (
    LYAPUNOV_EXPONENT,
    RolloutMetrics,
    VPTResult,
    invariant_drift,
    kdv_energy,
    kdv_mass,
    lotka_volterra_invariant,
    normalize_to_lyapunov_time,
    relative_l2_error,
    relative_l2_over_time,
    spectral_error,
    valid_prediction_time,
)

__all__ = [
    # rollout
    "OneStepForecaster",
    "autoregressive_rollout",
    "autoregressive_rollout_python_loop",
    "autoregressive_rollout_with_history",
    "make_history_slider",
    # metrics
    "LYAPUNOV_EXPONENT",
    "RolloutMetrics",
    "VPTResult",
    "invariant_drift",
    "kdv_energy",
    "kdv_mass",
    "lotka_volterra_invariant",
    "normalize_to_lyapunov_time",
    "relative_l2_error",
    "relative_l2_over_time",
    "spectral_error",
    "valid_prediction_time",
]
