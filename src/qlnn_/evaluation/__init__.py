"""P4 evaluation subpackage — rollout + pre-registered metric suite.

The autoregressive rollout path required by `ODE_PDE_PRE_REG.md` §3.2
+ §5. Three modules:

  - `rollout.py` — generic `autoregressive_rollout(...)` adapter
    that wraps any one-step forecaster (QLNN, classical MLP,
    Neural-ODE baseline, rf_qrc) under a single callable protocol.
  - `rollout_metrics.py` (planned next commit) — the locked metric
    suite: relative-L2, valid-prediction-time (VPT), spectral
    error (FFT-based PSD L2), invariant drift.

NOTE: 1-step / h-step MAE is BANNED as a headline per the pre-reg.
Permitted only as a sanity diagnostic, never as a paper claim.
"""

from qlnn_.evaluation.rollout import (
    OneStepForecaster,
    autoregressive_rollout,
    autoregressive_rollout_with_history,
    make_history_slider,
)

__all__ = [
    "OneStepForecaster",
    "autoregressive_rollout",
    "autoregressive_rollout_with_history",
    "make_history_slider",
]
