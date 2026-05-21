"""P4 commit 3c — per-family adapters into the OneStepForecaster protocol.

Each forecaster family has a different "predict next state" calling
convention; this module wraps each into the canonical
`(history: (T, d), dt) → next_state: (d,)` signature so the
generic `autoregressive_rollout(...)` from `rollout.py` is
family-agnostic.

Three adapter factories:

  1. `make_vector_forecaster_adapter(trained_model)` — wraps a
     gradient-trained `VectorForecaster` (the 4 registry quantum
     ansätze: data_reuploading, hardware_efficient,
     strongly_entangling, brickwall). The model already accepts
     `(T, d)` and emits `(d,)`; the adapter just strips/ignores `dt`
     (intrinsic to the training cadence).

  2. `make_rf_qrc_adapter(trained_rf_qrc)` — wraps the closed-form
     Tikhonov-ridge-trained `RFQRCForecaster`. Subtlety: rf_qrc has
     an internal leaky-integrator state that's recomputed from the
     input sequence each call. To respect that, we feed the FULL
     history each rollout step and take the LAST prediction. O(T)
     per step but T is small (≤ 24).

  3. `make_classical_mlp_adapter(trained_mlp)` — wraps a classical
     MLP forecaster (P5 baseline). Accepts a flat-vectorized history
     and emits the next state vector. Placeholder signature; the
     concrete MLP module lands in P5.

All adapters return a callable that satisfies
`qlnn_.evaluation.rollout.OneStepForecaster`. The adapter does NOT
own the model; the caller binds the trained parameters at
adapter-construction time, then passes the adapter to
`autoregressive_rollout(...)`.

Per-family training paths (already implemented):
  - Vector QLNN families: `train_vector_forecaster` (gradient
    descent + adam; module `forecaster_training.py`).
  - rf_qrc: `RFQRCForecaster.fit(X, Y)` (closed-form ridge).

Per-family fit data convention (consistent across all adapters):
  X : (n_pairs, T, d)  → reshape to (n_pairs, T·d) for rf_qrc
  Y : (n_pairs, d)
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# VectorForecaster (QLNN-cell + vector head) adapter
# ---------------------------------------------------------------------------


def make_vector_forecaster_adapter(
    trained_model,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Adapter for `VectorForecaster` (gradient-trained QLNN family).

    Args:
      trained_model : a trained `VectorForecaster` (Equinox module).

    Returns:
      Callable `(history, dt) → next_state` matching the
      OneStepForecaster protocol. The `dt` argument is ignored — the
      model's step is intrinsic to its training-data cadence (the
      `step_dt` it was built with). Pre-reg §3.2: "the same sampling
      cadence used to train the forecaster is used to roll out."

    The adapter is stateless and JAX-traceable; can be wrapped in
    `jax.jit` for fast rollout.
    """

    def one_step(history: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
        # `dt` is intentionally unused — see docstring.
        return trained_model(history)

    return one_step


# ---------------------------------------------------------------------------
# rf_qrc adapter — closed-form ridge + leaky integrator
# ---------------------------------------------------------------------------


def make_rf_qrc_adapter(
    trained_rf_qrc,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Adapter for `RFQRCForecaster` (closed-form Tikhonov ridge).

    Subtlety: rf_qrc has an internal leaky-integrator state
    `r(t) = (1-eps)·r(t-1) + eps·r_hat(t)` (paper Eq. 2, p.4).
    Each call to `.predict(X)` recomputes this state from scratch
    starting from `r(0) = 0`. To respect the temporal context we
    feed the FULL history window each rollout step and take the
    LAST predicted state.

    This is O(T) per rollout step (vs O(1) for stateless models),
    but T is small (≤ 24) so the overhead is negligible.

    The adapter converts JAX arrays to numpy (rf_qrc internals are
    pure numpy via numpy.linalg.solve in fit) and back. Output is
    a JAX `jnp.ndarray` for compatibility with the rollout loop.
    """

    def one_step(history: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
        # `dt` unused: rf_qrc's step is intrinsic to its training cadence.
        h_np = np.asarray(history, dtype=np.float64)
        y_np = trained_rf_qrc.predict(h_np)              # (T, d) predictions
        # Last prediction is the "next state" — Y[t] is the prediction at
        # one step beyond x[t] when trained on the conventional (x_t, x_{t+1})
        # pairs (see forecaster_training.prepare_windows).
        return jnp.asarray(y_np[-1])

    return one_step


# ---------------------------------------------------------------------------
# Classical MLP adapter (P5 baseline placeholder)
# ---------------------------------------------------------------------------


def make_classical_mlp_adapter(
    trained_mlp,
    *,
    flatten_history: bool = True,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Adapter for a generic classical MLP forecaster (P5 baseline).

    The mandatory pre-reg §6 plain-MLP baseline isn't built yet
    (lands in P5). This factory establishes the adapter signature
    so the P5 module just plugs in.

    Args:
      trained_mlp     : callable model with signature
                        `mlp(x_flat_or_history) → next_state`. P5
                        will define the concrete module shape.
      flatten_history : if True (default), pass `history.reshape(-1)`
                        to the MLP. If False, pass the 2-D `(T, d)`
                        unchanged (for MLPs that handle their own
                        flattening).

    Returns: `OneStepForecaster`-protocol callable.
    """

    def one_step(history: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
        if flatten_history:
            x = history.reshape(-1)
        else:
            x = history
        return trained_mlp(x)

    return one_step


# ---------------------------------------------------------------------------
# Identity adapter (the persistence floor — never reported as a win)
# ---------------------------------------------------------------------------


def make_persistence_adapter() -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Persistence baseline: predict next = current.

    Per pre-reg §6 "persistence is a triviality floor; a model
    below the floor is discarded, not reported as a win." This
    adapter exists so the floor can be plotted on the same axes as
    the trained models, never as a competitor in H1.
    """

    def one_step(history: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
        return history[-1]

    return one_step


def make_linear_extrapolation_adapter() -> Callable[
    [jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Linear-extrapolation floor: `next = x[-1] + (x[-1] - x[-2])`.

    The second triviality floor from pre-reg §6. Same role as
    persistence — never reported as a win, plotted for context.
    Requires T ≥ 2; the rollout's window_length is always ≥ 2.
    """

    def one_step(history: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * history[-1] - history[-2]

    return one_step
