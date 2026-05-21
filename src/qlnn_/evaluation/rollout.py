"""P4 — Generic autoregressive rollout for ODE/PDE forecasters.

The pre-reg's §3.2 defines the forecaster task as **autoregressive
multi-step rollout** (NOT 1-step / h-step MAE, which is banned as a
headline per §5 and Risk R3 the persistence trap). The pre-reg's
primary endpoint is the rollout relative-L2 error over the full
pre-registered horizon.

Every forecaster family in our roster has a different "predict the
next state" calling convention:
  - Diffrax QLNN forecaster (`src/qlnn_/models/qlnn_forecaster.py`)
    consumes a `(T, F)` history window + per-step times, internally
    integrates with Diffrax, and returns ONE delta around persistence.
  - rf_qrc (`src/qlnn_/circuits/rf_qrc.py`) consumes a state vector,
    applies a fixed-frozen reservoir feature map, then a closed-form
    ridge readout.
  - The classical MLP forecaster (P5) flattens a history window into
    a single vector and produces a next-state vector via a few dense
    layers.
  - The mandatory non-liquid Neural-ODE baseline (P5) is a Diffrax-
    integrated MLP cell with no quantum + no learnable time-constants.

This module establishes a **single callable protocol** all of them
implement — `OneStepForecaster` — and the generic rollout loop that
slides the history window and accumulates predicted states.

The protocol is deliberately minimal:

    next_state = model(history, dt)

where `history` is `(T, d)` (last T past states, oldest first) and
`dt` is the step size to advance by. Each family ships an adapter
that wraps its native predict method into this signature.

This keeps the rollout loop **family-agnostic** so the same
`autoregressive_rollout(...)` produces the relative-L2 / VPT /
spectral / invariant-drift metrics for every model class. The
rollout metric suite consumes the output trajectory; the rollout
loop produces it.

Design choices (declared, P4):

1. **Sliding-window history.** At each step, we drop the oldest
   state and append the predicted state. Constant window length
   throughout the rollout — matches how the forecaster was trained.
2. **Equispaced grid.** The rollout uses a single `dt` for every
   step. Pre-reg §5 specifies fixed horizons in physical time; the
   step grid is set by the reference numerical solution's grid
   (typically 0.01 sec for ODEs, 0.005 for PDE time-stepping). Per
   pre-reg §3.2: "the same sampling cadence used to train the
   forecaster is used to roll out."
3. **No teacher forcing in rollout.** Pre-reg explicitly excludes
   teacher forcing during evaluation (Risk R3). The rollout sees
   only its own past predictions after the initial history window.
4. **JIT-friendly.** The implementation uses `jax.lax.scan` so the
   inner loop compiles once and the gradient is well-defined
   (useful for back-propagating rollout-loss training in P6+).

Out of scope for P4 commit 1 (this file):
  - The rollout metric suite (relative-L2, VPT, spectral, invariant).
    That's `rollout_metrics.py` — separate commit, separate module.
  - Per-family adapters mapping (QLNN, rf_qrc, MLP, Neural-ODE) into
    `OneStepForecaster`. Those live in the demo CLI and per-family
    wrapper functions — commit 3.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# The protocol every forecaster family adapts to
# ---------------------------------------------------------------------------


@runtime_checkable
class OneStepForecaster(Protocol):
    """Callable that maps a history window + step size → next state.

    Signature:
        next_state = model(history, dt)

    Args:
      history : (T, d) JAX array — the last T past states, oldest
                first. T is the model's training window length.
      dt      : scalar JAX array — the step size to advance by.
                Some families (Diffrax-based) honor it as the
                physical-time advance; others (memoryless MLPs)
                ignore it and rely on the training-time cadence.

    Returns:
      next_state : (d,) JAX array — the predicted state at the
                   moment one `dt` beyond `history[-1]`.

    The protocol is intentionally lax (`runtime_checkable`): any
    callable matching the signature is acceptable. Test fixtures
    use plain functions; production code uses Equinox modules with
    bound parameters via `functools.partial` or closures.
    """

    def __call__(
        self,
        history: jnp.ndarray,
        dt: jnp.ndarray,
    ) -> jnp.ndarray: ...


# ---------------------------------------------------------------------------
# The generic rollout loop
# ---------------------------------------------------------------------------


def autoregressive_rollout(
    model: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    history0: jnp.ndarray,
    n_steps: int,
    dt: float,
) -> jnp.ndarray:
    """Slide `model` forward `n_steps` from `history0` by sliding window.

    At each step i ∈ [0, n_steps):
      1. Call `next = model(history, dt)`.
      2. Slide: `history = concat([history[1:], next[None, :]])`.
      3. Append `next` to the output trajectory.

    Args:
      model    : callable matching `OneStepForecaster` — wraps the
                 family's native predict method into the canonical
                 (history, dt) → next_state signature.
      history0 : (T, d) initial history window. T is the model's
                 training window length; d is the state dimension.
      n_steps  : number of rollout steps (≥ 1).
      dt       : per-step physical-time advance (positive scalar).

    Returns:
      trajectory : (n_steps, d) — only the rolled-out predictions,
                   NOT including the initial history. This convention
                   matches the pre-reg's "predicted rollout" definition.

    Implementation:
      Uses `jax.lax.scan` for JIT-friendliness; the inner loop
      compiles once. Side-effect-free; deterministic given the model
      and `history0`.

    Raises:
      ValueError on bad shape / non-positive n_steps / dt.
    """
    if history0.ndim != 2:
        raise ValueError(
            f"history0 must be 2-D (T, d), got shape {history0.shape}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    dt_arr = jnp.asarray(dt)

    def step_fn(history: jnp.ndarray, _step_index: jnp.ndarray):
        next_state = model(history, dt_arr)        # (d,)
        # Slide the window: drop the oldest, append the prediction.
        new_history = jnp.concatenate(
            [history[1:], next_state[None, :]], axis=0)
        return new_history, next_state

    # scan along an unused index range; we only need the count.
    _, trajectory = jax.lax.scan(
        step_fn, history0, jnp.arange(n_steps))
    return trajectory                              # (n_steps, d)


def autoregressive_rollout_with_history(
    model: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    history0: jnp.ndarray,
    n_steps: int,
    dt: float,
) -> jnp.ndarray:
    """Like `autoregressive_rollout` but prepends the initial history.

    Useful when downstream code (e.g. visualization, spectral error)
    needs the full trajectory including the warm-up window so the
    time axis is continuous from t=0 onwards.

    Returns: (T + n_steps, d) where T = history0.shape[0].
    """
    rollout = autoregressive_rollout(model, history0, n_steps, dt)
    return jnp.concatenate([history0, rollout], axis=0)


# ---------------------------------------------------------------------------
# A non-JIT slider for debugging + callbacks
# ---------------------------------------------------------------------------


def make_history_slider(
    model: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    history0: jnp.ndarray,
    dt: float,
) -> Callable[[], jnp.ndarray]:
    """Return a Python-side closure that advances one step per call.

    Use when the user needs to interleave rollout with side effects
    (logging, early-stop checks, per-step instrumentation). NOT
    JIT-compatible (it mutates closure state) — for performance use
    `autoregressive_rollout` instead.

    Example:
        step = make_history_slider(model, hist0, dt=0.01)
        for i in range(100):
            x_next = step()    # one (d,) prediction at a time
    """
    state = {"history": history0}
    dt_arr = jnp.asarray(dt)

    def step():
        h = state["history"]
        x_next = model(h, dt_arr)
        state["history"] = jnp.concatenate(
            [h[1:], x_next[None, :]], axis=0)
        return x_next

    return step
