"""P4 commit 3b — supervised training loop for VectorForecaster.

Trains a `VectorForecaster` on one-step-ahead pairs `(history, next_state)`
drawn from a long numerical-RK4 reference trajectory.

Pipeline:

  1. `prepare_windows(trajectory, window_length)` — slice the
     `(N, d)` reference trajectory into `(N - window_length, T, d)`
     histories + `(N - window_length, d)` next-state targets.

  2. `train_test_split(trajectory, train_frac=0.7)` — chronological
     split (per pre-reg §3.2 "the same sampling cadence used to train
     the forecaster is used to roll out"). Test trajectory is the
     tail, NOT shuffled.

  3. `train_vector_forecaster(model, X_windows, Y_targets, *, steps,
     lr)` — optax-adam minimization of MSE over the train windows.
     JIT-compiled inner step.

NOTE: rf_qrc has its own closed-form ridge train path
(`RFQRCForecaster.fit(X, Y)`) so it's NOT routed through this
module. Per-family adapters in commit 3c handle the dispatch
between this gradient-trained path and the rf_qrc ridge path.

Output: a trained `VectorForecaster` whose `model(history)` produces
the predicted next state. The autoregressive rollout adapter then
wraps `model` into the `OneStepForecaster` protocol from
`qlnn_.evaluation.rollout`.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_windows(
    trajectory: np.ndarray, window_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a trajectory into history windows + next-state targets.

    Args:
      trajectory    : `(N, d)` — the canonical RK4 reference field.
      window_length : T — the model's input history length.

    Returns:
      X_windows : `(N - window_length, T, d)` — sliding history windows.
      Y_targets : `(N - window_length, d)` — the state one step
                  beyond each window.

    Example: with N=10, T=3, the windows are
      [t=0..2, t=1..3, ..., t=6..8]  (7 windows)
    and the targets are
      [traj[3], traj[4], ..., traj[9]]  (7 targets).

    Raises ValueError on shape / size errors.
    """
    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory must be 2-D (N, d), got shape {trajectory.shape}")
    if window_length < 2:
        raise ValueError(
            f"window_length must be >= 2, got {window_length}")
    N = trajectory.shape[0]
    if N <= window_length:
        raise ValueError(
            f"trajectory length {N} must exceed window_length {window_length} "
            f"by at least 1 (need ≥ 1 target after the window)")

    n_windows = N - window_length
    d = trajectory.shape[1]
    X_windows = np.empty((n_windows, window_length, d),
                          dtype=np.float64)
    for i in range(n_windows):
        X_windows[i] = trajectory[i:i + window_length]
    Y_targets = trajectory[window_length:]   # (n_windows, d)
    return X_windows, Y_targets


def train_test_split(
    trajectory: np.ndarray, *, train_frac: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split (NOT shuffled — pre-reg §3.2 binding).

    Args:
      trajectory : `(N, d)` reference trajectory.
      train_frac : fraction in [0, 1]; default 0.7.

    Returns: (train_trajectory, test_trajectory).
    The split is by ROW INDEX (chronological), not random sampling.
    Tests the model's ability to forecast the tail given only the head.
    """
    if not 0 < train_frac < 1:
        raise ValueError(
            f"train_frac must be in (0, 1), got {train_frac}")
    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory must be 2-D (N, d), got shape {trajectory.shape}")
    n_train = int(trajectory.shape[0] * train_frac)
    if n_train < 2:
        raise ValueError(
            f"train fraction {train_frac} gives {n_train} samples — "
            f"need >= 2 for a non-trivial split")
    return trajectory[:n_train], trajectory[n_train:]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def mse_loss(
    model, X_windows: jnp.ndarray, Y_targets: jnp.ndarray,
) -> jnp.ndarray:
    """Mean squared error of model predictions vs targets.

    The model is `vmap`'d over the leading window axis. Both arrays
    are JAX `jnp.ndarray`s for JIT-friendliness.
    """
    preds = jax.vmap(model)(X_windows)              # (n_windows, d)
    err = preds - Y_targets                          # (n_windows, d)
    return jnp.mean(err ** 2)


def train_vector_forecaster(
    model,
    X_windows: jnp.ndarray,
    Y_targets: jnp.ndarray,
    *,
    steps: int = 1000,
    lr: float = 1e-3,
    batch_size: int | None = None,
    log_every: int = 0,
    seed: int = 0,
) -> tuple["VectorForecaster", list[float]]:
    """Train `model` to predict one-step-ahead via optax-adam MSE.

    Args:
      model       : `VectorForecaster` (Equinox module).
      X_windows   : `(n_windows, T, d)` history windows.
      Y_targets   : `(n_windows, d)` next-state targets.
      steps       : optimizer steps.
      lr          : adam learning rate (default 1e-3).
      batch_size  : optional mini-batch (default None = full-batch).
      log_every   : if > 0, append loss to history every k steps;
                    else only at the end.
      seed        : RNG seed for mini-batch shuffling (ignored if
                    batch_size is None).

    Returns:
      (trained_model, loss_history). Loss history is the MSE on the
      full training set sampled at the requested cadence.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")

    X = jnp.asarray(X_windows, dtype=jnp.float32)
    Y = jnp.asarray(Y_targets, dtype=jnp.float32)
    n = X.shape[0]

    opt = optax.adam(lr)
    # eqx.partition isolates trainable leaves from static config.
    diff_model, static_model = eqx.partition(model, eqx.is_array)
    opt_state = opt.init(diff_model)

    @eqx.filter_jit
    def step_fn(diff_m, opt_s, x_batch, y_batch):
        def loss_for_diff(dm):
            m = eqx.combine(dm, static_model)
            return mse_loss(m, x_batch, y_batch)
        loss, grads = jax.value_and_grad(loss_for_diff)(diff_m)
        updates, opt_s = opt.update(grads, opt_s)
        diff_m = eqx.apply_updates(diff_m, updates)
        return diff_m, opt_s, loss

    history: list[float] = []
    rng = np.random.default_rng(seed)
    for step in range(steps):
        if batch_size is None or batch_size >= n:
            x_b, y_b = X, Y
        else:
            idx = rng.choice(n, size=batch_size, replace=False)
            x_b, y_b = X[idx], Y[idx]

        diff_model, opt_state, loss = step_fn(
            diff_model, opt_state, x_b, y_b)

        if log_every > 0 and (step + 1) % log_every == 0:
            history.append(float(loss))
        elif log_every == 0 and step == steps - 1:
            history.append(float(loss))

    trained = eqx.combine(diff_model, static_model)
    return trained, history
