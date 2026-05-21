"""P5 commit 5b — Matched-baselines sweep dispatcher.

Extends `p4_forecaster_demo.py` with the 3 new baseline families
required by pre-reg §6 for the H1 verdict:

  - `plain_neuralode` : non-liquid Diffrax-integrated MLP cell
    (PlainNeuralODEForecaster, P5 commit 1) — THE MANDATORY H1 CONTRAST.
  - `plain_mlp`       : feedforward MLP forecaster (no Diffrax,
    no quantum) — capacity-matched classical control.
  - `skyline`         : known-structure RHS with fitted free params,
    RK4 rollout — the upper-bound used by the skyline guard.

Per-family training paths:
  - plain_neuralode: optax-adam supervised one-step training (same
    pipeline as VectorForecaster — they share the (T, d) → (d,)
    call signature, so the same loss + optimizer + adapter work).
  - plain_mlp: same training pipeline as plain_neuralode.
  - skyline: closed-form least-squares fit of the known structural
    coefficients; NO gradient descent.

Outputs the same per-cell schema as P4: metrics.json + field.npz +
seeds_summary.json with nested metrics matching pre-reg §5 metric
suite (relative_l2, vpt, spectral_error, invariant_drift).

The H1 verdict (commit 5d) reads the P4 forecaster results +
this module's baseline results and computes the (Δ_smooth − Δ_broad)
paired-bootstrap CI per pre-reg §7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from qlnn_.evaluation.forecaster_adapters import (
    make_persistence_adapter,
    make_vector_forecaster_adapter,
)
from qlnn_.evaluation.rollout import (
    autoregressive_rollout,
    autoregressive_rollout_python_loop,
)
from qlnn_.evaluation.rollout_metrics import (
    LYAPUNOV_EXPONENT,
    invariant_drift,
    lotka_volterra_invariant,
    relative_l2_error,
    spectral_error,
    valid_prediction_time,
)
from qlnn_.models.plain_mlp_forecaster import (
    PlainMLPForecaster, PlainMLPForecasterConfig,
)
from qlnn_.models.plain_neuralode_forecaster import (
    PlainNeuralODEForecaster, PlainNeuralODEForecasterConfig,
)
from qlnn_.training.forecaster_training import (
    prepare_windows,
    train_test_split,
    train_vector_forecaster,
)
from qlnn_.training.p4_forecaster_demo import (
    P4SweepConfig,
    SYSTEMS_P4,
)
from qlnn_.training.skyline_baseline import (
    fit_skyline, rollout_skyline,
)


def _simulate(name, *, n_points, seed):
    from quantum_liquid_neuralode.data_processing.synthetic_ode import simulate
    return simulate(name, n_points=n_points, seed=seed)


P5_BASELINE_FAMILIES = ("plain_neuralode", "plain_mlp", "skyline")


# ---------------------------------------------------------------------------
# Per-family trainers
# ---------------------------------------------------------------------------


def _train_plain_neuralode(
    X_train: np.ndarray, Y_train: np.ndarray,
    *, d: int, cfg: P4SweepConfig, seed: int,
) -> tuple[Any, list[float], int]:
    """Train PlainNeuralODEForecaster — the MANDATORY H1 contrast."""
    fc_cfg = PlainNeuralODEForecasterConfig(
        input_dim=d,
        hidden_dim=cfg.num_qubits,            # match QLNN hidden dim
        mlp_hidden=cfg.num_qubits,            # capacity-matched cell
        step_dt=0.05,
        delta_scale_init=cfg.delta_scale_init,
        delta_scale_min=cfg.delta_scale_min,
    )
    model = PlainNeuralODEForecaster(fc_cfg, key=jax.random.PRNGKey(seed))
    trained, history = train_vector_forecaster(
        model, X_train, Y_train,
        steps=cfg.train_steps, lr=cfg.learning_rate,
        log_every=max(1, cfg.train_steps // 5), seed=seed)
    diff_model, _ = eqx.partition(trained, eqx.is_array)
    pcount = sum(
        int(np.asarray(leaf).size)
        for leaf in jax.tree_util.tree_leaves(diff_model))
    return trained, history, pcount


def _train_plain_mlp(
    X_train: np.ndarray, Y_train: np.ndarray,
    *, d: int, cfg: P4SweepConfig, seed: int,
) -> tuple[Any, list[float], int]:
    """Train PlainMLPForecaster — capacity-matched classical control."""
    fc_cfg = PlainMLPForecasterConfig(
        input_dim=d,
        window_length=cfg.window_length,
        hidden_dim=cfg.num_qubits * 2,        # ~ matched capacity
        n_hidden_layers=2,
        delta_scale_init=cfg.delta_scale_init,
        delta_scale_min=cfg.delta_scale_min,
    )
    model = PlainMLPForecaster(fc_cfg, key=jax.random.PRNGKey(seed))
    trained, history = train_vector_forecaster(
        model, X_train, Y_train,
        steps=cfg.train_steps, lr=cfg.learning_rate,
        log_every=max(1, cfg.train_steps // 5), seed=seed)
    return trained, history, model.num_parameters()


# ---------------------------------------------------------------------------
# Per-cell dispatch
# ---------------------------------------------------------------------------


def train_and_rollout_baseline_cell(
    system: str, family: str, seed: int,
    *, cfg: P4SweepConfig | None = None,
) -> dict[str, Any]:
    """One (system, family, seed) baseline cell for the P5 sweep.

    Same output schema as `p4_forecaster_demo.train_and_rollout_one_cell`
    so the per-seed metrics.json / field.npz layout is identical and
    `make_p5_h1_verdict_figure.py` can read both directories with the
    same code path.
    """
    if system not in SYSTEMS_P4:
        raise ValueError(f"unknown system {system!r}")
    if family not in P5_BASELINE_FAMILIES:
        raise ValueError(f"unknown P5 baseline family {family!r}")
    cfg = cfg or P4SweepConfig()

    t, Y, sys_obj = _simulate(system, n_points=cfg.n_points, seed=0)
    sampled_dt = float(sys_obj.dt * sys_obj.sample_every)

    Y_train_traj, Y_test_traj = train_test_split(
        Y, train_frac=cfg.train_frac)
    X_train_windows, Y_train_targets = prepare_windows(
        Y_train_traj, cfg.window_length)
    d = Y.shape[1]

    # Per-family training.
    if family == "plain_neuralode":
        model, hist, pcount = _train_plain_neuralode(
            X_train_windows, Y_train_targets,
            d=d, cfg=cfg, seed=seed)
        adapter = make_vector_forecaster_adapter(model)
        use_jit = True
    elif family == "plain_mlp":
        model, hist, pcount = _train_plain_mlp(
            X_train_windows, Y_train_targets,
            d=d, cfg=cfg, seed=seed)
        adapter = make_vector_forecaster_adapter(model)
        use_jit = True
    elif family == "skyline":
        # Closed-form structural fit; no gradient training.
        fit_info = fit_skyline(system, Y_train_traj, dt=sampled_dt)
        # Skyline's rollout is its OWN procedure (not autoregressive
        # window-sliding) — it's RK4 of the fitted RHS. We compute
        # directly here and build a "synthetic" trajectory for the
        # metric suite.
        adapter = None
        use_jit = False
        hist = []
        pcount = sum(int(c.size) for c in fit_info["coeffs_per_component"])
    else:
        raise AssertionError(f"unreachable: {family}")

    # Rollout from the test trajectory's initial window.
    history0 = jnp.asarray(
        Y_test_traj[:cfg.window_length], dtype=jnp.float32)
    n_rollout_steps = min(
        cfg.rollout_steps,
        Y_test_traj.shape[0] - cfg.window_length)
    if n_rollout_steps < 5:
        raise ValueError(
            f"rollout_steps={cfg.rollout_steps} leaves only "
            f"{n_rollout_steps} test timesteps for {system}")

    if family == "skyline":
        # RK4 the fitted RHS from the last point of history0.
        y0 = np.asarray(history0[-1])
        traj = rollout_skyline(
            fit_info["rhs_fn"], y0, n_steps=n_rollout_steps, dt=sampled_dt)
        pred = traj
    else:
        if use_jit:
            adapter_jit = jax.jit(adapter)
            traj = autoregressive_rollout(
                adapter_jit, history0, n_steps=n_rollout_steps,
                dt=sampled_dt)
        else:
            traj = autoregressive_rollout_python_loop(
                adapter, history0, n_steps=n_rollout_steps, dt=sampled_dt)
        pred = np.asarray(traj, dtype=np.float64)

    ref = np.asarray(
        Y_test_traj[cfg.window_length:cfg.window_length + n_rollout_steps],
        dtype=np.float64)

    rel_l2 = relative_l2_error(pred, ref)
    lyap = LYAPUNOV_EXPONENT.get(system)
    vpt = valid_prediction_time(
        pred, ref, dt=sampled_dt,
        threshold=cfg.vpt_threshold,
        lyapunov_exponent=lyap)
    spec = spectral_error(pred, ref)

    inv_drift = None
    if system == "lotka_volterra":
        try:
            drift_curve = invariant_drift(pred, lotka_volterra_invariant)
            inv_drift = float(drift_curve[-1])
        except ValueError:
            inv_drift = float("inf")

    # Persistence floor (same as P4 — for context, not H1).
    pers_adapter = make_persistence_adapter()
    pers_traj = autoregressive_rollout_python_loop(
        pers_adapter, history0, n_steps=n_rollout_steps, dt=sampled_dt)
    pers_relL2 = relative_l2_error(
        np.asarray(pers_traj, dtype=np.float64), ref)

    return {
        "system": system,
        "family": family,
        "seed": int(seed),
        "n_train_points": int(Y_train_traj.shape[0]),
        "n_test_points": int(Y_test_traj.shape[0]),
        "sampled_dt": sampled_dt,
        "trainable_params": int(pcount),
        "train_loss_history": [float(x) for x in hist],
        "rollout_steps": int(n_rollout_steps),
        "dt_step": sampled_dt,
        "relative_l2": float(rel_l2),
        "vpt_step": int(vpt.vpt_step),
        "vpt_time": float(vpt.vpt_time),
        "vpt_lyapunov": (None if vpt.vpt_lyapunov is None
                         else float(vpt.vpt_lyapunov)),
        "spectral_error": float(spec),
        "invariant_drift_final": inv_drift,
        "persistence_floor_relative_l2": float(pers_relL2),
        "u_pred": pred,
        "u_ref": ref,
        "rel_l2_curve": np.asarray(vpt.rel_l2_curve, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Aggregation across seeds (re-uses P4's schema)
# ---------------------------------------------------------------------------


def _t_ci95(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 2:
        return {"mean": float(arr.mean()), "std": 0.0,
                "min": float(arr.min()), "max": float(arr.max()),
                "n_seeds": int(n), "ci95_half_width": 0.0,
                "ci95_low": float(arr.mean()),
                "ci95_high": float(arr.mean())}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    half = t_crit * std / float(np.sqrt(n))
    return {"mean": mean, "std": std,
            "min": float(arr.min()), "max": float(arr.max()),
            "n_seeds": int(n), "ci95_half_width": half,
            "ci95_low": mean - half, "ci95_high": mean + half}


def summarize_p5(results: list[dict]) -> dict[str, Any]:
    """Aggregate per-seed results for one (system, family) group."""
    if not results:
        return {}
    r0 = results[0]
    metrics: dict[str, Any] = {
        "relative_l2": _t_ci95([r["relative_l2"] for r in results]),
        "vpt_time": _t_ci95([r["vpt_time"] for r in results]),
        "spectral_error": _t_ci95([r["spectral_error"] for r in results]),
        "persistence_floor_relative_l2": _t_ci95(
            [r["persistence_floor_relative_l2"] for r in results]),
    }
    if r0.get("vpt_lyapunov") is not None:
        metrics["vpt_lyapunov"] = _t_ci95(
            [r["vpt_lyapunov"] for r in results])
    if r0.get("invariant_drift_final") is not None:
        metrics["invariant_drift_final"] = _t_ci95(
            [r["invariant_drift_final"] for r in results])
    return {
        "system": r0["system"],
        "family": r0["family"],
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "trainable_params": int(r0["trainable_params"]),
        "n_train_points": int(r0["n_train_points"]),
        "n_test_points": int(r0["n_test_points"]),
        "rollout_steps": int(r0["rollout_steps"]),
        "sampled_dt": float(r0["sampled_dt"]),
        "metrics": metrics,
    }
