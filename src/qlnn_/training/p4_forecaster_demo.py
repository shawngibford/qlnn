"""P4 dispatch module — one (system, family, seed) cell at a time.

Orchestrates the per-cell pipeline:
  1. Integrate the canonical ODE reference (synthetic_ode.simulate).
  2. Chronological train/test split (pre-reg §3.2 binding).
  3. Prepare sliding-window training pairs.
  4. Train the family-specific forecaster:
       - VectorForecaster-based: gradient-descent + adam.
       - rf_qrc: closed-form Tikhonov ridge.
  5. Build the OneStepForecaster adapter.
  6. Autoregressive rollout from the test trajectory's initial window.
  7. Compute the pre-reg §5 metric suite (relative-L2, VPT, spectral
     error, invariant drift).
  8. Return a result dict for serialization.

5 quantum-forecaster families exercised:
  - 4 VectorForecaster ansätze: `data_reuploading`, `hardware_efficient`,
    `strongly_entangling`, `brickwall`.
  - 1 reservoir family: `rf_qrc` (own train path).

3 ODE systems: `lotka_volterra`, `van_der_pol`, `lorenz`.

Pre-reg §5 metrics:
  - relative_l2 over rollout horizon (PRIMARY)
  - VPT (in Lyapunov times for Lorenz; physical-time for non-chaotic)
  - spectral_error (FFT PSD L2)
  - invariant_drift (LV only; others have no invariant)

The compute budget is moderate-quantum-heavy: each VectorForecaster
trains ~200 optax steps with JIT-compiled inner step. Expect ~70 min
CPU wall-clock for the full 5 fams × 3 systems × 3 seeds = 45 cells
sweep. rf_qrc cells are fast (<1 min); the 4 quantum families
dominate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from qlnn_.circuits import AnsatzConfig
from qlnn_.circuits.rf_qrc import RFQRCConfig, RFQRCForecaster
from qlnn_.evaluation.forecaster_adapters import (
    make_persistence_adapter,
    make_rf_qrc_adapter,
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
from qlnn_.models.vector_forecaster import (
    VectorForecaster, VectorForecasterConfig,
)
from qlnn_.training.forecaster_training import (
    prepare_windows,
    train_test_split,
    train_vector_forecaster,
)


# Lazy-import synthetic_ode to keep this module lightweight at top-level.
def _simulate(name, *, n_points, seed):
    from quantum_liquid_neuralode.data_processing.synthetic_ode import simulate
    return simulate(name, n_points=n_points, seed=seed)


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class P4SweepConfig:
    """Per-cell hyperparameters for the P4 sweep.

    Kept centralized so every cell uses identical settings (a binding
    requirement under pre-reg §6 "equal documented HPO budget").
    """

    # Trajectory
    n_points: int = 800           # per-system points (4000 default in
                                   # synthetic_ode; we use 800 for time budget)
    train_frac: float = 0.7       # chronological split

    # Forecaster window
    window_length: int = 8

    # VectorForecaster hyperparams (same across the 4 quantum ansätze)
    num_qubits: int = 3
    num_layers: int = 1
    train_steps: int = 200
    learning_rate: float = 5e-3
    delta_scale_init: float = 0.1
    delta_scale_min: float = 0.01

    # rf_qrc hyperparams
    rfqrc_num_qubits: int = 4
    rfqrc_leak_rate: float = 0.5
    rfqrc_beta: float = 1e-4      # Tikhonov ridge

    # Rollout
    rollout_steps: int = 200      # horizon in step count (relative to
                                   # the sampled dt of the system)
    vpt_threshold: float = 0.3


SYSTEMS_P4 = ("lotka_volterra", "van_der_pol", "lorenz")

VECTOR_QLNN_FAMILIES = (
    "data_reuploading",
    "hardware_efficient",
    "strongly_entangling",
    "brickwall",
)

# P7.11: τ-ablated variants of the 4 vector-QLNN families. Use a
# `non_liquid_` prefix so the dispatcher can route them to the
# NonLiquidVectorForecaster builder while reusing the same ansatz
# name for the encoder. rf_qrc is already non-liquid by construction
# (fixed leak_rate), so no non_liquid_rf_qrc variant is needed.
NON_LIQUID_QLNN_FAMILIES = tuple(
    f"non_liquid_{name}" for name in VECTOR_QLNN_FAMILIES
)

ALL_FAMILIES_P4 = (
    VECTOR_QLNN_FAMILIES + ("rf_qrc",) + NON_LIQUID_QLNN_FAMILIES
)


# ---------------------------------------------------------------------------
# Per-cell dispatch
# ---------------------------------------------------------------------------


def _train_vector_forecaster_family(
    family: str, X_train: np.ndarray, Y_train: np.ndarray,
    *, cfg: P4SweepConfig, seed: int,
) -> tuple[Any, list[float], dict[str, int]]:
    """Train one of the 4 VectorForecaster-based quantum ansätze.

    Returns (trained_model, loss_history, param_counts).
    """
    d = X_train.shape[-1]
    ansatz = AnsatzConfig(
        name=family, num_qubits=cfg.num_qubits,
        num_layers=cfg.num_layers)
    fc_cfg = VectorForecasterConfig(
        input_dim=d,
        num_qubits=cfg.num_qubits,
        num_layers=cfg.num_layers,
        step_dt=0.05,                            # nominal (intrinsic to
                                                  # training cadence; adapter
                                                  # ignores `dt` arg)
        delta_scale_init=cfg.delta_scale_init,
        delta_scale_min=cfg.delta_scale_min,
        ansatz=ansatz,
    )
    model = VectorForecaster(fc_cfg, key=jax.random.PRNGKey(seed))
    trained, history = train_vector_forecaster(
        model, X_train, Y_train,
        steps=cfg.train_steps, lr=cfg.learning_rate,
        log_every=max(1, cfg.train_steps // 5), seed=seed)
    # Param count: sum sizes of trainable arrays in the model pytree.
    import equinox as eqx
    diff_model, _ = eqx.partition(trained, eqx.is_array)
    param_count = sum(
        int(np.asarray(leaf).size)
        for leaf in jax.tree_util.tree_leaves(diff_model))
    return trained, history, {"trainable_params": param_count}


def _train_non_liquid_vector_forecaster_family(
    family: str, X_train: np.ndarray, Y_train: np.ndarray,
    *, cfg: P4SweepConfig, seed: int,
) -> tuple[Any, list[float], dict[str, int]]:
    """P7.11: train one of the 4 non-liquid VectorForecaster variants.

    Strips the `non_liquid_` prefix to recover the underlying ansatz
    name, then instantiates a NonLiquidVectorForecaster (same encoder
    + decoder + Diffrax integration as VectorForecaster, but with the
    τ-ablated NonLiquidQuantumCell).
    """
    from qlnn_.models.non_liquid_vector_forecaster import (
        NonLiquidVectorForecaster, NonLiquidVectorForecasterConfig,
    )

    if not family.startswith("non_liquid_"):
        raise ValueError(
            f"non-liquid family must start with 'non_liquid_', got {family!r}")
    base_ansatz = family[len("non_liquid_"):]
    if base_ansatz not in VECTOR_QLNN_FAMILIES:
        raise ValueError(
            f"non-liquid family {family!r} → base ansatz {base_ansatz!r} "
            f"is not in VECTOR_QLNN_FAMILIES {VECTOR_QLNN_FAMILIES!r}")

    d = X_train.shape[-1]
    ansatz = AnsatzConfig(
        name=base_ansatz, num_qubits=cfg.num_qubits,
        num_layers=cfg.num_layers)
    fc_cfg = NonLiquidVectorForecasterConfig(
        input_dim=d,
        num_qubits=cfg.num_qubits,
        num_layers=cfg.num_layers,
        step_dt=0.05,
        delta_scale_init=cfg.delta_scale_init,
        delta_scale_min=cfg.delta_scale_min,
        ansatz=ansatz,
    )
    model = NonLiquidVectorForecaster(fc_cfg, key=jax.random.PRNGKey(seed))
    trained, history = train_vector_forecaster(
        model, X_train, Y_train,
        steps=cfg.train_steps, lr=cfg.learning_rate,
        log_every=max(1, cfg.train_steps // 5), seed=seed)
    import equinox as eqx
    diff_model, _ = eqx.partition(trained, eqx.is_array)
    param_count = sum(
        int(np.asarray(leaf).size)
        for leaf in jax.tree_util.tree_leaves(diff_model))
    return trained, history, {"trainable_params": param_count}


def _train_rfqrc(
    X_train_windows: np.ndarray, Y_train_targets: np.ndarray,
    *, cfg: P4SweepConfig, seed: int,
) -> tuple[RFQRCForecaster, list[float], dict[str, int]]:
    """Train rf_qrc via closed-form Tikhonov ridge.

    The pre-reg's training pairs are (history_window, next_state).
    rf_qrc's fit expects (X, Y) where X is a sequence and Y is the
    corresponding targets. We feed the FLATTENED last state of each
    window as the X-row (rf_qrc is memoryless beyond its leaky
    integrator; the integrator is recomputed from this sequence).
    """
    # Flatten history windows into "current state" sequence:
    # X_row[i] = X_train_windows[i, -1, :] (the last frame of each window)
    X_seq = np.asarray(X_train_windows[:, -1, :], dtype=np.float64)
    Y_seq = np.asarray(Y_train_targets, dtype=np.float64)
    d = X_seq.shape[1]
    cfg_rf = RFQRCConfig(
        num_qubits=cfg.rfqrc_num_qubits,
        input_dim=d,
        leak_rate=cfg.rfqrc_leak_rate,
        beta=cfg.rfqrc_beta,
        alpha_seed=seed,
    )
    fc = RFQRCForecaster(cfg_rf)
    fc.fit(X_seq, Y_seq)
    history = []                                  # closed-form: no curve
    return fc, history, {"trainable_params": fc.n_trained_params}


def _eval_rollout_cell(
    adapter: Callable, history0: jnp.ndarray, dt_step: float,
    *, family: str, ref_test: np.ndarray, cfg: P4SweepConfig,
    system: str,
) -> dict[str, Any]:
    """Roll out `n_steps` from history0; compute the §5 metric suite."""
    n_steps = min(cfg.rollout_steps, ref_test.shape[0] - history0.shape[0])
    if n_steps < 5:
        raise ValueError(
            f"rollout_steps={cfg.rollout_steps} leaves only {n_steps} test "
            f"timesteps for {system}; reduce window or n_points")

    # rf_qrc returns numpy → use the python-loop variant.
    if family == "rf_qrc":
        traj = autoregressive_rollout_python_loop(
            adapter, history0, n_steps=n_steps, dt=dt_step)
    else:
        # JIT the adapter for speed.
        adapter_jit = jax.jit(adapter)
        traj = autoregressive_rollout(
            adapter_jit, history0, n_steps=n_steps, dt=dt_step)

    pred = np.asarray(traj, dtype=np.float64)
    # Test reference window aligned with rollout: skip the initial
    # history window (already used) and take the next n_steps rows.
    ref = ref_test[history0.shape[0]:history0.shape[0] + n_steps]
    ref = np.asarray(ref, dtype=np.float64)

    rel_l2 = relative_l2_error(pred, ref)
    lyap = LYAPUNOV_EXPONENT.get(system)
    vpt = valid_prediction_time(
        pred, ref, dt=dt_step,
        threshold=cfg.vpt_threshold,
        lyapunov_exponent=lyap)
    spec = spectral_error(pred, ref)

    # invariant drift for Lotka-Volterra ONLY (the other systems lack
    # a closed-form invariant per pre-reg §5).
    inv_drift = None
    if system == "lotka_volterra":
        try:
            drift_curve = invariant_drift(pred, lotka_volterra_invariant)
            inv_drift = float(drift_curve[-1])
        except ValueError:
            # If any predicted state has u<=0 or v<=0, the invariant is
            # undefined — bail out gracefully.
            inv_drift = float("inf")

    return {
        "rollout_steps": int(n_steps),
        "dt_step": float(dt_step),
        "relative_l2": float(rel_l2),
        "vpt_step": int(vpt.vpt_step),
        "vpt_time": float(vpt.vpt_time),
        "vpt_lyapunov": (None if vpt.vpt_lyapunov is None
                         else float(vpt.vpt_lyapunov)),
        "spectral_error": float(spec),
        "invariant_drift_final": inv_drift,
        "u_pred": pred,
        "u_ref": ref,
        "rel_l2_curve": np.asarray(vpt.rel_l2_curve, dtype=np.float64),
    }


def train_and_rollout_one_cell(
    system: str, family: str, seed: int,
    *, cfg: P4SweepConfig | None = None,
) -> dict[str, Any]:
    """One (system, family, seed) cell of the P4 matrix.

    Returns a dict with the training history, parameter counts,
    rollout metrics, and the predicted/reference field arrays.
    Serialized to per-seed `metrics.json` + `field.npz` by the CLI.
    """
    if system not in SYSTEMS_P4:
        raise ValueError(f"unknown P4 system {system!r}")
    if family not in ALL_FAMILIES_P4:
        raise ValueError(f"unknown P4 family {family!r}")
    cfg = cfg or P4SweepConfig()

    # 1. Integrate the canonical reference.
    t, Y, sys_obj = _simulate(system, n_points=cfg.n_points, seed=0)
    # Note: simulation seed=0 (fixed, deterministic per system); the
    # `seed` argument is for forecaster init + training shuffle.
    sampled_dt = float(sys_obj.dt * sys_obj.sample_every)

    # 2. Train/test chronological split.
    Y_train_traj, Y_test_traj = train_test_split(Y, train_frac=cfg.train_frac)

    # 3. Prepare training windows.
    X_train_windows, Y_train_targets = prepare_windows(
        Y_train_traj, cfg.window_length)

    # 4. Train the family-specific forecaster.
    if family == "rf_qrc":
        model, hist, pcount = _train_rfqrc(
            X_train_windows, Y_train_targets, cfg=cfg, seed=seed)
        adapter = make_rf_qrc_adapter(model)
    elif family in NON_LIQUID_QLNN_FAMILIES:
        # P7.11: τ-ablated quantum forecaster (matches the
        # NonLiquidQuantumCell + same Diffrax scaffold). Uses the
        # standard vector-forecaster adapter since the call signature
        # is identical: (T, d) → (d,).
        model, hist, pcount = _train_non_liquid_vector_forecaster_family(
            family, X_train_windows, Y_train_targets,
            cfg=cfg, seed=seed)
        adapter = make_vector_forecaster_adapter(model)
    else:
        model, hist, pcount = _train_vector_forecaster_family(
            family, X_train_windows, Y_train_targets,
            cfg=cfg, seed=seed)
        adapter = make_vector_forecaster_adapter(model)

    # 5. Build initial rollout window from the test trajectory's head.
    history0 = jnp.asarray(
        Y_test_traj[:cfg.window_length], dtype=jnp.float32)

    # 6. Roll out + score.
    rollout_result = _eval_rollout_cell(
        adapter, history0, sampled_dt,
        family=family, ref_test=Y_test_traj, cfg=cfg, system=system)

    # 7. Persistence floor baseline (for figure context, not H1).
    pers_adapter = make_persistence_adapter()
    pers_traj = autoregressive_rollout_python_loop(
        pers_adapter, history0, n_steps=rollout_result["rollout_steps"],
        dt=sampled_dt)
    pers_relL2 = relative_l2_error(
        np.asarray(pers_traj, dtype=np.float64),
        rollout_result["u_ref"])

    return {
        "system": system,
        "family": family,
        "seed": int(seed),
        "n_train_points": int(Y_train_traj.shape[0]),
        "n_test_points": int(Y_test_traj.shape[0]),
        "sampled_dt": sampled_dt,
        "trainable_params": int(pcount["trainable_params"]),
        "train_loss_history": [float(x) for x in hist],
        **{k: rollout_result[k] for k in (
            "rollout_steps", "dt_step", "relative_l2",
            "vpt_step", "vpt_time", "vpt_lyapunov",
            "spectral_error", "invariant_drift_final")},
        "persistence_floor_relative_l2": float(pers_relL2),
        # Arrays for figure / npz:
        "u_pred": rollout_result["u_pred"],
        "u_ref": rollout_result["u_ref"],
        "rel_l2_curve": rollout_result["rel_l2_curve"],
    }


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


def _t_ci95(values: list[float]) -> dict[str, float]:
    """t-based 95% CI; uses the same schema as p3_8_review_demo."""
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


def summarize_p4(results: list[dict]) -> dict[str, Any]:
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
    # vpt_lyapunov only for chaotic systems
    if r0.get("vpt_lyapunov") is not None:
        metrics["vpt_lyapunov"] = _t_ci95(
            [r["vpt_lyapunov"] for r in results])
    # invariant_drift only for LV
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
