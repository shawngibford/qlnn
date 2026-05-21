"""P7.5 commit 1 — Solver-task H1 verdict module.

CRITICAL: this module computes the PRIMARY (pre-reg §7 GATING) H1
verdict that P5 was missing. P5 computed the forecaster-task H1
verdict, but pre-reg §7 says:

  "H1 CONFIRMED iff ... it holds on the **SOLVER** task
   (forecaster reported as corroborating/contradicting, not gating)."

This module:
  1. Re-uses the existing P3.6 multi-state QLNN solver results (4
     quantum families × 3 ODE systems × 3 seeds — already on disk).
  2. Trains a classical-PINN-as-solver baseline (Lagaris hard-IC +
     physics residual + MLP), matching the QLNN solver's per-component
     decomposition + Chebyshev coordinate map for apples-to-apples.
  3. Builds per-cell `CellRecord`s for the H1 verdict module.
  4. Reports the SOLVER-task H1 outcome — the PRIMARY paper headline.

Pre-reg §6 model table row: "Classical | classical PINN | Solver-task
classical control." This is exactly the right baseline for the
solver-task H1 contrast (the Neural-ODE pure-quantum-free analog
on the solver task, since the forecaster-task Neural-ODE doesn't
have a direct solver-task equivalent — physics-residual-trained
MLP IS the classical PINN).

Reuse map (NO new sweeps for the QLNN side — data is on disk):
  - QLNN solver results: results/p3_6_multi_state/{family}_{system}/
    seed_N/metrics.json — already committed.
  - Reference trajectories: synthetic_ode.simulate (deterministic,
    canonical configs).
  - Vector ODE PINN: `qlnn_.training.classical_pinn_solver
    .build_classical_pinn_vector_ode` + `vector_ode_pinn_trial`
    (P5 commit 4 / `f2932e9`).
  - Lagaris hard-IC trial solution: from multi_state_solver — same
    `u(t) = u₀ + (t − t₀) · N(t)` form.

Output: `results/p7_5_solver_h1/{system}_classical_pinn/seed_N/
metrics.json` per baseline cell, plus the verdict at
`results/p7_5_solver_h1/h1_analysis_solver_task.json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qlnn_.training.classical_pinn_solver import (
    MLPConfig,
    _mlp_apply_vector,
    matched_mlp_config_vector_ode,
    init_classical_pinn_weights,
)
from qlnn_.training.multi_state_solver import (
    VECTOR_ODES,
    _affine_to_chebyshev,
    _reference_trajectory,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "results" / "p7_5_solver_h1"
P36_RESULTS = REPO_ROOT / "results" / "p3_6_multi_state"


# ---------------------------------------------------------------------------
# Classical PINN solver — physics-residual training on vector ODE
# ---------------------------------------------------------------------------


def _make_classical_pinn_vector_residual_loss(
    cfg: MLPConfig,
    rhs_jax: Callable,
    *, t0: float, t1: float, u0: jnp.ndarray,
):
    """Vector-residual loss for the classical PINN (mirrors
    `multi_state_solver._make_vector_residual_loss` but with the
    classical MLP forward in place of the quantum circuit).

    Trial solution (Lagaris hard-IC):
        u(t) = u₀ + (t − t₀) · MLP(t̃)
    where t̃ = affine_to_chebyshev(t).

    The "per-component" decomposition lives INSIDE the MLP's vector
    output (cfg.output_dim = d) — a single MLP outputs the full d-vector
    versus the QLNN's d independent scalar circuits. This is the
    natural way to do it for the classical PINN.
    """

    def u_of_t(t, w):
        x = _affine_to_chebyshev(t, t0, t1)
        n = _mlp_apply_vector(x, w, cfg)               # (d,)
        return u0 + (t - t0) * n                        # (d,)

    du_dt = jax.jacrev(u_of_t, argnums=0)

    def loss(w, t_colloc):
        u = jax.vmap(lambda tt: u_of_t(tt, w))(t_colloc)              # (T, d)
        ut = jax.vmap(lambda tt: du_dt(tt, w))(t_colloc)               # (T, d)
        f = jax.vmap(rhs_jax)(t_colloc, u)                              # (T, d)
        return jnp.mean((ut - f) ** 2)

    return loss, u_of_t


def train_classical_pinn_solver_one_cell(
    system: str, seed: int,
    *,
    n_colloc: int = 60,
    steps: int = 1500,
    lr: float = 0.02,
    target_param_count: int | None = None,
) -> dict[str, Any]:
    """One (system, seed) classical-PINN-as-solver training run.

    Args:
      system : 'lotka_volterra' / 'van_der_pol' / 'lorenz'.
      seed : RNG seed.
      n_colloc : interior collocation count on [t0, t1].
      steps : optax-adam optimization steps.
      lr : learning rate.
      target_param_count : if None, default 60 (close to QLNN
        chebyshev_dqc's 120 per-component × 1 component decomposition
        within factor of 2; documented).

    Returns metrics dict with the same shape as the P3.6 vector-
    solver output schema (relative_l2, MAE, per-component MAE,
    field arrays for plotting).
    """
    if system not in VECTOR_ODES:
        raise ValueError(
            f"unknown system {system!r}; expected one of {list(VECTOR_ODES)}")
    bench = VECTOR_ODES[system]
    d = bench.dim
    u0 = jnp.asarray(bench.u0)

    # Capacity-matched MLP: target ~60 params (the QLNN per-component
    # default is ~30; one PINN must match the full vector → use ~60
    # so the per-d budget is comparable).
    if target_param_count is None:
        target_param_count = 60
    cfg = matched_mlp_config_vector_ode(
        target_param_count, output_dim=d, hidden_layers=2)
    w = init_classical_pinn_weights(cfg, seed=seed)

    loss_fn, u_of_t = _make_classical_pinn_vector_residual_loss(
        cfg, bench.rhs_jax, t0=bench.t0, t1=bench.t1, u0=u0)

    # Interior collocation (avoid Chebyshev singular ±1 endpoints).
    t_colloc = jnp.linspace(bench.t0, bench.t1, n_colloc + 2)[1:-1]
    opt = optax.adam(lr)
    opt_state = opt.init(w)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    hist: list[float] = []
    for _ in range(steps):
        val, grads = loss_and_grad(w, t_colloc)
        updates, opt_state = opt.update(grads, opt_state)
        w = optax.apply_updates(w, updates)
        hist.append(float(val))

    # Eval grid (interior of Chebyshev domain).
    t_eval = jnp.linspace(bench.t0, bench.t1, 102)[1:-1]
    u_pred = jax.vmap(lambda tt: u_of_t(tt, w))(t_eval)                # (T, d)
    ref = _reference_trajectory(system, np.asarray(t_eval))             # (T, d)
    err = np.asarray(u_pred) - ref
    mae_per_component = np.mean(np.abs(err), axis=0).tolist()
    mae = float(np.mean(np.abs(err)))
    rel_l2 = float(
        np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-12))

    # Train-side relative-L2 (the underfit guard input).
    u_train = jax.vmap(lambda tt: u_of_t(tt, w))(t_colloc)              # (T_c, d)
    ref_train = _reference_trajectory(system, np.asarray(t_colloc))
    err_train = np.asarray(u_train) - ref_train
    rel_l2_train = float(
        np.linalg.norm(err_train) / max(np.linalg.norm(ref_train), 1e-12))

    return {
        "family": "classical_pinn",
        "system": system,
        "seed": int(seed),
        "dim": int(d),
        "regime": bench.regime,
        "steps": int(steps),
        "lr": float(lr),
        "n_colloc": int(n_colloc),
        "trainable_params": int(cfg.total_params()),
        "config_str": (f"MLPConfig(input_dim=1, output_dim={d}, "
                       f"hidden_layers={cfg.hidden_layers}, "
                       f"hidden_width={cfg.hidden_width}, "
                       f"target~{target_param_count})"),
        "final_loss": float(hist[-1]),
        "mae": mae,
        "mae_per_component": mae_per_component,
        "relative_l2": rel_l2,
        "train_relative_l2": rel_l2_train,
        "t_eval": np.asarray(t_eval, dtype=np.float64),
        "u_pred": np.asarray(u_pred, dtype=np.float64),
        "u_ref": ref.astype(np.float64),
        "loss_history": [float(v) for v in hist],
    }


# ---------------------------------------------------------------------------
# Best-QLNN-per-cell extraction from P3.6 data
# ---------------------------------------------------------------------------


_P36_FAMILIES = (
    "chebyshev_dqc", "te_qpinn_fnn", "te_qpinn_qnn", "qcpinn",
)


def load_p36_qlnn_best(system: str, seed: int) -> tuple[float | None, str | None]:
    """Return (best_relative_l2, best_family_name) for the QLNN solver
    on (system, seed) from P3.6 multi-state results.

    Reads `results/p3_6_multi_state/{family}_{system}/seed_N/metrics.json`
    for each of the 4 P3.6 families and returns the MINIMUM relative_l2.
    """
    best_v = None
    best_fam = None
    for family in _P36_FAMILIES:
        p = P36_RESULTS / f"{family}_{system}" / f"seed_{seed}" / "metrics.json"
        if not p.exists():
            continue
        import json
        m = json.loads(p.read_text())
        v = float(m["relative_l2"])
        if best_v is None or v < best_v:
            best_v = v
            best_fam = family
    return best_v, best_fam
