"""P3.8 — Review-iteration dispatch (corrected re-runs + classical PINN).

Orchestrates the empirical work the peer-review audit requires:

(A) **PDE re-runs with audit-corrected configs** + matched classical-PINN
    parallel runs:
    - heat: steps=1200 (closes the gate-vs-sweep step-budget gap; same
      config as P3.7's convergence gate test).
    - burgers_smooth: steps=1500 (gate-target step count; tests
      whether relL2<0.30 was missed at 600 steps due to under-training).
    - allen_cahn: n_x_colloc=64, n_t_colloc=32, steps=1800 (tests
      whether the "broadband failure" was sub-Nyquist aliasing
      vs a real regime property — front width ≈0.085 vs Δx≈2π/64≈0.098,
      ~1.2 collocation points per front; >10× P3.7's 28×28 ratio).

(B) **Lorenz extended to T=5** (~5 Lyapunov times — halfway to the
    pre-reg's 10 LTE; balances cost/coverage) for the 4 quantum
    families AND the classical PINN. Tests whether the P3.6
    "universal failure" reflects transient nonlinear difficulty
    or genuine chaotic-regime inaccessibility.

(C) **Diagnostics**:
    - BC violation: max_t |u(t, x_near_0) - u(t, x_near_2π)| for each
      PDE solver's trained model. Quantifies how well the Lagaris
      trial form enforces periodicity implicitly.
    - Predict-mean baseline relL2 for Lorenz: replaces the predict-
      zero floor that overstated the "failure" interpretation in P3.6.
    - Spectral PSD of every reference solution: empirically grounds
      the H1 regime tags (broadband vs low-frequency).

This module DOES NOT modify any P3.6/P3.7 module or test contract.
It is a thin orchestrator that imports the existing builders
(quantum AND classical) and writes to a separate `results/p3_8_review/`
directory so the original P3.5/P3.6/P3.7 results stay committed as
historical record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qlnn_.training.classical_pinn_solver import (
    MLPConfig,
    build_classical_pinn_1d,
    build_classical_pinn_2d,
    init_classical_pinn_weights,
    matched_mlp_config,
)
from qlnn_.training.multi_state_solver import (
    VECTOR_ODES,
    _build_per_component,
    _make_vector_residual_loss,
    _reference_trajectory,
)
from qlnn_.training.pde_demo import (
    PDE_BENCH,
    _reference_field,
)
from qlnn_.training.pde_residual_loss import (
    ChebyshevDQC2DConfig,
    build_chebyshev_dqc_2d,
    init_pde_solver_params,
    make_pde_residual_loss,
)
from qlnn_.training.solver_demo import FAMILIES as SCALAR_FAMILIES

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "results" / "p3_8_review"


# ---------------------------------------------------------------------------
# Corrected per-PDE configs (audit-driven overrides on PDE_BENCH defaults)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectedPDEConfig:
    name: str                # one of PDE_BENCH keys
    n_t_colloc: int
    n_x_colloc: int
    steps: int
    audit_reason: str


CORRECTED_PDE_CONFIGS: dict[str, CorrectedPDEConfig] = {
    "heat": CorrectedPDEConfig(
        name="heat", n_t_colloc=24, n_x_colloc=24, steps=1200,
        audit_reason=("matches the convergence-gate config (P3.7's "
                       "test_heat_equation_gate_recovers_analytic_at_seed_0); "
                       "closes the gate-vs-sweep step-budget gap")),
    "burgers_smooth": CorrectedPDEConfig(
        name="burgers_smooth", n_t_colloc=28, n_x_colloc=28, steps=1500,
        audit_reason=("uses PDE_BENCH's intended step count (P3.7 sweep "
                       "ran at 600); tests whether relL2<0.30 gate was "
                       "missed at 600 steps due to under-training")),
    "allen_cahn": CorrectedPDEConfig(
        name="allen_cahn", n_t_colloc=32, n_x_colloc=64, steps=1800,
        audit_reason=("Δx = 2π/64 ≈ 0.098 ~ 1.2× the equilibrium front "
                       "width √2·ε ≈ 0.085; >10× the 28×28 resolution that "
                       "P3.7 used. Tests whether 'broadband failure' was "
                       "sub-Nyquist aliasing vs a real regime property")),
}


# ---------------------------------------------------------------------------
# Per-PDE train + eval (quantum and classical share the same scaffolding)
# ---------------------------------------------------------------------------


def _train_pde_one(
    circuit: Callable, weights_init: dict,
    pde_residual: Callable, ic_fn: Callable,
    *, t0, t1, x0, x1, n_t_colloc, n_x_colloc,
    steps: int, lr: float = 0.02, seed: int = 0,
) -> tuple[dict, list[float]]:
    """Shared train loop used by both quantum and classical PINN PDE
    runs. Identical to `pde_residual_loss.train_pde_solver`'s loop,
    duplicated here to avoid coupling to that module (the gate-test
    contract there is immutable)."""
    p = {"w": weights_init, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    t_colloc = jnp.linspace(t0, t1, n_t_colloc + 2)[1:-1]
    x_colloc = jnp.linspace(x0, x1, n_x_colloc + 2)[1:-1]
    T, X = jnp.meshgrid(t_colloc, x_colloc, indexing="ij")
    tx_colloc = jnp.stack([T.ravel(), X.ravel()], axis=1)

    loss_fn, u_of_tx = make_pde_residual_loss(
        circuit, pde_residual, ic_fn,
        t0=t0, t1=t1, x0=x0, x1=x1)
    opt = optax.adam(lr)
    opt_state = opt.init(p)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    hist: list[float] = []
    for _ in range(steps):
        val, grads = loss_and_grad(p, tx_colloc)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        hist.append(float(val))

    return p, hist, u_of_tx


def _eval_pde_field(u_of_tx: Callable, p: dict,
                     t0: float, t1: float, x0: float, x1: float,
                     n_t_eval: int = 50, n_x_eval: int = 50,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the trained model on the interior eval grid."""
    t_eval = jnp.linspace(t0, t1, n_t_eval + 2)[1:-1]
    x_eval = jnp.linspace(x0, x1, n_x_eval + 2)[1:-1]
    Te, Xe = jnp.meshgrid(t_eval, x_eval, indexing="ij")
    u_pred = jax.vmap(jax.vmap(lambda tt, xx: u_of_tx(tt, xx, p)))(Te, Xe)
    return (np.asarray(t_eval), np.asarray(x_eval),
            np.asarray(u_pred, dtype=np.float64))


def _bc_violation(u_of_tx: Callable, p: dict,
                   t0: float, t1: float, x_period_lo: float,
                   x_period_hi: float, n_probes: int = 20) -> float:
    """Periodic-BC violation: max_t |u(t, x_lo + ε) - u(t, x_hi - ε)|
    relative to max|u(t, ·)| over the same probes. The Lagaris hard-IC
    trial solution does NOT structurally enforce periodicity; this
    diagnostic quantifies the implicit-enforcement gap."""
    eps = 1e-3 * (x_period_hi - x_period_lo)
    ts = jnp.linspace(t0 + (t1 - t0) * 0.05, t1, n_probes)
    u_lo = jax.vmap(lambda tt: u_of_tx(tt, jnp.asarray(x_period_lo + eps), p))(ts)
    u_hi = jax.vmap(lambda tt: u_of_tx(tt, jnp.asarray(x_period_hi - eps), p))(ts)
    abs_diff = float(jnp.max(jnp.abs(u_lo - u_hi)))
    max_u = float(jnp.max(jnp.abs(jnp.concatenate([u_lo, u_hi]))))
    return abs_diff / max(max_u, 1e-8)


def train_one_pde_quantum(
    pde_name: str, seed: int,
    *, n_t_qubits: int = 4, n_x_qubits: int = 4, num_layers: int = 5,
) -> dict[str, Any]:
    """Re-run a single PDE at the audit-corrected config using the
    quantum Chebyshev-DQC 2D solver."""
    if pde_name not in CORRECTED_PDE_CONFIGS:
        raise ValueError(f"unknown PDE {pde_name!r}")
    cc = CORRECTED_PDE_CONFIGS[pde_name]
    bench = PDE_BENCH[pde_name]

    cfg = ChebyshevDQC2DConfig(
        n_t_qubits=n_t_qubits, n_x_qubits=n_x_qubits, num_layers=num_layers)
    circuit = build_chebyshev_dqc_2d(cfg)
    weights_init = (init_pde_solver_params(cfg.weight_shape, seed=seed))["w"]

    p, hist, u_of_tx = _train_pde_one(
        circuit, weights_init, bench.pde_residual, bench.ic_fn,
        t0=bench.t0, t1=bench.t1, x0=bench.x0, x1=bench.x1,
        n_t_colloc=cc.n_t_colloc, n_x_colloc=cc.n_x_colloc,
        steps=cc.steps, seed=seed)

    t_eval, x_eval, u_pred = _eval_pde_field(
        u_of_tx, p, bench.t0, bench.t1, bench.x0, bench.x1)
    u_ref = _reference_field(bench, t_eval, x_eval)
    err = u_pred - u_ref
    mae = float(np.mean(np.abs(err)))
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(u_ref), 1e-12))
    bc_viol = _bc_violation(u_of_tx, p, bench.t0, bench.t1, bench.x0, bench.x1)

    return {
        "pde": pde_name, "model": "chebyshev_dqc_2d",
        "seed": int(seed), "regime": bench.regime,
        "steps": int(cc.steps),
        "n_t_colloc": int(cc.n_t_colloc), "n_x_colloc": int(cc.n_x_colloc),
        "pqc_params": int(np.prod(cfg.weight_shape)),
        "classical_params": 0,
        "config_str": (f"n_t={n_t_qubits}, n_x={n_x_qubits}, L={num_layers}; "
                        f"colloc={cc.n_x_colloc}×{cc.n_t_colloc}; "
                        f"steps={cc.steps}"),
        "audit_reason": cc.audit_reason,
        "final_loss": float(hist[-1]),
        "mae": mae, "relative_l2": rel_l2,
        "bc_violation": bc_viol,
        "t_eval": t_eval, "x_eval": x_eval,
        "u_pred": u_pred, "u_ref": u_ref,
        "loss_history": hist,
    }


def train_one_pde_classical(
    pde_name: str, seed: int,
    *, target_param_count: int = 120, hidden_layers: int = 2,
) -> dict[str, Any]:
    """Capacity-matched classical MLP-PINN on the same PDE / config."""
    if pde_name not in CORRECTED_PDE_CONFIGS:
        raise ValueError(f"unknown PDE {pde_name!r}")
    cc = CORRECTED_PDE_CONFIGS[pde_name]
    bench = PDE_BENCH[pde_name]

    mlp_cfg = matched_mlp_config(
        target_param_count, input_dim=2, hidden_layers=hidden_layers)
    circuit = build_classical_pinn_2d(mlp_cfg)
    weights_init = init_classical_pinn_weights(mlp_cfg, seed=seed)

    p, hist, u_of_tx = _train_pde_one(
        circuit, weights_init, bench.pde_residual, bench.ic_fn,
        t0=bench.t0, t1=bench.t1, x0=bench.x0, x1=bench.x1,
        n_t_colloc=cc.n_t_colloc, n_x_colloc=cc.n_x_colloc,
        steps=cc.steps, seed=seed)

    t_eval, x_eval, u_pred = _eval_pde_field(
        u_of_tx, p, bench.t0, bench.t1, bench.x0, bench.x1)
    u_ref = _reference_field(bench, t_eval, x_eval)
    err = u_pred - u_ref
    mae = float(np.mean(np.abs(err)))
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(u_ref), 1e-12))
    bc_viol = _bc_violation(u_of_tx, p, bench.t0, bench.t1, bench.x0, bench.x1)

    return {
        "pde": pde_name, "model": "classical_pinn",
        "seed": int(seed), "regime": bench.regime,
        "steps": int(cc.steps),
        "n_t_colloc": int(cc.n_t_colloc), "n_x_colloc": int(cc.n_x_colloc),
        "pqc_params": 0,
        "classical_params": int(mlp_cfg.total_params()),
        "config_str": (f"MLP H={mlp_cfg.hidden_width}, L={mlp_cfg.hidden_layers}, "
                        f"target≈{target_param_count}; "
                        f"colloc={cc.n_x_colloc}×{cc.n_t_colloc}; "
                        f"steps={cc.steps}"),
        "audit_reason": "capacity-matched classical baseline",
        "final_loss": float(hist[-1]),
        "mae": mae, "relative_l2": rel_l2,
        "bc_violation": bc_viol,
        "t_eval": t_eval, "x_eval": x_eval,
        "u_pred": u_pred, "u_ref": u_ref,
        "loss_history": hist,
    }


# ---------------------------------------------------------------------------
# Lorenz extended (T=5) for the P3.6 re-do
# ---------------------------------------------------------------------------


# Lorenz LTE (largest Lyapunov exponent ≈ 0.906 for σ=10, ρ=28, β=8/3).
# Pre-reg specifies 10 LTE horizons (~11 time units); we use T=5 ≈ 5.5 LTE
# as a balance of cost/coverage, ~2.5× longer than P3.6's T=2.
_LORENZ_T_EXTENDED = 5.0
_LORENZ_LYAPUNOV_EXP = 0.906


def train_one_lorenz_quantum(
    family: str, seed: int,
    *, steps_override: int | None = None,
) -> dict[str, Any]:
    """Lorenz at T=5 (~5.5 Lyapunov times) using the per-component
    quantum solver — same scaffolding as P3.6 multi_state_solver but
    with the extended time horizon."""
    bench = VECTOR_ODES["lorenz"]
    if steps_override is None:
        steps_override = SCALAR_FAMILIES[family][1]

    # Override the time horizon to t1=5.0 (P3.6 used 2.0)
    t1 = _LORENZ_T_EXTENDED

    circuits, p_init, counts = _build_per_component(
        family, bench.dim, seed)
    loss_fn, u_of_t = _make_vector_residual_loss(
        circuits, bench.rhs_jax,
        t0=bench.t0, t1=t1, u0=jnp.asarray(bench.u0))

    t_colloc = jnp.linspace(bench.t0, t1, 60 + 2)[1:-1]
    opt = optax.adam(0.02)
    opt_state = opt.init(p_init)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    p = p_init
    hist: list[float] = []
    for _ in range(steps_override):
        val, grads = loss_and_grad(p, t_colloc)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        hist.append(float(val))

    t_eval = jnp.linspace(bench.t0, t1, 102)[1:-1]
    u_pred = jax.vmap(lambda tt: u_of_t(tt, p))(t_eval)
    ref = _reference_trajectory("lorenz", np.asarray(t_eval))
    # Override the synthetic_ode reference (which was for t∈[0,2]) by
    # re-integrating to t1=5.0; _reference_trajectory uses the bench
    # t1 internally — but our bench has t1=2.0. So we re-do it here:
    ref = _lorenz_reference_to_t1(np.asarray(t_eval), t1)

    err = np.asarray(u_pred) - ref
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-12))
    mae = float(np.mean(np.abs(err)))
    # Predict-mean baseline: the relative-L2 of using ref.mean(axis=0)
    # as a constant prediction. For chaotic systems with attractors far
    # from origin, this is a much more honest baseline than predict-zero.
    mean_pred = np.broadcast_to(ref.mean(axis=0, keepdims=True), ref.shape)
    rel_l2_mean_baseline = float(
        np.linalg.norm(ref - mean_pred) / max(np.linalg.norm(ref), 1e-12))

    return {
        "system": "lorenz", "model": family,
        "seed": int(seed), "regime": bench.regime,
        "dim": int(bench.dim),
        "t1": float(t1), "lyapunov_times": float(t1 * _LORENZ_LYAPUNOV_EXP),
        "steps": int(steps_override),
        "pqc_params": int(counts["pqc_params"]),
        "classical_params": int(counts["classical_params"]),
        "config_str": str(counts["config"]),
        "final_loss": float(hist[-1]),
        "mae": mae, "relative_l2": rel_l2,
        "relative_l2_predict_mean_baseline": rel_l2_mean_baseline,
        "t_eval": np.asarray(t_eval), "u_pred": np.asarray(u_pred),
        "u_ref": ref, "loss_history": hist,
    }


def _lorenz_reference_to_t1(t_eval: np.ndarray, t1: float) -> np.ndarray:
    """Re-integrate Lorenz over [0, t1] at fine dt and interpolate onto
    the requested eval grid. Mirrors `multi_state_solver._reference_
    trajectory` but lets the caller pick t1 (P3.6 used a fixed t1
    inside the function)."""
    from quantum_liquid_neuralode.data_processing import synthetic_ode
    sys = synthetic_ode.get_system("lorenz")
    n_steps = int(np.ceil(t1 / sys.dt)) + 1
    y = np.asarray(sys.y0, dtype=np.float64).copy()
    out = np.empty((n_steps, y.size), dtype=np.float64)
    t = 0.0
    for k in range(n_steps):
        out[k] = y
        if k == n_steps - 1:
            break
        k1 = sys.rhs(t, y, sys.params)
        k2 = sys.rhs(t + sys.dt / 2, y + sys.dt / 2 * k1, sys.params)
        k3 = sys.rhs(t + sys.dt / 2, y + sys.dt / 2 * k2, sys.params)
        k4 = sys.rhs(t + sys.dt, y + sys.dt * k3, sys.params)
        y = y + (sys.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += sys.dt
    t_ref = np.arange(n_steps) * sys.dt
    ref = np.empty((t_eval.size, y.size), dtype=np.float64)
    for j in range(y.size):
        ref[:, j] = np.interp(t_eval, t_ref, out[:, j])
    return ref


# ---------------------------------------------------------------------------
# Aggregation (mirrors the existing schema)
# ---------------------------------------------------------------------------


def _t_ci95(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 2:
        return {"mean": float(arr.mean()), "std": 0.0,
                "min": float(arr.min()), "max": float(arr.max()),
                "n_seeds": int(n), "ci95_half_width": 0.0,
                "ci95_low": float(arr.mean()), "ci95_high": float(arr.mean())}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    half = t_crit * std / float(np.sqrt(n))
    return {"mean": mean, "std": std,
            "min": float(arr.min()), "max": float(arr.max()),
            "n_seeds": int(n), "ci95_half_width": half,
            "ci95_low": mean - half, "ci95_high": mean + half}


def summarize(results: list[dict]) -> dict[str, Any]:
    """Aggregate a list of per-seed results into the standard schema."""
    if not results:
        return {}
    r0 = results[0]
    key = "pde" if "pde" in r0 else "system"
    name = r0[key]
    metrics: dict[str, Any] = {
        "mae": _t_ci95([r["mae"] for r in results]),
        "relative_l2": _t_ci95([r["relative_l2"] for r in results]),
        "final_loss": _t_ci95([r["final_loss"] for r in results]),
    }
    if "bc_violation" in r0:
        metrics["bc_violation"] = _t_ci95(
            [r["bc_violation"] for r in results])
    if "relative_l2_predict_mean_baseline" in r0:
        metrics["relative_l2_predict_mean_baseline"] = _t_ci95(
            [r["relative_l2_predict_mean_baseline"] for r in results])
    return {
        key: name, "model": r0["model"],
        "regime": r0["regime"],
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "pqc_params": int(r0["pqc_params"]),
        "classical_params": int(r0["classical_params"]),
        "config_str": r0["config_str"],
        "steps": int(r0["steps"]),
        "metrics": metrics,
        "audit_reason": r0.get("audit_reason", ""),
    }
