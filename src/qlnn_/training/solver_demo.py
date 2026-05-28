"""P3.5 — Solver comparison demo library.

Trains each of the 4 P3 solver families
(`chebyshev_dqc`, `te_qpinn_fnn`, `te_qpinn_qnn`, `qcpinn`) on a small
ODE benchmark via the same physics-residual loss
(`physics_residual_loss.make_residual_loss`, with the Lagaris hard-IC
trial solution). Returns per-run metrics + the predicted curves;
`scripts/run_solver_demo.py` is the thin CLI on top of this.

The 4 circuits do NOT share a config or pytree shape, so the demo
provides a per-family dispatch that adapts each family's
`init_*_weights(cfg)` to the uniform
`p = {"w": native_weights, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}`
pytree contract that `make_residual_loss` expects (and that
`tests/qlnn_/test_te_qpinn.py::test_qnn_drop_in_interop_with_make_residual_loss`
already proves works for the trainable-embedding families).

This module DELIBERATELY duplicates the optax train loop from
`physics_residual_loss.train_solver` instead of importing it. Reason:
`train_solver` hard-codes the chebyshev-shaped param init via
`init_solver_params(weight_shape)`, and refactoring it to accept a
pre-built pytree would touch the P3 acceptance-gate contract
(commit 77009ce). Keeping the gate code untouched is worth the ~30
duplicated lines.

NOT used inside the gate test; see the smoke test in
`tests/qlnn_/test_solver_demo.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qlnn_.circuits.qcpinn import (
    QCPINNConfig,
    build_qcpinn,
    init_qcpinn_weights,
    n_trainable_pqc_params,
)
from qlnn_.circuits.te_qpinn import (
    TEQPINNFnnConfig,
    TEQPINNQnnConfig,
    build_te_qpinn_fnn,
    build_te_qpinn_qnn,
    init_te_qpinn_fnn_weights,
    init_te_qpinn_qnn_weights,
)
from qlnn_.training.physics_residual_loss import (
    ChebyshevDQCConfig,
    build_chebyshev_dqc,
    make_residual_loss,
)


# ---------------------------------------------------------------------------
# ODE benchmarks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ODEBench:
    name: str
    t0: float
    t1: float
    u0: float
    rhs: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]   # f(t, u) -> du/dt
    exact: Callable[[jnp.ndarray], jnp.ndarray]              # exact u(t)
    description: str


ODES: dict[str, ODEBench] = {
    "expdecay": ODEBench(
        name="expdecay",
        t0=0.0, t1=2.0, u0=1.0,
        rhs=lambda t, u: -u,
        exact=lambda t: jnp.exp(-t),
        description="u' = -u,  u(0)=1  on  t in [0, 2];  exact  u = e^{-t}",
    ),
    "logistic": ODEBench(
        name="logistic",
        t0=0.0, t1=4.0, u0=0.5,
        rhs=lambda t, u: u * (1.0 - u),
        # u0 = 0.5 ⇒ exact u(t) = 1/(1 + e^{-t}); standard logistic curve.
        exact=lambda t: 1.0 / (1.0 + jnp.exp(-t)),
        description="u' = u(1-u),  u(0)=0.5  on  t in [0, 4];  exact  u = sigmoid(t)",
    ),
}


# ---------------------------------------------------------------------------
# Per-family dispatch: build circuit + init the `{w, s, b}` pytree
# ---------------------------------------------------------------------------


def _chebyshev_factory(seed: int) -> tuple[Callable, dict, dict[str, int]]:
    cfg = ChebyshevDQCConfig(num_qubits=4, num_layers=5)   # gate-matched
    circuit = build_chebyshev_dqc(cfg)
    w = 0.1 * jax.random.normal(jax.random.PRNGKey(seed), cfg.weight_shape)
    p = {"w": w, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    counts = {
        "pqc_params": int(np.prod(cfg.weight_shape)),
        "classical_params": 0,
        "config": f"n={cfg.num_qubits}, L={cfg.num_layers}",
    }
    return circuit, p, counts


def _te_qpinn_fnn_factory(seed: int) -> tuple[Callable, dict, dict[str, int]]:
    cfg = TEQPINNFnnConfig(num_qubits=4, num_layers=5, fnn_hidden_dim=16)
    circuit = build_te_qpinn_fnn(cfg)
    native = init_te_qpinn_fnn_weights(cfg, seed=seed)
    p = {"w": native, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    fnn_params = (1 * 16 + 16 + 16 * cfg.num_qubits + cfg.num_qubits)
    counts = {
        "pqc_params": int(cfg.n_pqc_rotations),
        "classical_params": fnn_params,
        "config": (f"n={cfg.num_qubits}, L={cfg.num_layers}, "
                   f"H_fnn={cfg.fnn_hidden_dim}"),
    }
    return circuit, p, counts


def _te_qpinn_qnn_factory(seed: int) -> tuple[Callable, dict, dict[str, int]]:
    cfg = TEQPINNQnnConfig(num_qubits=4, num_layers=5, num_embed_layers=3)
    circuit = build_te_qpinn_qnn(cfg)
    native = init_te_qpinn_qnn_weights(cfg, seed=seed)
    p = {"w": native, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    counts = {
        "pqc_params": int(cfg.n_trained_params),
        "classical_params": 0,
        "config": (f"n={cfg.num_qubits}, L_var={cfg.num_layers}, "
                   f"K_embed={cfg.num_embed_layers}"),
    }
    return circuit, p, counts


def _qcpinn_factory(seed: int) -> tuple[Callable, dict, dict[str, int]]:
    cfg = QCPINNConfig(num_qubits=5, num_layers=1, topology="Cascade",
                       pre_hidden=50, post_hidden=50,
                       input_dim=1, output_dim=1)
    circuit = build_qcpinn(cfg)
    native = init_qcpinn_weights(cfg, seed=seed)
    p = {"w": native, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}
    classical = sum(int(np.prod(native[k].shape))
                    for k in native
                    if k.startswith("pre_") or k.startswith("post_"))
    counts = {
        "pqc_params": int(n_trainable_pqc_params(native, cfg)),
        "classical_params": classical,
        "config": f"n={cfg.num_qubits}, L={cfg.num_layers}, {cfg.topology}",
    }
    return circuit, p, counts


# (family-name → factory) + the per-family training-step budget.
#
# FAIRNESS NOTE (2026-05-28, PRE_REG_AMENDMENT A15): the step budget is
# now UNIFORM across all 4 families at 2000 steps. Previously each
# family had its own budget (chebyshev_dqc 1200, te_qpinn_fnn 1500,
# te_qpinn_qnn 2000, qcpinn 1500) justified by "equal compute per
# loss point" (te_qpinn_qnn's trainable embedding adds a second circuit
# eval). The audit surfaced this as an undocumented unfair-comparison
# concern — equal-compute-per-step ≠ equal-iterations, and the family
# with the largest step budget (te_qpinn_qnn) also wins 3 of 4 ODE
# solver systems. Equalizing to the maximum (2000) gives every family
# the strongest possible shot.
#
# Trade-off: this triples chebyshev_dqc's compute relative to its prior
# config. Per the kuramoto+KdV smoke (2026-05-28), per-cell wall-clock
# is dominated by per-step cost, not iteration count, so the absolute
# cost increase is modest (~30-60%) and the comparison is now fair.
_UNIFORM_SOLVER_STEPS = 2000
FAMILIES: dict[str, tuple[Callable[[int], tuple[Callable, dict, dict]], int]] = {
    "chebyshev_dqc": (_chebyshev_factory, _UNIFORM_SOLVER_STEPS),
    "te_qpinn_fnn":  (_te_qpinn_fnn_factory, _UNIFORM_SOLVER_STEPS),
    "te_qpinn_qnn":  (_te_qpinn_qnn_factory, _UNIFORM_SOLVER_STEPS),
    "qcpinn":        (_qcpinn_factory, _UNIFORM_SOLVER_STEPS),
}

# Wong palette assignment, matching scripts/make_paper_figures.py style.
FAMILY_COLORS: dict[str, str] = {
    "chebyshev_dqc": "#0072B2",     # cool blue
    "te_qpinn_fnn":  "#D55E00",     # vermilion
    "te_qpinn_qnn":  "#CC79A7",     # pink
    "qcpinn":        "#009E73",     # cool green (Wong-complete)
}


# ---------------------------------------------------------------------------
# Train loop  (Lagaris hard-IC + interior collocation; mirrors train_solver)
# ---------------------------------------------------------------------------


def _train(
    circuit: Callable,
    rhs: Callable,
    p_init: dict,
    *,
    t0: float, t1: float, u0: float,
    n_colloc: int, steps: int, lr: float,
) -> tuple[dict, list[float], Callable]:
    """Optax-adam train loop on the physics residual. Returns
    (final_params, loss_history, u_of_t_callable)."""
    # Interior collocation — Chebyshev-singular bare ±1 excluded; same
    # convention as physics_residual_loss.train_solver.
    t_colloc = jnp.linspace(t0, t1, n_colloc + 2)[1:-1]
    loss_fn, u_of_t = make_residual_loss(
        circuit, rhs, t0=t0, t1=t1, u0=u0)
    opt = optax.adam(lr)
    opt_state = opt.init(p_init)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    p = p_init
    hist: list[float] = []
    for _ in range(steps):
        val, grads = loss_and_grad(p, t_colloc)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        hist.append(float(val))
    return p, hist, u_of_t


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def train_one_family(
    family: str,
    ode: str,
    seed: int,
    *,
    steps: int | None = None,
    n_colloc: int = 60,
    lr: float = 0.02,
) -> dict[str, Any]:
    """Train one family on one ODE with one seed; return a metrics dict.

    Keys returned:
      family, ode, seed, steps, lr, n_colloc,
      pqc_params, classical_params, config_str,
      final_loss, mae, rmse,
      t_eval, u_pred, exact            (numpy arrays, len 100)
      loss_history                     (list[float], len `steps`)
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from "
                         f"{list(FAMILIES)}")
    if ode not in ODES:
        raise ValueError(f"unknown ode {ode!r}; choose from {list(ODES)}")

    factory, default_steps = FAMILIES[family]
    if steps is None:
        steps = default_steps
    bench = ODES[ode]

    circuit, p_init, counts = factory(seed)
    p_final, hist, u_of_t = _train(
        circuit, bench.rhs, p_init,
        t0=bench.t0, t1=bench.t1, u0=bench.u0,
        n_colloc=n_colloc, steps=steps, lr=lr)

    # Interior eval grid — excludes the Chebyshev-singular bare ±1
    # endpoints; same convention as solver_prototype_ode.
    t_eval = jnp.linspace(bench.t0, bench.t1, 102)[1:-1]
    u_pred = jax.vmap(lambda tt: u_of_t(tt, p_final))(t_eval)
    exact = bench.exact(t_eval)
    err = u_pred - exact
    mae = float(jnp.mean(jnp.abs(err)))
    rmse = float(jnp.sqrt(jnp.mean(err ** 2)))

    return {
        "family": family,
        "ode": ode,
        "seed": int(seed),
        "steps": int(steps),
        "lr": float(lr),
        "n_colloc": int(n_colloc),
        "pqc_params": int(counts["pqc_params"]),
        "classical_params": int(counts["classical_params"]),
        "config_str": str(counts["config"]),
        "final_loss": float(hist[-1]),
        "mae": mae,
        "rmse": rmse,
        "t_eval": np.asarray(t_eval, dtype=np.float64),
        "u_pred": np.asarray(u_pred, dtype=np.float64),
        "exact":  np.asarray(exact,  dtype=np.float64),
        "loss_history": [float(v) for v in hist],
    }


def run_sweep(
    families: list[str] | None = None,
    odes: list[str] | None = None,
    seeds: list[int] | None = None,
    *,
    steps_override: int | None = None,
    n_colloc: int = 60,
    lr: float = 0.02,
) -> list[dict[str, Any]]:
    """Run the full (families × odes × seeds) Cartesian sweep.

    `steps_override` (if not None) replaces each family's default step
    count — used by the smoke test to slash runtime.
    """
    families = list(families) if families is not None else list(FAMILIES)
    odes = list(odes) if odes is not None else list(ODES)
    seeds = list(seeds) if seeds is not None else [0, 1, 2]
    out: list[dict[str, Any]] = []
    for fam in families:
        for ode in odes:
            for s in seeds:
                out.append(train_one_family(
                    fam, ode, s,
                    steps=steps_override, n_colloc=n_colloc, lr=lr))
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers (mirror the project's seeds_summary.json schema)
# ---------------------------------------------------------------------------


def _t_ci95(values: list[float]) -> dict[str, float]:
    """Mean ± 95% t-CI summary (matches results/*/seeds_summary.json)."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 2:
        return {"mean": float(arr.mean()), "std": 0.0,
                "min": float(arr.min()), "max": float(arr.max()),
                "n_seeds": int(n),
                "ci95_half_width": 0.0,
                "ci95_low": float(arr.mean()),
                "ci95_high": float(arr.mean())}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    # t-critical at 95% for small n; for n=3 t* ≈ 4.303 (per scipy.stats.t.ppf)
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
              6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    half = t_crit * std / float(np.sqrt(n))
    return {
        "mean": mean, "std": std,
        "min": float(arr.min()), "max": float(arr.max()),
        "n_seeds": int(n),
        "ci95_half_width": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def summarize_seeds(results: list[dict]) -> dict[str, Any]:
    """Aggregate one (family, ode) run-group across seeds into the
    project's `seeds_summary.json` schema."""
    if not results:
        return {}
    fam = results[0]["family"]
    ode = results[0]["ode"]
    assert all(r["family"] == fam and r["ode"] == ode for r in results)
    seeds = [r["seed"] for r in results]
    metrics = {
        m: _t_ci95([r[m] for r in results])
        for m in ("mae", "rmse", "final_loss")
    }
    return {
        "family": fam,
        "ode": ode,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "pqc_params": int(results[0]["pqc_params"]),
        "classical_params": int(results[0]["classical_params"]),
        "config_str": results[0]["config_str"],
        "steps": int(results[0]["steps"]),
        "metrics": metrics,
    }
