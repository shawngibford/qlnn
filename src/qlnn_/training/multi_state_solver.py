"""P3.6 — Multi-state ODE solver via per-component scalar circuits.

Extends the P3 scalar solver path to **vector-state ODEs** (the 5
canonical systems in `synthetic_ode.py` are 2–12-dimensional). The
minimum-faithful extension per the plan: for a `d`-component ODE we
instantiate `d` independent scalar circuits of the chosen family, each
with its own weight pytree, and stack their outputs into the
trial-solution vector. No AnsatzProtocol refactor, no quantum
entanglement across components — those would be real architectural
decisions for a separate phase.

Trial solution (per-component Lagaris hard-IC):

    u_k(t) = u0[k] + (t − t0) · ( s_k · circuit_k(x(t), w_k) + b_k )
    u(t) = stack_k u_k(t)  ∈ ℝ^d

Residual loss:  L(p) = mean_t ‖ u_t(t) − f(t, u(t)) ‖₂²
where u_t is taken with `jax.jacrev(u_of_t, argnums=0)` (locked
convention — same as the scalar gate `solver_prototype_ode`).

NOT used inside the P3 acceptance gate; see the smoke test in
`tests/qlnn_/test_multi_state_solver.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from quantum_liquid_neuralode.data_processing import synthetic_ode
from qlnn_.training.physics_residual_loss import _affine_to_chebyshev
from qlnn_.training.solver_demo import FAMILIES as _SCALAR_FAMILIES


# ---------------------------------------------------------------------------
# Vector ODE benchmarks (subset of synthetic_ode.SYSTEMS; the H1-relevant 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorODESystem:
    name: str            # one of {lotka_volterra, van_der_pol, lorenz}
    dim: int             # state dimension
    u0: np.ndarray       # initial condition (d,)
    t0: float
    t1: float
    rhs_jax: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]  # f(t, u) → du/dt
    regime: str          # H1 bin: smooth_periodic | broadband_multiscale
    description: str


def _lv_rhs(t, u):
    sys = synthetic_ode.get_system("lotka_volterra")
    p = sys.params
    x, z = u[0], u[1]
    return jnp.array([p["alpha"] * x - p["beta"] * x * z,
                      p["delta"] * x * z - p["gamma"] * z])


def _vdp_rhs(t, u):
    sys = synthetic_ode.get_system("van_der_pol")
    mu = sys.params["mu"]
    x, v = u[0], u[1]
    return jnp.array([v, mu * (1.0 - x * x) * v - x])


def _lorenz_rhs(t, u):
    sys = synthetic_ode.get_system("lorenz")
    p = sys.params
    x, y, z = u[0], u[1], u[2]
    return jnp.array([p["sigma"] * (y - x),
                      x * (p["rho"] - z) - y,
                      x * y - p["beta"] * z])


VECTOR_ODES: dict[str, VectorODESystem] = {
    "lotka_volterra": VectorODESystem(
        name="lotka_volterra", dim=2,
        u0=np.array([10.0, 5.0]),
        # ~one predator-prey cycle at these params; smooth + periodic.
        t0=0.0, t1=5.0,
        rhs_jax=_lv_rhs,
        regime="smooth_periodic",
        description=("Lotka-Volterra (prey, predator), α=1.1, β=0.4, "
                     "δ=0.1, γ=0.4 — H1 SMOOTH/PERIODIC"),
    ),
    "van_der_pol": VectorODESystem(
        name="van_der_pol", dim=2,
        u0=np.array([2.0, 0.0]),
        # ~one stiff relaxation cycle at μ=5; intermediate / borderline.
        t0=0.0, t1=10.0,
        rhs_jax=_vdp_rhs,
        regime="smooth_periodic",   # borderline; P1 lists it under
                                    # SMOOTH/PERIODIC even at μ=5.
        description=("Van der Pol stiff relaxation oscillator, μ=5.0 — "
                     "borderline SMOOTH/PERIODIC (stiff but periodic)"),
    ),
    "lorenz": VectorODESystem(
        name="lorenz", dim=3,
        u0=np.array([1.0, 1.0, 1.0]),
        # ~2 Lyapunov times; chaotic regime is P1's BROADBAND/MULTISCALE.
        t0=0.0, t1=2.0,
        rhs_jax=_lorenz_rhs,
        regime="broadband_multiscale",
        description="Lorenz σ=10, ρ=28, β=8/3 — H1 BROADBAND/CHAOTIC",
    ),
}


# ---------------------------------------------------------------------------
# Per-component scalar-circuit dispatch (one circuit per state component)
# ---------------------------------------------------------------------------


def _build_per_component(
    family: str,
    dim: int,
    seed: int,
) -> tuple[list[Callable], dict, dict[str, int]]:
    """Build `dim` independent scalar circuits (same family, decorrelated
    seeds) and a unified pytree keyed by `c0..c{dim-1}` with each sub-key
    holding the family's native `{w, s, b}`.

    Returns (circuits, pytree, counts_summary). `counts_summary` reports
    the PER-COMPONENT param counts × dim, plus a `config_str` describing
    the family default config.
    """
    factory, _ = _SCALAR_FAMILIES[family]
    circuits: list[Callable] = []
    p: dict[str, dict] = {}
    last_counts: dict[str, Any] = {}
    for k in range(dim):
        # Decorrelate per-component initial weights by perturbing the seed
        # — keeps each component's randomness independent even if circuit
        # structures are identical.
        c, pk, counts = factory(seed * 1009 + k)
        circuits.append(c)
        p[f"c{k}"] = pk
        last_counts = counts
    per_comp_pqc = int(last_counts.get("pqc_params", 0))
    per_comp_cls = int(last_counts.get("classical_params", 0))
    summary = {
        "pqc_params": dim * per_comp_pqc,
        "classical_params": dim * per_comp_cls,
        "per_component_pqc_params": per_comp_pqc,
        "per_component_classical_params": per_comp_cls,
        "dim": dim,
        "config": f"per-component: {last_counts.get('config', '?')}; dim={dim}",
    }
    return circuits, p, summary


# ---------------------------------------------------------------------------
# Train loop (per-component Lagaris hard-IC + interior collocation)
# ---------------------------------------------------------------------------


def _make_vector_residual_loss(
    circuits: list[Callable],
    rhs_jax: Callable,
    *,
    t0: float, t1: float, u0: jnp.ndarray,
):
    """Vector-residual loss with per-component Lagaris hard-IC."""

    dim = len(circuits)

    def u_of_t(t, p):
        x = _affine_to_chebyshev(t, t0, t1)
        comps = []
        for k in range(dim):
            pk = p[f"c{k}"]
            n = pk["s"] * circuits[k](x, pk["w"]) + pk["b"]
            comps.append(u0[k] + (t - t0) * n)
        return jnp.stack(comps)

    du_dt = jax.jacrev(u_of_t, argnums=0)

    def loss(p, t_colloc):
        u = jax.vmap(lambda tt: u_of_t(tt, p))(t_colloc)          # (T, d)
        ut = jax.vmap(lambda tt: du_dt(tt, p))(t_colloc)          # (T, d)
        f = jax.vmap(rhs_jax)(t_colloc, u)                         # (T, d)
        return jnp.mean((ut - f) ** 2)

    return loss, u_of_t


def train_one_vector(
    family: str,
    system: str,
    seed: int,
    *,
    steps: int | None = None,
    n_colloc: int = 60,
    lr: float = 0.02,
) -> dict[str, Any]:
    """Train one family on one vector-ODE system at one seed. Returns
    the same metrics dict shape as `solver_demo.train_one_family`, with
    an extra `dim` field and per-component `mae_per_component` array."""
    if family not in _SCALAR_FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from "
                         f"{list(_SCALAR_FAMILIES)}")
    if system not in VECTOR_ODES:
        raise ValueError(f"unknown system {system!r}; choose from "
                         f"{list(VECTOR_ODES)}")

    bench = VECTOR_ODES[system]
    if steps is None:
        steps = _SCALAR_FAMILIES[family][1]
    u0 = jnp.asarray(bench.u0)

    circuits, p_init, counts = _build_per_component(family, bench.dim, seed)
    loss_fn, u_of_t = _make_vector_residual_loss(
        circuits, bench.rhs_jax, t0=bench.t0, t1=bench.t1, u0=u0)

    t_colloc = jnp.linspace(bench.t0, bench.t1, n_colloc + 2)[1:-1]
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

    # Interior eval grid (Chebyshev-singular ±1 excluded)
    t_eval = jnp.linspace(bench.t0, bench.t1, 102)[1:-1]
    u_pred = jax.vmap(lambda tt: u_of_t(tt, p))(t_eval)            # (T, d)

    # Numerical reference via the existing canonical numpy RK4 in
    # synthetic_ode (deterministic given seed + canonical params).
    ref = _reference_trajectory(system, np.asarray(t_eval))
    err = np.asarray(u_pred) - ref
    mae_per_component = np.mean(np.abs(err), axis=0).tolist()
    mae = float(np.mean(np.abs(err)))
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-12))

    return {
        "family": family,
        "system": system,
        "seed": int(seed),
        "dim": int(bench.dim),
        "regime": bench.regime,
        "steps": int(steps),
        "lr": float(lr),
        "n_colloc": int(n_colloc),
        "pqc_params": int(counts["pqc_params"]),
        "classical_params": int(counts["classical_params"]),
        "per_component_pqc_params": int(counts["per_component_pqc_params"]),
        "config_str": str(counts["config"]),
        "final_loss": float(hist[-1]),
        "mae": mae,
        "mae_per_component": mae_per_component,
        "relative_l2": rel_l2,
        "t_eval": np.asarray(t_eval, dtype=np.float64),
        "u_pred": np.asarray(u_pred, dtype=np.float64),
        "u_ref":  ref.astype(np.float64),
        "loss_history": [float(v) for v in hist],
    }


# ---------------------------------------------------------------------------
# Reference solution via the canonical numpy RK4 in synthetic_ode
# ---------------------------------------------------------------------------


def _reference_trajectory(system: str, t_eval: np.ndarray) -> np.ndarray:
    """Numerically integrate the chosen system at fine dt over [t0, t1]
    using the existing synthetic_ode RK4, then linearly interpolate to
    the requested eval grid. Deterministic given the canonical config."""
    sys = synthetic_ode.get_system(system)
    bench = VECTOR_ODES[system]
    # Re-integrate from the canonical IC over [t0, t1] at the system's
    # native dt. The canonical synthetic_ode integrator integrates from
    # y0 with no burn-in (we override burn_in=0 and sample_every=1 for
    # the reference path).
    n_steps = int(np.ceil((bench.t1 - bench.t0) / sys.dt)) + 1
    y = np.asarray(sys.y0, dtype=np.float64).copy()
    out = np.empty((n_steps, y.size), dtype=np.float64)
    t = bench.t0
    for k in range(n_steps):
        out[k] = y
        if k == n_steps - 1:
            break
        k1 = sys.rhs(t, y, sys.params)
        k2 = sys.rhs(t + sys.dt / 2, y + sys.dt / 2 * k1, sys.params)
        k3 = sys.rhs(t + sys.dt / 2, y + sys.dt / 2 * k2, sys.params)
        k4 = sys.rhs(t + sys.dt, y + sys.dt * k3, sys.params)
        y = y + (sys.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + sys.dt
    t_ref = bench.t0 + np.arange(n_steps) * sys.dt
    # Linear interpolation per component onto the requested t_eval grid.
    ref = np.empty((t_eval.size, y.size), dtype=np.float64)
    for j in range(y.size):
        ref[:, j] = np.interp(t_eval, t_ref, out[:, j])
    return ref


# ---------------------------------------------------------------------------
# Sweep + aggregation (mirrors solver_demo.run_sweep / summarize_seeds)
# ---------------------------------------------------------------------------


def run_vector_sweep(
    families: list[str] | None = None,
    systems: list[str] | None = None,
    seeds: list[int] | None = None,
    *,
    steps_override: int | None = None,
    n_colloc: int = 60,
    lr: float = 0.02,
) -> list[dict[str, Any]]:
    families = list(families) if families is not None else list(_SCALAR_FAMILIES)
    systems = list(systems) if systems is not None else list(VECTOR_ODES)
    seeds = list(seeds) if seeds is not None else [0, 1, 2]
    out: list[dict[str, Any]] = []
    for fam in families:
        for sysname in systems:
            for s in seeds:
                out.append(train_one_vector(
                    fam, sysname, s,
                    steps=steps_override, n_colloc=n_colloc, lr=lr))
    return out


def _t_ci95(values: list[float]) -> dict[str, float]:
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
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    half = t_crit * std / float(np.sqrt(n))
    return {"mean": mean, "std": std,
            "min": float(arr.min()), "max": float(arr.max()),
            "n_seeds": int(n),
            "ci95_half_width": half,
            "ci95_low": mean - half, "ci95_high": mean + half}


def summarize_vector_seeds(results: list[dict]) -> dict[str, Any]:
    if not results:
        return {}
    fam = results[0]["family"]
    sysname = results[0]["system"]
    return {
        "family": fam,
        "system": sysname,
        "regime": results[0]["regime"],
        "dim": results[0]["dim"],
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "pqc_params": int(results[0]["pqc_params"]),
        "classical_params": int(results[0]["classical_params"]),
        "config_str": results[0]["config_str"],
        "steps": int(results[0]["steps"]),
        "metrics": {
            "mae": _t_ci95([r["mae"] for r in results]),
            "relative_l2": _t_ci95([r["relative_l2"] for r in results]),
            "final_loss": _t_ci95([r["final_loss"] for r in results]),
        },
    }
