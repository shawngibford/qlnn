"""P3.7 — PDE solver demo dispatch (heat + Burgers + Allen-Cahn).

Mirrors the structure of `solver_demo.py` (per-system dispatch +
per-seed train + summary aggregation) but for PDEs through the
`pde_residual_loss.py` infrastructure. Only one model family
(`chebyshev_dqc_2d`) is exercised in this phase — cross-family on
PDEs is P6 territory.

3 PDE benchmarks:
- `heat`: analytic reference  u(t,x) = e^{-νt}sin(x);  the convergence
  gate target re-run at 3 seeds for a clean per-seed CI alongside the
  real-PDE demos.
- `burgers_smooth`: reference = `data/pde/burgers_smooth.npz` field
  (H1 SMOOTH/PERIODIC bin).
- `allen_cahn`: reference = `data/pde/allen_cahn.npz` field
  (H1 BROADBAND/MULTISCALE bin — tests whether Lorenz's universal
  failure pattern reproduces on a PDE with sharp fronts).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from qlnn_.training.pde_residual_loss import (
    ChebyshevDQC2DConfig,
    build_chebyshev_dqc_2d,
    init_pde_solver_params,
    make_pde_residual_loss,
    train_pde_solver,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PDE_DATA_DIR = REPO_ROOT / "data" / "pde"


# ---------------------------------------------------------------------------
# Per-PDE config (residual, IC, domain, training budget)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDEBench:
    name: str
    regime: str                    # H1 bin
    t0: float
    t1: float
    x0: float
    x1: float
    pde_residual: Callable         # rhs(t,x,u,ut,ux,uxx[,uxxx]) → scalar
    ic_fn: Callable[[jnp.ndarray], jnp.ndarray]
    steps: int                     # training step budget
    n_t_colloc: int
    n_x_colloc: int
    description: str
    has_analytic: bool             # if True, reference comes from analytic_ref(t,x)
    analytic_ref: Callable | None  # u(t,x) for closed-form references
    npz_basename: str | None       # for npz-backed references
    # P7.8: KdV-class PDEs need u_xxx (triple-nested autodiff). The
    # residual function then takes 7 args instead of 6. P3.7 gate
    # established jacrev² works; P7.8 KdV gate established jacrev³
    # works at the canonical 4+4 qubit, 5-layer Chebyshev-DQC config.
    needs_uxxx: bool = False


# --- heat (analytic reference) ---------------------------------------------


_HEAT_NU = 0.1


def _heat_residual(t, x, u, ut, ux, uxx):
    return ut - _HEAT_NU * uxx


def _heat_ic(x):
    return jnp.sin(x)


def _heat_analytic(t, x):
    return jnp.exp(-_HEAT_NU * t) * jnp.sin(x)


# --- burgers_smooth + burgers_shock (npz references) -----------------------

# Two viscosity values matched to the committed npz field generators
# (data/pde/burgers_smooth.npz at nu=0.12; data/pde/burgers_shock.npz at
# nu=0.004 → sharp gradients near the inviscid shock time t*≈1). Same
# equation + IC; only the viscosity parameter differs. The shock regime
# is tagged BROADBAND/MULTISCALE per pre-reg §4.
_BURGERS_NU = 0.12
_BURGERS_SHOCK_NU = 0.004


def _burgers_residual(t, x, u, ut, ux, uxx):
    return ut + u * ux - _BURGERS_NU * uxx


def _burgers_shock_residual(t, x, u, ut, ux, uxx):
    return ut + u * ux - _BURGERS_SHOCK_NU * uxx


def _burgers_ic(x):
    return jnp.sin(x)


# --- allen_cahn (npz reference) --------------------------------------------


_AC_EPS = 0.06


def _allen_cahn_residual(t, x, u, ut, ux, uxx):
    # u_t = ε² u_xx + u − u³;  written as residual u_t − rhs = 0.
    return ut - ((_AC_EPS ** 2) * uxx + u - u ** 3)


# --- kdv (npz reference) ---------------------------------------------------
# Korteweg-de Vries: u_t + 6 u u_x + u_xxx = 0  (canonical form)
# Periodic domain x ∈ [0, 40), T = 5.
# Reference at data/pde/kdv.npz: 1-soliton sech² IC (c=4, centered x0=20).
#   u_soliton(t, x) = (c/2) · sech²( √c/2 · (x − x0 − c·t) )  (modulo periodicity)
# Pre-reg §4 PDE BROADBAND/MULTISCALE bin.


def _kdv_residual(t, x, u, ut, ux, uxx, uxxx):
    """KdV residual: u_t + 6·u·u_x + u_xxx."""
    return ut + 6.0 * u * ux + uxxx


_KDV_C = 4.0
_KDV_X0 = 20.0


def _kdv_ic(x):
    """Exact 1-soliton sech² initial condition (matches kdv.npz)."""
    arg = jnp.sqrt(_KDV_C) / 2.0 * (x - _KDV_X0)
    return (_KDV_C / 2.0) * (1.0 / jnp.cosh(arg)) ** 2


def _allen_cahn_ic(x, Lx=2.0 * jnp.pi, eps=_AC_EPS):
    """Periodic two-front IC — IDENTICAL to pde_systems._allen_cahn ic
    so the reference field's IC matches exactly. Note: pde_systems uses
    0.6·√2·ε narrower fronts so the simulation EVOLVES; our solver
    starts at the SAME perturbed-narrow IC and tries to recover the
    relaxed reference field."""
    w = 0.6 * jnp.sqrt(2.0) * eps
    return (jnp.tanh((x - Lx / 4.0) / w)
            - jnp.tanh((x - 3.0 * Lx / 4.0) / w) - 1.0)


# --- registry --------------------------------------------------------------


PDE_BENCH: dict[str, PDEBench] = {
    "heat": PDEBench(
        name="heat",
        regime="smooth_periodic",
        t0=0.0, t1=1.0, x0=0.0, x1=2.0 * float(jnp.pi),
        pde_residual=_heat_residual,
        ic_fn=_heat_ic,
        steps=1200,
        n_t_colloc=24, n_x_colloc=24,
        description=(f"Heat: u_t = {_HEAT_NU}·u_xx, u(0,x)=sin(x); "
                     f"exact u(t,x)=e^{{-{_HEAT_NU}t}}·sin(x)"),
        has_analytic=True,
        analytic_ref=_heat_analytic,
        npz_basename=None,
    ),
    "burgers_smooth": PDEBench(
        name="burgers_smooth",
        regime="smooth_periodic",
        t0=0.0, t1=2.0, x0=0.0, x1=2.0 * float(jnp.pi),
        pde_residual=_burgers_residual,
        ic_fn=_burgers_ic,
        steps=1500,
        n_t_colloc=28, n_x_colloc=28,
        description=(f"Burgers smooth: u_t + u·u_x = {_BURGERS_NU}·u_xx, "
                     f"u(0,x)=sin(x); reference = data/pde/"
                     f"burgers_smooth.npz"),
        has_analytic=False,
        analytic_ref=None,
        npz_basename="burgers_smooth.npz",
    ),
    "burgers_shock": PDEBench(
        name="burgers_shock",
        regime="broadband_multiscale",
        t0=0.0, t1=2.0, x0=0.0, x1=2.0 * float(jnp.pi),
        pde_residual=_burgers_shock_residual,
        ic_fn=_burgers_ic,
        # Sharper gradient near t* ≈ 1 (inviscid shock time at IC=sin(x))
        # → finer collocation grid + extra train steps. Matches the
        # allen_cahn-style budget.
        steps=2400,
        n_t_colloc=32, n_x_colloc=64,
        description=(f"Burgers shock: u_t + u·u_x = {_BURGERS_SHOCK_NU}·u_xx,"
                     f" u(0,x)=sin(x); reference = data/pde/"
                     f"burgers_shock.npz"),
        has_analytic=False,
        analytic_ref=None,
        npz_basename="burgers_shock.npz",
    ),
    "allen_cahn": PDEBench(
        name="allen_cahn",
        regime="broadband_multiscale",
        t0=0.0, t1=8.0, x0=0.0, x1=2.0 * float(jnp.pi),
        pde_residual=_allen_cahn_residual,
        ic_fn=_allen_cahn_ic,
        steps=1800,
        n_t_colloc=28, n_x_colloc=28,
        description=(f"Allen-Cahn: u_t = ε²·u_xx + u - u³ (ε={_AC_EPS}); "
                     f"IC periodic two-front (narrow→relax); "
                     f"reference = data/pde/allen_cahn.npz"),
        has_analytic=False,
        analytic_ref=None,
        npz_basename="allen_cahn.npz",
    ),
    "kdv": PDEBench(
        name="kdv",
        regime="broadband_multiscale",
        t0=0.0, t1=5.0, x0=0.0, x1=40.0,
        pde_residual=_kdv_residual,
        ic_fn=_kdv_ic,
        # KdV with c=4 soliton over T=5 propagates ~20 units; the
        # domain is 40 units so the soliton crosses ~half the box.
        # u_xxx (triple-nested autodiff) is gate-tested PASS at
        # results/p7_8_kdv_gate/gate_result.json. Per-point cost ratio
        # jacrev³/jacrev² ≈ 0.5× thanks to XLA fusion, so train cost
        # is dominated by collocation count (32×32 = 1024 points).
        steps=2400,
        n_t_colloc=32, n_x_colloc=32,
        description=("KdV: u_t + 6·u·u_x + u_xxx = 0; "
                     "1-soliton sech² IC (c=4, x0=20); "
                     "periodic on [0, 40); reference = data/pde/kdv.npz"),
        has_analytic=False,
        analytic_ref=None,
        npz_basename="kdv.npz",
        needs_uxxx=True,
    ),
}


# ---------------------------------------------------------------------------
# Reference field interpolation (npz → eval grid)
# ---------------------------------------------------------------------------


def _reference_field(bench: PDEBench, t_eval: np.ndarray,
                      x_eval: np.ndarray) -> np.ndarray:
    """Build the reference u(t, x) field on the eval grid.

    For PDEs with an analytic solution, evaluate it directly. For the
    npz-backed PDEs, linearly interpolate the committed numerical
    reference (n_frames × n_x) onto the requested eval grid.
    """
    if bench.has_analytic:
        T, X = np.meshgrid(np.asarray(t_eval), np.asarray(x_eval),
                            indexing="ij")
        return np.asarray(
            bench.analytic_ref(jnp.asarray(T), jnp.asarray(X)),
            dtype=np.float64)
    # npz-backed reference
    path = PDE_DATA_DIR / bench.npz_basename
    d = np.load(path)
    u_ref_full = np.asarray(d["u"], dtype=np.float64)    # (n_frames, n_x)
    t_ref = np.asarray(d["t"], dtype=np.float64)
    x_ref = np.asarray(d["x"], dtype=np.float64)
    # Bilinear: first interpolate along t for each x-column, then along x.
    # Use scipy-free 2-step linear interpolation via numpy.interp.
    n_t_eval, n_x_eval = len(t_eval), len(x_eval)
    # Step 1: interp along t for each reference x-column → (n_t_eval, n_x_ref)
    tmp = np.empty((n_t_eval, x_ref.size), dtype=np.float64)
    for jx in range(x_ref.size):
        tmp[:, jx] = np.interp(t_eval, t_ref, u_ref_full[:, jx])
    # Step 2: interp along x at each eval-t → (n_t_eval, n_x_eval)
    out = np.empty((n_t_eval, n_x_eval), dtype=np.float64)
    for it in range(n_t_eval):
        out[it, :] = np.interp(x_eval, x_ref, tmp[it, :])
    return out


# ---------------------------------------------------------------------------
# Train one PDE / one seed / record metrics
# ---------------------------------------------------------------------------


def train_one_pde(
    pde_name: str,
    seed: int,
    *,
    steps_override: int | None = None,
    n_t_qubits: int = 4,
    n_x_qubits: int = 4,
    num_layers: int = 5,
    lr: float = 0.02,
    n_t_eval: int = 50,
    n_x_eval: int = 50,
) -> dict[str, Any]:
    if pde_name not in PDE_BENCH:
        raise ValueError(f"unknown pde {pde_name!r}; choose from "
                         f"{list(PDE_BENCH)}")
    bench = PDE_BENCH[pde_name]
    steps = steps_override if steps_override is not None else bench.steps

    cfg = ChebyshevDQC2DConfig(
        n_t_qubits=n_t_qubits, n_x_qubits=n_x_qubits,
        num_layers=num_layers)
    circuit = build_chebyshev_dqc_2d(cfg)

    res = train_pde_solver(
        circuit, bench.pde_residual, bench.ic_fn,
        t0=bench.t0, t1=bench.t1, x0=bench.x0, x1=bench.x1,
        weight_shape=cfg.weight_shape,
        n_t_colloc=bench.n_t_colloc, n_x_colloc=bench.n_x_colloc,
        n_t_eval=n_t_eval, n_x_eval=n_x_eval,
        steps=steps, lr=lr, seed=seed)

    u_pred = np.asarray(res.u_pred)               # (n_t_eval, n_x_eval)
    u_ref = _reference_field(bench, np.asarray(res.t_eval),
                              np.asarray(res.x_eval))
    err = u_pred - u_ref
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(u_ref), 1e-12))

    return {
        "pde": pde_name,
        "seed": int(seed),
        "regime": bench.regime,
        "steps": int(steps),
        "lr": float(lr),
        "n_t_qubits": int(n_t_qubits),
        "n_x_qubits": int(n_x_qubits),
        "num_layers": int(num_layers),
        "n_t_colloc": int(bench.n_t_colloc),
        "n_x_colloc": int(bench.n_x_colloc),
        "pqc_params": int(np.prod(cfg.weight_shape)),
        "final_loss": float(res.final_loss),
        "mae": mae,
        "rmse": rmse,
        "relative_l2": rel_l2,
        "t_eval": np.asarray(res.t_eval, dtype=np.float64),
        "x_eval": np.asarray(res.x_eval, dtype=np.float64),
        "u_pred": u_pred,
        "u_ref": u_ref,
        "loss_history": [float(v) for v in res.loss_history],
    }


def run_pde_sweep(
    pdes: list[str] | None = None,
    seeds: list[int] | None = None,
    *,
    steps_override: int | None = None,
) -> list[dict[str, Any]]:
    pdes = list(pdes) if pdes is not None else list(PDE_BENCH)
    seeds = list(seeds) if seeds is not None else [0, 1, 2]
    out: list[dict[str, Any]] = []
    for name in pdes:
        for s in seeds:
            out.append(train_one_pde(name, s, steps_override=steps_override))
    return out


# ---------------------------------------------------------------------------
# Per-PDE seeds_summary aggregation (matches project schema)
# ---------------------------------------------------------------------------


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


def summarize_pde_seeds(results: list[dict]) -> dict[str, Any]:
    if not results:
        return {}
    name = results[0]["pde"]
    bench = PDE_BENCH[name]
    return {
        "pde": name,
        "regime": bench.regime,
        "description": bench.description,
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "pqc_params": int(results[0]["pqc_params"]),
        "config_str": (f"n_t={results[0]['n_t_qubits']}, "
                        f"n_x={results[0]['n_x_qubits']}, "
                        f"L={results[0]['num_layers']}, "
                        f"steps={results[0]['steps']}"),
        "metrics": {
            "mae": _t_ci95([r["mae"] for r in results]),
            "relative_l2": _t_ci95([r["relative_l2"] for r in results]),
            "final_loss": _t_ci95([r["final_loss"] for r in results]),
        },
    }
