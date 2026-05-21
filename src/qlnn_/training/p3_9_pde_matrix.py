"""P3.9 — PDE multi-family dispatch (4 quantum × 3 PDEs).

Sibling to `p3_8_review_demo.py`. Whereas P3.8 ran only
`chebyshev_dqc_2d` + classical_pinn on the PDE side, P3.9 closes the
audit gap by adding the three remaining PINN-style families to the
PDE matrix:

  - chebyshev_dqc_2d  (P3.7 original — already in p3_8_review)
  - qcpinn_2d         (P3.9 commit 1, src/qlnn_/circuits/pde_2d/)
  - te_qpinn_fnn_2d   (P3.9 commit 2)
  - te_qpinn_qnn_2d   (P3.9 commit 3)

rf_qrc is out of scope per `.planning/P3_9_DESIGN.md` (frozen-
reservoir closed-form ridge — architecturally different from
PINN-style residual training; deferred to P4 as a forecaster).

Output layout (mirrors p3_8_review's per-{system,model}/seed_N/
structure):
  results/p3_9_pde_matrix/{pde}_{family}/seed_{seed}/{
    metrics.json, field.npz}
  results/p3_9_pde_matrix/{pde}_{family}/seeds_summary.json
  results/p3_9_pde_matrix/config.json
  results/p3_9_pde_matrix/provenance.json

Uses the audit-corrected configs from `CORRECTED_PDE_CONFIGS`
(p3_8_review_demo.py) — heat 1200 steps, Burgers 1500, AC 64×32
× 1800. This makes P3.9 results directly comparable to P3.8's
chebyshev_dqc_2d + classical_pinn numbers (same per-PDE training
budget, same eval grid).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qlnn_.circuits.pde_2d import (
    QCPINN2DConfig,
    TEQPINNFnn2DConfig,
    TEQPINNQnn2DConfig,
    build_qcpinn_2d,
    build_te_qpinn_fnn_2d,
    build_te_qpinn_qnn_2d,
    init_qcpinn_2d_solver_params,
    init_te_qpinn_fnn_2d_solver_params,
    init_te_qpinn_qnn_2d_solver_params,
)
from qlnn_.training.p3_8_review_demo import (
    CORRECTED_PDE_CONFIGS,
    _bc_violation,
    _eval_pde_field,
)
from qlnn_.training.pde_demo import PDE_BENCH, _reference_field
from qlnn_.training.pde_residual_loss import (
    ChebyshevDQC2DConfig,
    build_chebyshev_dqc_2d,
    init_pde_solver_params,
    make_pde_residual_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "results" / "p3_9_pde_matrix"


# ---------------------------------------------------------------------------
# Per-family dispatch table: each entry produces (circuit, init_p, info)
# ---------------------------------------------------------------------------


def _make_chebyshev(seed: int) -> tuple[Callable, dict, dict]:
    """Original P3.7 family — re-used unchanged for P3.9 matrix parity."""
    cfg = ChebyshevDQC2DConfig(n_t_qubits=4, n_x_qubits=4, num_layers=5)
    circuit = build_chebyshev_dqc_2d(cfg)
    p = init_pde_solver_params(cfg.weight_shape, seed=seed)
    info = {
        "model": "chebyshev_dqc_2d",
        "pqc_params": int(np.prod(cfg.weight_shape)),
        "classical_params": 0,
        "config_str": "n_t=4, n_x=4, L=5, total magnetization readout",
    }
    return circuit, p, info


def _make_qcpinn(seed: int) -> tuple[Callable, dict, dict]:
    cfg = QCPINN2DConfig(num_qubits=5, num_layers=1, topology="Cascade",
                         pre_hidden=50, post_hidden=50)
    circuit = build_qcpinn_2d(cfg)
    p = init_qcpinn_2d_solver_params(cfg, seed=seed)
    # Param accounting: PQC scalars vs classical (pre/post-NN) scalars.
    pqc_keys = ("pqc_rot", "pqc_crx", "pqc_pair")
    cls_keys = ("pre_W1", "pre_b1", "pre_W2", "pre_b2",
                "post_W1", "post_b1", "post_W2", "post_b2")
    pqc_n = sum(int(jnp.asarray(p["w"][k]).size)
                for k in pqc_keys if k in p["w"])
    cls_n = sum(int(jnp.asarray(p["w"][k]).size)
                for k in cls_keys if k in p["w"])
    info = {
        "model": "qcpinn_2d",
        "pqc_params": pqc_n,
        "classical_params": cls_n,
        "config_str": ("n=5, L=1, Cascade topology; "
                        "pre_hidden=50 + post_hidden=50; "
                        "per-qubit ⟨Z⟩ readout + post-NN scalar"),
    }
    return circuit, p, info


def _make_te_qpinn_fnn(seed: int) -> tuple[Callable, dict, dict]:
    cfg = TEQPINNFnn2DConfig(n_t_qubits=2, n_x_qubits=2,
                              num_layers=5, fnn_hidden_dim=16)
    circuit = build_te_qpinn_fnn_2d(cfg)
    p = init_te_qpinn_fnn_2d_solver_params(cfg, seed=seed)
    fnn_keys = ("fnn_t_W1", "fnn_t_b1", "fnn_t_W2", "fnn_t_b2",
                "fnn_x_W1", "fnn_x_b1", "fnn_x_W2", "fnn_x_b2")
    pqc_n = int(jnp.asarray(p["w"]["pqc_W"]).size)
    cls_n = sum(int(jnp.asarray(p["w"][k]).size) for k in fnn_keys)
    info = {
        "model": "te_qpinn_fnn_2d",
        "pqc_params": pqc_n,
        "classical_params": cls_n,
        "config_str": ("n_t=2, n_x=2 (n=4), L=5; "
                        "two FNN heads (H=16); ⟨⊗ Z⟩ readout"),
    }
    return circuit, p, info


def _make_te_qpinn_qnn(seed: int) -> tuple[Callable, dict, dict]:
    cfg = TEQPINNQnn2DConfig(n_t_qubits=2, n_x_qubits=2,
                              num_layers=5, num_embed_layers=3)
    circuit = build_te_qpinn_qnn_2d(cfg)
    p = init_te_qpinn_qnn_2d_solver_params(cfg, seed=seed)
    pqc_n = cfg.n_trained_params
    info = {
        "model": "te_qpinn_qnn_2d",
        "pqc_params": pqc_n,
        "classical_params": 0,
        "config_str": ("n_t=2, n_x=2 (n=4), L=5, K=3; "
                        "split-qubit U_embed + Σ Z readout"),
    }
    return circuit, p, info


QUANTUM_FAMILIES: dict[str, Callable[[int], tuple[Callable, dict, dict]]] = {
    "chebyshev_dqc_2d": _make_chebyshev,
    "qcpinn_2d": _make_qcpinn,
    "te_qpinn_fnn_2d": _make_te_qpinn_fnn,
    "te_qpinn_qnn_2d": _make_te_qpinn_qnn,
}


# ---------------------------------------------------------------------------
# Per-cell train + eval
# ---------------------------------------------------------------------------


def _train_pde_one_generic(
    circuit: Callable, init_params: dict,
    pde_residual: Callable, ic_fn: Callable,
    *, t0, t1, x0, x1, n_t_colloc, n_x_colloc,
    steps: int, lr: float = 0.02,
) -> tuple[dict, list[float], Callable]:
    """Shared train loop — accepts any {w, s, b} pytree.

    Same structure as `pde_residual_loss.train_pde_solver` and
    `p3_8_review_demo._train_pde_one` but takes a pre-initialized
    pytree so we can support both qcpinn_2d (w is a dict) and
    chebyshev_dqc_2d (w is a tensor) without branching.
    """
    p = init_params
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


def train_one_cell(
    pde_name: str, family: str, seed: int,
) -> dict[str, Any]:
    """One (PDE, family, seed) cell of the P3.9 matrix.

    Returns the full results dict including the field arrays so the
    caller can write to disk.
    """
    if pde_name not in CORRECTED_PDE_CONFIGS:
        raise ValueError(f"unknown PDE {pde_name!r}; "
                          f"expected one of {list(CORRECTED_PDE_CONFIGS)}")
    if family not in QUANTUM_FAMILIES:
        raise ValueError(f"unknown family {family!r}; "
                          f"expected one of {list(QUANTUM_FAMILIES)}")

    cc = CORRECTED_PDE_CONFIGS[pde_name]
    bench = PDE_BENCH[pde_name]
    circuit, p0, info = QUANTUM_FAMILIES[family](seed)

    p, hist, u_of_tx = _train_pde_one_generic(
        circuit, p0, bench.pde_residual, bench.ic_fn,
        t0=bench.t0, t1=bench.t1, x0=bench.x0, x1=bench.x1,
        n_t_colloc=cc.n_t_colloc, n_x_colloc=cc.n_x_colloc,
        steps=cc.steps)

    t_eval, x_eval, u_pred = _eval_pde_field(
        u_of_tx, p, bench.t0, bench.t1, bench.x0, bench.x1)
    u_ref = _reference_field(bench, t_eval, x_eval)
    err = u_pred - u_ref
    mae = float(np.mean(np.abs(err)))
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(u_ref), 1e-12))
    bc_viol = _bc_violation(
        u_of_tx, p, bench.t0, bench.t1, bench.x0, bench.x1)

    return {
        "pde": pde_name,
        "model": info["model"],
        "seed": int(seed),
        "regime": bench.regime,
        "steps": int(cc.steps),
        "n_t_colloc": int(cc.n_t_colloc),
        "n_x_colloc": int(cc.n_x_colloc),
        "pqc_params": int(info["pqc_params"]),
        "classical_params": int(info["classical_params"]),
        "config_str": info["config_str"],
        "audit_reason": cc.audit_reason,
        "final_loss": float(hist[-1]),
        "mae": mae,
        "relative_l2": rel_l2,
        "bc_violation": bc_viol,
        "t_eval": t_eval,
        "x_eval": x_eval,
        "u_pred": u_pred,
        "u_ref": u_ref,
        "loss_history": hist,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _t_ci95(values: list[float]) -> dict[str, float]:
    """t-based 95% CI; n=3 seeds default. Same as p3_8_review_demo."""
    a = np.asarray(values, dtype=np.float64)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return {"mean": mean, "std": 0.0, "ci95": 0.0, "n": int(n)}
    std = float(a.std(ddof=1))
    # t_{0.025, n-1}; for n=3, t=4.303; for n=5, t=2.776; n=10, t=2.262.
    t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
               6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    t = t_table.get(n, 2.0)  # asymptotic fall-back
    ci = t * std / np.sqrt(n)
    return {"mean": mean, "std": std, "ci95": float(ci), "n": int(n)}


def summarize(results: list[dict]) -> dict[str, Any]:
    """Aggregate per-seed results into a {mean, std, ci95} summary."""
    if not results:
        return {}
    first = results[0]
    return {
        "pde": first["pde"],
        "model": first["model"],
        "regime": first["regime"],
        "n_seeds": len(results),
        "relative_l2": _t_ci95([r["relative_l2"] for r in results]),
        "mae": _t_ci95([r["mae"] for r in results]),
        "final_loss": _t_ci95([r["final_loss"] for r in results]),
        "bc_violation": _t_ci95([r["bc_violation"] for r in results]),
        "pqc_params": int(first["pqc_params"]),
        "classical_params": int(first["classical_params"]),
        "steps": int(first["steps"]),
        "n_t_colloc": int(first["n_t_colloc"]),
        "n_x_colloc": int(first["n_x_colloc"]),
        "config_str": first["config_str"],
        "audit_reason": first["audit_reason"],
    }
