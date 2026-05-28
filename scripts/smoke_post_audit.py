"""Post-audit smoke — fill in the per-cell wall-time gaps the original
KdV + kuramoto smokes left open.

Measures:
  - 3 new A17 qcpinn variants (qcpinn_balanced, qcpinn_quantum,
    qcpinn_full_q) on kuramoto at 50 steps → extrapolated to 2000.
  - classical PINN solver on kuramoto at 50 steps → 2000 (closes the
    cross-side parity question after A15's lift of cPINN budget
    from 1500 → 2000 steps).

Skips: forecaster cells. At 200 steps they were demonstrably fast
(~seconds); at 2000 steps the ~5 hr estimate for the full ~90-cell
matrix is conservative and stays as-is in ADVISOR_BRIEF.

Output: results/smoke_post_audit/smoke_post_audit_runtimes.json
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "smoke_post_audit"

N_COLLOC = 60
SMOKE_STEPS = 50
PROD_STEPS = 2000      # uniform per A15
SEED = 0
LR = 0.02


def _smoke_qlnn_kuramoto(family: str) -> dict:
    """Smoke one QLNN family on kuramoto. Mirrors smoke_kuramoto_p6.py."""
    rec: dict = {"family": family, "task": "ode_solver", "system": "kuramoto",
                 "error": None}
    try:
        from qlnn_.training.multi_state_solver import (
            VECTOR_ODES, _build_per_component, _make_vector_residual_loss,
        )
        bench = VECTOR_ODES["kuramoto"]
        u0 = jnp.asarray(bench.u0)
        circuits, p_init, counts = _build_per_component(
            family, bench.dim, SEED)
        loss_fn, _ = _make_vector_residual_loss(
            circuits, bench.rhs_jax, t0=bench.t0, t1=bench.t1, u0=u0)
        t_colloc = jnp.linspace(bench.t0, bench.t1, N_COLLOC + 2)[1:-1]
        opt = optax.adam(LR)
        opt_state = opt.init(p_init)
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
        rec["pqc_params"] = int(counts["pqc_params"])
        rec["classical_params"] = int(counts["classical_params"])

        # iter 0: JIT compile
        p = p_init
        hist = []
        t0 = time.perf_counter()
        val, grads = loss_and_grad(p, t_colloc)
        val.block_until_ready()
        for leaf in jax.tree_util.tree_leaves(grads):
            leaf.block_until_ready()
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        jit_sec = time.perf_counter() - t0
        hist.append(float(val))

        # iters 1..N-1
        t1 = time.perf_counter()
        for _ in range(SMOKE_STEPS - 1):
            val, grads = loss_and_grad(p, t_colloc)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            hist.append(float(val))
        if hist:
            float(hist[-1])
        steady_per_iter = (time.perf_counter() - t1) / max(SMOKE_STEPS - 1, 1)

        rec["jit_sec"] = jit_sec
        rec["steady_per_iter_ms"] = steady_per_iter * 1000.0
        rec["est_full_cell_hours"] = (jit_sec + PROD_STEPS * steady_per_iter) / 3600.0
        rec["loss_step0"] = hist[0]
        rec["loss_step_final"] = hist[-1]
        rec["loss_decreased"] = hist[0] > hist[-1]
    except Exception:
        rec["error"] = traceback.format_exc()
    return rec


def _smoke_cpinn_kuramoto() -> dict:
    """Smoke the classical PINN solver on kuramoto at 2000 steps."""
    rec: dict = {"family": "classical_pinn", "task": "ode_solver",
                 "system": "kuramoto", "error": None}
    try:
        from qlnn_.training.multi_state_solver import VECTOR_ODES
        from qlnn_.training.classical_pinn_solver import (
            matched_mlp_config_vector_ode,
            init_classical_pinn_weights,
        )
        from qlnn_.training.p7_5_solver_h1 import (
            _make_classical_pinn_vector_residual_loss,
        )

        bench = VECTOR_ODES["kuramoto"]
        d = bench.dim
        u0 = jnp.asarray(bench.u0)
        cfg = matched_mlp_config_vector_ode(60, output_dim=d, hidden_layers=2)
        w = init_classical_pinn_weights(cfg, seed=SEED)
        loss_fn, _ = _make_classical_pinn_vector_residual_loss(
            cfg, bench.rhs_jax, t0=bench.t0, t1=bench.t1, u0=u0)
        t_colloc = jnp.linspace(bench.t0, bench.t1, N_COLLOC + 2)[1:-1]
        opt = optax.adam(LR)
        opt_state = opt.init(w)
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
        rec["classical_params"] = int(cfg.total_params())

        hist = []
        # iter 0: JIT
        t0 = time.perf_counter()
        val, grads = loss_and_grad(w, t_colloc)
        val.block_until_ready()
        for leaf in jax.tree_util.tree_leaves(grads):
            leaf.block_until_ready()
        updates, opt_state = opt.update(grads, opt_state)
        w = optax.apply_updates(w, updates)
        jit_sec = time.perf_counter() - t0
        hist.append(float(val))

        # iters 1..N-1
        t1 = time.perf_counter()
        for _ in range(SMOKE_STEPS - 1):
            val, grads = loss_and_grad(w, t_colloc)
            updates, opt_state = opt.update(grads, opt_state)
            w = optax.apply_updates(w, updates)
            hist.append(float(val))
        if hist:
            float(hist[-1])
        steady_per_iter = (time.perf_counter() - t1) / max(SMOKE_STEPS - 1, 1)

        rec["jit_sec"] = jit_sec
        rec["steady_per_iter_ms"] = steady_per_iter * 1000.0
        rec["est_full_cell_hours"] = (jit_sec + PROD_STEPS * steady_per_iter) / 3600.0
        rec["loss_step0"] = hist[0]
        rec["loss_step_final"] = hist[-1]
        rec["loss_decreased"] = hist[0] > hist[-1]
    except Exception:
        rec["error"] = traceback.format_exc()
    return rec


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # A17 new qcpinn variants
    for fam in ("qcpinn_balanced", "qcpinn_quantum", "qcpinn_full_q"):
        print(f"[smoke_post_audit] running {fam} on kuramoto ...", flush=True)
        rec = _smoke_qlnn_kuramoto(fam)
        records.append(rec)
        if rec["error"]:
            print(f"  CRASH:\n{rec['error']}", flush=True)
        else:
            print(f"  {fam}: JIT {rec['jit_sec']:.1f}s, "
                  f"steady {rec['steady_per_iter_ms']:.1f}ms/it, "
                  f"est full @2000 = {rec['est_full_cell_hours']:.2f} hr, "
                  f"PQC {rec['pqc_params']}, classical {rec['classical_params']}",
                  flush=True)

    # classical PINN at 2000 steps (post-A15 budget)
    print(f"[smoke_post_audit] running classical_pinn on kuramoto ...",
          flush=True)
    rec = _smoke_cpinn_kuramoto()
    records.append(rec)
    if rec["error"]:
        print(f"  CRASH:\n{rec['error']}", flush=True)
    else:
        print(f"  classical_pinn: JIT {rec['jit_sec']:.1f}s, "
              f"steady {rec['steady_per_iter_ms']:.1f}ms/it, "
              f"est full @2000 = {rec['est_full_cell_hours']:.2f} hr, "
              f"params {rec['classical_params']}", flush=True)

    # Summary table
    print()
    print("=" * 82, flush=True)
    print("Post-audit smoke summary", flush=True)
    print("=" * 82, flush=True)
    print(f"  {'family':<22} {'JIT (s)':>9} {'steady (ms/it)':>15} "
          f"{'est full (hr)':>14} {'loss↓':>5}", flush=True)
    print(f"  {'-' * 22} {'-' * 9} {'-' * 15} {'-' * 14} {'-' * 5}",
          flush=True)
    for r in records:
        if r["error"]:
            print(f"  {r['family']:<22} {'CRASH':>9} {'-':>15} {'-':>14} {'-':>5}",
                  flush=True)
            continue
        print(f"  {r['family']:<22} {r['jit_sec']:>9.2f} "
              f"{r['steady_per_iter_ms']:>15.2f} "
              f"{r['est_full_cell_hours']:>14.3f} "
              f"{'yes' if r['loss_decreased'] else 'NO':>5}", flush=True)

    out_json = OUT_DIR / "smoke_post_audit_runtimes.json"
    payload = {
        "config": {"system": "kuramoto", "n_colloc": N_COLLOC,
                   "smoke_steps": SMOKE_STEPS, "prod_steps": PROD_STEPS,
                   "seed": SEED, "lr": LR},
        "records": records,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[smoke_post_audit] wrote {out_json.relative_to(REPO_ROOT)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
