"""Heat IC-robustness smoke — co-author's idea pilot.

Trains chebyshev_dqc_2d at seed 0 on the heat equation with 5 different
initial conditions (1 baseline + 4 variants). Same PDE operator, same
periodic BC, same time + space domain, same collocation grid, same
training budget — only the IC varies.

Reports per-IC:
  - JIT compile time + steady-state per-iter cost (for wall-clock
    extrapolation)
  - Final training loss
  - Held-out relL² against the analytic Fourier-series reference

If the QLNN's relative ordering across ICs matches what an analytical
"hardness" estimate would predict (sin < multifreq < gaussian < highfreq
< step), the family handles IC variation gracefully. If it inverts the
ordering, the family is IC-specific and the H1 verdict is partly an
IC-fixture artifact.

Output: results/smoke_heat_ic/smoke_heat_ic_runtimes.json
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "smoke_heat_ic"

# Canonical heat config (matches PDE_BENCH["heat"]).
N_T_COLLOC = 24
N_X_COLLOC = 24
SMOKE_STEPS = 50
PROD_STEPS = 1200
SEED = 0
LR = 0.02

# 5 ICs: baseline + 4 variants (single-mode, multi-mode, localized,
# discontinuous).
IC_VARIANTS = (
    "heat",            # sin(x) — baseline; only low-frequency content
    "heat_multifreq",  # sin(x)+0.5·sin(3x)+0.25·sin(5x) — mid-broadband
    "heat_gaussian",   # Gaussian bump — localized, all Fourier modes
    "heat_highfreq",   # sin(8x) — single high-frequency mode, fast decay
    "heat_step",       # square wave — discontinuous, hardest IC
)


def _smoke_one_ic(variant: str) -> dict:
    """Train chebyshev_dqc_2d on one heat IC for SMOKE_STEPS, report
    JIT + steady-state cost + final relL² against analytic reference.
    """
    rec: dict = {"variant": variant, "error": None}
    try:
        from qlnn_.training.pde_demo import PDE_BENCH
        from qlnn_.training.pde_residual_loss import make_pde_residual_loss
        from qlnn_.training.p3_9_pde_matrix import QUANTUM_FAMILIES

        bench = PDE_BENCH[variant]

        # Same circuit factory the production matrix runner uses.
        circuit, p, _info = QUANTUM_FAMILIES["chebyshev_dqc_2d"](SEED)

        # Collocation grid identical to _train_pde_one_generic.
        t_colloc = jnp.linspace(bench.t0, bench.t1, N_T_COLLOC + 2)[1:-1]
        x_colloc = jnp.linspace(bench.x0, bench.x1, N_X_COLLOC + 2)[1:-1]
        T, X = jnp.meshgrid(t_colloc, x_colloc, indexing="ij")
        tx_colloc = jnp.stack([T.ravel(), X.ravel()], axis=1)

        loss_fn, u_of_tx = make_pde_residual_loss(
            circuit, bench.pde_residual, bench.ic_fn,
            t0=bench.t0, t1=bench.t1, x0=bench.x0, x1=bench.x1,
            need_uxxx=bench.needs_uxxx,
        )
        opt = optax.adam(LR)
        opt_state = opt.init(p)
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        hist: list[float] = []

        # iter 0: JIT compile + first eval
        t0 = time.perf_counter()
        val, grads = loss_and_grad(p, tx_colloc)
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
            val, grads = loss_and_grad(p, tx_colloc)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            hist.append(float(val))
        if hist:
            float(hist[-1])
        steady_per_iter = (time.perf_counter() - t1) / max(SMOKE_STEPS - 1, 1)

        # Evaluate relL² against analytic reference at SMOKE_STEPS.
        t_eval = jnp.linspace(bench.t0, bench.t1, 33)[1:-1]
        x_eval = jnp.linspace(bench.x0, bench.x1, 33)[1:-1]
        T_e, X_e = jnp.meshgrid(t_eval, x_eval, indexing="ij")
        tx_eval = jnp.stack([T_e.ravel(), X_e.ravel()], axis=1)
        u_pred = jax.vmap(lambda tx: u_of_tx(tx[0], tx[1], p))(tx_eval)
        u_ref = jax.vmap(lambda tx: bench.analytic_ref(tx[0], tx[1]))(tx_eval)
        err = np.asarray(u_pred) - np.asarray(u_ref)
        rel_l2_at_smoke = float(
            np.linalg.norm(err) / max(np.linalg.norm(np.asarray(u_ref)), 1e-12))

        rec.update({
            "jit_sec": jit_sec,
            "steady_per_iter_ms": steady_per_iter * 1000.0,
            "smoke_steps": SMOKE_STEPS,
            "prod_steps": PROD_STEPS,
            "est_full_cell_hours": (
                jit_sec + PROD_STEPS * steady_per_iter) / 3600.0,
            "loss_step0": hist[0],
            "loss_step_final": hist[-1],
            "loss_decreased": hist[0] > hist[-1],
            "rel_l2_at_smoke_steps": rel_l2_at_smoke,
        })
    except Exception:
        rec["error"] = traceback.format_exc()
    return rec


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for variant in IC_VARIANTS:
        print(f"[smoke_heat_ic] running {variant} ...", flush=True)
        rec = _smoke_one_ic(variant)
        records.append(rec)
        if rec["error"]:
            print(f"  CRASH:\n{rec['error']}", flush=True)
        else:
            print(f"  {variant}: JIT {rec['jit_sec']:.1f}s, "
                  f"steady {rec['steady_per_iter_ms']:.1f}ms/it, "
                  f"loss {rec['loss_step0']:.3e}→{rec['loss_step_final']:.3e}, "
                  f"relL²@{SMOKE_STEPS}={rec['rel_l2_at_smoke_steps']:.4f}",
                  flush=True)

    # Summary table — the key signal is the relL² ordering across ICs.
    print()
    print("=" * 96, flush=True)
    print("Heat IC-robustness smoke summary", flush=True)
    print("=" * 96, flush=True)
    print(f"  PDE: heat (u_t = 0.1·u_xx), periodic BC on [0,2π], "
          f"t ∈ [0,1], 24×24 collocation", flush=True)
    print(f"  Family: chebyshev_dqc_2d, seed {SEED}, "
          f"{SMOKE_STEPS} steps (extrapolated → {PROD_STEPS})", flush=True)
    print()
    print(f"  {'IC':<18} {'JIT (s)':>9} {'steady (ms/it)':>15} "
          f"{'est full (hr)':>14} {'final loss':>11} {'relL²':>9}",
          flush=True)
    print(f"  {'-' * 18} {'-' * 9} {'-' * 15} {'-' * 14} {'-' * 11} {'-' * 9}",
          flush=True)
    for r in records:
        if r["error"]:
            print(f"  {r['variant']:<18} {'CRASH':>9} "
                  f"{'-':>15} {'-':>14} {'-':>11} {'-':>9}", flush=True)
            continue
        print(f"  {r['variant']:<18} "
              f"{r['jit_sec']:>9.2f} "
              f"{r['steady_per_iter_ms']:>15.2f} "
              f"{r['est_full_cell_hours']:>14.3f} "
              f"{r['loss_step_final']:>11.3e} "
              f"{r['rel_l2_at_smoke_steps']:>9.4f}", flush=True)
    print()

    # Headline signal: the ordering.
    successful = [r for r in records if not r["error"]]
    if successful:
        successful_sorted = sorted(
            successful, key=lambda r: r["rel_l2_at_smoke_steps"])
        print("relL² ordering (best → worst):",
              " < ".join(r["variant"] for r in successful_sorted),
              flush=True)
        print(flush=True)
        print(
            "If chebyshev_dqc_2d is IC-robust, the relL² ordering after\n"
            f"only {SMOKE_STEPS} steps should track the IC's intrinsic\n"
            "hardness (more Fourier content → harder). If the ordering is\n"
            "inverted or chaotic, the family is IC-sensitive and the H1\n"
            "verdict numbers depend on the specific IC chosen.", flush=True)

    out_json = OUT_DIR / "smoke_heat_ic_runtimes.json"
    payload = {
        "config": {
            "pde": "heat",
            "n_t_colloc": N_T_COLLOC,
            "n_x_colloc": N_X_COLLOC,
            "smoke_steps": SMOKE_STEPS,
            "prod_steps": PROD_STEPS,
            "seed": SEED,
            "lr": LR,
            "ic_variants": list(IC_VARIANTS),
        },
        "records": records,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[smoke_heat_ic] wrote {out_json.relative_to(REPO_ROOT)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
