"""Heat IC-robustness full head-to-head matrix smoke.

5 ICs × (4 QLNN families + classical PINN) × 3 seeds = 75 cells.

Same operator (∂_t u = 0.1 · ∂_xx u), same periodic BC, same domain
[0, 2π] × [0, 1], same 24×24 collocation grid, same training budget
(50 steps for the smoke; production would be 1200). Only the IC varies.

For each cell we measure:
  - JIT compile time + steady-state per-iter cost
  - Final training loss
  - relL² against the analytic Fourier-series reference at SMOKE_STEPS

Reports a 5×5 matrix of (mean ± std) relL² across seeds — the
head-to-head answer to "is the IC-robustness signal real, and which
family handles which IC best?"

Output: results/smoke_heat_ic_matrix/runtimes.json
        + a printed 5×5 table per metric.
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
OUT_DIR = REPO_ROOT / "results" / "smoke_heat_ic_matrix"

N_T_COLLOC = 24
N_X_COLLOC = 24
SMOKE_STEPS = 50
PROD_STEPS = 1200
LR = 0.02

IC_VARIANTS = (
    "heat",            # baseline: sin(x)
    "heat_multifreq",  # sin(x) + 0.5·sin(3x) + 0.25·sin(5x)
    "heat_gaussian",   # Gaussian bump at π
    "heat_highfreq",   # sin(8x) — single high-frequency mode
    "heat_step",       # discontinuous square wave
)

# 4 QLNN families that ship in QUANTUM_FAMILIES + the classical PINN.
QLNN_FAMILIES = (
    "chebyshev_dqc_2d",
    "qcpinn_2d",
    "te_qpinn_fnn_2d",
    "te_qpinn_qnn_2d",
)
SEEDS = (0, 1, 2)


def _smoke_qlnn(family: str, variant: str, seed: int) -> dict:
    """Smoke one (QLNN family, heat IC variant, seed) cell at SMOKE_STEPS."""
    rec: dict = {"family": family, "variant": variant, "seed": seed,
                 "error": None}
    try:
        from qlnn_.training.pde_demo import PDE_BENCH
        from qlnn_.training.pde_residual_loss import make_pde_residual_loss
        from qlnn_.training.p3_9_pde_matrix import QUANTUM_FAMILIES

        bench = PDE_BENCH[variant]
        circuit, p, _info = QUANTUM_FAMILIES[family](seed)

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
        t0 = time.perf_counter()
        val, grads = loss_and_grad(p, tx_colloc)
        val.block_until_ready()
        for leaf in jax.tree_util.tree_leaves(grads):
            leaf.block_until_ready()
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        jit_sec = time.perf_counter() - t0
        hist.append(float(val))

        t1 = time.perf_counter()
        for _ in range(SMOKE_STEPS - 1):
            val, grads = loss_and_grad(p, tx_colloc)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            hist.append(float(val))
        if hist:
            float(hist[-1])
        steady_per_iter = (time.perf_counter() - t1) / max(SMOKE_STEPS - 1, 1)

        # Eval relL² against analytic reference at SMOKE_STEPS.
        t_eval = jnp.linspace(bench.t0, bench.t1, 33)[1:-1]
        x_eval = jnp.linspace(bench.x0, bench.x1, 33)[1:-1]
        T_e, X_e = jnp.meshgrid(t_eval, x_eval, indexing="ij")
        u_pred = jax.vmap(lambda tx: u_of_tx(tx[0], tx[1], p))(
            jnp.stack([T_e.ravel(), X_e.ravel()], axis=1))
        u_ref = bench.analytic_ref(T_e, X_e).ravel()
        err = np.asarray(u_pred) - np.asarray(u_ref)
        rel_l2 = float(
            np.linalg.norm(err) / max(np.linalg.norm(np.asarray(u_ref)), 1e-12))

        rec.update({
            "jit_sec": jit_sec,
            "steady_per_iter_ms": steady_per_iter * 1000.0,
            "loss_step0": hist[0],
            "loss_step_final": hist[-1],
            "rel_l2": rel_l2,
        })
    except Exception:
        rec["error"] = traceback.format_exc()
    return rec


def _smoke_cpinn(variant: str, seed: int) -> dict:
    """Smoke one (classical PINN, heat IC variant, seed) cell."""
    rec: dict = {"family": "classical_pinn", "variant": variant,
                 "seed": seed, "error": None}
    try:
        from qlnn_.training.pde_demo import PDE_BENCH
        from qlnn_.training.pde_residual_loss import make_pde_residual_loss
        from qlnn_.training.classical_pinn_solver import (
            matched_mlp_config,
            init_classical_pinn_weights,
            build_classical_pinn_2d,
        )

        bench = PDE_BENCH[variant]
        mlp_cfg = matched_mlp_config(120, input_dim=2, hidden_layers=2)
        circuit = build_classical_pinn_2d(mlp_cfg)
        weights = init_classical_pinn_weights(mlp_cfg, seed=seed)
        # The classical PINN's pytree wraps weights in the same
        # {w, s, b} shape so it plugs into make_pde_residual_loss.
        p = {"w": weights, "s": jnp.asarray(1.0), "b": jnp.asarray(0.0)}

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
        t0 = time.perf_counter()
        val, grads = loss_and_grad(p, tx_colloc)
        val.block_until_ready()
        for leaf in jax.tree_util.tree_leaves(grads):
            leaf.block_until_ready()
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        jit_sec = time.perf_counter() - t0
        hist.append(float(val))

        t1 = time.perf_counter()
        for _ in range(SMOKE_STEPS - 1):
            val, grads = loss_and_grad(p, tx_colloc)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            hist.append(float(val))
        if hist:
            float(hist[-1])
        steady_per_iter = (time.perf_counter() - t1) / max(SMOKE_STEPS - 1, 1)

        t_eval = jnp.linspace(bench.t0, bench.t1, 33)[1:-1]
        x_eval = jnp.linspace(bench.x0, bench.x1, 33)[1:-1]
        T_e, X_e = jnp.meshgrid(t_eval, x_eval, indexing="ij")
        u_pred = jax.vmap(lambda tx: u_of_tx(tx[0], tx[1], p))(
            jnp.stack([T_e.ravel(), X_e.ravel()], axis=1))
        u_ref = bench.analytic_ref(T_e, X_e).ravel()
        err = np.asarray(u_pred) - np.asarray(u_ref)
        rel_l2 = float(
            np.linalg.norm(err) / max(np.linalg.norm(np.asarray(u_ref)), 1e-12))

        rec.update({
            "jit_sec": jit_sec,
            "steady_per_iter_ms": steady_per_iter * 1000.0,
            "loss_step0": hist[0],
            "loss_step_final": hist[-1],
            "rel_l2": rel_l2,
        })
    except Exception:
        rec["error"] = traceback.format_exc()
    return rec


def _print_matrix(records: list[dict], metric: str, title: str,
                  fmt: str = "{:>8.4f}") -> None:
    families = list(QLNN_FAMILIES) + ["classical_pinn"]
    print()
    print("=" * 92, flush=True)
    print(title, flush=True)
    print("=" * 92, flush=True)
    header_cols = " ".join(f"{ic:>13}" for ic in IC_VARIANTS)
    print(f"  {'family':<18}  {header_cols}", flush=True)
    print(f"  {'-' * 18}  {' '.join(['-' * 13] * len(IC_VARIANTS))}",
          flush=True)
    for fam in families:
        cells = []
        for ic in IC_VARIANTS:
            seed_vals = [r[metric] for r in records
                         if not r["error"]
                         and r["family"] == fam and r["variant"] == ic]
            if not seed_vals:
                cells.append("    CRASH   ")
                continue
            mean = float(np.mean(seed_vals))
            std = float(np.std(seed_vals, ddof=1)) if len(seed_vals) >= 2 else 0.0
            cells.append(f"{mean:>6.3f}±{std:.2f}  ")
        print(f"  {fam:<18}  {' '.join(cells)}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    total_cells = (len(QLNN_FAMILIES) + 1) * len(IC_VARIANTS) * len(SEEDS)
    cell_n = 0
    overall_start = time.perf_counter()

    for ic in IC_VARIANTS:
        for family in QLNN_FAMILIES:
            for seed in SEEDS:
                cell_n += 1
                t_cell = time.perf_counter()
                print(f"[{cell_n:>2}/{total_cells}] "
                      f"{family:<18} {ic:<16} seed_{seed} ...",
                      end=" ", flush=True)
                rec = _smoke_qlnn(family, ic, seed)
                records.append(rec)
                if rec["error"]:
                    print(f"CRASH", flush=True)
                else:
                    print(f"relL²={rec['rel_l2']:.4f} "
                          f"({time.perf_counter() - t_cell:.0f}s)",
                          flush=True)
        for seed in SEEDS:
            cell_n += 1
            t_cell = time.perf_counter()
            print(f"[{cell_n:>2}/{total_cells}] "
                  f"{'classical_pinn':<18} {ic:<16} seed_{seed} ...",
                  end=" ", flush=True)
            rec = _smoke_cpinn(ic, seed)
            records.append(rec)
            if rec["error"]:
                print(f"CRASH", flush=True)
            else:
                print(f"relL²={rec['rel_l2']:.4f} "
                      f"({time.perf_counter() - t_cell:.0f}s)",
                      flush=True)

    overall_min = (time.perf_counter() - overall_start) / 60.0

    _print_matrix(records, "rel_l2",
                  f"relL² @ {SMOKE_STEPS} steps (mean ± std across "
                  f"{len(SEEDS)} seeds; lower is better)")
    _print_matrix(records, "loss_step_final",
                  "Final training loss (mean ± std; lower is better)",
                  fmt="{:>10.3e}")

    out_json = OUT_DIR / "runtimes.json"
    payload = {
        "config": {
            "pde": "heat",
            "n_t_colloc": N_T_COLLOC,
            "n_x_colloc": N_X_COLLOC,
            "smoke_steps": SMOKE_STEPS,
            "prod_steps": PROD_STEPS,
            "lr": LR,
            "ic_variants": list(IC_VARIANTS),
            "qlnn_families": list(QLNN_FAMILIES),
            "seeds": list(SEEDS),
        },
        "overall_wall_clock_min": overall_min,
        "records": records,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(flush=True)
    print(f"[smoke_heat_ic_matrix] wrote {out_json.relative_to(REPO_ROOT)}",
          flush=True)
    print(f"[smoke_heat_ic_matrix] total wall-clock: {overall_min:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
