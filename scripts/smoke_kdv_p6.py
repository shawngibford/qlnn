"""KdV smoke — measure real per-iter cost for all 4 quantum families
before the M3 sweep commits the box to ~14-16 hr.

This is throwaway. Writes to results/smoke_kdv/, NOT to the M3 canonical
results/p6_kuramoto_kdv/ path. The M3 runner scaffold
(scripts/run_p7_8_h1_kuramoto_kdv.py) is untouched.

For each KdV quantum family:
  1. Build the circuit + pytree at seed 0.
  2. JIT a value_and_grad of the KdV residual loss at canonical
     32×32 collocation (with jacrev³ for u_xxx).
  3. Time iter 0 separately (JIT compile + first eval) from
     iters 1..N-1 (steady state).
  4. Extrapolate to the full 2400-step cell:
        est_full_sec = jit_sec + 2400 * steady_state_sec_per_iter
  5. Capture the final loss after 50 steps as a "is it actually
     decreasing?" sanity check.

Then print a summary table and refined M3 estimate, and write a JSON
record so the numbers survive the session.
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
OUT_DIR = REPO_ROOT / "results" / "smoke_kdv"

# Canonical KdV smoke config. Collocation matches the M3 production config
# (CORRECTED_PDE_CONFIGS["kdv"].n_t_colloc/n_x_colloc); steps is a smoke
# budget chosen so iters 1..N give a stable steady-state estimate while
# total wall-clock stays in the 5-10 min/family ballpark.
N_T_COLLOC = 32
N_X_COLLOC = 32
SMOKE_STEPS = 50
SEED = 0
LR = 0.02

# Used for extrapolation. Must match CORRECTED_PDE_CONFIGS["kdv"].steps.
PROD_STEPS = 2400

# Reference for the kuramoto side (untouched by this smoke).
HOURS_PER_KURAMOTO_CELL_PRIOR = 7.0
N_KURAMOTO_QLNN_CELLS = 12      # 4 families × 3 seeds
SEC_PER_CLASSICAL_PINN_CELL = 1.5


def _smoke_one_family(family: str) -> dict:
    """Run the smoke for one KdV quantum family. Returns a record dict.

    On crash, captures the traceback and returns a partial dict with
    error=<str> so the caller can keep going.
    """
    rec: dict = {"family": family, "error": None}
    try:
        from qlnn_.training.p3_9_pde_matrix import QUANTUM_FAMILIES
        from qlnn_.training.pde_demo import PDE_BENCH
        from qlnn_.training.pde_residual_loss import make_pde_residual_loss

        bench = PDE_BENCH["kdv"]
        circuit, p0, info = QUANTUM_FAMILIES[family](SEED)
        rec["pqc_params"] = int(info["pqc_params"])
        rec["classical_params"] = int(info["classical_params"])
        rec["config_str"] = info["config_str"]

        # Collocation grid identical to _train_pde_one_generic.
        t_colloc = jnp.linspace(bench.t0, bench.t1, N_T_COLLOC + 2)[1:-1]
        x_colloc = jnp.linspace(bench.x0, bench.x1, N_X_COLLOC + 2)[1:-1]
        T, X = jnp.meshgrid(t_colloc, x_colloc, indexing="ij")
        tx_colloc = jnp.stack([T.ravel(), X.ravel()], axis=1)

        loss_fn, _u_of_tx = make_pde_residual_loss(
            circuit, bench.pde_residual, bench.ic_fn,
            t0=bench.t0, t1=bench.t1, x0=bench.x0, x1=bench.x1,
            need_uxxx=bench.needs_uxxx,
        )
        opt = optax.adam(LR)
        opt_state = opt.init(p0)
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        p = p0
        hist: list[float] = []

        # iter 0: JIT compile + first eval
        t_jit_start = time.perf_counter()
        val, grads = loss_and_grad(p, tx_colloc)
        val.block_until_ready()  # force JAX to actually finish
        for leaf in jax.tree_util.tree_leaves(grads):
            leaf.block_until_ready()
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        jit_sec = time.perf_counter() - t_jit_start
        hist.append(float(val))

        # iters 1..N-1: steady state
        t_steady_start = time.perf_counter()
        for _ in range(SMOKE_STEPS - 1):
            val, grads = loss_and_grad(p, tx_colloc)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            hist.append(float(val))
        # Block on the final result.
        if hist:
            float(hist[-1])
        steady_total_sec = time.perf_counter() - t_steady_start
        steady_per_iter_sec = steady_total_sec / max(SMOKE_STEPS - 1, 1)

        est_full_sec = jit_sec + PROD_STEPS * steady_per_iter_sec
        est_full_hours = est_full_sec / 3600.0

        rec.update({
            "jit_sec": jit_sec,
            "steady_per_iter_sec": steady_per_iter_sec,
            "steady_per_iter_ms": steady_per_iter_sec * 1000.0,
            "smoke_steps": SMOKE_STEPS,
            "prod_steps": PROD_STEPS,
            "est_full_cell_sec": est_full_sec,
            "est_full_cell_hours": est_full_hours,
            "loss_step0": hist[0] if hist else None,
            "loss_step_final": hist[-1] if hist else None,
            "loss_decreased": (
                (hist[0] > hist[-1]) if len(hist) >= 2 else None),
        })
    except Exception:
        rec["error"] = traceback.format_exc()
    return rec


def _print_table(records: list[dict]) -> None:
    print()
    print("=" * 78, flush=True)
    print("KdV smoke — per-family wall-clock measurement", flush=True)
    print("=" * 78, flush=True)
    print(f"  collocation     : {N_T_COLLOC} × {N_X_COLLOC} (canonical)",
          flush=True)
    print(f"  smoke steps     : {SMOKE_STEPS}", flush=True)
    print(f"  extrapolate to  : {PROD_STEPS} (CORRECTED_PDE_CONFIGS['kdv'])",
          flush=True)
    print(f"  seed            : {SEED}", flush=True)
    print()
    print(f"  {'family':<22}  {'JIT (s)':>9}  {'steady (ms/it)':>15}  "
          f"{'est full (hr)':>14}  {'loss↓':>5}", flush=True)
    print(f"  {'-' * 22}  {'-' * 9}  {'-' * 15}  {'-' * 14}  {'-' * 5}",
          flush=True)
    for r in records:
        if r["error"]:
            print(f"  {r['family']:<22}  {'CRASH':>9}  "
                  f"{'-':>15}  {'-':>14}  {'-':>5}", flush=True)
            continue
        print(f"  {r['family']:<22}  "
              f"{r['jit_sec']:>9.2f}  "
              f"{r['steady_per_iter_ms']:>15.2f}  "
              f"{r['est_full_cell_hours']:>14.2f}  "
              f"{'yes' if r['loss_decreased'] else 'NO':>5}", flush=True)
    print()


def _print_m3_estimate(records: list[dict]) -> None:
    successful = [r for r in records if not r["error"]]
    failed = [r for r in records if r["error"]]
    print("M3 estimate refinement", flush=True)
    print("-" * 78, flush=True)
    if failed:
        print(f"  ⚠️  {len(failed)} of {len(records)} families CRASHED — "
              f"do NOT proceed with M3 until investigated:", flush=True)
        for r in failed:
            print(f"      - {r['family']}", flush=True)
        print(flush=True)
    if not successful:
        print("  No successful families. Cannot estimate.", flush=True)
        return

    # KdV portion: sum of per-family hours × 3 seeds. If some families
    # crashed, extrapolate using the surviving mean as the per-family
    # estimate for the missing ones (conservative, prints with caveat).
    sum_hr_per_seed = sum(r["est_full_cell_hours"] for r in successful)
    mean_hr_per_cell = sum_hr_per_seed / len(successful)
    if failed:
        # Use the surviving mean for the missing families.
        sum_hr_per_seed_extrapolated = (
            sum_hr_per_seed + mean_hr_per_cell * len(failed))
    else:
        sum_hr_per_seed_extrapolated = sum_hr_per_seed
    kdv_qlnn_total_hr = sum_hr_per_seed_extrapolated * 3  # 3 seeds
    kdv_classical_hr = (3 * SEC_PER_CLASSICAL_PINN_CELL) / 3600.0
    kdv_total_hr = kdv_qlnn_total_hr + kdv_classical_hr

    # Kuramoto portion (untouched; uses prior estimate).
    kuramoto_qlnn_total_hr = (HOURS_PER_KURAMOTO_CELL_PRIOR
                              * N_KURAMOTO_QLNN_CELLS)
    kuramoto_classical_hr = (3 * SEC_PER_CLASSICAL_PINN_CELL) / 3600.0
    kuramoto_total_hr = kuramoto_qlnn_total_hr + kuramoto_classical_hr

    m3_total_hr = kdv_total_hr + kuramoto_total_hr

    print(f"  KdV QLNN  : 4 families × 3 seeds × ~{mean_hr_per_cell:.2f} "
          f"hr/cell = {kdv_qlnn_total_hr:.1f} hr",
          flush=True)
    print(f"  KdV cPINN : 3 seeds × ~1.5 sec = "
          f"{kdv_classical_hr * 3600:.1f} sec (negligible)",
          flush=True)
    print(f"  → KdV portion total : {kdv_total_hr:.1f} hr",
          flush=True)
    print(flush=True)
    print(f"  Kuramoto (UNCHANGED, prior estimate): "
          f"12 cells × {HOURS_PER_KURAMOTO_CELL_PRIOR:.1f} hr = "
          f"{kuramoto_qlnn_total_hr:.1f} hr  + "
          f"{kuramoto_classical_hr * 3600:.1f} sec cPINN",
          flush=True)
    print(f"  → Kuramoto portion total : {kuramoto_total_hr:.1f} hr",
          flush=True)
    print(flush=True)
    print(f"  ╔═════════════════════════════════════════════╗", flush=True)
    print(f"  ║  REFINED M3 TOTAL ESTIMATE : {m3_total_hr:>5.1f} hr  "
          f"({m3_total_hr / 24:>4.2f} days)  ║",
          flush=True)
    print(f"  ╚═════════════════════════════════════════════╝", flush=True)
    print(flush=True)
    print(f"  (Prior estimate from P6_LAUNCH_PLAN.md: 14-16 hr. "
          f"Compare and decide.)",
          flush=True)
    if failed:
        print(flush=True)
        print(f"  NOTE: {len(failed)} families crashed; the KdV "
              f"portion uses the surviving mean as a placeholder. "
              f"Investigate crashes before launching M3.",
              flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    families = ["chebyshev_dqc_2d", "qcpinn_2d",
                "te_qpinn_fnn_2d", "te_qpinn_qnn_2d"]
    records: list[dict] = []
    overall_start = time.perf_counter()
    for fam in families:
        print(f"[smoke_kdv] running {fam} ...", flush=True)
        rec = _smoke_one_family(fam)
        records.append(rec)
        if rec["error"]:
            print(f"[smoke_kdv] {fam} CRASHED — continuing", flush=True)
            # Print the traceback inline so it's visible without
            # reading the JSON.
            print(rec["error"], flush=True)
        else:
            print(f"[smoke_kdv] {fam}: JIT {rec['jit_sec']:.1f}s, "
                  f"steady {rec['steady_per_iter_ms']:.1f}ms/it, "
                  f"est full {rec['est_full_cell_hours']:.2f} hr, "
                  f"loss {rec['loss_step0']:.3e} → "
                  f"{rec['loss_step_final']:.3e}",
                  flush=True)
    overall_sec = time.perf_counter() - overall_start

    _print_table(records)
    _print_m3_estimate(records)

    out_json = OUT_DIR / "smoke_kdv_runtimes.json"
    payload = {
        "config": {
            "n_t_colloc": N_T_COLLOC,
            "n_x_colloc": N_X_COLLOC,
            "smoke_steps": SMOKE_STEPS,
            "prod_steps": PROD_STEPS,
            "seed": SEED,
            "lr": LR,
        },
        "overall_smoke_wall_clock_sec": overall_sec,
        "records": records,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[smoke_kdv] wrote {out_json.relative_to(REPO_ROOT)}",
          flush=True)
    print(f"[smoke_kdv] total smoke wall-clock: "
          f"{overall_sec:.1f} sec ({overall_sec / 60:.1f} min)",
          flush=True)

    # Exit 0 even on partial crashes — the user wanted the surviving
    # data + a clear printout. Crashes are surfaced in the table and
    # the JSON record.
    return 0


if __name__ == "__main__":
    sys.exit(main())
