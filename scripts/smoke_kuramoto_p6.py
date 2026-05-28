"""Kuramoto smoke — measure real per-iter cost for all 4 ODE-solver
quantum families on the 12D kuramoto system, mirroring smoke_kdv_p6.py.

The point: the prior 7 hr/kuramoto-cell estimate from Agent C is the
single dominant term in the M3 budget (~84 hr / total 97 hr after the
KdV smoke). If kuramoto is also overestimated, M3 collapses to
overnight wall-clock.

Methodology:
  1. Per family, JIT a value_and_grad of the per-component vector
     residual loss at the canonical 60-point t-collocation.
  2. Time iter 0 separately (JIT compile) from iters 1..N-1.
  3. Extrapolate to the FAMILY-specific production step budget pulled
     from solver_demo.FAMILIES (chebyshev 1200, te_qpinn_fnn 1500,
     te_qpinn_qnn 2000, qcpinn 1500).
  4. Write throwaway results to results/smoke_kuramoto/.

The runner scaffold scripts/run_p7_8_h1_kuramoto_kdv.py is untouched.
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
OUT_DIR = REPO_ROOT / "results" / "smoke_kuramoto"

# Canonical smoke config — collocation matches train_one_vector default;
# step budget is small enough to measure steady-state per-iter in ~5-15 min
# per family.
N_COLLOC = 60
SMOKE_STEPS = 50
SEED = 0
LR = 0.02

# Reference for the (already-smoked) KdV side.
HOURS_PER_KDV_CELL_MEASURED = 1.13  # mean of 4 families from KdV smoke
N_KDV_QLNN_CELLS = 12              # 4 families × 3 seeds
SEC_PER_CLASSICAL_PINN_CELL = 1.5


def _smoke_one_family(family: str, prod_steps: int) -> dict:
    """Run the smoke for one family on kuramoto. Returns a record dict.

    prod_steps comes from solver_demo.FAMILIES[family][1].
    On crash: capture traceback, return partial dict with error=<str>.
    """
    rec: dict = {"family": family, "prod_steps": prod_steps, "error": None}
    try:
        from qlnn_.training.multi_state_solver import (
            VECTOR_ODES, _build_per_component, _make_vector_residual_loss,
        )

        bench = VECTOR_ODES["kuramoto"]
        u0 = jnp.asarray(bench.u0)

        circuits, p_init, counts = _build_per_component(
            family, bench.dim, SEED)
        loss_fn, _u_of_t = _make_vector_residual_loss(
            circuits, bench.rhs_jax, t0=bench.t0, t1=bench.t1, u0=u0)

        t_colloc = jnp.linspace(bench.t0, bench.t1, N_COLLOC + 2)[1:-1]
        opt = optax.adam(LR)
        opt_state = opt.init(p_init)
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        p = p_init
        hist: list[float] = []
        rec["pqc_params"] = int(counts["pqc_params"])
        rec["classical_params"] = int(counts["classical_params"])
        rec["dim"] = int(bench.dim)
        rec["per_component_pqc"] = int(counts["per_component_pqc_params"])

        # iter 0: JIT compile + first eval
        t_jit_start = time.perf_counter()
        val, grads = loss_and_grad(p, t_colloc)
        val.block_until_ready()
        for leaf in jax.tree_util.tree_leaves(grads):
            leaf.block_until_ready()
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        jit_sec = time.perf_counter() - t_jit_start
        hist.append(float(val))

        # iters 1..N-1: steady state
        t_steady_start = time.perf_counter()
        for _ in range(SMOKE_STEPS - 1):
            val, grads = loss_and_grad(p, t_colloc)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            hist.append(float(val))
        if hist:
            float(hist[-1])
        steady_total_sec = time.perf_counter() - t_steady_start
        steady_per_iter_sec = steady_total_sec / max(SMOKE_STEPS - 1, 1)

        est_full_sec = jit_sec + prod_steps * steady_per_iter_sec
        est_full_hours = est_full_sec / 3600.0

        rec.update({
            "jit_sec": jit_sec,
            "steady_per_iter_sec": steady_per_iter_sec,
            "steady_per_iter_ms": steady_per_iter_sec * 1000.0,
            "smoke_steps": SMOKE_STEPS,
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
    print("=" * 82, flush=True)
    print("Kuramoto smoke — per-family wall-clock measurement", flush=True)
    print("=" * 82, flush=True)
    print(f"  system          : kuramoto (12D coupled phase oscillators)",
          flush=True)
    print(f"  collocation     : {N_COLLOC} t-points", flush=True)
    print(f"  smoke steps     : {SMOKE_STEPS}", flush=True)
    print(f"  seed            : {SEED}", flush=True)
    print()
    print(f"  {'family':<18}  {'prod_steps':>10}  {'JIT (s)':>9}  "
          f"{'steady (ms/it)':>15}  {'est full (hr)':>14}  {'loss↓':>5}",
          flush=True)
    print(f"  {'-' * 18}  {'-' * 10}  {'-' * 9}  {'-' * 15}  "
          f"{'-' * 14}  {'-' * 5}",
          flush=True)
    for r in records:
        if r["error"]:
            print(f"  {r['family']:<18}  {r['prod_steps']:>10}  "
                  f"{'CRASH':>9}  {'-':>15}  {'-':>14}  {'-':>5}",
                  flush=True)
            continue
        print(f"  {r['family']:<18}  {r['prod_steps']:>10}  "
              f"{r['jit_sec']:>9.2f}  "
              f"{r['steady_per_iter_ms']:>15.2f}  "
              f"{r['est_full_cell_hours']:>14.2f}  "
              f"{'yes' if r['loss_decreased'] else 'NO':>5}", flush=True)
    print()


def _print_m3_estimate(records: list[dict]) -> None:
    successful = [r for r in records if not r["error"]]
    failed = [r for r in records if r["error"]]
    print("M3 total estimate refinement (KdV measured + kuramoto measured)",
          flush=True)
    print("-" * 82, flush=True)
    if failed:
        print(f"  ⚠️  {len(failed)} of {len(records)} kuramoto families "
              f"CRASHED:", flush=True)
        for r in failed:
            print(f"      - {r['family']}", flush=True)
        print(flush=True)
    if not successful:
        print("  No successful families. Cannot estimate.", flush=True)
        return

    # Kuramoto portion: sum per-family hours × 3 seeds.
    sum_hr_per_seed = sum(r["est_full_cell_hours"] for r in successful)
    mean_hr_per_cell = sum_hr_per_seed / len(successful)
    if failed:
        sum_hr_per_seed_ext = (
            sum_hr_per_seed + mean_hr_per_cell * len(failed))
    else:
        sum_hr_per_seed_ext = sum_hr_per_seed
    kuramoto_qlnn_total_hr = sum_hr_per_seed_ext * 3
    kuramoto_classical_hr = (3 * SEC_PER_CLASSICAL_PINN_CELL) / 3600.0
    kuramoto_total_hr = kuramoto_qlnn_total_hr + kuramoto_classical_hr

    # KdV portion (already measured).
    kdv_qlnn_total_hr = HOURS_PER_KDV_CELL_MEASURED * N_KDV_QLNN_CELLS
    kdv_classical_hr = (3 * SEC_PER_CLASSICAL_PINN_CELL) / 3600.0
    kdv_total_hr = kdv_qlnn_total_hr + kdv_classical_hr

    m3_total_hr = kdv_total_hr + kuramoto_total_hr

    print(f"  Kuramoto QLNN  : 4 families × 3 seeds × ~"
          f"{mean_hr_per_cell:.2f} hr/cell = {kuramoto_qlnn_total_hr:.1f} hr",
          flush=True)
    print(f"  Kuramoto cPINN : 3 seeds × ~1.5 sec = "
          f"{kuramoto_classical_hr * 3600:.1f} sec",
          flush=True)
    print(f"  → Kuramoto portion total : {kuramoto_total_hr:.2f} hr",
          flush=True)
    print(flush=True)
    print(f"  KdV (MEASURED earlier): 12 cells × "
          f"{HOURS_PER_KDV_CELL_MEASURED:.2f} hr/cell = "
          f"{kdv_qlnn_total_hr:.2f} hr  + cPINN ~5 sec",
          flush=True)
    print(f"  → KdV portion total : {kdv_total_hr:.2f} hr",
          flush=True)
    print(flush=True)
    print(f"  ╔═══════════════════════════════════════════════════════╗",
          flush=True)
    print(f"  ║  REFINED M3 TOTAL ESTIMATE : {m3_total_hr:>6.2f} hr  "
          f"({m3_total_hr / 24:>4.2f} days)  ║",
          flush=True)
    print(f"  ╚═══════════════════════════════════════════════════════╝",
          flush=True)
    print(flush=True)
    print(f"  (Prior P6_LAUNCH_PLAN estimate: 14-16 hr. Agent-C estimate: "
          f"~97 hr.)",
          flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-family production step counts from solver_demo.FAMILIES
    family_steps = {
        "chebyshev_dqc":  1200,
        "te_qpinn_fnn":   1500,
        "te_qpinn_qnn":   2000,
        "qcpinn":         1500,
    }

    records: list[dict] = []
    overall_start = time.perf_counter()
    for fam, prod_steps in family_steps.items():
        print(f"[smoke_kuramoto] running {fam} (prod_steps={prod_steps}) ...",
              flush=True)
        rec = _smoke_one_family(fam, prod_steps)
        records.append(rec)
        if rec["error"]:
            print(f"[smoke_kuramoto] {fam} CRASHED — continuing", flush=True)
            print(rec["error"], flush=True)
        else:
            print(f"[smoke_kuramoto] {fam}: JIT {rec['jit_sec']:.1f}s, "
                  f"steady {rec['steady_per_iter_ms']:.1f}ms/it, "
                  f"est full {rec['est_full_cell_hours']:.2f} hr, "
                  f"loss {rec['loss_step0']:.3e} → "
                  f"{rec['loss_step_final']:.3e}",
                  flush=True)
    overall_sec = time.perf_counter() - overall_start

    _print_table(records)
    _print_m3_estimate(records)

    out_json = OUT_DIR / "smoke_kuramoto_runtimes.json"
    payload = {
        "config": {
            "system": "kuramoto",
            "n_colloc": N_COLLOC,
            "smoke_steps": SMOKE_STEPS,
            "seed": SEED,
            "lr": LR,
            "family_prod_steps": family_steps,
        },
        "overall_smoke_wall_clock_sec": overall_sec,
        "records": records,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[smoke_kuramoto] wrote {out_json.relative_to(REPO_ROOT)}",
          flush=True)
    print(f"[smoke_kuramoto] total smoke wall-clock: "
          f"{overall_sec:.1f} sec ({overall_sec / 60:.1f} min)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
