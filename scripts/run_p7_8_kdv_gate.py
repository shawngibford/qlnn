"""P7.8 KdV mechanism gate — does `jacrev(jacrev(jacrev(QNode)))` work?

KdV equation: `u_t + 6·u·u_x + u_xxx = 0`. The third spatial derivative
`u_xxx` requires composing reverse-mode autodiff THREE times through a
PennyLane QNode. The P3.7 nested-autodiff gate proved `jacrev²` works
on `default.qubit` with the JAX interface; `jacrev³` is the natural
extension but has NOT been gate-tested.

This script answers two questions:

  Q1 (mechanism): does
      d3u_dx3 = jacrev(jacrev(jacrev(u_of_tx, argnums=1), argnums=1), argnums=1)
  produce FINITE non-trivial values at random init for the 2D
  Chebyshev-DQC circuit used in P3.7+?

  Q2 (cost): is the JIT compile time + per-step cost tractable (i.e.,
  not 10× slower than `jacrev²`)? If 10×, KdV is impractical at our
  compute budget.

Both questions must pass for KdV to enter the n=27 H1 verdict. If
either fails, KdV is documented as deferred to the follow-up paper
with the exact failure mode.

Outputs:
  results/p7_8_kdv_gate/gate_result.json
    — {jacrev3_finite: bool, jacrev3_nontrivial: bool,
       jacrev3_seconds_per_point: float,
       jacrev2_seconds_per_point: float,  (baseline for comparison)
       max_abs_uxxx: float, n_test_points: int,
       verdict: PASS | FAIL_FINITE | FAIL_NONTRIVIAL | FAIL_PROHIBITIVE_COST}
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from qlnn_.training.pde_residual_loss import (
    ChebyshevDQC2DConfig,
    build_chebyshev_dqc_2d,
    init_pde_solver_params,
    _affine_to_chebyshev_axis,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p7_8_kdv_gate"

# Same config as the P3.7 PDE gate (4 t-qubits, 4 x-qubits, 5 layers).
# Why: keeps the gate result interpretable against P3.7's jacrev²
# gate that already passed at this exact circuit shape.
N_T_QUBITS = 4
N_X_QUBITS = 4
N_LAYERS = 5

# A handful of (t, x) probe points inside the domain. We don't need
# many — the gate is "does the math work at all" not "is the model
# accurate".
N_TEST_POINTS = 5

# Cost-prohibitive threshold: jacrev³ must not be more than 5× slower
# per point than jacrev². If it is, KdV at 2400 steps × 64×32 colloc
# is intractable at our compute budget.
COST_RATIO_THRESHOLD = 5.0


def _git_prov() -> dict:
    try:
        c = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        b = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
        d = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip())
    except Exception:
        c, b, d = "unknown", "unknown", True
    return {"git_commit": c, "git_branch": b, "git_dirty": d}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"P7.8 KdV mechanism gate — start {start}", flush=True)
    print(f"  circuit: chebyshev_dqc_2d (n_t_q={N_T_QUBITS}, "
          f"n_x_q={N_X_QUBITS}, n_layers={N_LAYERS})", flush=True)

    cfg = ChebyshevDQC2DConfig(
        n_t_qubits=N_T_QUBITS, n_x_qubits=N_X_QUBITS, num_layers=N_LAYERS)
    circuit = build_chebyshev_dqc_2d(cfg)
    weights = init_pde_solver_params(cfg.weight_shape, seed=0)["w"]
    s = jnp.asarray(1.0)
    b_off = jnp.asarray(0.0)

    # KdV-like domain (matches data/pde/kdv.npz: t ∈ [0, 5], x ∈ [0, 40])
    t0_dom, t1_dom = 0.0, 5.0
    x0_dom, x1_dom = 0.0, 40.0

    def u_of_tx(t, x):
        t_chev = _affine_to_chebyshev_axis(t, t0_dom, t1_dom)
        x_chev = _affine_to_chebyshev_axis(x, x0_dom, x1_dom)
        # Lagaris hard-IC trial. The IC doesn't matter for the gate;
        # it's the autodiff topology of the circuit + the (t-t0) tail.
        n = s * circuit(t_chev, x_chev, weights) + b_off
        return (t - t0_dom) * n          # IC = 0 for the gate's purposes

    # First derivative w.r.t. x
    du_dx = jax.jacrev(u_of_tx, argnums=1)
    # Second derivative (the P3.7 gate's mechanism)
    d2u_dx2 = jax.jacrev(du_dx, argnums=1)
    # Third derivative (the KdV-specific test)
    d3u_dx3 = jax.jacrev(d2u_dx2, argnums=1)

    # Probe interior points
    rng = np.random.default_rng(42)
    t_probes = jnp.asarray(rng.uniform(t0_dom + 0.5, t1_dom - 0.5, N_TEST_POINTS))
    x_probes = jnp.asarray(rng.uniform(x0_dom + 1.0, x1_dom - 1.0, N_TEST_POINTS))

    print("  --- Compiling + evaluating jacrev² (baseline) ---", flush=True)
    t_pre = time.time()
    uxx_jit = jax.jit(d2u_dx2)
    uxx_first = float(uxx_jit(t_probes[0], x_probes[0]))     # JIT compile
    jit2 = time.time() - t_pre
    print(f"    jacrev² JIT compile : {jit2:.2f}s", flush=True)

    t_pre = time.time()
    uxx_vals = jnp.asarray([float(uxx_jit(t_probes[i], x_probes[i]))
                            for i in range(N_TEST_POINTS)])
    sec2 = (time.time() - t_pre) / N_TEST_POINTS
    print(f"    jacrev² per point   : {sec2*1000:.1f}ms  values: "
          f"{[f'{v:.4f}' for v in uxx_vals.tolist()]}", flush=True)

    print("  --- Compiling + evaluating jacrev³ (KdV gate) ---", flush=True)
    t_pre = time.time()
    try:
        uxxx_jit = jax.jit(d3u_dx3)
        uxxx_first = float(uxxx_jit(t_probes[0], x_probes[0]))   # JIT compile
        jit3 = time.time() - t_pre
        print(f"    jacrev³ JIT compile : {jit3:.2f}s", flush=True)

        t_pre = time.time()
        uxxx_vals = jnp.asarray([float(uxxx_jit(t_probes[i], x_probes[i]))
                                 for i in range(N_TEST_POINTS)])
        sec3 = (time.time() - t_pre) / N_TEST_POINTS
        print(f"    jacrev³ per point   : {sec3*1000:.1f}ms  values: "
              f"{[f'{v:.4f}' for v in uxxx_vals.tolist()]}", flush=True)

        finite = bool(np.all(np.isfinite(np.asarray(uxxx_vals))))
        max_abs = float(np.max(np.abs(np.asarray(uxxx_vals))))
        nontrivial = max_abs > 1e-6  # not numerically zero
        cost_ratio = sec3 / max(sec2, 1e-9)
        cost_ok = cost_ratio < COST_RATIO_THRESHOLD

        if not finite:
            verdict = "FAIL_FINITE"
        elif not nontrivial:
            verdict = "FAIL_NONTRIVIAL"
        elif not cost_ok:
            verdict = "FAIL_PROHIBITIVE_COST"
        else:
            verdict = "PASS"

        result = {
            "verdict": verdict,
            "jacrev3_finite": finite,
            "jacrev3_nontrivial": nontrivial,
            "max_abs_uxxx": max_abs,
            "max_abs_uxx_baseline": float(np.max(np.abs(np.asarray(uxx_vals)))),
            "jacrev3_seconds_per_point": sec3,
            "jacrev2_seconds_per_point": sec2,
            "cost_ratio_3rd_over_2nd": cost_ratio,
            "cost_ratio_threshold": COST_RATIO_THRESHOLD,
            "jacrev3_jit_compile_seconds": jit3,
            "jacrev2_jit_compile_seconds": jit2,
            "n_test_points": N_TEST_POINTS,
            "circuit_config": {
                "n_t_qubits": N_T_QUBITS, "n_x_qubits": N_X_QUBITS,
                "num_layers": N_LAYERS,
            },
        }
    except Exception as e:
        verdict = "FAIL_EXCEPTION"
        result = {
            "verdict": verdict,
            "exception_type": type(e).__name__,
            "exception_msg": str(e)[:1000],
            "jacrev2_seconds_per_point": sec2,
            "jacrev2_jit_compile_seconds": jit2,
            "max_abs_uxx_baseline": float(np.max(np.abs(np.asarray(uxx_vals)))),
            "circuit_config": {
                "n_t_qubits": N_T_QUBITS, "n_x_qubits": N_X_QUBITS,
                "num_layers": N_LAYERS,
            },
        }
        print(f"    jacrev³ FAILED: {type(e).__name__}: {str(e)[:300]}",
              flush=True)

    prov = {
        **_git_prov(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "wall_clock_start_utc": start,
        "wall_clock_end_utc": _dt.datetime.utcnow().isoformat() + "Z",
    }
    out_path = OUT / "gate_result.json"
    out_path.write_text(json.dumps({**result, "provenance": prov},
                                   indent=2) + "\n")

    print("=" * 70, flush=True)
    print(f"P7.8 KdV mechanism gate: {verdict}", flush=True)
    if verdict == "PASS":
        print("  → KdV is feasible. Queue the full 12-cell sweep.",
              flush=True)
    else:
        print(f"  → KdV deferred. Reason: {verdict}.", flush=True)
        if verdict == "FAIL_PROHIBITIVE_COST":
            print(f"  Cost ratio jacrev³/jacrev² = "
                  f"{result['cost_ratio_3rd_over_2nd']:.1f}× "
                  f"(threshold {COST_RATIO_THRESHOLD}×).",
                  flush=True)
    print(f"Result: {out_path}", flush=True)


if __name__ == "__main__":
    main()
