"""Heat IC-robustness FULL PRODUCTION matrix.

Same 5 ICs × 5 model classes × 3 seeds = 75-cell grid as
`scripts/smoke_heat_ic_matrix.py`, but at the PRODUCTION step budget
(1200 steps per CORRECTED_PDE_CONFIGS) instead of the 50-step smoke.

Mirrors the M3 runner pattern from `scripts/run_p7_8_h1_kuramoto_kdv.py`:
  - Resumable: skip any (family, IC, seed) cell whose `metrics.json`
    already exists.
  - Per-cell error-isolated: a crashed cell writes `error.json` with
    traceback and the sweep continues.
  - --confirm gate so an accidental invocation prints the plan and
    exits 1 instead of burning ~8 hr.
  - --max-cells N for stage-testing.
  - --dry-run prints the cell plan without running.

Output: results/heat_ic_matrix_full/<ic>_<family>/seed_N/metrics.json
        + results/heat_ic_matrix_full/<ic>_<family>/seed_N/field.npz

Per-cell est. wall-clock at 1200 steps (extrapolated from the
post-audit smoke + smoke_heat_ic_matrix measurements):

  chebyshev_dqc_2d  :  ~15 min/cell  (JIT 27s + 700 ms × 1200)
  qcpinn_2d         :  ~1 min/cell   (fastest)
  te_qpinn_fnn_2d   :  ~5.5 min/cell
  te_qpinn_qnn_2d   :  ~7.5 min/cell
  classical_pinn    :  ~1 min/cell

Total nominal: ~7-8 hr serial wall-clock on Apple Silicon.
Embarrassingly parallel by cell on Anvil GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "results" / "heat_ic_matrix_full"

# === Persistent JAX compilation cache =====================================
# The 50-step smoke surfaced an XLA recompile pathology on
# `chebyshev_dqc_2d × heat_gaussian` where individual cells could take
# 20+ min each instead of the expected ~60s. Each cell creates fresh
# QNodes via the factory pattern, which defeats JAX's in-memory
# JIT cache.
#
# Persistent disk cache solves this — JAX writes compiled XLA modules
# to a directory and reuses them across processes / fresh QNode
# instances when the (function, shape, dtype) signature matches.
# First chebyshev cell still pays the compile cost; subsequent
# chebyshev cells (across seeds, across ICs) hit cache and start
# in seconds.
#
# Must be set BEFORE jax is imported — using os.environ is the most
# robust way (jax.config.update also works but only after import).
_JAX_CACHE_DIR = REPO_ROOT / ".jax_compilation_cache"
_JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_JAX_CACHE_DIR))

import numpy as np

# All 5 ICs registered in PDE_BENCH (heat baseline + 4 variants).
IC_VARIANTS = (
    "heat",
    "heat_multifreq",
    "heat_gaussian",
    "heat_highfreq",
    "heat_step",
)

# 4 baseline PDE 2D QLNN families + classical PINN.
QLNN_FAMILIES = (
    "chebyshev_dqc_2d",
    "qcpinn_2d",
    "te_qpinn_fnn_2d",
    "te_qpinn_qnn_2d",
)
SEEDS = (0, 1, 2)

# Per-family wall-clock estimates at 1200 steps, in hours.
EST_HOURS = {
    "chebyshev_dqc_2d":  15 / 60.0,
    "qcpinn_2d":         1 / 60.0,
    "te_qpinn_fnn_2d":   5.5 / 60.0,
    "te_qpinn_qnn_2d":   7.5 / 60.0,
    "classical_pinn":    1 / 60.0,
}


@dataclass(frozen=True)
class Cell:
    ic: str
    family: str
    seed: int
    out_dir: Path
    est_hours: float


def _build_cells() -> list[Cell]:
    cells: list[Cell] = []
    for ic in IC_VARIANTS:
        for family in QLNN_FAMILIES:
            for seed in SEEDS:
                cells.append(Cell(
                    ic=ic, family=family, seed=seed,
                    out_dir=OUT_ROOT / f"{ic}_{family}" / f"seed_{seed}",
                    est_hours=EST_HOURS[family],
                ))
        for seed in SEEDS:
            cells.append(Cell(
                ic=ic, family="classical_pinn", seed=seed,
                out_dir=OUT_ROOT / f"{ic}_classical_pinn" / f"seed_{seed}",
                est_hours=EST_HOURS["classical_pinn"],
            ))
    return cells


def _print_plan(cells: list[Cell]) -> None:
    print("=" * 78, flush=True)
    print("Heat IC-robustness FULL matrix — production-step sweep", flush=True)
    print("=" * 78, flush=True)
    total_hr = sum(c.est_hours for c in cells)
    print(f"  total cells           : {len(cells)}", flush=True)
    print(f"  total est. wall-clock : {total_hr:.1f} hr "
          f"(~{total_hr / 24:.2f} days serial)", flush=True)
    print(f"  output root           : "
          f"{OUT_ROOT.relative_to(REPO_ROOT)}/", flush=True)
    print(flush=True)
    print(f"  {'idx':>3}  {'ic':<16}  {'family':<20}  "
          f"{'seed':>4}  {'est_hr':>7}", flush=True)
    print(f"  {'-' * 3}  {'-' * 16}  {'-' * 20}  {'-' * 4}  {'-' * 7}",
          flush=True)
    for i, c in enumerate(cells):
        print(f"  {i:>3}  {c.ic:<16}  {c.family:<20}  "
              f"{c.seed:>4}  {c.est_hours:>7.2f}", flush=True)
    print("=" * 78, flush=True)


# --- Result serialization (same convention as the M3 runner) ----------------

_BULKY_KEYS = ("t_eval", "x_eval", "u_pred", "u_ref",
               "loss_history", "mae_per_component")


def _dispatch_one(cell: Cell) -> dict:
    """Call the right training entry point for this cell."""
    if cell.family == "classical_pinn":
        from qlnn_.training.p3_8_review_demo import train_one_pde_classical
        return train_one_pde_classical(cell.ic, cell.seed)
    else:
        from qlnn_.training.p3_9_pde_matrix import train_one_cell
        return train_one_cell(cell.ic, cell.family, cell.seed)


def _save_cell_result(cell: Cell, result: dict, wall_clock_sec: float) -> None:
    bulky = {}
    scalars = {}
    for k, v in result.items():
        if k in _BULKY_KEYS and v is not None:
            if isinstance(v, list):
                scalars[k] = v
            else:
                bulky[k] = np.asarray(v)
        else:
            scalars[k] = v
    scalars["wall_clock_sec"] = float(wall_clock_sec)
    scalars["cell_ic"] = cell.ic
    scalars["cell_family"] = cell.family
    scalars["cell_seed"] = int(cell.seed)
    scalars["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"
    if bulky:
        np.savez_compressed(cell.out_dir / "field.npz", **bulky)
    (cell.out_dir / "metrics.json").write_text(
        json.dumps(scalars, indent=2, default=str))


def _save_cell_error(cell: Cell, tb_text: str, wall_clock_sec: float) -> None:
    payload = {
        "ic": cell.ic, "family": cell.family, "seed": int(cell.seed),
        "wall_clock_sec_before_crash": float(wall_clock_sec),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "traceback": tb_text,
    }
    (cell.out_dir / "error.json").write_text(
        json.dumps(payload, indent=2, default=str))


def _execute(cells: list[Cell]) -> None:
    n = len(cells)
    n_done = 0
    n_skipped = 0
    n_crashed = 0
    sweep_start = time.perf_counter()
    for i, cell in enumerate(cells, start=1):
        cell.out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = cell.out_dir / "metrics.json"
        if metrics_path.exists():
            print(f"[{i:>2}/{n}] SKIP  {cell.ic:<16} {cell.family:<20} "
                  f"seed_{cell.seed}  (metrics.json exists)", flush=True)
            n_skipped += 1
            continue
        started_at = datetime.now().strftime("%H:%M:%S")
        print(f"[{i:>2}/{n}] START {cell.ic:<16} {cell.family:<20} "
              f"seed_{cell.seed}  est={cell.est_hours:.2f}hr  "
              f"at {started_at}", flush=True)
        t_start = time.perf_counter()
        try:
            result = _dispatch_one(cell)
            elapsed = time.perf_counter() - t_start
            _save_cell_result(cell, result, elapsed)
            rl2 = result.get("relative_l2")
            rl2_str = f"relL²={rl2:.4f}" if rl2 is not None else ""
            print(f"[{i:>2}/{n}] OK    {cell.ic:<16} {cell.family:<20} "
                  f"seed_{cell.seed}  wall={elapsed / 60:.1f}min  "
                  f"{rl2_str}", flush=True)
            n_done += 1
        except Exception:
            elapsed = time.perf_counter() - t_start
            tb_text = traceback.format_exc()
            _save_cell_error(cell, tb_text, elapsed)
            print(f"[{i:>2}/{n}] CRASH {cell.ic:<16} {cell.family:<20} "
                  f"seed_{cell.seed}  wall={elapsed / 60:.1f}min  "
                  f"→ error.json", flush=True)
            print(tb_text, flush=True)
            n_crashed += 1
    total_hr = (time.perf_counter() - sweep_start) / 3600.0
    print("=" * 78, flush=True)
    print(f"Heat IC matrix sweep done: {n_done} new / {n_skipped} skipped / "
          f"{n_crashed} crashed  out of {n}  in {total_hr:.2f} hr",
          flush=True)
    if n_crashed:
        print("⚠️  inspect error.json files for tracebacks", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the cell plan and exit.")
    ap.add_argument("--max-cells", type=int, default=None, metavar="N",
                    help="Stage only the first N cells (for staging tests).")
    ap.add_argument("--confirm", action="store_true",
                    help="REQUIRED to start training (protects against "
                          "accidental ~8 hr runs).")
    args = ap.parse_args()
    cells = _build_cells()
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    _print_plan(cells)
    if args.dry_run:
        print("--dry-run set: exiting without execution.", flush=True)
        return 0
    if not args.confirm:
        print("REFUSING TO START: --confirm not passed.", flush=True)
        print(f"    python {Path(__file__).name} --confirm", flush=True)
        return 1
    print("CONFIRMED: dispatching to _execute() ...", flush=True)
    _execute(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
