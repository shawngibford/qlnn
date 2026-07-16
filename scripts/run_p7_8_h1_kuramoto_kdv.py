"""P6 M0/G8 scaffold — kuramoto + KdV solver-task runner.

Status: SCAFFOLD ONLY. No training executes unless `--confirm` is
passed. Per `P6_LAUNCH_PLAN.md` §4, this is the M0 prep deliverable
for the M3 milestone (deferred-system completion). M3 will actually
spend the ~14-16 hours of compute this script schedules.

Why a parallel script (not `--systems` flag on `run_p7_8_h1_n24.py`):
  `run_p7_8_h1_n24.py` is a *verdict aggregator* — it reads existing
  per-cell `metrics.json` artifacts and runs a paired-bootstrap H1
  verdict. The work to add kuramoto + KdV is *upstream training*
  (writing the `metrics.json` artifacts that an aggregator would
  later consume). Conflating the two concerns in one script would
  obscure the cost (~16 hr training vs ~1 sec aggregation) and
  require restructuring the n=24 verdict runner. Per the plan, the
  cleaner choice is a parallel script. The n=24 aggregator is
  untouched; once kuramoto/KdV per-cell metrics exist, a follow-up
  aggregator (or a `--systems` extension at that point) folds them
  in.

What this script schedules (per pre-reg §4 hardness ladder + A11):
  - ODE side, kuramoto: 4 QLNN families × 3 seeds = 12 cells
      → results/p3_6_multi_state/{family}_kuramoto/seed_N/
    plus classical PINN baseline: 1 × 3 = 3 cells
      → results/p7_5_solver_h1/kuramoto_classical_pinn/seed_N/
  - PDE side, kdv: 4 QLNN families × 3 seeds = 12 cells
      → results/p3_9_pde_matrix/kdv_{family}/seed_N/  (3 fams)
      → results/p3_8_review/kdv_chebyshev_dqc_2d/seed_N/ (cheby)
    plus classical PINN baseline: 1 × 3 = 3 cells
      → results/p3_8_review/kdv_classical_pinn/seed_N/

  Total: 30 cells. With Agent C's per-cell estimates (~7 hr/cell
  kuramoto, ~8 hr/cell kdv) and 4 QLNN families on each system,
  the ceiling is dominated by the 24 QLNN cells. The classical
  PINN cells are ~1.5 sec each (negligible).

  Output is staged under a NEW directory to avoid polluting the
  existing P7.8 n=24 verdict tree:
      results/p6_kuramoto_kdv/<system>_<ansatz>/seed_<n>/

  Once cells complete and pass integrity, M5 will symlink or
  re-aggregate into `results/p7_8_solver_h1_n27` (or whatever the
  M3-completed combined verdict directory is named).

Hard constraints:
  - Default: print plan and refuse to execute (saves 16 hr of
    accidental compute).
  - `--dry-run`: print plan and exit, no execution attempted.
  - `--confirm`: required alongside any execution request.
  - `--max-cells N`: stage only the first N cells (for smoke).
  - Existing `run_p7_8_h1_n24.py` is untouched.

Ansatz parity:
  The ansatz sets mirror `run_p7_8_h1_n24.py`:
    ODE_QLNN_FAMILIES = (chebyshev_dqc, te_qpinn_fnn, te_qpinn_qnn,
                         qcpinn)
    PDE_QLNN_LOC keys = (chebyshev_dqc_2d, qcpinn_2d, te_qpinn_fnn_2d,
                         te_qpinn_qnn_2d)
  Seeds = (0, 1, 2) — matches n=24 verdict.

Blocker tracked (see Report when M0 ran):
  - `VECTOR_ODES` in `src/qlnn_/training/multi_state_solver.py` does
    NOT include kuramoto. The kuramoto generator at
    `src/quantum_liquid_neuralode/data_processing/synthetic_ode.py:
    _kuramoto_system` exists but is not wired into the H1 solver
    path. M3 prerequisite: register kuramoto as a `VectorODESystem`
    (12D, regime="smooth_periodic" per h1_verdict.py:32). KdV is
    fully wired (`PDE_BENCH["kdv"]` with needs_uxxx=True).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "results" / "p6_kuramoto_kdv"

# --- Parity with run_p7_8_h1_n24.py -------------------------------------
# Keep these in sync with the n=24 verdict runner. If those drift, this
# scaffold drifts with them — Update both together.
ODE_QLNN_FAMILIES = ("chebyshev_dqc", "te_qpinn_fnn", "te_qpinn_qnn",
                     "qcpinn")
PDE_QLNN_FAMILIES = ("chebyshev_dqc_2d", "qcpinn_2d", "te_qpinn_fnn_2d",
                     "te_qpinn_qnn_2d")
SEEDS = (0, 1, 2)

# Per-cell wall-clock estimates. Updated 2026-05-28 from the smoke
# measurements at scripts/smoke_kuramoto_p6.py + smoke_kdv_p6.py.
# Smokes ran the actual training loop on Apple-Silicon-CPU JAX, then
# extrapolated to per-family production step budgets.
# Prior Agent-C estimates (7.0 / 8.0 hr) were ~9× too pessimistic.
HOURS_PER_KURAMOTO_CELL = 0.80   # mean across 4 families; range 0.14-2.29
HOURS_PER_KDV_CELL = 1.13        # mean across 4 families; range 0.15-2.24
SEC_PER_CLASSICAL_PINN_CELL = 1.5     # negligible


@dataclass(frozen=True)
class Cell:
    """One unit of compute. Maps 1:1 to a `metrics.json` artifact."""
    task: str        # "ode_solver" | "pde_solver"
    system: str      # "kuramoto" | "kdv"
    ansatz: str      # one of *_QLNN_FAMILIES or "classical_pinn"
    seed: int
    est_hours: float
    out_dir: Path    # results/p6_kuramoto_kdv/<system>_<ansatz>/seed_N/


def _build_cells(out_root: Path = OUT_ROOT) -> list[Cell]:
    cells: list[Cell] = []

    # ODE side: kuramoto QLNN
    for fam in ODE_QLNN_FAMILIES:
        for seed in SEEDS:
            cells.append(Cell(
                task="ode_solver", system="kuramoto", ansatz=fam, seed=seed,
                est_hours=HOURS_PER_KURAMOTO_CELL,
                out_dir=out_root / f"kuramoto_{fam}" / f"seed_{seed}"))

    # ODE side: kuramoto classical PINN baseline
    for seed in SEEDS:
        cells.append(Cell(
            task="ode_solver", system="kuramoto", ansatz="classical_pinn",
            seed=seed, est_hours=SEC_PER_CLASSICAL_PINN_CELL / 3600.0,
            out_dir=out_root / "kuramoto_classical_pinn" / f"seed_{seed}"))

    # PDE side: kdv QLNN
    for fam in PDE_QLNN_FAMILIES:
        for seed in SEEDS:
            cells.append(Cell(
                task="pde_solver", system="kdv", ansatz=fam, seed=seed,
                est_hours=HOURS_PER_KDV_CELL,
                out_dir=out_root / f"kdv_{fam}" / f"seed_{seed}"))

    # PDE side: kdv classical PINN baseline
    for seed in SEEDS:
        cells.append(Cell(
            task="pde_solver", system="kdv", ansatz="classical_pinn",
            seed=seed, est_hours=SEC_PER_CLASSICAL_PINN_CELL / 3600.0,
            out_dir=out_root / "kdv_classical_pinn" / f"seed_{seed}"))

    return cells


def _print_plan(cells: list[Cell]) -> None:
    print("=" * 72, flush=True)
    print("P6/G8 scaffold: kuramoto + KdV solver-task cells", flush=True)
    print("=" * 72, flush=True)
    print(f"  total cells           : {len(cells)}", flush=True)
    total_h = sum(c.est_hours for c in cells)
    print(f"  total est. wall-clock : {total_h:.1f} hr"
          f"  ({total_h / 24.0:.2f} days)", flush=True)
    root = cells[0].out_dir.parent.parent if cells else OUT_ROOT
    try:
        root_disp = root.relative_to(REPO_ROOT)
    except ValueError:
        root_disp = root
    print(f"  output root           : {root_disp}/", flush=True)
    print(flush=True)
    print(f"  {'idx':>3}  {'task':<11}  {'system':<10}  {'ansatz':<20}  "
          f"{'seed':>4}  {'est_hr':>7}", flush=True)
    print(f"  {'-' * 3}  {'-' * 11}  {'-' * 10}  {'-' * 20}  "
          f"{'-' * 4}  {'-' * 7}", flush=True)
    for i, c in enumerate(cells):
        print(f"  {i:>3}  {c.task:<11}  {c.system:<10}  {c.ansatz:<20}  "
              f"{c.seed:>4}  {c.est_hours:>7.2f}", flush=True)
    print(flush=True)
    print("Per-cell estimates (smoke-measured 2026-05-28):", flush=True)
    print(f"  kuramoto QLNN per cell : ~{HOURS_PER_KURAMOTO_CELL:.2f} hr "
          f"(mean of 4 families on 12D per-component scalar circuits)",
          flush=True)
    print(f"  kdv QLNN per cell      : ~{HOURS_PER_KDV_CELL:.2f} hr "
          f"(mean of 4 families with jacrev³ triple-nested autodiff)",
          flush=True)
    print(f"  classical PINN cells   : ~{SEC_PER_CLASSICAL_PINN_CELL:.1f} "
          f"sec (negligible)", flush=True)
    print("=" * 72, flush=True)


# --- M3 dispatcher -------------------------------------------------------
# Bulky array keys split out of metrics.json into field.npz. Everything
# else (scalars, strings, lists of floats) stays in metrics.json so a
# downstream aggregator can read it without loading numpy arrays.
_BULKY_KEYS = ("t_eval", "x_eval", "u_pred", "u_ref",
               "loss_history", "mae_per_component")


def _dispatch_one(cell: "Cell") -> dict:
    """Call the right training entry point for this cell. Returns the
    raw result dict — `_execute` handles I/O.

    Routing:
      (ode_solver, kuramoto, QLNN family)   → train_one_vector
      (ode_solver, kuramoto, classical_pinn) → train_classical_pinn_solver_one_cell
      (pde_solver, kdv, QLNN family)        → train_one_cell (p3_9_pde_matrix)
      (pde_solver, kdv, classical_pinn)     → train_one_pde_classical
    """
    if cell.task == "ode_solver":
        if cell.ansatz == "classical_pinn":
            from qlnn_.training.p7_5_solver_h1 import (
                train_classical_pinn_solver_one_cell,
            )
            return train_classical_pinn_solver_one_cell(
                cell.system, cell.seed)
        else:
            from qlnn_.training.multi_state_solver import train_one_vector
            return train_one_vector(cell.ansatz, cell.system, cell.seed)
    elif cell.task == "pde_solver":
        if cell.ansatz == "classical_pinn":
            from qlnn_.training.p3_8_review_demo import (
                train_one_pde_classical,
            )
            return train_one_pde_classical(cell.system, cell.seed)
        else:
            from qlnn_.training.p3_9_pde_matrix import train_one_cell
            return train_one_cell(cell.system, cell.ansatz, cell.seed)
    else:
        raise ValueError(f"unknown task {cell.task!r}")


def _save_cell_result(cell: "Cell", result: dict, wall_clock_sec: float) -> None:
    """Split bulky arrays into field.npz; write scalars + small lists +
    wall_clock_sec to metrics.json. Adds cell-identifying fields so each
    metrics.json is self-describing."""
    bulky = {}
    scalars = {}
    for k, v in result.items():
        if k in _BULKY_KEYS and v is not None:
            # numpy arrays go to npz; per-component MAE lists stay in JSON
            # (they're small and a downstream aggregator wants them inline).
            if isinstance(v, list):
                scalars[k] = v
            else:
                bulky[k] = np.asarray(v)
        else:
            scalars[k] = v

    scalars["wall_clock_sec"] = float(wall_clock_sec)
    scalars["task"] = cell.task
    scalars["cell_system"] = cell.system
    scalars["cell_ansatz"] = cell.ansatz
    scalars["cell_seed"] = int(cell.seed)
    scalars["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"

    if bulky:
        np.savez_compressed(cell.out_dir / "field.npz", **bulky)
    (cell.out_dir / "metrics.json").write_text(
        json.dumps(scalars, indent=2, default=str))


def _save_cell_error(cell: "Cell", tb_text: str, wall_clock_sec: float) -> None:
    """Write error.json on crash. Does not raise — sweep continues."""
    payload = {
        "task": cell.task,
        "system": cell.system,
        "ansatz": cell.ansatz,
        "seed": int(cell.seed),
        "wall_clock_sec_before_crash": float(wall_clock_sec),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "traceback": tb_text,
    }
    (cell.out_dir / "error.json").write_text(
        json.dumps(payload, indent=2, default=str))


def _execute(cells: list[Cell]) -> None:
    """M3 dispatcher. Resumable (skips cells that already have
    metrics.json), per-cell error-isolated (a crashed cell logs to
    error.json and the sweep continues), self-describing outputs
    (every metrics.json carries task/system/ansatz/seed/wall_clock).
    """
    n = len(cells)
    n_done = 0
    n_skipped = 0
    n_crashed = 0
    sweep_start = time.perf_counter()

    for i, cell in enumerate(cells, start=1):
        cell.out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = cell.out_dir / "metrics.json"
        if metrics_path.exists():
            print(f"[{i:>2}/{n}] SKIP  {cell.task:<11} {cell.system:<10} "
                  f"{cell.ansatz:<20} seed_{cell.seed}  "
                  f"(metrics.json already exists)",
                  flush=True)
            n_skipped += 1
            continue

        started_at = datetime.now().strftime("%H:%M:%S")
        print(f"[{i:>2}/{n}] START {cell.task:<11} {cell.system:<10} "
              f"{cell.ansatz:<20} seed_{cell.seed}  "
              f"est={cell.est_hours:.2f}hr  at {started_at}",
              flush=True)

        t_start = time.perf_counter()
        try:
            result = _dispatch_one(cell)
            elapsed = time.perf_counter() - t_start
            _save_cell_result(cell, result, elapsed)
            rl2 = result.get("relative_l2")
            rl2_str = f"relL²={rl2:.4f}" if rl2 is not None else ""
            print(f"[{i:>2}/{n}] OK    {cell.task:<11} {cell.system:<10} "
                  f"{cell.ansatz:<20} seed_{cell.seed}  "
                  f"wall={elapsed/3600:.2f}hr  {rl2_str}",
                  flush=True)
            n_done += 1
        except Exception:
            elapsed = time.perf_counter() - t_start
            tb_text = traceback.format_exc()
            _save_cell_error(cell, tb_text, elapsed)
            print(f"[{i:>2}/{n}] CRASH {cell.task:<11} {cell.system:<10} "
                  f"{cell.ansatz:<20} seed_{cell.seed}  "
                  f"wall={elapsed/3600:.2f}hr  → error.json",
                  flush=True)
            print(tb_text, flush=True)
            n_crashed += 1

    total_hr = (time.perf_counter() - sweep_start) / 3600.0
    print("=" * 72, flush=True)
    print(f"M3 sweep done: {n_done} new / {n_skipped} skipped / "
          f"{n_crashed} crashed  out of {n}  in {total_hr:.2f} hr",
          flush=True)
    if n_crashed:
        print("⚠️  inspect <out-root>/*/seed_*/error.json "
              "for crash tracebacks", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the cell plan and exit. No execution. "
                          "Default if --confirm is missing, but this "
                          "flag is explicit and is the recommended "
                          "smoke-test invocation.")
    ap.add_argument("--max-cells", type=int, default=None, metavar="N",
                    help="Stage only the first N cells (for smoke / "
                          "staging). Default: all cells.")
    ap.add_argument("--out", type=Path, default=OUT_ROOT, metavar="DIR",
                    help="Output root for all cells (default: "
                          "results/p6_kuramoto_kdv). The Anvil SLURM "
                          "array passes results/anvil/p6_kuramoto_kdv "
                          "so HPC-produced cells are segregated from "
                          "any laptop runs.")
    ap.add_argument("--cell-index", type=int, default=None, metavar="I",
                    help="Run exactly ONE cell, the I-th entry of the "
                          "deterministic _build_cells() ordering. Used "
                          "by the Anvil SLURM array (slurm/"
                          "01_kuramoto_kdv.sbatch) to map "
                          "SLURM_ARRAY_TASK_ID → cell. Mutually "
                          "composable with --dry-run for plan preview.")
    ap.add_argument("--confirm", action="store_true",
                    help="REQUIRED to start training. Without this flag "
                          "the script prints the plan and exits 1 with "
                          "a usage hint, so an accidental "
                          "`python scripts/run_p7_8_h1_kuramoto_kdv.py` "
                          "never burns 16 hr of compute.")
    args = ap.parse_args()

    cells = _build_cells(out_root=args.out)
    if args.cell_index is not None:
        if not (0 <= args.cell_index < len(cells)):
            print(f"FATAL: --cell-index {args.cell_index} out of range "
                  f"[0, {len(cells) - 1}]", flush=True)
            return 2
        cells = cells[args.cell_index : args.cell_index + 1]
    if args.max_cells is not None:
        cells = cells[: args.max_cells]

    _print_plan(cells)

    if args.dry_run:
        print("--dry-run set: exiting without execution.", flush=True)
        return 0

    if not args.confirm:
        print("REFUSING TO START: --confirm not passed. This protects "
              "against a 16-hr accidental run. Re-invoke with:",
              flush=True)
        print(f"    python {Path(__file__).name} --confirm",
              flush=True)
        print("or smoke-test without compute with:", flush=True)
        print(f"    python {Path(__file__).name} --dry-run", flush=True)
        return 1

    # Belt-and-suspenders: even with --confirm, M0 cannot execute
    # because the M3 dispatcher isn't wired yet. _execute() raises.
    print("CONFIRMED: dispatching to _execute() ...", flush=True)
    _execute(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
