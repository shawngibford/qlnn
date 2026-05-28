# Next steps — five phases from here to PRX Quantum submission

*One-page roadmap. Read `ADVISOR_BRIEF.md` first for the narrative;
this doc is the operational timeline.*

The session of 2026-05-28 landed eight commits on master (the M3
runner + five pre-registration amendments A15-A19 that close every
fairness concern an external reviewer could raise — see
`PRE_REG_AMENDMENT.md`). Every audit fix produces a re-run requirement.
We need an HPC environment to finish in days rather than weeks.

The single thing blocking everything is **Phase A: the ACCESS
allocation**, which requires a one-paragraph advisor letter.
Everything else is wall-clock + a working morning.

---

## Phase A — ACCESS allocation (1-3 business days, blocking)

**Action.** Submit an ACCESS Explore proposal at
<https://allocations.access-ci.org/get-your-first-project>.

**Required artifacts.**

- CV (≤ 3 pages).
- Project abstract — one-paragraph description of the workload,
  the systems being benchmarked, and the per-cell compute footprint
  (the smokes give us concrete numbers: ~0.8 hr/kuramoto cell,
  ~1.1 hr/KdV cell on Apple Silicon CPU; substantially faster on
  GPU per PennyLane Lightning + cuQuantum benchmarks).
- **Advisor letter on institutional letterhead** stating support
  for the Explore proposal, that the work is part of the student's
  dissertation, and that it is separate from the advisor's other
  funded grants.

**Outcome timeline.** ACCESS responds within one business day on
Explore. Credit-to-resource exchange takes up to one week.

**Gate.** Allocation granted, credits exchanged for "Anvil GPU" or
"Anvil AI" (H100) time.

---

## Phase B — Anvil environment provisioning (~2 hr)

**Actions.**

- `ssh anvil.rcac.purdue.edu`.
- Clone the repo under `/scratch/$USER/qlnn` (scratch has fast IO).
- Create a Python 3.11 venv from `pyproject.toml`.
- Load HPC modules:

  ```bash
  module load cuda/12 cuquantum
  pip install -e ".[dev]" pennylane-lightning-gpu
  ```
- Submit a 30-min smoke job:

  ```bash
  srun --gres=gpu:a100:1 -t 0:30 \
       PYTHONPATH=src python \
       scripts/run_p7_8_h1_kuramoto_kdv.py --max-cells 1 --confirm
  ```

**Gate.** The smoke cell writes
`results/p6_kuramoto_kdv/kuramoto_chebyshev_dqc/seed_0/metrics.json`
with `relative_l2 ≈ 0.0014` — matching the CPU integration smoke
already run on 2026-05-28.

---

## Phase C — Audit-driven re-run sweep (24-48 hr, embarrassingly parallel)

The audit produced 200 cells of re-runs across five workloads:

| Workload | Cells | Notes |
|---|---:|---|
| A15 — solver re-runs at uniform 2000 steps (7 QLNN families + classical PINN) | ~80 | includes the 3 A17 qcpinn variants |
| A16 — forecaster re-runs with un-aliased strongly_entangling | folded into A19 | |
| A17 — qcpinn quantum-attribution sub-experiment (3 variants × 8 systems × 3 seeds) | 72 | NEW work |
| A19 — forecaster all 5 sides at 2000 steps | 36 | post-A18 (brickwall removed) |
| M3 — kuramoto + KdV solver matrix | 30 | from the original launch plan |

**Action.** Submit as one big SLURM array job, one array task per
cell. The cells are independent, so the wall-clock collapses from
~145 CPU-hours serial to ~one cell-time on a GPU partition with
adequate parallelism.

**Resume protocol.** The M3 runner (`scripts/run_p7_8_h1_kuramoto_kdv.py`)
is resumable — any cell with a `metrics.json` already written is
skipped on re-invocation. Per-cell crashes write `error.json` and
the sweep continues.

**Gate.** Every cell has either `metrics.json` (success) or
`error.json` (with traceback investigated). Error count = 0 before
proceeding.

---

## Phase D — Verdict refresh + paper update (~2 hr)

**Actions.**

- Re-run the H1 aggregator on the refreshed cells (Δ_diff CI for
  solver task; Δ_combined 2×2 decomposition for forecaster).
- Update `paper/sections/05_h1_verdict.tex`, the master verdict
  table, and the `fig:h1-verdict` caption with refreshed numbers.
- Bump locked numbers + tolerances in
  `scripts/verify_paper_integrity.py` to match.
- Build paper + supplement; confirm 4-gate verification
  (pytest + integrity + main build + supplement build).

**Risk note.** The FALSIFIED verdict is robust — the CIs are
comfortably away from zero. The 2000-step budget can only sharpen
relL² on every model; the H1 contrast direction holds. But the
*magnitude* of the verdict numbers will refresh, and the paper text
needs the new numbers.

**Gate.** All 4 gates green; integrity exit-0.

---

## Phase E — PRX Quantum submission (~1 week polish)

**Actions.**

- Final figure pass; Overleaf import.
- Format check against PRX Quantum LaTeX template.
- Cover letter.
- arXiv preprint; Zenodo DOI for the code archive.
- Submit.

**Gate.** Submission confirmation email.

---

## Critical path summary

```
A (ACCESS) → B (Anvil setup) → C (re-run sweep) → D (paper update) → E (submit)
   1-3 d         ~2 hr           24-48 hr           ~2 hr        ~1 week
```

Everything past Phase A is wall-clock work that can be scheduled. Phase A
is the gate the advisor letter unblocks.
