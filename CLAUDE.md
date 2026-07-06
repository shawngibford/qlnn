# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Where to start

This is a **QLNN ODE/PDE solver+forecaster benchmark paper** (PRX
Quantum target). Read in this order:

1. [`ADVISOR_BRIEF.md`](ADVISOR_BRIEF.md) — plain-language narrative
   of the work, the headline finding, and the case for Anvil HPC.
2. [`NEXT_STEPS.md`](NEXT_STEPS.md) — five-phase timeline (ACCESS
   allocation → Anvil setup → audit re-runs → paper refresh → submit).
3. [`paper/main.pdf`](paper/) — the 25-page draft + 8-page supplement.
4. [`PRE_REG_AMENDMENT.md`](PRE_REG_AMENDMENT.md) — the 22
   pre-registration amendments (A1-A22) documenting every methodological
   choice and deviation.
5. [`ODE_PDE_PRE_REG.md`](ODE_PDE_PRE_REG.md) — the original
   pre-registration (foundational; do not modify without a corresponding
   amendment).

## Current state (master HEAD, post-2026-05-28 audit session)

- **Paper draft complete and integrity-gated:** main 25pp + supplement 8pp.
  Headline `fig:h1-verdict` lives in §5. Abstract and all body sections were
  de-overwritten 2026-06-15.
- **Two PRIMARY verdicts, both FALSIFIED**:
  - Solver-task H1 (n=24): Δ_diff ≈ −0.084 (CI includes 0; sign
    flipped from +0.032 at n=18 when broadband bin expanded).
  - Forecaster-task H1 (n=9): Δ_combined ≈ −0.501 (CI excludes 0
    negatively). Complete 2×2 mechanism decomposition with both
    algebraic identities holding per-cell exactly.
- **The major new finding** — the τ-isolation cross-check disagrees
  in sign between the two decomposition paths. The liquid-τ machinery
  is **substrate-dependent**: positive Δ on classical MLP hidden state,
  negative Δ on quantum cell hidden state. This is the headline
  *mechanism* result and the seed of a follow-up paper.
- **2026-05-28 audit session** added five amendments (A15-A19) that
  close every reviewer-fairness concern:
  - A15 — uniform 2000-step training budget across ALL solver models
    (QLNN families AND classical PINN).
  - A16 — un-aliased strongly_entangling from data_reuploading
    (PennyLane fallback was producing bit-identical outputs at n=3, L=1).
  - A17 — qcpinn quantum-parameter sweep (3 step-wise variants along
    PQC/(PQC+classical) ratio: 2% → 24% → 45% → 87%).
  - A18 — brickwall removed from empirical forecaster sweep (qubit 2
    structurally disconnected at n=3, L=1; T3 mechanism scalars kept
    as untrained-circuit diagnostic data).
  - A19 — cross-task budget parity (forecaster step budget raised
    200 → 2000 to match the solver side).
- **M3 runner wired:** `scripts/run_p7_8_h1_kuramoto_kdv.py` has a
  resumable, error-isolated `_execute()` dispatcher. Integration smoke
  passed (cell 0 — kuramoto chebyshev_dqc seed 0 — relL² 0.0014).
- **Refined compute estimates** from the 2026-05-28 smokes:
  ~0.8 hr/kuramoto-cell at the per-family budgets (0.9 hr at the new
  uniform 2000-step budget), ~1.1 hr/KdV-cell. Combined re-run budget
  for the committed scope (M3 + A15/A17 ODE + A16/A19 forecaster):
  ~216 cells / ~55 CPU-hours. Embarrassingly parallel. Optional
  PDE-side A17 extension adds ~36 cells / ~60 CPU-hours.
- **✅ UNBLOCKED (2026-07-06):** ACCESS allocation AWARDED — Anvil
  access granted. Phase A of NEXT_STEPS.md is closed. SLURM job-array
  infrastructure is committed under `slurm/` (see `slurm/README.md`).
  Next: fill `QLNN_ACCOUNT` in `slurm/config.env`, run
  `slurm/env_setup.sh` on an Anvil login node, pass the smoke gate,
  then `slurm/submit_all.sh` (222 cells, ~2-3 hr wall-clock).
- **Integrity gate exit-0** throughout the session. Locked numbers
  will refresh once the audit-driven re-runs land (Phase D of
  NEXT_STEPS.md).
- **GitHub:** `master` is the current branch (consolidated 2026-05-28).

## What NOT to do

- **Do not** launch the M3 sweep or any audit re-runs on the laptop
  — wait for the Anvil allocation. The runner is wired and ready to
  fire with `--confirm`, but the compute belongs on GPU.
- **Do not** modify `verify_paper_integrity.py`'s locked numbers
  until Phase D — the current numbers gate the *current* committed
  results, and the refresh is M5's job.
- **Do not** touch the pre-registration (`ODE_PDE_PRE_REG.md`)
  without a corresponding amendment entry in `PRE_REG_AMENDMENT.md`.
- **Do not** push to `main` on GitHub (placeholder). Push to `master`.

---

## Commands

### Setup (Python 3.11)
```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev]"
```

### Verify
```bash
PYTHONPATH=src .venv/bin/python -m pytest -q --tb=short
PYTHONPATH=src .venv/bin/python scripts/verify_paper_integrity.py
bash paper/build.sh
bash paper/build_supplement.sh
```

All four must exit-0 / build clean for the repo to be in a publishable
state. If tests drift, do not alter locked paper numbers to compensate;
fix the test/code issue or document the failure before submission.

### Smoke checks
```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_kdv_p6.py            # 4 KdV families
PYTHONPATH=src .venv/bin/python scripts/smoke_kuramoto_p6.py       # 4 kuramoto families
PYTHONPATH=src .venv/bin/python scripts/smoke_post_audit.py        # qcpinn variants + cPINN
```

### M3 + audit re-run launch (ACCESS AWARDED — run on Anvil, not laptop)
```bash
# On an Anvil login node (see slurm/README.md for the full sequence):
bash slurm/env_setup.sh                              # bootstrap + integrity gate
# edit slurm/config.env → set QLNN_ACCOUNT
sbatch -A $QLNN_ACCOUNT -p debug slurm/00_smoke.sbatch   # 5-cell smoke gate
cd $QLNN_ROOT/slurm && touch SMOKE_PASSED && ./submit_all.sh  # 222 cells

# Local plan preview (safe, no compute):
PYTHONPATH=src .venv/bin/python scripts/run_p7_8_h1_kuramoto_kdv.py --dry-run
```

The runner is resumable (skips cells whose `metrics.json` already
exists) and per-cell error-isolated (crashes write `error.json` and
the sweep continues).

## Architecture (post-pivot)

Active surface:

- `src/qlnn_/` — JAX + Equinox + Diffrax + PennyLane. Quantum
  circuit families (`circuits/`), liquid + non-liquid quantum cells
  (`cells/`), forecaster models (`models/`), training entry points
  (`training/{multi_state_solver,p3_9_pde_matrix,p3_8_review_demo,p4_forecaster_demo,p5_matched_baselines,p7_5_solver_h1,...}.py`),
  T3 diagnostics (`diagnostics/`).
- `src/quantum_liquid_neuralode/` — PyTorch baseline (LiquidCell,
  LiquidODForecaster, dataset utilities). Maintained for cross-stack
  comparison; the JAX side carries the current paper's claims.

Inactive / archived (preserved for `verify_paper_integrity` continuity
with the locked OD-frozen numbers):

- `archive/{src,scripts,tests,results}/` — pre-pivot bioreactor-OD
  artifacts.
- `archive/configs-od-era/` — the 585 YAMLs from the OD-era
  experiment-config sweep (none referenced by current code).
- `archive/superseded-2026-05-28/` — pivot-era planning docs
  (HANDOFF, NEXT_AGENT_PICKUP, P6_LAUNCH_PLAN, PURGE_PLAN)
  superseded by ADVISOR_BRIEF + NEXT_STEPS.

## Key dependencies

`torch`, `torchdiffeq` (classical baseline); `jax`, `jaxlib`,
`equinox`, `diffrax`, `optax`, `pennylane` (quantum side); `numpy`,
`scipy`. PennyLane device = `default.qubit` with JAX interface
(effective `diff_method = backprop`) — the only configuration that
composes through the nested Diffrax + `jax.jacrev` autodiff path at
the qubit counts used here.

`jax_enable_x64` is intentionally off (Diffrax dtype-promotion
constraint). Reverse-mode `jax.jacrev` is required through Diffrax's
`custom_vjp`.
