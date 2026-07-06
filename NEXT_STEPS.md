# Next steps — strengthened path to PRX Quantum submission

Read `ADVISOR_BRIEF.md` first. This document is the operational path.

## Current state

- Paper: `paper/main.pdf` **25 pp** + `paper/supplement.pdf` **8 pp**.
- Results: **405** completed `metrics.json`, **0** `error.json`.
- Integrity: `scripts/verify_paper_integrity.py` passes.
- Scientific verdict: pre-registered QLNN advantage hypothesis is
  **FALSIFIED** under matched controls.
- Main open decision: advisor/coauthor approval of the falsification
  framing and the Anvil-first strengthened path.

## Default path: strengthened submission

### Phase A — ACCESS allocation ✅ DONE (awarded 2026-07-06)

**AWARDED.** ACCESS allocation granted with Anvil access. Phase A is
closed. Remaining residue: put the awarded account string into
`slurm/config.env` (`QLNN_ACCOUNT`) before submitting jobs.

### Phase B — Anvil setup ⏩ NOW UNBLOCKED

**Action.** The full SLURM job-array infrastructure is committed under
`slurm/` (see `slurm/README.md`). On an Anvil login node:

```bash
bash slurm/env_setup.sh        # clone → venv → integrity gate must pass
vim slurm/config.env           # set QLNN_ACCOUNT to the awarded account
cd $QLNN_ROOT/slurm
sbatch -A $QLNN_ACCOUNT -p debug 00_smoke.sbatch   # 5 representative cells
# inspect logs/smoke_*.out — all five must end "OK", then:
touch SMOKE_PASSED
```

**Gate.** Smoke writes valid `metrics.json` files and no `error.json`;
`SMOKE_PASSED` marker created.

### Phase C — Audit re-run matrix

**Action.** After the smoke gate passes, submit the five job arrays +
dependent aggregation with one command:

```bash
cd $QLNN_ROOT/slurm && ./submit_all.sh
```

All five arrays run concurrently (222 cells, ~115 core-hr, ~2-3 hr
wall-clock). Per-workload scripts and index decodes are documented in
`slurm/README.md`.

Committed scope:

| Workload | Cells | Est. CPU-hr |
|---|---:|---:|
| M3: kuramoto + KdV completion at uniform 2000 steps | ~30 | ~24 |
| A15: original ODE solver cells at uniform 2000 steps | ~60 | ~9 |
| A17: qcpinn ODE-side quantum-attribution variants | 45 | ~15 |
| A16/A19: forecaster re-runs after ansatz/budget fixes | ~90 | ~5 |
| **Subtotal** | **~225** | **~53** |

Optional PDE-side qcpinn attribution extension: ~36 cells / ~40 CPU-hr.

**Gate.** Every scheduled cell has `metrics.json`; `error.json` count is
zero after investigation.

### Phase D — Verdict refresh

**Action.**

- Re-run the H1 aggregators on the refreshed matrix.
- Update paper tables, captions, and locked integrity values.
- Rebuild main and supplement.

**Gate.**

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_paper_integrity.py
bash paper/build.sh
bash paper/build_supplement.sh
```

All must pass.

### Phase E — Submission staging

**Action.**

- Final advisor/coauthor sign-off.
- PRX Quantum cover letter review.
- arXiv source bundle and Zenodo/code archive.
- Submit.

**Gate.** Submission confirmation.

## Minimum fallback

The current paper is already integrity-gated and defensible. If speed is
preferred over strengthening:

```text
advisor sign-off -> final format pass -> PRX/arXiv submission
```

Risk: reviewers may ask about the audit re-runs that the strengthened
path would close before submission.

## Stop condition

This project stops when the strengthened package is submitted or when
the minimum fallback is submitted. No new systems, ansatz families, or
mechanism studies go into this manuscript after that point.
