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

### Phase A — ACCESS allocation

**Action.** Submit an ACCESS Explore proposal for Purdue Anvil.

Required:

- CV, <= 3 pages.
- Project abstract from `ACCESS_APPLICATION.md`.
- Advisor letter on institutional letterhead confirming support,
  dissertation relevance, and separation from advisor-funded grants.

**Gate.** ACCESS allocation granted and exchanged for Anvil GPU/AI time.

### Phase B — Anvil setup

**Action.** Clone the repo under Anvil scratch, create a Python 3.11
environment, install the project, and run one smoke cell.

Target:

```bash
PYTHONPATH=src python scripts/run_p7_8_h1_kuramoto_kdv.py --dry-run
PYTHONPATH=src python scripts/run_p7_8_h1_kuramoto_kdv.py --max-cells 1 --confirm
```

**Gate.** Smoke writes one valid `metrics.json` and no `error.json`.

### Phase C — Audit re-run matrix

**Action.** Run the remaining audit-driven cells as a SLURM array, one
cell per task.

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
