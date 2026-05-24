# Archive — legacy documents from the pre-pivot bioreactor-OD program

These files describe **superseded planning phases and the now-archived
bioreactor optical-density (OD) program**. They are preserved here for
reproducibility, audit trail, and provenance — not as current
documentation. For the **active program** (QLNN ODE/PDE solver +
forecaster, paper draft in `paper/main.tex`), read the top-level docs:

- [`../README.md`](../README.md) — repo front page
- [`../NEXT_AGENT_PICKUP.md`](../NEXT_AGENT_PICKUP.md) — one-line state
  + commands for new contributors
- [`../HANDOFF.md`](../HANDOFF.md) — operational pickup detail
- [`../ODE_PDE_PRE_REG.md`](../ODE_PDE_PRE_REG.md) — current
  pre-registration (v1)
- [`../PRE_REG_AMENDMENT.md`](../PRE_REG_AMENDMENT.md) — disclosed
  amendments
- `../paper/main.tex` — paper draft (§1–§8 complete; supplement in
  progress)

## Why these are archived

The original three-claim study (reproducibility / expressivity /
sample-efficiency on a single 778-row bioreactor fermentation run) was
honestly reassessed as a **rigorous null on an n=1 dataset** — not
publishable as a positive result. The project pivoted to a controlled
ODE → PDE solver+forecaster benchmark with strong matched baselines
and a falsifiable inductive-bias hypothesis. The new pre-registration
in `ODE_PDE_PRE_REG.md` supersedes the OD program; `verify_paper_integrity.py`
still gates the archived numbers for continuity, but they no longer
appear in the paper.

## What is in here

### Archived program docs

| File | Was | Status |
|---|---|---|
| `PAPER_SUMMARY.md` | Canonical numbers + final verdicts for the 3 OD claims | Integrity-gated, preserved unchanged |
| `PROJECT_DOSSIER.md` | Self-contained snapshot of the OD program | Preserved unchanged |
| `hypothesis.md` | v2 pre-registration of the OD program | Superseded by `../ODE_PDE_PRE_REG.md` |

### Phase audit trail (peer-review-style reviews that drove design changes)

| File | Phase |
|---|---|
| `REVIEW_step1_classical.md` | Phase A — classical Liquid-ODE baseline |
| `REVIEW_step23_quantum.md` | Phase A/B/C — JAX QLNN subpackage |
| `REVIEW_methodology.md` | Peer-review-style audit that drove the QWGAN-GP drop |
| `REVIEW_integration.md` | Cross-stack integration audit |
| `REVIEW_SYNTHESIS.md` | Synthesis that prioritized Phase A/B/C work |
| `REVIEW_step56.md` | Fresh review of Steps 5/6 (effective dimension + sample efficiency) |

### Step-specific plans and corrections

| File | Purpose |
|---|---|
| `STEP5_MONOTONICITY_NOTE.md` | Post-hoc correction to the Claim 2 (effective-dimension) monotonicity criterion |
| `STEP6_PLAN.md` | Plan for Step 6 (sample efficiency / Claim 3) |
| `spec.md` | Historical spec (pre-QWGAN-drop) |

### Superseded design documents

| File | Superseded by |
|---|---|
| `OPTION_B_SEARCH_DESIGN.md` | Pivoted away from the Option-B 5-gate optimization on OD data; see `../ODE_PDE_PRE_REG.md` |
| `UNIFIED_MATRIX_DESIGN.md` | The 529-config matrix design served the archived program; the post-pivot matrix is described in the current pre-reg |

## Integrity

The numbers cited in the archived `PAPER_SUMMARY.md` are still verified
by `scripts/verify_paper_integrity.py` (exit code 0) against the
committed JSONs under `../results/`. They are frozen and will not be
edited; if a discrepancy is ever introduced, the integrity check
catches it.
