# Peer-review synthesis — 5-reviewer audit, 2026-05-28

*Internal pre-submission peer review. Five agents, five personas
(methodology / implementation / reproducibility / paper writing /
novelty + venue). Read alongside `ADVISOR_BRIEF.md` and
`NEXT_STEPS.md`.*

## Aggregate verdict

| Reviewer | Persona | Verdict |
|---|---|---|
| 1 | Methodology + statistical rigor (Bowles-Schuld skeptic) | **MAJOR REVISIONS** |
| 2 | Implementation + circuit faithfulness (paper-author critic) | **MAJOR — 3 critical bugs** |
| 3 | Reproducibility + integrity coverage | **YES with caveats** (moderate risk) |
| 4 | Paper writing + figure quality | **50-50 desk reject** |
| 5 | Novelty + venue fit (PRX Q editor) | **DESK REJECT — reposition** |

Worst-of-5 wins: **the paper is not ready for *PRX Quantum*
submission today.** The science is defensible, the methodology is
rigorous, the integrity gate is exemplary — but four of five
reviewers identify specific gaps that a real reviewer panel would
catch. Repositioning to *Quantum* or *Phys. Rev. Research* (with
the same paper) is a viable path; finishing the Phase C re-runs +
adding ~2 days of polish makes *PRX Quantum* defensible.

## The five top-priority must-fixes (deduplicated across reviewers)

These are items that **two or more** reviewers independently flagged
as blocking publication.

### 1. The PRIMARY verdicts are computed on pre-A15/A19 budgets

> *"The amendment's prediction that 'the FALSIFIED verdict is not
> expected to change qualitatively' is unverified assertion, not
> evidence."* — Reviewer 1

> *"The PRIMARY verdict uses default Adam on both, but the
> classical side may not have been tuned at the same intensity."*
> — Reviewer 5

The solver-task n=24 verdict (Δ_diff = −0.084) and the forecaster
n=9 verdict (Δ_combined = −0.501) were computed before A15
equalized the solver step budget across QLNN + classical, before
A16 un-aliased strongly_entangling, and before A19 raised the
forecaster budget 200 → 2000. The paper's headline numbers are
on the table the audit identified as unfair. **Until the Phase C
re-runs land and the integrity gate is bumped to the refreshed
numbers, the paper is structurally incomplete.**

This was already Phase C of `NEXT_STEPS.md`. The peer-review pass
confirms it is not optional.

### 2. The forecaster verdict is underpowered (n_broad = 3)

> *"A paired bootstrap CI at n=3 per regime is at the absolute
> lower limit of credibility… classical bootstrap theory recommends
> n ≥ 5 per stratum for percentile CIs."* — Reviewer 1

> *"Sample size is so small that changing which systems are
> included changes the verdict sign."* — Reviewer 5

The PRIMARY FORECASTER FALSIFIED verdict draws its broadband bin
from **Lorenz × 3 seeds only** — n_broad = 3. Two reviewers
independently flag this as the dominant statistical concern. The
percentile-method CI at n=3 is poorly calibrated; the resampling
distribution has only 27 unique resamples per regime.

Fix paths (any one is acceptable):
- Expand the forecaster sweep to include FitzHugh-Nagumo and
  Allen-Cahn / KdV as additional broadband cells (raise n_broad
  from 3 to 6 or 9). This is largely covered by the M3 + A19
  re-runs once Phase C lands.
- Report BCa-corrected CIs alongside percentile CIs as a sensitivity
  analysis.
- Explicitly downgrade the forecaster verdict from PRIMARY to
  "sensitivity analysis" and let the solver task carry the headline.

### 3. The τ-cross-check disagreement has no joint significance test

> *"The two verdicts are reported separately but the paper does NOT
> test whether the difference between them (ΔΔ_τ = −0.449) is
> statistically significant."* — Reviewer 1

> *"The mechanism story is 'here's what we observed' not 'here's
> why it happens.' Without that, the falsification is a null
> result."* — Reviewer 5

The A13 amendment reports Δ_τ_via_classical = +0.115 and
Δ_τ_via_quantum = −0.334 — signs disagree, both algebraic
identities hold exactly. This is positioned as the headline
mechanism finding. But there is no bootstrap CI on the difference
between the two τ contributions, and there is no theory for why
liquid-τ machinery would behave oppositely on the two substrates.

Per-cell decomposition shows the disagreement is driven by Lorenz
seed-variance (Δ_τ_via_quantum: LV ≈ +0.10 mean, Lorenz +0.36 mean
but s2 = +0.067, s0 = +0.594). A formal joint test would reveal
whether this is signal or noise.

Reviewer 5 explicitly names this as the deciding gap: "this could
be the anchor for a follow-up mechanistic paper — but as currently
reported it's a curiosity, not an explanation."

### 4. Paper-text inconsistencies that will trigger desk-review concerns

> *"Abstract says 25×, every other section says 24× (actual 24.4×).
> Looks like a proof error and invites scrutiny of data integrity."*
> — Reviewer 4

> *"Abstract claims nineteen amendments; the body still says 'A1–A12,
> plus the upcoming A13.' The paper was last drafted before A13–A19
> actually landed."* — Reviewer 4

Two concrete inconsistencies a desk editor will catch in 5 minutes:

| Where | Says | Reality |
|---|---|---|
| Abstract | "25× tighter seed variance" | Computed: 24.4× — every other section says 24× |
| Abstract | "nineteen amendments documented openly" | Body §2 + §7 say "A1–A12, plus the upcoming A13" |

Plus: **Hasani 2021 is cited in-text but has no `\bibitem` entry**.
This and 4 other must-cite references missing from a 10-entry
bibliography (Krishnapriyan 2021, Thanasilp barren-plateau,
Pérez-Salinas 2020 reuploading universality, Huang QML).

Action: ~2 hours of polish. Reviewer 4 says this single fix flips
desk-reject risk from 50-50 to clean accept.

### 5. Implementation bugs analogous to the audit-caught A16 alias

> *"This is the SAME class of bug as the caught strongly_entangling
> alias: gate structure silently differs at small dimensions."*
> — Reviewer 2

Reviewer 2 — the implementation specialist — found **three** new
issues in the same class as the strongly_entangling fallback bug
that the prior audit (A16) caught:

1. **te_qpinn_fnn vs te_qpinn_qnn readout divergence.**
   `src/qlnn_/circuits/te_qpinn.py:169` returns
   `qml.expval(qml.prod(*PauliZ))` (product of Z's) while line 306
   returns `qml.expval(qml.sum(*PauliZ))` (sum of Z's). Two
   different observables in two ostensibly-paired families. Unitarily
   non-equivalent; the trained model fits different objectives.
2. **brickwall at n=3 forms a linear chain.** The alternating-CNOT
   pattern at (num_qubits=3, num_layers=2) gives layer-0 CNOT(0,1)
   and layer-1 CNOT(1,2) — qubit 0 and qubit 2 have no direct
   coupling path. Same silent-topology bug as strongly_entangling's
   PennyLane fallback.
3. **2D PDE hard-IC silently assumes t₀ = 0.**
   `src/qlnn_/training/pde_residual_loss.py:25` uses
   `u(t,x) = u₀(x) + t · [s·circuit(...) + b]` instead of the
   correct `(t − t₀) · [...]`. All current PDEs have t₀ = 0 so the
   bug hasn't surfaced. A latent correctness violation.

These should be fixed as A20-A22 amendments + code patches before
the Phase C re-run sweep.

## Five should-fix items (revision-level, not desk-block)

These items would block a *Quantum*-tier review but the worst-case
penalty at *PRX Quantum* is "address in revision."

### 6. HPO symmetry only validated at 3 anchor cells

A9 tested HPO symmetry at LV s2, VdP s1, Lorenz s2 only — 3 of 24
solver cells. The reviewer wants extension to the full ladder, or
explicit text acknowledging anchor-cell-only scope.

### 7. No power analysis or minimum-sample-size justification

The pre-reg specifies n_iter ≥ 10,000 for bootstrap but doesn't
justify why n_smooth = 12 / n_broad = 12 / n_total = 9 are
sufficient. Standard practice would be a power simulation showing
n=24 reliably detects Δ ≈ 0.1.

### 8. Multiple-comparisons caveat absent

The §5 master-verdict table reports 11 H1 outcomes (5 sensitivity
points + 2×2 decomposition + 3 τ-cross-check variants). All
FALSIFIED in the same direction is robustness, not selection bias
— but a Bonferroni-style caveat in a table footnote would
preempt the question.

### 9. Percentile bootstrap CI not BCa-corrected

At n=3-9 per regime the percentile method is known to be
poorly calibrated. BCa is the standard correction. Reporting both
side-by-side would be honest.

### 10. Dependency pinning + reproducibility envelope

Reviewer 3's headline action: produce a `requirements.lock` (via
`pip-compile`), add a startup assertion that
`jax_enable_x64 is False`, add per-seed `provenance.json`, and
gate `data/pde/manifest.json` SHA-256s in the integrity script.
Without these, the integrity-gated numbers don't survive 12-18
months of upstream package drift.

## Strengths the swarm agreed on

Three things all five reviewers called out explicitly:

1. **The `verify_paper_integrity.py` gate is world-class.** 73
   numeric gates, comprehensive coverage of every paper claim,
   includes algebraic-identity verification on the 2×2
   decomposition. Reviewer 3 calls this "exemplary for QML
   benchmarking." Reviewer 5 names it as one of the strongest
   selling points of the work as-is.
2. **The pre-registration discipline is honest.** 19 amendments
   transparently documented. The FALSIFIED verdict is not spun.
   Limitations section is candid. Bowles-Schuld 2024-compliant
   in spirit.
3. **The 2×2 LTC decomposition (A12 + A13) is methodologically
   novel.** Even with the τ-disagreement noise concern, the
   decomposition framework itself is a genuine contribution.

## Honest "ship now or strengthen first?" — revised

`ADVISOR_BRIEF.md` says "minimum viable paper, yes." After the
peer-review pass: **the paper is submittable to *Quantum* or
*Phys. Rev. Research* today with ~2 hours of polish on the
text-level issues (24×/25×, bibliography, amendment count). It is
NOT submittable to *PRX Quantum* today** — at minimum it needs the
Phase C re-runs (to refresh the headline numbers from the
pre-amendment budgets) and ideally a joint test on the
τ-disagreement.

The honest matrix:

| Path | Time | PRX Q likelihood | Lower-tier likelihood |
|---|---|---:|---:|
| Submit as-is | 0 hr | 5-10 % | 35-50 % (Quantum, PRR) |
| 2 hr text-polish only | 2 hr | 10-15 % | 50-65 % |
| Polish + Phase C re-runs | ~3 days | 25-40 % | 70-80 % |
| Polish + Phase C + τ-mechanism work + n=27 | ~1-2 weeks | 50-70 % | 80-90 % |

The "polish + Phase C" row is where the project is heading per
`NEXT_STEPS.md`. The "+ τ-mechanism work" extension is exactly
what Follow-up Paper #2 in `FOLLOW_UP_PAPERS.md` proposes. Doing
that work *before* the current submission instead of after would
materially raise PRX Quantum's acceptance odds.

## Recommended next actions

In priority order:

1. **Apply the 2-hour text polish today.** Harmonize 24× vs 25×,
   add missing bibtex entries (Hasani 2021 is the must-do; add
   Krishnapriyan, Thanasilp, Pérez-Salinas, Huang while the
   bibliography is open), update body §2 + §7 to cite all 19
   amendments not just A1–A12. This eliminates the 50-50
   desk-reject risk Reviewer 4 flagged.
2. **File A20–A22 amendments + code patches** for the three new
   implementation bugs Reviewer 2 found. The brickwall and 2D
   hard-IC fixes mirror A16 in structure (silent small-n
   correctness violations). The te_qpinn readout fix needs a
   readout choice + a justification — recommend
   `qml.sum(*PauliZ)` for both fnn and qnn (matches Chebyshev-DQC
   precedent and the paper's "scalar Z sum" notation).
3. **Wait for Phase C re-runs.** The pre-A15/A19 headline numbers
   are blocking. Anvil allocation → re-run sweep → verdict refresh
   → bump integrity gate. This is the path `NEXT_STEPS.md`
   already describes.
4. **Decide the venue question.** With Phase C done + the text
   polish + the τ-mechanism work, *PRX Quantum* moves from 5-10%
   to 50-70% acceptance odds. Without the τ-mechanism work,
   repositioning to *Quantum* or *Phys. Rev. Research* is the
   higher-EV move (40-65 % vs 25-40 % at PRX Q).
5. **Tighten reproducibility envelope.** `pip-compile` to a
   `requirements.lock`, x64 startup assertion, per-seed
   provenance, manifest SHA-256 gate. ~1 day of work; mostly
   automatable; closes Reviewer 3's concerns.

## What this swarm did NOT do

- No code modified. No paper edited. No amendments filed. Only
  findings.
- All actions above are queued for next session(s), not executed
  today.
- The five reviewer transcripts live in the agent task outputs
  for this session and are not committed verbatim; this synthesis
  is the consolidated deliverable.

---

*Generated by a 5-agent peer-review swarm spawned 2026-05-28.
For the agent prompts, see the corresponding entry in the session
plan file. The reviews are deliberately adversarial: real
referees find problems.*
