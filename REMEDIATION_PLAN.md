# Remediation plan — close the peer-review gaps

*Companion to [`PEER_REVIEW_SYNTHESIS.md`](PEER_REVIEW_SYNTHESIS.md).
This doc turns every reviewer finding into a tiered action item with
explicit owner, effort, status, and gate condition.*

## How to read this doc

Five tiers, ordered by effort × blocker status, not by importance:

| Tier | Scope | Status | Effort |
|---|---|---|---|
| **1** | Paper-text polish (numbers / citations / amendment count) | In session | ~2 hr |
| **2** | Three implementation bugs as A20–A22 amendments | In session | ~4 hr |
| **3** | Phase C audit re-run sweep on Anvil | Deferred (ACCESS allocation) | ~3 days wall-clock |
| **4** | τ-mechanism work + sample expansion | Advisor decision | ~1–2 weeks |
| **5** | Reproducibility hardening (lockfile, provenance, Docker) | Partly in-session | ~1 day |

Tiers 1, 2, and in-session Tier 5 items get executed in the same
session that produces this doc. Tiers 3, 4, and deferred Tier 5
items wait on the gates listed in their sections.

## Tier 1 — Paper-text polish (~2 hr, in session)

Closes Reviewer 4's desk-reject risk. Each item is < 30 minutes.

### 1.1 — Headline number harmonization (24× / 25× / 24.4×)

**Reviewer flag:** R4 critical #1.

**Problem.** Abstract says "25× tighter seed variance". Every other
section says "24×". Computed value is 24.4× (FHN seed-variance ratio).

**Fix.** Pick "24×" (matches body + rounds correctly per
`scripts/verify_paper_integrity.py:357-360`). Change abstract in
`paper/main.tex` and audit `paper/sections/{01,03,05,07,08}*.tex`
for any 25× stragglers.

**Acceptance.** `grep -n "25[\\\\$]\\?times" paper/main.tex
paper/sections/*.tex` returns no hits in the seed-variance claim.

### 1.2 — Bibliography expansion

**Reviewer flag:** R4 critical #2; R5 critical #1 (novelty positioning).

**Problem.** 10 unique bibtex entries for a 17-page benchmark is
light. Hasani 2021 (the foundational liquid-time-constant
reference) is cited in-text but has no `\bibitem` entry.

**Fix.** Add these to the `.bib` source (filename TBD by inspection):

| Reference | Why | Where cited |
|---|---|---|
| Hasani 2021 (Liquid Time-Constant Networks) | foundational LTC reference; currently in-text but no bibtex | §4, §6 |
| Krishnapriyan 2021 (PINN failure modes) | establishes PINN limitations on chaotic systems | §3, §7 |
| Pérez-Salinas 2020 (universal data reuploading) | original data-reuploading universality result | §2, §6 |
| Thanasilp 2024 (random quantum data) | recent barren-plateau / random-data theory | §6, §7 |
| Huang 2025 review (QML systematic review) | benchmark + positioning context | §1, §7 |

**Acceptance.** `paper/main.bbl` rebuilds with ≥ 15 distinct entries
after `bash paper/build.sh`.

### 1.3 — Amendment-count sync

**Reviewer flag:** R4 critical #3; R1 minor #8.

**Problem.** Abstract claims "nineteen amendments documented openly".
Body `paper/sections/02_methods.tex` says
"Eleven amendments to the pre-registration are documented openly...
(A1--A11)" and `paper/sections/07_discussion.tex` says "Twelve
pre-registration amendments (A1--A12, plus the upcoming A13)".

**Fix.** Update body to say:

```
Nineteen amendments to the pre-registration are documented openly
in PRE_REG_AMENDMENT.md (A1–A19). A1–A14 were filed during the
original analysis pass; A15–A19 were filed 2026-05-28 during an
internal audit that surfaced (a) cross-side training-budget parity,
(b) the PennyLane strongly_entangling fallback that silently
aliased data_reuploading, (c) a qcpinn quantum-attribution
sub-experiment, (d) brickwall's structural connectivity deficit at
small n, and (e) cross-task forecaster/solver budget parity. The
verdict refresh under the A15–A19 configuration is reported as
Phase C deferred to the supplementary compute campaign.
```

**Acceptance.** No "upcoming A13" string anywhere in the paper.

### 1.4 — Nice-to-have polish (if time permits)

- Forward-reference §6 → §4.5 LTC-decomposition setup (R4 minor #1).
- Hedge "first ever" in §8 conclusions (R4 minor #3).
- `\appendix` / `\subsection` labels in `paper/supplement.tex`
  (R4 minor #4).

## Tier 2 — Three implementation bugs as A20–A22 (~4 hr, in session)

Closes Reviewer 2's critical findings. All three are silent
small-n / paired-family correctness violations analogous to A16
(the strongly_entangling PennyLane fallback).

### A20 — te_qpinn readout consistency

**Reviewer flag:** R2 critical #1, #3.

**Problem.** `src/qlnn_/circuits/te_qpinn.py:169` uses
`qml.expval(qml.prod(*[qml.PauliZ(k) for k in range(n)]))` for
te_qpinn_fnn (product of Z's; observable ∈ [−1, 1]).
`src/qlnn_/circuits/te_qpinn.py:306` uses
`qml.expval(qml.sum(*[qml.PauliZ(q) for q in range(n)]))` for
te_qpinn_qnn (sum of Z's; observable ∈ [−n, n]). Two ostensibly-
paired families with structurally different readouts. The trained
models fit different objectives.

**Fix.** Change line 169 to `qml.sum(...)` matching te_qpinn_qnn
and the Chebyshev-DQC precedent at
`src/qlnn_/training/physics_residual_loss.py:112`.

**Justification.** Berger 2025 Eq. 13 writes `O = ⊗ᵢ Zᵢ`, which the
fnn code interprets as the tensor-product observable
`⟨∏ᵢ Zᵢ⟩` (product expectation). The wider QPINN literature
(Kyriienko 2021, Zhou 2024) uses `Σᵢ ⟨Zᵢ⟩` (sum of single-qubit
expectations) — and the qnn variant ALREADY uses this. Picking
`qml.sum` for both:
1. Restores fnn/qnn paired-family equivalence.
2. Matches Chebyshev-DQC precedent in the same codebase.
3. Aligns with the "magnetization readout" notation used in
   §2 methods.

**Amendment text.** A20 documents the choice + disclosure that all
prior te_qpinn_fnn solver results used the product readout and
need re-running. Lands as part of the Tier 3 Phase C sweep at
~5 CPU-hr added cost.

### A21 — brickwall n=3 connectivity disclosure (no code change)

**Reviewer flag:** R2 critical #2.

**Problem.** At (num_qubits=3, num_layers=2), the alternating-CNOT
pattern gives layer-0 `CNOT(0,1)` and layer-1 `CNOT(1,2)`. Qubit 0
↔ qubit 2 has no direct CNOT path; the ansatz reduces to a linear
chain. At n=4 connectivity is restored. This is a silent
small-n topology change — the same class of bug as A16.

**Resolution.** Brickwall is already REMOVED from the empirical
forecaster sweep via **A18**. A21 strengthens the A18 disclosure
with the deeper diagnosis: not just qubit-2-disconnected at L=1
(A18's reasoning), but *structurally a linear chain at L=2 unless
n ≥ 4*. The T3 mechanism scalars remain valid as untrained-circuit
data, but the connectivity caveat now appears alongside them.

**Code change.** None. A21 is amendment-text only.

### A22 — 2D PDE hard-IC factor

**Reviewer flag:** R2 minor #5.

**Problem.** `src/qlnn_/training/pde_residual_loss.py:25` (or near)
uses `u(t, x) = u₀(x) + t · [s · circuit(...) + b]`. The Lagaris
hard-IC trial solution should be `(t − t₀) · [...]` so that
`u(t₀, x) = u₀(x)` holds at the time origin. The current form
silently assumes t₀ = 0.

**Fix.** Change the formula in `pde_residual_loss.py` and the 2D
variant in `pde_2d/`. All current PDEs do have t₀ = 0, so no
committed number changes.

**Amendment.** A22 documents the latent-bug fix + the invariance
of all current numbers (because every PDE has t₀ = 0 today).
Provides protection against future PDEs with t₀ ≠ 0.

**Side-effects.** Any test that explicitly tests t₀ ≠ 0 (probably
none) needs review. Run pytest suite to confirm.

## Tier 3 — Phase C compute on Anvil (deferred)

**Gate.** ACCESS Explore allocation + advisor letter on institutional
letterhead. See [`NEXT_STEPS.md`](NEXT_STEPS.md) Phase A.

**Workload.** ~225 cells / ~53 CPU-hours. Embarrassingly parallel
on GPU.

**Impact of Tier 2 fixes on this scope.** A20 adds ~36 ODE +
~24 PDE te_qpinn_fnn cells with the corrected readout (~5 CPU-hr
extra). A21 + A22 add no compute (disclosure / latent-fix
respectively).

**Refreshed Tier 3 total.** ~258 cells / ~58 CPU-hours. Still well
under one Anvil weekend.

## Tier 4 — τ-mechanism work + sample expansion (advisor decision, ~1–2 weeks)

**Gate.** Venue decision. If the paper targets *PRX Quantum*, this
work is required (R5 critical #2, R1 critical #3). If the paper
ships to *Quantum* or *Phys. Rev. Research* as a rigorous null,
this becomes Follow-up Paper #2 per [`FOLLOW_UP_PAPERS.md`](FOLLOW_UP_PAPERS.md).

Items:

| Item | Reviewer | Effort | Outcome |
|---|---|---|---|
| BCa CI on Δ_τ_via_quantum − Δ_τ_via_classical (joint significance test) | R1 critical #3 | ~2 hr | Closes "post-hoc interpretation" concern |
| Expand forecaster n_broad from 3 to 6+ (add FHN + Allen-Cahn as forecaster targets) | R1 critical #2; R5 critical #3 | ~1 day code + Phase C re-run | Closes "underpowered forecaster verdict" |
| Theoretical posit for τ substrate-dependence (linearized derivation in supplement) | R5 critical #2 | ~3 days analytical | Closes "mechanism observed not explained" |
| KL-to-Haar trend scaled to n ≥ 27 for p < 0.05 | R5 missing-20% #2 | ~Phase C re-run + aggregator | Closes "tentative trend" status |

If Tier 4 lands fully, R5's PRX Q acceptance probability shifts
from 5-10 % to 50-70 %.

## Tier 5 — Reproducibility hardening

Closes Reviewer 3's caveats.

### 5.1 — In-session (~30 min total)

| Item | Effort | Reviewer |
|---|---|---|
| Startup guard: `assert not jax.config.read("jax_enable_x64")` in `scripts/verify_paper_integrity.py` | 5 min | R3 critical #3 |
| Gate `data/pde/manifest.json` SHA-256 in `verify_paper_integrity.py` | 15 min | R3 critical #4 |
| Document fresh-clone reproduction recipe in `CLAUDE.md` Commands section | 10 min | R3 minor #2 |

### 5.2 — Deferred (~1 day)

| Item | Effort | Reviewer |
|---|---|---|
| `pip-compile` to `requirements.lock` (full lockfile) | ~2 hr | R3 critical #2 |
| Per-seed `provenance.json` (git SHA + package versions + platform) | ~4 hr (touches many runners) | R3 critical #5 |
| Docker image + `reproduce_paper.sh` end-to-end recipe | ~1 day | R3 minor #1 |

## Closure matrix

How each PEER_REVIEW_SYNTHESIS finding maps to a tier:

| Finding | Source reviewer | Tier | Status |
|---|---|---|---|
| Pre-A15/A19 verdicts stale | R1 critical #1; R5 implicit | 3 | Gated on Anvil |
| Forecaster n_broad = 3 underpowered | R1 critical #2; R5 critical #3 | 4 | Advisor decision |
| τ-cross-check no joint test | R1 critical #3 | 4 | Advisor decision |
| 24× vs 25× / Hasani / amendment count | R4 critical #1, #2, #3 | 1 | In session |
| te_qpinn fnn/qnn readout divergence | R2 critical #1, #3 | 2 | In session (A20) |
| brickwall n=3 linear chain | R2 critical #2 | 2 | In session (A21) |
| 2D hard-IC t₀ assumption | R2 minor #5 | 2 | In session (A22) |
| Dependency pinning | R3 critical #2 | 5 | Deferred |
| JAX x64 guard | R3 critical #3 | 5 | In session |
| Per-seed provenance | R3 critical #5 | 5 | Deferred |
| Manifest SHA-256 gate | R3 critical #4 | 5 | In session |
| HPO symmetry only 3 anchors | R1 minor #4 | 4 (sample expansion) | Advisor decision |
| Power analysis missing | R1 minor #5 | 1 (small text addition) | In session |
| Multiple-comparisons caveat | R1 minor #6 | 1 (table footnote) | In session |
| BCa vs percentile CI | R1 minor #7 | 4 | Advisor decision |
| Narrative forward-ref §6→§4.5 | R4 minor #1 | 1 | In session |
| "First ever" hedging | R4 minor #3 | 1 | In session |
| Supplement Appendix labels | R4 minor #4 | 1 | In session |

## Session decision points (for the advisor meeting)

Three questions only the advisor can answer:

1. **Venue.** Stay at *PRX Quantum* (requires Tier 4 work + Phase C),
   or reposition to *Quantum* / *Phys. Rev. Research* (where the
   current paper is already submittable after Tier 1 + 2)?
2. **Timeline.** 3-day strengthening (Tier 1 + 2 + Phase C only,
   ship to *Quantum*/PRR), or 1–2 week strengthening (also Tier 4,
   ship to *PRX Quantum*)?
3. **ACCESS letter.** When can the one-paragraph advisor letter on
   institutional letterhead be drafted and signed? This is the
   single gate that unblocks Phase C and consequently Tier 4.

## What this plan does NOT do

- Does not launch any compute. Phase C is deferred to Anvil.
- Does not promise Tier 4. That is the venue decision.
- Does not file A20–A22 amendments in `PRE_REG_AMENDMENT.md` before
  the actual code patches land — amendments and code go together,
  same commit family.
- Does not edit `ODE_PDE_PRE_REG.md`. The pre-registration is
  immutable; only amendments are added.

---

*Generated as the closure plan for the 2026-05-28 peer-review pass.
For the unconsolidated reviewer transcripts, see the agent task
outputs of this session. Master HEAD `44e8175` as of writing.*
