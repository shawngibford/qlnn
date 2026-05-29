# ACCESS Explore allocation — application materials

*Two ready-to-edit drafts. Replace bracketed placeholders with
your details before submitting.*

---

## 1. Project abstract (for the ACCESS allocations portal)

Paste into the "Project abstract" field of the ACCESS Explore
proposal. Target length: 200-300 words. The portal allows up to
~3,000 characters.

> ### A pre-registered benchmark of quantum-enhanced neural networks for differential-equation modeling
>
> This project is a controlled, pre-registered head-to-head benchmark
> of **Quantum Liquid Neural Networks (QLNNs)** against matched
> classical baselines on a structured hardness ladder of ordinary
> and partial differential equations. The work tests a falsifiable
> hypothesis from the quantum-machine-learning literature (a
> Schuld-Fourier regime-dependent inductive bias) and follows the
> rigorous methodological framework established by Bowles & Schuld
> 2024 for QML benchmarking. The current paper draft (17 pages main
> + 7 pages supplement) is integrity-gated by a mechanical
> reproducibility script that verifies every cited number against a
> committed JSON record; 22 pre-registration amendments transparently
> document every methodological choice.
>
> The compute request supports a 200-cell embarrassingly-parallel
> sweep across (a) the kuramoto + KdV completion of the 9-system
> pre-registered ladder, (b) audit-driven re-runs at uniform training
> budgets across all quantum and classical model classes (per-reg
> amendments A15–A19), (c) a quantum-attribution sub-experiment
> across three step-wise qcpinn variants, and (d) a per-cell
> initial-condition robustness check on the heat equation. Total
> footprint ≈ 200 CPU-hours serial; parallelized by cell on a multi-
> GPU partition, expected wall-clock is ~10 hours. Software stack:
> JAX + Equinox + Diffrax + PennyLane (`lightning.gpu` with NVIDIA
> cuQuantum SDK).
>
> All code, results, and pre-registration documentation are
> publicly available at `https://github.com/shawngibford/qlnn`.
> Target publication: *PRX Quantum* (Q4 2026).

**Notes for you to edit:**

- Replace the GitHub link if your username or repo name differs.
- If your advisor prefers a different target journal, swap "PRX
  Quantum" for whatever venue they suggest.
- Trim the parenthetical software-stack line if you want to free up
  characters for other detail.

---

## 2. Advisor letter (institutional letterhead format)

Paste into a Word/Google Docs document on your university's
letterhead. Your advisor signs and you upload it as the
"Letter of Support" attachment in the ACCESS Explore application.

> **[University Letterhead]**
>
> [Date]
>
> ACCESS Allocation Review Committee
> National Science Foundation
>
> Re: ACCESS Explore allocation in support of [Student Name]'s
> doctoral research
>
> Dear Reviewers,
>
> I write in support of the ACCESS Explore allocation request
> submitted by [Student Name], a doctoral candidate in my research
> group at [University / Department]. The proposed project — a
> pre-registered benchmark of quantum-enhanced neural networks
> against matched classical baselines on a hardness ladder of
> differential equations — constitutes a substantive component of
> [Student Name]'s dissertation work and is being conducted
> primarily by them.
>
> The compute request supports the completion of approximately
> 200 independent training cells (each one a 1,000–2,000-step
> optimization on a small PDE/ODE problem), which together form
> the empirical basis for the paper currently in preparation. The
> workload is embarrassingly parallel by cell and well-suited to
> Anvil's GPU partition; estimated total resource use is well
> within the Explore tier ceiling.
>
> This proposed work is **separate from my other funded research
> grants** and does not duplicate compute resources available through
> those grants. The research program is transparent and openly
> published at <https://github.com/shawngibford/qlnn>, including
> the full pre-registration, methodological amendments, and
> reproducibility infrastructure.
>
> I support this application and confirm [Student Name]'s
> qualifications to execute the proposed work.
>
> Sincerely,
>
> [Advisor Name]
> [Title, Department]
> [University]
> [Contact email]

**Notes for your advisor (or for you to edit before sending):**

- The "separate from my other funded grants" line is required by
  ACCESS for Explore-tier eligibility. Your advisor confirms it.
- "Approximately 200 independent training cells" matches the Phase
  C scope in `NEXT_STEPS.md`. Adjust if your scope changes before
  submission.
- The "no duplication" line addresses the standard Explore-tier
  question about overlap with existing allocations.

---

## Submission checklist

Before clicking "Submit" in the ACCESS allocations portal, verify
you have:

- [ ] ACCESS account created with institutional email
- [ ] Up-to-date CV (≤ 3 pages PDF)
- [ ] Project abstract (above, edited to voice)
- [ ] Advisor letter (above, on letterhead, signed by advisor)
- [ ] Target resource selected: **Anvil GPU** (or **Anvil AI** if
      you want H100 access — your call based on availability)
- [ ] Credit estimate filled in: **400,000 credits** (the Explore
      tier maximum; you don't need to justify the full amount, the
      committee scales the actual award)

Expected outcome timeline (per ACCESS documentation):
- **Decision**: 1 business day after submission
- **Credit-to-resource exchange**: up to 1 week after award

The award is for the duration of the doctoral project (Explore
allocations without a grant-citation default to 12 months and are
renewable).
