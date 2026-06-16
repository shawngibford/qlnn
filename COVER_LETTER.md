# Cover letter — PRX Quantum submission

*Stage-ready, NOT submitted. Submit only after advisor approval per
`NEXT_STEPS.md` Phase E. Paste the body below into the PRX Quantum
submission form's cover-letter field, or attach as a PDF.*

---

**To the Editors, PRX Quantum**

**Re: Submission of "Regime-dependent advantage in quantum
physics-informed neural networks: a pre-registered comparison on an
ODE/PDE hardness ladder"**

Dear Editors,

I am pleased to submit the enclosed manuscript for consideration as a
research article in PRX Quantum. The work is original, has not been
published elsewhere, and is not under consideration at any other
journal.

**What the paper does.** Quantum machine learning is in the middle of
a benchmarking-credibility crisis: a large fraction of claimed
"quantum advantages" do not survive matched-budget, properly
controlled re-evaluation, as Bowles, Ahmed, and Schuld and others have
documented. This manuscript confronts that problem head-on for the
fast-growing subfield of quantum physics-informed neural networks
(QPINNs). I **pre-registered** a falsifiable hypothesis — that QPINNs
enjoy a regime-dependent advantage favoring smooth/periodic dynamics
over broadband/multiscale dynamics — and then executed a head-to-head
benchmark of four QPINN families against capacity-matched classical
PINN and Neural-ODE baselines across a hardness ladder of four ODEs
and four PDEs, at three seeds each, under matched training budgets,
paired-bootstrap confidence intervals, an underfit guard, and a
known-structure skyline.

**The headline result.** The pre-registered hypothesis is
**FALSIFIED** across two PRIMARY verdicts (the full-ladder solver task
at n = 24, with the point-estimate sign inverted relative to the
original hypothesis, and the forecaster task at n = 9 with the
confidence interval excluding zero negatively), corroborated by three
additional layered sensitivity points. A rigorously executed
falsification of a widely assumed advantage is, in my view, exactly
the kind of result the field most needs and most lacks — and PRX
Quantum's mandate for significant, reproducible quantum-science results
is the natural home for it.

**Beyond the negative result, three contributions of independent
interest:**

1. **A substrate-dependent mechanism finding.** A τ-isolation
   cross-check disagrees in sign between the two paths of a complete
   2×2 mechanism decomposition: the liquid-time-constant machinery is
   beneficial on a classical MLP hidden state but detrimental on a
   quantum-cell hidden state. This is, to my knowledge, a new
   observation about where quantum substrates change the behavior of
   liquid neural-ODE dynamics, and it seeds a concrete follow-up.

2. **A positive sub-finding the larger sample size surfaced.** The
   trainable-embedding QPINN with a QNN encoder shows a 24× tighter
   seed-to-seed standard deviation and 2.5× lower mean error than the
   classical PINN on FitzHugh–Nagumo — a regime-dependent *structural*
   advantage on stiff fast–slow dynamics that was invisible at smaller
   n.

3. **Reproducibility infrastructure.** The complete benchmark
   framework is released open-source, including a
   `verify_paper_integrity.py` script that gates every cited number in
   the manuscript against a committed JSON results file. Every numeric
   claim in the paper is machine-verifiable against its source data.
   The full pre-registration and 19 documented amendments accompany
   the release, so every methodological choice and deviation is on the
   record.

**Fit for PRX Quantum.** The manuscript advances the methodological
foundations of quantum machine learning by demonstrating, end to end,
what a pre-registered, falsifiable, reproducibility-gated QML benchmark
looks like — and by reporting honestly when the hypothesis does not
hold. Both the negative primary result and the positive structural
sub-finding are significant for researchers building and evaluating
quantum models for scientific computing.

The manuscript is a 28-page main text with an 8-page supplement. It is
a single-author submission. I confirm there are no competing financial
interests.

Thank you for considering this work. I would be glad to suggest
qualified referees on request, and I welcome the editors' and
reviewers' scrutiny — the entire benchmark is built to be checked.

Sincerely,

Shawn Gibford
Independent
gibfords@gmail.com

---

## Pre-submission checklist (mirrors `ARXIV_SUBMISSION.md`)

- [ ] Confirm author affiliation line matches `paper/main.tex`
      (currently "Independent"; update both if the advisor relationship
      changes it before submission).
- [ ] Decide whether to name suggested referees (PRX Quantum allows
      it; optional). Candidates from the cited literature on QML
      benchmarking methodology.
- [ ] Confirm the falsification framing is acceptable to the advisor
      before submitting (this is the paper's central claim).
- [ ] `bash paper/build.sh && bash paper/build_supplement.sh` exit 0.
- [ ] `PYTHONPATH=src python scripts/verify_paper_integrity.py` exits 0.
- [ ] Export this letter to PDF if the form requires an upload rather
      than pasted text.
