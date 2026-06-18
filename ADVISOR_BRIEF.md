# Advisor brief — QLNN ODE/PDE benchmark

*Current draft: `paper/main.pdf` (25 pp) + `paper/supplement.pdf`
(8 pp). Integrity gate: `scripts/verify_paper_integrity.py` passes.*

## Status

This is a pre-registered benchmark of Quantum Liquid Neural Networks
(QLNNs) against matched classical baselines on ODE/PDE solver and
forecaster tasks. It is no longer the earlier bioreactor-OD project.

Current state:

- Results on disk: **405 completed cells, 0 error cells**.
- Pre-registration amendments: **22** documented amendments.
- Every paper number is mechanically checked against committed JSON
  results.
- The minimum paper is close to advisor handoff now.
- The strengthened paper needs Anvil/ACCESS re-runs first.

## Result

**The pre-registered quantum-advantage hypothesis is falsified under
matched controls.** The QLNN does not show a robust regime-dependent
advantage over classical baselines on the current ODE/PDE benchmark.

This is still publishable: it is a controlled, pre-registered,
reproducibility-gated null result in a field where many advantage claims
do not survive matched-budget comparison.

## Strongest positive finding

The mechanism result is better than the original headline:

**Learned liquid time constants are substrate-dependent.** They help on
a classical hidden state but hurt on a quantum-cell hidden state. The
2x2 decomposition identities hold, so this is not a bookkeeping error.
It is the cleanest follow-up-paper seed.

## Recommendation

Default path: **strengthened submission**.

1. Get advisor support for an ACCESS Explore request.
2. Run the audit-driven re-run matrix on Purdue Anvil.
3. Refresh the verdict numbers and paper text.
4. Submit to *PRX Quantum*.

Minimum fallback: submit the current integrity-gated paper after advisor
and coauthor sign-off, with audit re-runs documented as deferred.

## Stop condition for this project

Stop adding scope when:

- Anvil re-runs finish with **0 errors**.
- Integrity gate is refreshed and green.
- Main paper and supplement build clean.
- Advisor signs off on the falsification framing.
- PRX Quantum/arXiv package is staged.

New ideas after that go to follow-up papers, not this manuscript.

## Follow-up papers

1. **Training hardening.** QNG + causal training + deeper
   data-reuploading. Tests whether the null survives literature-best
   optimization.
2. **τ mechanism.** Explain why liquid time constants help classical
   substrates and hurt quantum substrates.
3. **Domain breadth.** New ChemE/BioE/PSE benchmark on process-relevant
   ODE/PDE systems, with a separate pre-registration.

## Advisor ask

Approve an ACCESS Explore request for Anvil so the remaining audit
re-runs happen on HPC rather than on a laptop. The needed letter is one
short institutional-letterhead statement that the work is part of the
student's dissertation and separate from the advisor's other funded
grants.
