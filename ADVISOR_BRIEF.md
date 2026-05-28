# Advisor brief — quantum-vs-classical benchmark on differential equations

*Single-page summary. The full draft is at `paper/main.pdf` (17 pp) +
`paper/supplement.pdf` (7 pp).*

## What we set out to test

A specific, pre-registered claim from the quantum-machine-learning
literature: that a quantum-enhanced neural network should beat its
classical counterpart on smooth, well-behaved differential equations
(and tie on chaotic ones). We built a head-to-head benchmark on two
tasks — *solving* differential equations and *forecasting* their
trajectories — across a structured ladder of problems running from
smooth to chaotic.

## Headline finding

**The predicted quantum advantage did not appear.** Under matched
parameter counts, multiple random seeds, and proper statistical
comparison, the quantum model is either *worse* than its classical
counterpart (on forecasting) or *statistically indistinguishable* from
it (on solving). This is exactly the kind of rigorous null the field
has been asking for — most prior "quantum advantage in ML" claims have
not held up under matched conditions. Ours is the first systematic
test on differential-equation tasks.

## Three sub-findings

1. **Forecasting task: the classical model wins clearly.** Across the
   systems and seeds we tested, the classical baseline reliably beats
   the quantum one. The statistical confidence interval excludes "tie".

2. **Solving task: no clear winner.** Across roughly two dozen
   system/seed combinations, we cannot distinguish the two models. The
   direction of the point estimate has even flipped as we added more
   problems to the ladder — a clear sign there is no robust effect.

3. **An unexpected mechanism finding.** A specific architectural piece
   borrowed from *liquid neural networks* — a learned per-neuron time
   constant — behaves in opposite directions depending on what kind of
   network it is attached to. On the classical hidden state it helps;
   on the quantum hidden state it actively hurts. We cannot yet
   explain this. It is genuinely new and probably deserves its own
   paper.

## What we have done

- Pre-registered the hypothesis, success/failure criteria, and
  baselines in writing *before* running the comparison (14 amendments
  documented openly as the work evolved).
- Built and ran the full benchmark: four quantum model families
  against four classical baselines on a structured problem ladder.
- Wired the entire paper to a mechanical integrity gate — every
  number in the draft is checked against a JSON record on disk on
  every build.
- Drafted the full paper (17 pages main + 7 pages supplement).

## What we want to finish before submitting

- One overnight + morning of remaining compute (about 23 hours on a
  laptop CPU) to complete the last two systems on the problem ladder,
  bringing coverage from 8 of 9 systems to all 9.
- A one-hour paper-update pass: refresh the headline numbers, the
  master verdict table, and the integrity gate.
- Submit to *PRX Quantum*.

## Where we could go from here

- **A training-hardening follow-up paper.** Three specific
  interventions — a quantum-aware optimizer, a "causal" training
  schedule, and deeper data-encoding — have been identified as
  candidates that might change the verdict. If they do, the story is
  "default optimization underestimated quantum potential"; if they do
  not, the null becomes much stronger. Either outcome is publishable.
- **A mechanism-investigation paper on the time-constant finding.**
  The substrate-dependent behavior above is a more fundamental
  question than the original benchmark and currently has no
  explanation in the literature.
- **(Longer-term) hardware execution** on a real quantum device.
  Out of scope for the current submission but a natural revision
  ask.
