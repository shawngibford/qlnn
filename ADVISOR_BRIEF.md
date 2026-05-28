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

---

## Why we should move the remaining compute to Purdue's Anvil

The story above is honest about the paper's current state. What it
does not yet show is that **a deeper audit of our own work, completed
this week, surfaced four issues that materially expand the compute
budget needed to ship the paper at the rigor we have been holding it
to.** I want to make the case for requesting an ACCESS allocation on
Purdue's Anvil supercomputer to run the remainder on GPU rather than
on my laptop CPU.

### What the audit found

A read-only review pass by four specialist agents over the codebase
this week identified:

1. **Training-budget asymmetry on the solver task.** Four quantum
   families had been given different numbers of training iterations
   (1200, 1500, 2000, 1500) justified informally as "equal compute
   per step". This is not equal-iterations fairness and was never
   pre-registered. The family with the largest budget (te_qpinn_qnn)
   is also the family that wins three of four solver systems. The
   fix is to equalize at 2000 steps and re-run the solver cells.
2. **A circuit-aliasing bug.** Our "strongly entangling" circuit and
   our "data reuploading" circuit were producing bit-identical
   numerical outputs on every forecasting cell — at 16-digit
   precision. Root-caused to a PennyLane template fallback at our
   small qubit count. The fix is one line of code but it requires
   re-running every cell that involves either circuit.
3. **A classical-parameter confound on a key quantum family.** The
   qcpinn family wins on one system at relL² 0.006, but does it with
   706 classical neural-network parameters alongside 15 quantum ones.
   The "win" is mostly the MLP, not the quantum circuit. Fix: add
   three step-wise variants that progressively shift mass from
   classical to quantum (24% / 45% / 87% quantum), so the attribution
   becomes testable.
4. **One ansatz is structurally broken at our config.** The
   "brickwall" forecaster ansatz leaves one of three qubits entirely
   disconnected at our depth budget. Loss decreases during training
   but the circuit cannot represent three-dimensional dynamics.
   Removing it from the empirical sweep is the only honest call.

All four findings are documented in formal pre-registration amendments
(A15–A18) with full code changes committed and pushed to GitHub. The
paper's headline verdict will not change qualitatively, but the
numbers backing it will be refreshed with a fair comparison.

### What this implies for compute

Combining the re-runs the audit forces, plus the originally-planned
M3 sweep to bring our pre-registration coverage from 8 of 9 systems
to all 9 of 9:

| Workload | Cells | Est. CPU-hours |
|---|---:|---:|
| Original M3: kuramoto + KdV solver sweep | 30 | ~23 |
| A15 re-run: solver cells at equalized training budget | ~80 | ~50 |
| A16 re-run: strongly_entangling forecaster cells | ~18 | ~3 |
| A17 new: qcpinn quantum-parameter sweep (3 variants × 8 systems × 3 seeds) | 72 | ~70 |
| **Total** | **~200** | **~145** |

That is roughly **six straight days of CPU compute on my laptop**
(Apple Silicon, JAX `default.qubit`). The actual wall-clock would be
longer because the laptop needs to do everything else I do on it.

### Why Anvil specifically

Purdue's Anvil supercomputer (NSF-funded ACCESS resource at the
Rosen Center for Advanced Computing) is sized exactly for this kind
of workload:

- **GPU partition**: 16 nodes, each with four NVIDIA A100 Tensor
  Core GPUs (40 GB each), delivering 1.5 PF of single-precision
  performance. A 2025 NSF NAIRR upgrade added 21 more nodes with
  four NVIDIA H100 SXM GPUs each (80 GB), bringing the total to 84
  H100s.
- **CPU partition** ("Sub-cluster A"): 1,000 nodes, each with 128
  cores (two AMD Milan CPUs at 2.45 GHz) and 256 GB of memory.
- **Embarrassingly parallel fit**: our experiment is independent per
  (system, family, seed) cell, so we can launch every cell as a
  separate SLURM job and finish the entire 200-cell matrix in roughly
  the time of one cell — call it overnight rather than a week.
- **GPU speedup expected**: PennyLane ships a `lightning.gpu` device
  backed by NVIDIA's cuQuantum SDK, with proven JAX interoperability.
  Independent benchmarks have demonstrated PennyLane at scale on
  Frontier (the world's largest supercomputer until 2024). The same
  software path that runs on my laptop runs on Anvil GPUs with a
  module load and a device-string change.

### How we apply

Anvil is allocated through ACCESS, a National Science Foundation
program. Graduate students are explicitly eligible for the **Explore
tier** with an advisor letter on institutional letterhead. The
process:

1. Create an ACCESS account at the allocations portal using my
   Purdue / institutional email.
2. Submit an Explore project: title, ≤3-page CV, one-page abstract
   describing the workload, and your letter of support.
3. Outcome typically within **one business day**. Credit-to-resource
   exchange takes up to one week.
4. The Explore tier ceiling is 400,000 ACCESS credits, which
   translates to ~6,000 GPU-hours or ~334,000 CPU-core-hours on
   Anvil. Our entire 200-cell matrix at the worst-case CPU estimate
   uses ~145 × 128 ≈ 18,500 core-hours — a fraction of one Explore
   allocation. With GPU acceleration it is a sliver.

If we get a no on the first pass we can re-request with a tighter
abstract. If we get a yes we can be running on Anvil this week.

### What I would need from you

A short letter on institutional letterhead, ~one paragraph, stating
that the proposed Anvil workload is being conducted by me as part of
my dissertation, that you support the request, and that it is
separate from your other funded grants. ACCESS publishes a template;
I can send you a draft to sign.

### The ask in one sentence

**Authorize me to request an ACCESS Explore allocation on Anvil so
the audit re-runs and the kuramoto + KdV sweep can finish on GPU
this week rather than holding the paper open for another six days of
my laptop CPU.**

---

## Sources

- [Anvil overview — Purdue RCAC](https://www.rcac.purdue.edu/compute/anvil)
- [Anvil specs — RCAC Knowledge Base](https://www.rcac.purdue.edu/knowledge/anvil/overview)
- [Anvil GPU resource — ACCESS](https://allocations.access-ci.org/resources/anvil.purdue.access-ci.org)
- [Anvil AI now available to ACCESS researchers (H100 upgrade)](https://www.rcac.purdue.edu/news/7268)
- [ACCESS allocations: getting your first project](https://allocations.access-ci.org/get-your-first-project)
- [How to get onto Anvil through ACCESS (graduate-student path)](https://www.rcac.purdue.edu/knowledge/anvil/access/anvil_through_access?all=true)
- [PennyLane Lightning + NVIDIA cuQuantum on GPU](https://pennylane.ai/blog/2022/07/lightning-fast-simulations-with-pennylane-and-the-nvidia-cuquantum-sdk/)
- [Hybrid quantum programming with PennyLane Lightning on HPC platforms (arXiv:2403.02512)](https://arxiv.org/html/2403.02512v1)
