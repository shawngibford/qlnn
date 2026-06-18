# Follow-up papers

Current manuscript ships first. These are separate projects with
separate pre-registrations where needed.

## 1. Training-hardening paper

**Working title.** *Hardening quantum PINN training on a
pre-registered QLNN benchmark.*

**Question.** Does the falsified QLNN advantage remain falsified after
literature-best training interventions?

**Interventions.**

- Quantum Natural Gradient.
- Causal training / time-ordered residual weighting.
- Deeper data-reuploading (`L=5`) to increase Fourier capacity.

**Publishable either way.**

- If the verdict flips: default Adam underestimated quantum potential.
- If it stays falsified: stronger null under best-known training.

**Scale.** ~3-4 days development, ~2 hr Anvil GPU.

## 2. τ mechanism paper

**Working title.** *Substrate-dependent learned time constants in
quantum and classical liquid neural dynamics.*

**Question.** Why does the liquid time-constant mechanism help on a
classical hidden state but hurt on a quantum-cell hidden state?

**Core work.**

- τ-distribution analysis.
- Frequency-domain decomposition.
- Third-substrate control, e.g. random-feature reservoir.
- Linearized dynamics derivation for the τ-leak term on classical vs
  quantum substrates.

**Why this is strongest.** The current paper already proves the sign
disagreement in a 2x2 decomposition. The missing piece is mechanism,
not evidence that the anomaly exists.

**Scale.** ~1 day development plus focused experiments.

## 3. Domain-breadth paper

**Working title.** *Quantum-enhanced solvers and forecasters for
chemical, biochemical, and process-systems engineering.*

**Question.** Does the current falsification/regime map hold on systems
that ChemE, BioE, and PSE researchers actually use?

**Scope.** New pre-registered benchmark, not an extension of the current
paper. Candidate ladder remains balanced across smooth and broadband
dynamics.

| Domain | Example systems |
|---|---|
| ChemE | Van de Vusse CSTR, saponification CSTR, heat-exchanger PDEs |
| BioE | Monod chemostat, yeast fermentation, Contois kinetics, Fisher-KPP |
| PSE | convection-diffusion-reaction, fixed-bed reactor, reduced ADM1 |

**Scale.** Larger than the other two: roughly 1-2 days scaffolding plus
Anvil compute. Needs its own `FOLLOWUP_PRE_REG.md`.

## Boundary

None of these three projects should be folded into the current PRX
Quantum submission. The current project stops at the strengthened
submission package.
