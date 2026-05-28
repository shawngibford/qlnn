# Follow-up papers — pipeline after the current submission

The current paper is the **pre-registered benchmark** of QLNN solver +
forecaster against matched classical baselines on a 9-system
mathematical-physics ladder. Three follow-up papers have been
identified and lightly scoped during the audit work of 2026-05-28.
This document is the seed for each — kept here so the next contributor
(or my future self) does not have to re-derive the design.

None of the content below is in the current paper's scope or its
pre-registration. The current paper ships first.

---

## Follow-up #1 — Training-hardening paper

**Working title.** *Hardening quantum PINN training: QNG + causal
weighting + deeper data-reuploading on the QLNN benchmark.*

**Premise.** The current paper's FALSIFIED verdict uses default Adam
on every cell. Three interventions identified by the H3 mechanism
analysis as plausibly relevant:

1. **Quantum Natural Gradient** (Stokes 2020 + follow-ups) — replace
   optax Adam with PennyLane's QNG. Drop-in.
2. **Causal training** (Wang et al. arXiv:2203.07404) — weight
   collocation points by inverse time-order. Directly targets the
   van der Pol stiff failure observed in P3.6.
3. **Deeper data-reuploading depth L=5** (Schuld 2021) — widens the
   Fourier K_max ceiling. Targets the H3 KL-to-Haar trend mechanism.

**Story.** Re-run a subset of the current benchmark at literature-best
training. Either the verdict flips (CONFIRMED at HPO-best) — "default
optimization underestimated quantum potential" — or it stays FALSIFIED
— "even with literature-best techniques, no advantage." Either outcome
is publishable.

**Compute.** ~3-4 days dev + ~2 hr Anvil GPU compute.

---

## Follow-up #2 — Substrate-dependent τ mechanism paper

**Working title.** *Substrate-dependent behavior of learned time
constants: a mechanism investigation.*

**Premise.** The current paper's headline mechanism finding — the
τ-cross-check disagreement (Δ_τ on classical MLP = +0.115; Δ_τ on
quantum cell = −0.334; signs disagree, both algebraic identities
hold per-cell exactly) — has **no explanation in the literature**.
Why does a learned per-neuron time constant help on a classical
hidden state but hurt on a quantum hidden state?

**Approach.** A full mechanism investigation:
- τ-distribution analysis (do quantum cells push τ toward `tau_min`?
  If so, the cell decouples from history and becomes near-memoryless.)
- Frequency-domain analysis (do the τ-modulated quantum cells
  preferentially suppress low-frequency components?)
- Substrate-isolation experiments (run liquid τ on a third substrate:
  e.g., a random-feature reservoir, to see if the sign depends on
  trainability or on representation).
- Theory: derive the τ-leak contribution to the linearized cell
  dynamics for both substrates; show the sign asymmetry analytically.

**Story.** A first-principles explanation of a benchmark anomaly. More
fundamental than the original H1 hypothesis.

**Compute.** Smaller — analysis sits on existing cells. ~1 day dev +
~hours of focused experiments.

---

## Follow-up #3 — Domain-breadth paper (ChemE/BioE/PSE) — **NEW, scoped 2026-05-28**

**Working title.** *Quantum-enhanced solvers and forecasters for
chemical, biochemical, and process-systems engineering: a
pre-registered benchmark.*

**Premise.** The current paper's benchmark systems are mathematical-
physics (Lorenz, Burgers, KdV) and generic dynamical (Lotka-Volterra,
van der Pol, kuramoto, FHN). These are the right systems for testing
the H1 hypothesis but they leave a gap: the ChemE / BioE / PSE
communities that would actually deploy a quantum-enhanced
differential-equation solver have not seen a head-to-head benchmark
on systems *they* care about.

The current paper does not claim to address those audiences. This
follow-up paper does, with a new pre-registration on a domain-specific
ladder.

### Candidate system ladder (web-verified 2026-05-28)

#### 3a. Van de Vusse CSTR — Chemical Engineering ODE benchmark

Continuous Stirred Tank Reactor with the Van de Vusse reaction scheme
(Van de Vusse, *Chem. Eng. Sci.*, 1964): parallel + series

```
A → B → C    and    2A → D
```

Isothermal 2-state mole-balance ODE:

```
dC_A / dt = (q/V)(C_A,in − C_A) − k₁·C_A − k₃·C_A²
dC_B / dt = −(q/V)·C_B + k₁·C_A − k₂·C_B
```

Non-monotonic steady-state map; multiple equilibria for certain
residence-time windows. Classic nonlinear control benchmark
(Pannocchia 2003; APMonitor).

- Regime tag: **SMOOTH** with stiff transitions near a fold.
- State dim: 2 (4 for full thermal form; we use isothermal).
- Reference: Cox-Matthews integrating-factor RK4 (already in the
  codebase for PDE generators).

#### 3b. Monod chemostat — Biochemical Engineering ODE benchmark

Continuous bioreactor: biomass X consumes growth-limiting substrate S
with Monod kinetics (Monod 1950; Novick & Szilard 1950 — the
foundational microbial growth law):

```
dS / dt = D·(S_in − S) − (1/Y)·μ(S)·X
dX / dt = (μ(S) − D)·X

with μ(S) = μ_max·S / (K_s + S)
```

- Regime tag: **SMOOTH** (stable steady state for D < μ_max; washout
  boundary at D → μ_max is interesting but borderline).
- State dim: 2.
- Reference: numerical RK4.
- Bonus thematic continuity: the project's earlier OD bioreactor work
  used this same Monod growth law implicitly.

#### 3c, 3d. Convection–Diffusion–Reaction PDE — Process Systems Engineering benchmark (two regime variants from one equation)

1D linear CDR equation on a finite domain:

```
∂u/∂t + v·∂u/∂x = D·∂²u/∂x² − k·u
```

Two dimensionless parameters:

- Péclet number **Pe = v·L / D** — advection vs diffusion balance.
- Damköhler number **Da = k·L / v** — reaction vs advection balance.

Two systems for the price of one equation:

- **CDR-low-Pe** (Pe = 1, diffusion-dominated): smooth concentration
  profile, exponential decay → **SMOOTH** bin.
- **CDR-high-Pe** (Pe = 100, advection-dominated): sharp reaction
  front, large gradients → **BROADBAND** bin.

Sources: Agud Albesa et al., *Math. Methods Appl. Sci.* 2023;
*ScienceDirect* "Effective models for reactive flow"; hplgit
scaling-book chapter on convection-diffusion. Standard analytical
solutions exist at the Pe / Da extremes; method-of-lines /
Crank-Nicolson for the reference field.

### Regime balance

Adding all four systems keeps the H1 bins roughly balanced:

| Bin | Current systems | + Follow-up systems |
|---|---|---|
| SMOOTH | LV, VdP, kuramoto, heat, burgers-smooth | + CSTR, Monod, CDR-low |
| BROADBAND | Lorenz, FHN, AC, burgers-shock, KdV | + CDR-high |

### Implementation effort

| Task | Effort |
|---|---|
| Add CSTR + Monod to `multi_state_solver.VECTOR_ODES` | ~30 min each |
| Add CDR-low / CDR-high to `pde_demo.PDE_BENCH` (one residual fn, two configs) | ~1 hr |
| Reference fields (RK4 for ODEs already exists; Fourier-spectral / Cox-Matthews for CDR already exists) | ~30 min wiring |
| Pre-register the new ladder + decision rule (separate `FOLLOWUP_PRE_REG.md`) | ~2 hr |
| Smoke-test each new cell | ~30 min |
| **Total scaffold effort** | **~5 hr** |

### Compute estimate

Per-cell wall-time scales with state dim. Using the post-audit smoke
numbers as the basis:

| System | State dim | Per-cell est. | ×7 QLNN families × 3 seeds + cPINN |
|---|---:|---:|---:|
| CSTR | 2 | ~0.17 hr | ~3.6 hr |
| Monod | 2 | ~0.17 hr | ~3.6 hr |
| CDR-low | 1D PDE | ~0.8 hr | ~10 hr |
| CDR-high | 1D PDE | ~1.5 hr (sharp fronts → more steps) | ~18 hr |
| **Total addition** | | | **~35 hr** |

A modest add to a future Anvil sweep. Fits in the same ACCESS Explore
allocation as the current paper's compute.

### Status

- Equations verified (2026-05-28 web search; canonical sources cited
  above).
- Code scaffolding not yet written.
- Pre-registration not yet drafted.
- Compute not yet scheduled.

**Next step for this follow-up:** after the current paper is submitted,
draft `FOLLOWUP_PRE_REG.md` and add the four systems to the codebase.
The work is shovel-ready.

---

## Cross-cutting future direction

- **Hardware execution.** Pre-reg disclaimed simulator-only. PRX
  Quantum routinely publishes simulator-only QML benchmarks. A
  small-scale hardware run (e.g., 4-qubit ionQ Aria, IBM Eagle 127q
  with circuit transpilation) is a natural revision-stage extension
  if reviewers request it. Out of scope for the current submission;
  out of scope for any of the three follow-up papers above unless
  budget + access materializes.
