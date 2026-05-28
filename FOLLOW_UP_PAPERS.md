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

### Candidate system ladder — 12 systems, 4 per domain (web-verified 2026-05-28)

After a 2026-05-28 co-author conversation, the original 4-system
Option A ladder expanded to a balanced **12-system ladder**: 4 per
domain (ChemE / BioE / PSE), tagged 6 SMOOTH + 6 BROADBAND to mirror
the current paper's H1 regime split.

#### ChemE — 4 systems

**3a. Van de Vusse CSTR** *(ODE, 2-state, SMOOTH)*

Van de Vusse 1964 reaction scheme: `A → B → C` and `2A → D`
(parallel + series).

```
dC_A / dt = (q/V)(C_A,in − C_A) − k₁·C_A − k₃·C_A²
dC_B / dt = −(q/V)·C_B + k₁·C_A − k₂·C_B
```

Non-monotonic steady-state map; classic nonlinear control benchmark.
Sources: Van de Vusse 1964; Pannocchia 2003; APMonitor wiki.

**3b. Saponification CSTR** *(ODE, 2-state, SMOOTH)*

NaOH + ethyl acetate → sodium acetate + ethanol, exothermic.

```
dC_NaOH / dt = (q/V)(C_NaOH,in − C_NaOH) − k(T)·C_NaOH·C_EtAc
dT / dt = (q/V)(T_in − T) + (−ΔH/ρc_p)·k(T)·C_NaOH·C_EtAc
         − UA(T − T_c)/(V·ρ·c_p)
```

with Arrhenius `k(T) = k₀·exp(−E_a/RT)`. Classic ChemE
heat-coupled-reactor benchmark used in undergraduate kinetics labs.
Sources: Engineering LibreTexts 6.4; *Optimizing the Sodium Hydroxide
Conversion in CSTR* (MDPI 2021); MDPI *Processes* 9/873.

**3c. Heat exchanger hot-side** *(1D PDE, BROADBAND)*

Counter-current heat exchanger, hot-stream transport equation
(treated as a stand-alone scalar PDE):

```
∂T_h / ∂t + v_h · ∂T_h / ∂x = − k · (T_h − T_c) / (ρ_h · c_p,h)
```

where `T_c` is the cold-side temperature profile (held fixed at its
steady-state value for the standalone test; coupled-PDE variant is
the heat-exchanger cold-side below).

**3d. Heat exchanger cold-side** *(1D PDE, BROADBAND)*

Same setup, cold-stream side (note the *opposite* sign on `v_c` for
counter-current flow):

```
∂T_c / ∂t − v_c · ∂T_c / ∂x = + k · (T_h − T_c) / (ρ_c · c_p,c)
```

Sources: *PDE Observer Design for Counter-Current Heat Flows in a
Heat-Exchanger* (Hasan et al., IFAC 2017); Engineering LibreTexts
6.6; *PDE Control of Heat Exchangers by Input-Output Linearization*
(ScienceDirect 2024). The per-co-author preference: keep these as two
separate scalar PDE benchmarks rather than a single coupled-PDE
system.

#### BioE — 4 systems

**3e. Monod chemostat** *(ODE, 2-state, SMOOTH)*

Monod 1950 / Novick-Szilard 1950 — the foundational microbial growth
law:

```
dS / dt = D·(S_in − S) − (1/Y)·μ(S)·X
dX / dt = (μ(S) − D)·X      with μ(S) = μ_max·S / (K_s + S)
```

Sources: Monod, *Annual Review of Microbiology* 1950; Novick & Szilard,
*PNAS* 1950; Engineering LibreTexts 6.3.

**3f. Yeast fermentation with Andrews / Aiba kinetics** *(ODE, 3-state, SMOOTH)*

Saccharomyces cerevisiae ethanol fermentation with substrate inhibition
*and* product inhibition (Andrews 1968 for `μ(S)`; Aiba 1968 for
the product-inhibition term):

```
dX / dt = μ(S, P) · X − D · X
dS / dt = D · (S_in − S) − (1/Y_X/S) · μ(S, P) · X
dP / dt = (Y_P/X / Y_X/S) · μ(S, P) · X − D · P

μ(S, P) = μ_max · [S / (K_s + S + S²/K_i)] · (1 − P / P_max)
```

Three states (biomass X, substrate S, product P ≡ ethanol) with both
substrate and product inhibition. Industrially standard for ethanol
fermentation. Sources: Andrews, *Biotechnology and Bioengineering*
1968; Aiba et al., *Biotechnology and Bioengineering* 1968;
Thatipamala 1992 (Wiley).

**3g. Contois kinetics chemostat** *(ODE, 2-state, SMOOTH)*

Alternative to Monod where biomass concentration limits substrate
uptake (e.g., insoluble substrate degradation):

```
dS / dt = D·(S_in − S) − (1/Y)·μ(S, X)·X
dX / dt = (μ(S, X) − D)·X      with μ(S, X) = μ_max·S / (k_C·X + S)
```

Reduces to Monod in the low-biomass limit. Sources: Contois,
*Journal of General Microbiology* 1959; *Theoretical derivation of
the Contois equation* (ScienceDirect 2013); Nelson, *UoW chemical-
reactor-engineering notes*.

**3h. Fisher-KPP tumor growth** *(1D PDE, BROADBAND)*

Reaction-diffusion for tumor density with logistic growth:

```
∂u / ∂t = D · ∂²u / ∂x² + r · u · (1 − u/K)
```

Localized initial conditions evolve to *traveling-wave* solutions
with sharp fronts → broadband regime. Sources: Fisher 1937; KPP
1937; *A time-fractional Fisher-KPP equation for tumor growth*
(arXiv:2511.05312, 2025); Murray, *Mathematical Biology* (Springer).

#### PSE — 4 systems

**3i. Convection–Diffusion–Reaction, low-Péclet** *(1D PDE, SMOOTH)*

```
∂u/∂t + v·∂u/∂x = D·∂²u/∂x² − k·u      Pe = vL/D = 1
```

Diffusion-dominated regime, smooth concentration profile, exponential
decay. Sources: Agud Albesa et al., *Math. Methods Appl. Sci.* 2023;
hplgit scaling-book.

**3j. Convection–Diffusion–Reaction, high-Péclet** *(1D PDE, BROADBAND)*

Same equation, Pe = 100. Advection-dominated regime with sharp
reaction front. Two systems from one equation by Péclet-number
variation.

**3k. Fixed-bed reactor with axial dispersion** *(1D PDE, BROADBAND)*

Packed catalytic reactor with first-order reaction + axial dispersion:

```
∂C/∂t + v·∂C/∂x = D_ax·∂²C/∂x² − k·C(1 − ε)/ε
```

Closed-form analytic solution exists at low Damköhler number;
broadband regime emerges at high Damköhler. Distinct from the CDR
above in the reaction kinetics + the ε (void-fraction) coupling.
Sources: Levenspiel, *Chemical Reaction Engineering* (Wiley);
*Fogler, Elements of Chemical Reaction Engineering* (Prentice Hall).

**3l. Anaerobic digestion (simplified ADM1)** *(ODE, 4-state, BROADBAND)*

Reduced-order ADM1 (Batstone et al. 2002) tracking 4 key state
variables (substrate, acetogen biomass, methanogen biomass, methane
gas-phase concentration) with stiff fast-slow dynamics across acetogenesis
and methanogenesis time-scales. The full ADM1 is 24-state; we use a
literature-standard 4-state reduction.

```
dS / dt = D·(S_in − S) − (1/Y_a)·μ_a(S)·X_a
dX_a / dt = (Y_X_a · μ_a(S) − k_da − D) · X_a
dX_m / dt = (Y_X_m · μ_m(P_a, X_a) − k_dm − D) · X_m
dP_m / dt = (Y_P/X_m · μ_m(P_a, X_a) · X_m) − k_L·a·(P_m − P_m_eq)
```

Stiff fast-slow → BROADBAND. Sources: Batstone et al., *Water Science
and Technology* 2002 (ADM1); Bernard et al., *Biotechnology and
Bioengineering* 2001 (4-state reduction).

### Regime balance — 6 SMOOTH + 6 BROADBAND

| Bin | Systems |
|---|---|
| **SMOOTH** (6) | Van de Vusse CSTR, Saponification CSTR, Monod chemostat, Yeast fermentation (Andrews/Aiba), Contois chemostat, CDR low-Pe |
| **BROADBAND** (6) | Heat-exchanger hot-side, Heat-exchanger cold-side, Fisher-KPP tumor growth, CDR high-Pe, Fixed-bed reactor, Anaerobic digestion (ADM1) |

Matches the current paper's 4 SMOOTH + 4 BROADBAND split structurally,
so the H1 verdict aggregator can run on the new ladder without
methodological adjustment.

### Implementation effort

| Task | Effort |
|---|---|
| Add 6 new ODE systems to `multi_state_solver.VECTOR_ODES` | ~30 min × 6 = 3 hr |
| Add 6 new PDE systems to `pde_demo.PDE_BENCH` (6 residual fns + ICs + ref fields) | ~1 hr × 6 = 6 hr |
| Reference fields: ODE side uses existing RK4; PDE side needs Cox-Matthews integrating-factor RK4 for nonlinear (Fisher-KPP, ADM1) and Fourier-spectral for linear (CDR, heat-exchanger, fixed-bed) | ~3 hr wiring |
| Smoke-test each new cell (1 cell per system × 1 seed at low steps) | ~2 hr |
| Pre-register the new ladder + decision rule (`FOLLOWUP_PRE_REG.md`) | ~3 hr |
| **Total scaffold effort** | **~17 hr** |

### Compute estimate

Per-cell wall-time scales with state dim (ODE) or grid size (PDE).
Using the post-audit smoke numbers as the basis, and ×7 QLNN families
× 3 seeds + classical PINN per system:

| Domain | System | State / grid | Per-cell est. | × 7 QLNN + cPINN × 3 seeds |
|---|---|---|---:|---:|
| ChemE | Van de Vusse | 2-state | ~0.17 hr | ~3.6 hr |
| ChemE | Saponification | 2-state | ~0.17 hr | ~3.6 hr |
| ChemE | HX hot-side | 1D PDE | ~0.8 hr | ~17 hr |
| ChemE | HX cold-side | 1D PDE | ~0.8 hr | ~17 hr |
| BioE | Monod | 2-state | ~0.17 hr | ~3.6 hr |
| BioE | Yeast (Andrews) | 3-state | ~0.25 hr | ~5.3 hr |
| BioE | Contois | 2-state | ~0.17 hr | ~3.6 hr |
| BioE | Fisher-KPP | 1D PDE | ~1.2 hr (broadband fronts) | ~25 hr |
| PSE | CDR low-Pe | 1D PDE | ~0.8 hr | ~17 hr |
| PSE | CDR high-Pe | 1D PDE | ~1.5 hr (sharp fronts) | ~32 hr |
| PSE | Fixed-bed | 1D PDE | ~1.2 hr | ~25 hr |
| PSE | ADM1 (4-state) | 4-state ODE | ~0.34 hr | ~7.2 hr |
| **Total addition** | | | | **~160 hr** |

About **6.5 days of Anvil GPU wall-clock if run fully serial**;
embarrassingly parallel by cell, so on a 16-GPU partition it
collapses to ~10 hr wall-clock. Fits comfortably in an ACCESS
Explore allocation (~6,000 GPU-hours available).

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
