# Pre-registration — QLNN ODE/PDE solver & forecaster (v1)

**Status:** PRE-REGISTRATION v1 — committed 2026-05-19. This document is
the scientific keystone of the **post-pivot** program. It is locked
*before* any P6 training run. Any deviation found at analysis time is
disclosed in a "Deviations from pre-registration" section, never
silently absorbed.

**Authoritative spec:** the approved plan
`~/.claude/plans/i-want-to-automate-quiet-cerf.md`
("# PIVOT — QLNN ODE/PDE solver/forecaster"). This pre-registration
operationalizes that plan into falsifiable, gate-able science. Template
and pre-registration culture inherited from `hypothesis.md` (the
superseded bioreactor-OD pre-reg).

**Relationship to prior work:** the bioreactor-OD program
(`PAPER_SUMMARY.md`, `hypothesis.md` v2, three OD claims) is a rigorous
**null on an n=1 dataset** and is **archived, not deleted**.
`scripts/verify_paper_integrity.py` stays green; the OD numbers are
frozen and never regressed. This document does not modify, reinterpret,
or relitigate any OD claim. It opens a new, independent program.

---

## 1. Why this program exists (the persistence trap)

The OD task died because 1-hour-ahead OD forecasting is
**persistence-trivial**: a model that copies the last observation is a
near-optimal baseline, so "beating persistence" measured almost nothing.
Any new task must be **hard in a way persistence cannot exploit**. PDEs
are introduced specifically for this reason: rollout on a nonlinear
field has no trivial copy-forward solution. The metric set below
**bans 1-step MAE** as a headline endpoint for exactly this reason
(§5, Risk R3).

---

## 2. The falsifiable hypothesis

**Theoretical basis.** Schuld, Sweke, Meyer, *"The effect of data
encoding on the expressive power of variational quantum machine
learning models,"* Phys. Rev. A 103, 032430 (2021), arXiv:2008.08605:
a data-re-uploading PQC realizes a **truncated Fourier series** in the
encoded variable; the accessible frequency spectrum is set by the
encoding gate generators and enriched by re-uploading depth. A model
whose hypothesis class is a low-order Fourier series has a genuine
**inductive bias toward smooth, low-frequency, (quasi-)periodic
functions** and a corresponding **inability to cheaply represent
broadband / sharp / multiscale structure**.

**H1 (primary, falsifiable, directional).**
At matched parameter count and equal HPO budget, the QLNN's
performance *advantage over the mandatory non-liquid Neural-ODE
baseline* (the quantum-isolating contrast, §6) is **positive and
materially larger on the SMOOTH/PERIODIC regime than on the
BROADBAND/MULTISCALE/CHAOTIC regime**, on both the solver and the
forecaster task.

Pre-registered regime partition (assigned *now*, before any run, from
the dynamics — not from results):

| Regime | ODE/PDE systems | Why this regime |
|---|---|---|
| **SMOOTH/PERIODIC** (advantage predicted) | `lotka_volterra`, `van_der_pol` (limit cycle, μ=5), `kuramoto` (phase-locked), **viscous Burgers — smooth regime** (high ν, no shock) | Low-order, (quasi-)periodic or smooth — inside a truncated-Fourier hypothesis class |
| **BROADBAND/MULTISCALE/CHAOTIC** (no advantage or disadvantage predicted) | `lorenz` (chaotic), `fitzhugh_nagumo` (relaxation spikes), **KdV** (dispersive solitons), **Allen–Cahn** (sharp fronts), **viscous Burgers — shock regime** (low ν) | Broadband spectra / sharp fronts / sensitive dependence — outside a low-order Fourier class |

**H1 makes a sign-and-magnitude prediction**, so it can be **confirmed
or falsified**:

- **CONFIRMED** if mean QLNN-minus-NeuralODE advantage on SMOOTH/PERIODIC
  exceeds that on BROADBAND/MULTISCALE by a margin that clears the
  decision rule in §7, on the primary metric, on at least the solver
  task (forecaster corroborating).
- **FALSIFIED** if the advantage gap is zero, reversed, or
  within-noise — i.e. the QLNN's quantum component buys nothing
  regime-specific. **A falsification is a publishable result**: the
  contribution is the regime map × task × model + the mechanistic (T3)
  explanation, *not* a quantum-win headline.

**H2 (secondary, exploratory — NOT a gate).**
Does the locus of any quantum benefit differ between the
**physics-informed solver** and the **data-driven forecaster**? Stated
now so the analysis is honest, but H2 is reported descriptively; only
H1 carries a confirm/falsify verdict.

**H3 (mechanistic, secondary).**
Wherever H1's advantage gap appears (or fails to), the **T3
trainability/expressibility suite**
(`scripts/analyze_quantum_trainability.py`, committed `707debb`)
explains it: barren-plateau / gradient-variance / expressibility
metrics co-vary with the regime partition. H3 is the *mechanism*
behind H1; it is reported but does not itself gate.

---

## 3. The two tasks (locked definitions)

The same systems are evaluated under **two distinct tasks**. Keeping
both, contrasted, is a user-locked decision.

### 3.1 Solver (physics-informed, DQC-grounded)

Given the governing equation, IC, and BCs, train the model by
minimizing a **physics residual** (PDE/ODE residual + IC/BC penalty),
**without supervised trajectory targets**. Architecture lineage:
Kyriienko–Paine–Elfving DQC (PRA 103, 052416, arXiv:2011.10395) and
the TE-QPINN/QCPINN SOTA family (P3a spec cards). Evaluation is against
the held-out high-accuracy numerical reference solution (P2
generators). Solver autodiff (input-coordinate derivative through
PennyLane+Diffrax) uses **`jax.jacrev` only** (Diffrax `custom_vjp`
forbids forward-mode — locked gotcha); a 1-D toy ODE `u' = −u` with
known solution `u=e^{−t}` is the mandatory autodiff prototype gate
before any scale-up (P3 acceptance).

### 3.2 Forecaster (data-driven, autoregressive rollout)

Given an observed history window, predict the future trajectory by
**autoregressive rollout** for a pre-registered horizon (§5). The
model never sees the governing equations. This **replaces** the
persistence-trivial h-step OD protocol. Architecture: the existing
Diffrax QLNN forecaster + the RF-QRC SOTA forecaster family (P3a),
across the 10-ansatz roster.

The solver-vs-forecaster contrast is the H2 axis.

---

## 4. Systems (the hardness ladder — locked)

**ODE tier (reuse `src/quantum_liquid_neuralode/data_processing/synthetic_ode.py`
unchanged — the canonical 5):**
`lotka_volterra`, `fitzhugh_nagumo`, `van_der_pol` (μ=5), `lorenz`
(σ=10, ρ=28, β=8/3), `kuramoto` (K=2.0, N=12). RK4, fixed params as
registered in `get_system()`. The ODE tier is frozen at commit
`1eabdc2`'s `synthetic_ode.py`.

**PDE tier (new, P2 — `pde_systems.py`):** 1-D, method-of-lines,
numpy RK4, periodic domain unless noted:
- **Viscous Burgers** `u_t + u u_x = ν u_xx` — run in **two pre-declared
  regimes**: SMOOTH (ν large, no shock over the horizon) and SHOCK
  (ν small, shock forms). The regime split is part of H1.
- **Allen–Cahn** `u_t = ε² u_xx + u − u³` — sharp fronts (multiscale).
- **KdV** `u_t + 6 u u_x + u_xxx = 0` — dispersive solitons; mass and
  energy are conserved invariants used as a metric (§5).

PDE generators are validated against known behavior with the same rigor
as the 21/21 `synthetic_ode` tests: **Burgers shock-formation time**,
**Allen–Cahn front speed**, **KdV soliton mass & energy conservation**
(P2 acceptance). The artifact is an **npz field** (`u[t, x]`, grid,
IC/BC, invariants) — *not* the scalar qZETA CSV schema (that seam is
blocked; do not emit CSV).

---

## 5. Metric set (locked — 1-step MAE is BANNED as a headline)

All headline metrics are **rollout** metrics over a pre-registered
horizon. Fields use the **relative-L2** norm — the established
PDEBench (Takamoto et al., NeurIPS 2022, arXiv:2210.07182) / FNO
(Li et al., arXiv:2010.08895) standard.

**Primary endpoint (both tasks):** rollout **relative-L2 error**
`‖û − u‖₂ / ‖u‖₂` over the full pre-registered rollout horizon, vs the
numerical reference, mean over seeds. For ODE state vectors the field
norm is the Euclidean state norm; for PDEs it is the spatial-grid L2.

**Secondary endpoints (reported for every cell, never substituted for
the primary):**
- **Valid-prediction-time (VPT):** first rollout time at which
  relative-L2 exceeds a fixed threshold (ε = 0.3, pre-registered).
  For **chaotic** systems (Lorenz) VPT is reported in **Lyapunov
  times** (normalized by the system's largest Lyapunov exponent,
  computed from the reference trajectory) so the number is
  dynamics-meaningful.
- **Spectral error:** L2 error of the power spectral density of the
  rollout vs reference — directly probes the Fourier-bias mechanism
  (H1/H3): a Fourier-biased model should track low-k power and lose
  high-k power on broadband systems.
- **Invariant drift:** for KdV, relative drift of conserved mass and
  energy over the rollout; for Hamiltonian-like ODEs where an invariant
  exists, the analogous conserved-quantity drift.
- **Solver-only:** final physics-residual norm and IC/BC violation.

**Rollout horizons (pre-registered, per system, set from dynamics not
results):** ODE — 10 characteristic periods (periodic systems) or 10
Lyapunov times (Lorenz) or 10 relaxation cycles (FitzHugh–Nagumo);
PDE — to a fixed multiple of the characteristic time (shock-formation
time for Burgers, front-traversal for Allen–Cahn, one soliton-period
for KdV). Exact numeric horizons are fixed in the P2 generator configs
and frozen there before P6.

**Banned as headline:** 1-step / h-step MAE or RMSE. Permitted only as
a sanity diagnostic, never as an endpoint or in a decision rule
(Risk R3 — the persistence trap must not recur on PDEs).

**Statistics:** seeds {0,1,2,3,4}; report mean ± ddof=1 std AND 95%
t-CI. Head-to-head (QLNN vs Neural-ODE, the H1 contrast): paired
bootstrap over rollout test trajectories, n_iter ≥ 10000, two-sided;
per-seed p-values combined via Stouffer's Z-method (Whitlock 2005),
reusing `src/quantum_liquid_neuralode/evaluation/bootstrap.py`.

---

## 6. Model suite & the MANDATORY quantum-isolating baseline

Keeping "Quantum **Liquid** Neural Network" as the central identity
reintroduces a **quantum-vs-liquid confound**: a win could be the
liquid dynamics, not the quantum circuit. Therefore:

**MANDATORY: a non-liquid plain Neural-ODE baseline.** It is the
**primary H1 contrast** — H1 is stated as QLNN-minus-NeuralODE
advantage, *not* QLNN-minus-persistence. This baseline is
non-optional; the program is invalid without it. (Listed as Risk R1 in
the plan; promoted here to a binding pre-registration requirement.)

Full suite, every model run at **equal documented HPO budget** (same
search space size, same trial count, same selection rule — logged in
each run's provenance):

| Class | Model | Role |
|---|---|---|
| Quantum-liquid | QLNN over the **10-ansatz roster** (4 existing: `data_reuploading`, `hardware_efficient`, `strongly_entangling`, `brickwall`; 2 foundational: `chebyshev_dqc`, `lubasch_multicopy`; 4 SOTA: `te_qpinn_fnn`, `te_qpinn_qnn`, `qcpinn`, `rf_qrc`) | The system under test; each ansatz P3a-faithfulness-gated |
| **Non-liquid quantum-free** | **plain Neural-ODE (MANDATORY)** | **Isolates quantum-vs-liquid — the primary H1 contrast** |
| Classical | MLP | Capacity-matched classical control |
| Classical | classical PINN | Solver-task classical control |
| Floor | persistence, linear extrapolation | Triviality floor (a model below the floor is discarded, not reported as a win) |
| Skyline | known-structure model (true RHS, fit only free constants) | Upper bound — contextualizes every gap |

**Parameter matching:** every model compared under a decision rule is
matched within a factor of 2 in trainable parameter count; the matched
counts are logged per cell. **Underfitting control:** before any
reproducibility or advantage statement, a control run confirms each
model has the capacity to fit the *training* trajectory (train-side
relative-L2 below a pre-registered adequacy threshold); a model that
cannot fit training is reported as underfit, and no advantage/null
claim is drawn from it (Risk R: an apparent null that is really an
underfit is not a null).

---

## 7. Decision rules (pre-registered — applied mechanically)

Let `Δ_smooth` = mean (over SMOOTH/PERIODIC systems & seeds) of
[QLNN best-ansatz primary metric] − [Neural-ODE primary metric], in
the *improvement* direction (lower relative-L2 ⇒ larger Δ). Let
`Δ_broad` be the analogous quantity over BROADBAND/MULTISCALE/CHAOTIC.
Both are computed with paired-bootstrap CIs.

- **H1 CONFIRMED** iff `Δ_smooth − Δ_broad > 0` **and** the
  paired-bootstrap 95% CI of `(Δ_smooth − Δ_broad)` excludes 0
  **and** it holds on the **solver** task (forecaster reported as
  corroborating/contradicting, not gating).
- **H1 FALSIFIED** iff the 95% CI of `(Δ_smooth − Δ_broad)` includes 0
  or is negative. Published as a rigorous mechanistic null.
- **Underfit guard:** any cell failing the §6 underfitting control is
  excluded from Δ and flagged; if exclusions remove a regime's
  support, H1 is reported **inconclusive for that regime** (not
  silently confirmed/falsified).
- **Skyline guard:** if *no* model (including the known-structure
  skyline) achieves the adequacy threshold on a system, that system is
  declared **out-of-reach** and excluded from H1 with disclosure (the
  task, not the model, failed).

**Acceptance of the program (independent of H1's direction):** a
system × task × model **regime map**, a T3 mechanistic explanation
(H3), and an explicit confirm/falsify verdict on H1, with the
mandatory non-liquid Neural-ODE baseline isolating quantum-vs-liquid —
defensible to a hostile reviewer **whether the QLNN wins or is a
rigorous null.** No result direction is a project failure; only an
un-falsifiable or confounded design is.

---

## 8. Provenance & gating bindings

- Every numerical result carries a `provenance.json` with the git
  commit and the generator/data hash. PDE npz artifacts hash-locked
  per system before P6 (the per-dataset-lock pattern, reused).
- This file is committed **before any P6 training run**. P6 is gated
  and system-grouped: one go/no-go per system group; compute is
  multi-day; no sweep > 30 min without explicit user go-ahead.
- The P3a faithfulness gate (PDF-grounded spec cards, dual-agent
  cross-check) blocks any literature ansatz from entering P6 until its
  circuit is verified against the source PDF body (not abstracts/priors).
- `pytest` full suite stays green; `scripts/verify_paper_integrity.py`
  stays exit-0 throughout (OD claims frozen, superseded in framing,
  never regressed).

---

## 9. What this pre-registration does NOT claim

- No quantum-hardware advantage — all PQCs run on classical simulators
  (PennyLane, JAX).
- No claim beyond the locked systems, horizons, and metric set.
- No reinterpretation or revival of any bioreactor-OD claim; no QWGAN-GP
  (that drop is locked in `hypothesis.md` v2 and unaffected here).
- H2 (solver-vs-forecaster locus) and H3 (mechanism) are reported but
  do **not** carry confirm/falsify verdicts; only H1 does.

---

## 10. References (verified this program; see plan for fetch provenance)

- Schuld, Sweke, Meyer, PRA 103, 032430 (2021), arXiv:2008.08605 —
  re-uploading ≈ truncated Fourier series (H1 basis).
- Kyriienko, Paine, Elfving, PRA 103, 052416 (2021), arXiv:2011.10395 —
  DQC solver (`chebyshev_dqc`).
- Lubasch, Joo, Moinier, Kiffner, Jaksch, PRA 101, 010301 (2020),
  arXiv:1907.09032 — variational nonlinear (`lubasch_multicopy`).
- Berger, Hosters, Möller, Sci. Rep. (Nature) 2025,
  s41598-025-02959-z — TE-QPINN (`te_qpinn_fnn`, `te_qpinn_qnn`).
- QCPINN, arXiv:2503.16678 (`qcpinn`).
- RF-QRC, Phys. Rev. Research 6, 043082 (2024), arXiv:2405.03390
  (`rf_qrc`).
- Takamoto et al., PDEBench, NeurIPS 2022, arXiv:2210.07182 —
  relative-L2 convention.
- Li et al., FNO, arXiv:2010.08895 — Burgers/field-norm convention.
- Whitlock 2005, J. Evol. Biol. 18(5) — Stouffer Z meta-analysis.
- Abbas et al., Nat. Comput. Sci. 1, 403–409 (2021) — effective
  dimension (carried forward for T3/H3).
- COS pre-registration guidelines: https://www.cos.io/initiatives/prereg
</content>
</invoke>
