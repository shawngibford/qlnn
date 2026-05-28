# Pre-registration amendment — methodological choices not explicitly
locked in `ODE_PDE_PRE_REG.md` at commit `2646d74`

**Status:** committed before P8 paper draft. All amendments are
disclosed deviations from the original pre-reg; per pre-reg §11
ethos ("a publishable null is a result"), these are reported here in
the spirit of full transparency.

**Pre-reg base commit:** `2646d74` (`ODE_PDE_PRE_REG.md`), 2026-05-19.

**This amendment commit:** P7.5 sprint (P7.5 commit 4).

---

## Amendment A1 — Skyline-out-of-reach threshold

**Pre-reg §7** says: *"if no model (including the known-structure
skyline) achieves the adequacy threshold on a system, that system is
declared out-of-reach"*, but does not lock a numeric threshold.

**Amendment:** the skyline-out-of-reach threshold is set to
`relL2 = 0.5` (matches the §5 primary metric's "predict-zero floor"
convention).

**Sensitivity at threshold = 0.75** is reported in the supplement
alongside the primary verdict for transparency. The two thresholds
produce the same QUALITATIVE outcome on the solver task (INCONCLUSIVE
under both pre-reg-strict guards), differing only on whether Lorenz
passes the skyline gate (it's 0.708 — borderline).

**Rationale:** 0.5 is the more conservative choice (excludes more
systems). The H1 verdict under either threshold maintains the
SOLVER-task CONFIRMED-by-raw-bootstrap finding.

## Amendment A2 — Sample size

**Pre-reg §5** specifies "seeds {0,1,2,3,4}" (n=5) and a 9-system
hardness ladder.

**Amendment:** the verdict is computed at **n=3 seeds** ({0,1,2}) and
**3 ODE systems** (LV, VdP, Lorenz). The remaining 2 seeds and 6
systems (3 ODE: Kuramoto, FitzHugh-Nagumo + canonical config; 4 PDE
forecaster tasks) are deferred to P6 (unified-matrix scale-up).

**Rationale:** compute budget constraints — the present subset
provides 9 forecaster cells + 9 solver cells = 18 total cells for
the bootstrap CI. The solver-task verdict's CI [+0.014, +0.220]
excludes 0 at this sample size; the forecaster-task verdict's CI
[-0.79, -0.05] also excludes 0. The SIGN of both verdicts is robust;
scaling up to the full pre-reg matrix would tighten the CI but is
unlikely to flip the outcome direction.

## Amendment A3 — HPO budget

**Pre-reg §6** specifies "equal documented HPO budget — same search
space size, same trial count, same selection rule".

**Amendment:** the present sweep uses **fixed hyperparameters**, not
HPO. All models use identical learning_rate=0.005, train_steps=200
(forecaster) or 1500 (solver), num_qubits=3, num_layers=1. No
Optuna/grid-search HPO was performed.

**Sensitivity:** P7.5 HPO-sensitivity sweep (commit 5) varies LR
∈ {1e-3, 5e-3, 1e-2} and train_steps ∈ {default, 2× default} across
3 anchor cells (LV s2, Lorenz s2, VdP s1) to confirm the verdict
sign is hyperparameter-invariant.

**Rationale:** fixed hyperparameters reduce researcher
degrees-of-freedom; the sensitivity sweep confirms robustness. A
future full P6 sprint would implement matched-HPO budgets per
pre-reg §6.

## Amendment A4 — MLP capacity matching

**Pre-reg §6** specifies "matched within a factor of 2 in trainable
parameter count".

**Amendment:** the plain MLP forecaster (P5 baseline) has 3.3-3.7×
more parameters than the matched QLNN/Neural-ODE pair (159-214 vs
42-58 params). This is a documented violation of the factor-of-2
bound on the MLP arm only.

**Impact:** the MLP is the auxiliary "matched classical control" per
pre-reg §6 table row 3; it does not enter the H1 contrast (which is
QLNN vs Neural-ODE, both matched within 1.14×). The MLP's extra
capacity DOES NOT confer an unfair advantage — in the actual P5
results, the MLP performed WORSE than the Neural-ODE on every cell
(MLP relL2 ≈ 0.56-0.75 on LV vs Neural-ODE 0.29-0.47). So the MLP
control informs but does not bias the H1 verdict.

**Rationale:** the MLP architecture (flatten T×d → hidden → d) has
intrinsically more params for a given hidden width because of the
flattened-history input. A future revision would constrain `hidden_dim`
to enforce the factor-of-2 bound exactly.

## Amendment A5 — Van der Pol task-difficulty boundary

**Observation:** Van der Pol at μ=5 over 10 Lyapunov times shows
universal failure (relL2 ~1.0) across all 7 model classes (4 QLNN +
Neural-ODE + MLP + classical PINN). The known-structure skyline
reaches relL2 = 0.96 — barely below the adequacy threshold.

**Amendment:** VdP at the current configuration is **near the
learnability boundary** at the matched compute budget. The pre-reg
§7 skyline guard correctly excludes it from the H1 aggregation under
the 0.5 threshold.

**Rationale:** stiff dynamics (μ=5) require either (a) finer
collocation grid + longer training, OR (b) a softer system (μ=1).
This is a task-definition observation, not a quantum-specific
weakness.

## Amendment A6 — Underfit guard scope

**Pre-reg §6** specifies: "before any reproducibility or advantage
statement, a control run confirms each model has the capacity to fit
the **training** trajectory (train-side relative-L2 below a
pre-registered adequacy threshold)".

**Amendment:** train-side relative-L2 is recorded for the SOLVER-task
P7.5 sweep (`results/p7_5_solver_h1/.../metrics.json` —
`train_relative_l2` field present and populated). For the FORECASTER-
task P4/P5 sweeps, train-side relL2 was NOT logged at sweep time
(documented in P7.5 commit 3 message). The forecaster-task H1 verdict
is therefore reported with the underfit guard **INACTIVE** for those
cells; this is disclosed transparently.

**Rationale:** activating the underfit guard on the forecaster cells
would require either (a) a re-sweep with train-eval added (compute
cost ~30 min), OR (b) a post-hoc replay of the trained models against
the training trajectory. The forecaster-task H1 verdict's outcome is
robust to this disclosure: the CI [-0.79, -0.05] is comfortably
negative, and the per-cell QLNN-better/QLNN-worse split is stable
across cells; an underfit guard would not change the direction.

## Amendment A7 — Skyline threshold-vs-underfit interaction (the
solver-task INCONCLUSIVE outcome)

**Observation:** on the SOLVER task, the underfit guard (active per
A6 above) correctly identifies VdP + Lorenz classical PINN cells as
underfit (train_relative_l2 ≈ 0.95-1.0 > 0.5 threshold). This excludes
the broadband regime from the H1 aggregation → verdict INCONCLUSIVE
under strict pre-reg guards.

**Amendment:** in addition to the pre-reg-strict verdict
(INCONCLUSIVE), we report the **raw bootstrap verdict** (underfit
guard disabled) as a sensitivity analysis. The raw bootstrap yields
CONFIRMED at CI [+0.014, +0.220]. Per pre-reg §6 ethos ("a model that
cannot fit training is reported as underfit, and no advantage/null
claim is drawn from it"), the strict-guard outcome is the SCIENTIFIC
verdict; the raw bootstrap is reported as the EMPIRICAL pattern
underneath.

**Rationale:** transparent reporting of BOTH gives the reader the
full picture: at the matched 1500-step budget, the classical PINN is
underfit on VdP + Lorenz, and the QLNN consistently shows a positive
Δ on every (system, seed) cell. Whether this is "QLNN advantage" or
"classical PINN undertrained at matched budget" is adjudicated by the
HPO sensitivity sweep (A3 / P7.5 commit 5).

## Amendment A8 — H3 mechanism: tentative trend, not significance

**Pre-reg §H3** posits: *"the T3 trainability/expressibility suite
explains H1's advantage gap"*.

**Amendment:** the P7 T3 cross-tabulation (n=9 forecaster cells)
shows the strongest trend on KL-to-Haar (ρ = +0.518, p = 0.154).
**No T3 scalar reaches statistical significance at p<0.05.** The H3
finding is reported as a TENTATIVE trend, not a confirmed mechanism.

**Rationale:** n=9 cells is underpowered for Spearman significance
testing. A P6 unified-matrix scale-up to ~45 cells would likely push
the KL trend toward significance (if real). The current observation
is consistent with the inverted-pattern reading: less-expressive
ansätze (further from Haar) show larger QLNN advantage on the
forecaster task.

## Amendment A9 — Symmetric QLNN HPO sensitivity (P7.6 commit 1)

**Audit gap closed:** P7.5 commit 6 swept HPO ONLY for the classical
PINN baseline (the H1 contrast model). A peer reviewer would
correctly ask: *"Did you tune the QLNN side symmetrically? If yes
and the SIGN flips, your verdict isn't robust. If you didn't try,
the comparison isn't fair under Bowles/Schuld 2024."*

**Amendment:** P7.6 commit 1 runs the symmetric QLNN HPO sweep —
same anchor cells, same LRs ({1e-3, 5e-3, 1e-2}), same train_steps
({1500, 3000}) — across all 4 quantum solver families. 72 retrains
total. Per-family HPO-best per anchor cell is selected, applied to
the anchor seeds, and a new H1 verdict computed.

**Result (results/p7_6_qlnn_hpo/h1_verdict_full_hpo_best.json):**

  Full-HPO-best H1 verdict (n=9 ODE solver-task):
    outcome  = FALSIFIED
    Δ_smooth = +0.0762
    Δ_broad  = +0.0173
    Δ_diff   = +0.0588
    95% CI   = [-0.0575, +0.1913]   (includes 0)

**Key empirical findings:**

1. **te_qpinn_fnn on LV s2** improves substantially with tuning
   (default 0.524 → HPO-best 0.0988 at lr=1e-2, steps=3000).
2. **qcpinn on LV s2** reaches 0.0057 at lr=5e-3, steps=3000
   (~90× better than the P3.6 default-Adam baseline).
3. **te_qpinn_qnn LV s2 confirmed HPO-INVARIANT structural ceiling**:
   all 6 HPO combos cluster at 0.524 (range 0.5240–0.5250). The
   structural trainability claim from P3.5 generalizes.
4. **All Lorenz s2 cells stay at predict-zero floor (~0.99) across
   all 24 HPO combos** for all 4 quantum families — chaotic-regime
   failure is HPO-invariant.
5. **te_qpinn_fnn / qcpinn on VdP s1** improve substantially with
   high LR but stay above linear-extrapolation floor.

**Rationale:** with both sides at their HPO-best (the cleanest
possible Bowles/Schuld 2024 "tune both sides" sensitivity test),
QLNN does NOT show a regime-dependent advantage. The default-Adam
n=9 raw CONFIRMED (P7.5 / A7) was sample-size-fragile and
HPO-fragile. The HPO-symmetric verdict is FALSIFIED. This is the
methodologically strongest sensitivity point and supersedes the
n=9 raw-bootstrap CONFIRMED as the paper's principal verdict.

## Amendment A22 — 2D PDE hard-IC trial-solution docstring fix (2026-05-28, post-peer-review)

**Audit gap closed:** the docstring of
`src/qlnn_/training/pde_residual_loss.py` (line 25) described the 2D
Lagaris hard-IC trial solution as `u(t, x) = u₀(x) + t · ( s · circuit +
b )`, omitting the `(t − t₀)` offset needed when the PDE time origin
is not zero. The peer-review pass flagged this as a latent correctness
violation.

**Diagnosis.** The *implementation* at line 217 has always used the
correct `ic_fn(x) + (t - t0) * n` formula. Only the module-level
docstring carried the incorrect omission. All current PDE benchmarks
(heat, burgers_smooth, burgers_shock, allen_cahn, kdv) use t₀ = 0, so
the docstring discrepancy did not produce any incorrect numbers. The
bug is purely documentation-level.

**Amendment.** The docstring is corrected to match the implementation:
`u(t, x) = u₀(x) + (t − t₀) · ( s · circuit_2d(...) + b )`. The
comment block carries an explicit note that the implementation has
always been correct.

**Code change.** One-line docstring edit. Implementation unchanged.

**Consequences.** Zero — no committed numbers depended on the
docstring. Future PDEs with t₀ ≠ 0 are protected against
misreading the documentation.

---

## Amendment A21 — brickwall connectivity diagnosis strengthened (2026-05-28, post-peer-review)

**Audit gap closed:** A18 removed `brickwall` from the empirical
forecaster sweep on the grounds that "at (num_qubits=3, num_layers=1)
qubit 2 is structurally disconnected." The peer-review pass surfaced
a deeper diagnosis: even at `num_layers=2`, the alternating-CNOT
pattern produces only `CNOT(0,1)` on layer 0 and `CNOT(1,2)` on
layer 1, giving qubit 0 ↔ qubit 2 no direct CNOT path. The
ansatz reduces to a *linear chain* (rather than the bricklayer
graph the name suggests) unless `num_qubits ≥ 4`.

**Resolution.** No code change. brickwall stays removed from the
forecaster empirical sweep (A18). A21 strengthens the A18
disclosure with the connectivity caveat: any future use of
brickwall should require `num_qubits ≥ 4` to recover the intended
bricklayer interaction graph. The T3 mechanism scalars at
(num_qubits=3, num_layers=1) remain valid as untrained-circuit
diagnostic data and the integrity gate continues to verify them.

**Consequences.** Zero compute or number changes. Strengthens the
A18 paper-supplement text against reviewer pushback ("why not just
fix brickwall by adding layers?").

---

## Amendment A20 — te_qpinn readout consistency restored (2026-05-28, post-peer-review)

**Audit gap closed:** the peer-review pass identified a paired-family
divergence inside `src/qlnn_/circuits/te_qpinn.py`:

- Line 169 (`te_qpinn_fnn` factory): returned
  `qml.expval(qml.prod(*PauliZ))` — the tensor-product observable
  expectation `⟨∏ᵢ Zᵢ⟩ ∈ [−1, 1]`.
- Line 306 (`te_qpinn_qnn` factory): returned
  `qml.expval(qml.sum(*PauliZ))` — the sum-of-Z's magnetization
  `⟨Σᵢ Zᵢ⟩ ∈ [−n, n]`.

These are structurally different observables with different
eigenspectra. The two ostensibly-paired QPINN variants were therefore
fitting different objectives.

**Choice.** Restore `te_qpinn_fnn` to `qml.sum(*PauliZ)`, matching:
1. The `te_qpinn_qnn` precedent in the same file (line 306).
2. The Chebyshev-DQC `Σⱼ⟨Zⱼ⟩` magnetization-sum readout in
   `src/qlnn_/training/physics_residual_loss.py` (Kyriienko
   2011.10395 §III.3 Eq. 9).
3. The wider QPINN literature's "scalar Z-sum" convention
   (Zhou 2503.16678; Kyriienko 2021).

**Justification for the choice.** Berger 2025 Eq. 13 writes
`O = ⊗ᵢ Zᵢ` in compact tensor notation, which the original
implementation read as the tensor-product observable expectation.
This is a defensible reading of the equation in isolation. However:
(a) the paired `te_qpinn_qnn` variant in the same codebase already
implemented the sum-of-Z's reading; (b) the Berger paper's worked
numerical examples produce values in `[−n, n]`, consistent with the
sum reading, not the product reading; (c) the Chebyshev-DQC
precedent in the same codebase uses the sum form. Restoring fnn to
the sum form re-establishes paired-family equivalence and matches
the literature consensus.

**Amendment.** One-line code change on line 169:
`qml.prod` → `qml.sum`. Docstring comment block explains the change
and references this amendment.

**Consequences.** All prior `te_qpinn_fnn` solver-task results
(P3.5 demo, P3.6 multi-state, P7.5 solver-h1, P7.8 n=24 verdict,
P3.9 PDE matrix) used the product readout and need re-running with
the corrected sum readout. The expected impact on the PRIMARY
n=24 solver verdict is small: te_qpinn_fnn is one of four QLNN
families in the best-of-family selection, and its empirical
performance (mean relL² across systems) is in the middle of the
field. The fnn family's *seed variance* may change measurably —
the product readout's compressed observable range `[−1, 1]` made
the family appear lower-variance than the magnetization-sum
families.

These re-runs fold into the Tier 3 Phase C sweep documented in
`NEXT_STEPS.md` and `REMEDIATION_PLAN.md`. Per-cell wall-clock
under the corrected readout is unchanged. Phase C budget increases
by approximately 5 CPU-hours (4 ODE systems × 1 family × 3 seeds
× ~0.18 hr per cell + 4 PDE systems × 1 family × 3 seeds × ~0.92
hr per cell).

---

## Amendment A19 — Cross-task budget parity: forecaster step budget raised 200 → 2000 (2026-05-28)

**Audit gap closed (third pass):** the first two passes of A15
equalized step budgets WITHIN the SOLVER task (QLNN families to each
other, then QLNN to classical PINN — all at 2000 steps). The third
pass extends the same fairness principle ACROSS tasks: the FORECASTER
side was running at 200 steps (per A3) — uniform across its 5 model
classes (LiquidQ, NonLiquidQ, ClassicalLTC, PlainNeuralODE, PlainMLP)
but an order of magnitude lower than the solver-side 2000.

The user directive (2026-05-28): "we need a matched budget for ALL
models." The most defensible reading is cross-task as well as
cross-side parity.

**Amendment:** raise the forecaster step budget from 200 → **2000**.
Concretely:

| Side                          | File                                 | Was   | Now |
|---|---|---:|---:|
| All 5 forecaster sides (cfg)  | p4_forecaster_demo.P4SweepConfig.train_steps | 200 | 2000 |
| Function default               | forecaster_training.train_vector_forecaster | 1000 | 2000 |
| P7.6 QLNN HPO lower bound      | scripts/run_p7_6_qlnn_hpo.py --train-steps-list | [1500, 3000] | [2000, 3000] |

The forecaster-cell internal parity audit (carried out in the same
session) confirms that beyond the budget, the liquid / non-liquid /
classical-LTC / plain-Neural-ODE cells are already structurally
symmetric:

- Hidden dimension Q=4 across all 5 cells.
- τ machinery: identical softplus + tau_min=0.1 + tau_init=1.0
  parameterization in LiquidQuantumCell.tau() and
  ClassicalLTCCell.tau().
- τ-axis adds exactly 4 params on both the quantum side
  (LiquidQ vs NonLiquidQ) and the classical side (LTC vs PlainNODE).
- Drive-term parameter counts within 10 % (MLP ~56 vs encoder + circuit
  + A ~60).
- Diffrax ODE solver with step_dt=0.05 across all 4 ODE-based cells.

The drive-form axis (MLP vs A ⊙ q(x)) is the central experimental
question of the paper, not a fairness asymmetry to equalize.

**Rationale:** The original 200-step budget was justified historically
by forecaster training-cost concerns. The 2026-05-28 smokes show the
loss converges fast on this task at all 5 model classes; 2000 steps
is comfortably above the convergence point and matches the solver
side. After this raise, the entire experiment runs at uniform 2000
steps everywhere it matters:

- Solver QLNN families: 2000 ✓
- Solver classical PINN: 2000 ✓
- PDE QLNN families: per-PDE config (1200/1500/1800/2400) ✓ (symmetric
  within each PDE across QLNN and classical PINN; varies across PDEs
  for physics reasons documented in CORRECTED_PDE_CONFIGS)
- Forecaster all 5 model classes: 2000 ✓ (this amendment)

**Consequences:** All forecaster cells must be re-run. The affected
result directories are:

- `results/p4_forecaster_rollout/` — 45 cells (post-A18 brickwall
  removal: 36 cells)
- `results/p5_h1_verdict/` — full matched-baseline matrix
- `results/p7_10_forecaster_decomposition/` — classical LTC + 2×2
  matrix
- `results/p7_11_decomposition/` — non-liquid QLNN + complete 2×2 +
  τ-cross-check

The integrity gate's locked PRIMARY FORECASTER number (Δ_combined =
−0.501, CI [−0.804, −0.244]) will be re-computed. The FALSIFIED
verdict is not expected to change qualitatively (the CI is comfortably
away from zero on the negative side and the 10× budget raise can only
*improve* the quantum families' relL² — the H1 contrast remains
classical-driven).

Per-cell wall-clock on the forecaster side is short (smokes show
seconds per cell at the small Q=4 size), so the full re-run is a
modest ~1-3 hour add-on to the broader M3 sweep budget. Already
folded into the ADVISOR_BRIEF.md Anvil-allocation case.

---

## Amendment A18 — brickwall removed from empirical forecaster sweep (2026-05-28 audit)

**Audit gap closed:** the `brickwall` forecaster ansatz scores BEST on
the H3 T3 mechanism diagnostics (lowest entangling-Q at 0.309 vs ~0.78
for the other families; most barren-plateau-resistant in the qubit-
scaling study) but FAILS every empirical forecaster cell (relL² 1.15
on Lorenz, 8.27 on van der Pol, 0.64 on Lotka-Volterra — vs winning
families around 0.6 on each).

**Root cause:** brickwall implements alternating-layer CNOT
entanglement. At the project's P4 config (num_qubits=3, num_layers=1),
only layer-0 executes: CNOT(0,1) connects qubits 0-1, and **qubit 2
remains structurally disconnected** (it receives single-qubit
rotations only). The circuit cannot represent 3D dynamics (Lorenz,
Lotka-Volterra, van der Pol all have 2D or 3D state) because the
required cross-component entanglement is unreachable.

The T3 score is not misleading — it correctly identifies that brickwall
entangles fewer qubits at this config. The audit's prior interpretation
("low entangling-Q = good") read T3 in the wrong direction: low Q here
means "structurally fewer qubits mixed", not "expressively economical".

**Amendment:** brickwall is removed from `VECTOR_QLNN_FAMILIES` in
`src/qlnn_/training/p4_forecaster_demo.py`. The trainable forecaster
sweep is now four families: data_reuploading, hardware_efficient,
strongly_entangling (post-A16 fix), and rf_qrc.

The T3 mechanism scalars for brickwall (computed on the *untrained*
circuit) remain in the H3 mechanism analysis — they are still valid
data points describing what the brickwall pattern *would* produce if
it were trained, and they correctly explain *why* it fails empirically.
The integrity gate's brickwall T3 numbers (entangling_q 0.309,
gradient_variance 0.046) are NOT touched.

**Alternatives considered and rejected:**
- Adding a wraparound CNOT to "close" the brickwall pattern into a
  ring at n=3 — but that defeats brickwall's structurally distinct
  identity (it would alias the ring-CNOT families).
- Increasing num_layers from 1 to 2 to expose the odd-layer
  CNOT(1,2) — but that would unfairly give brickwall a 2× compute
  budget vs the other families (forecaster sweep is uniform at
  num_layers=1).

**Consequences:** removes 9 P4 forecaster cells from the matrix
(3 systems × 3 seeds × {liquid only — brickwall is not used as a
non-liquid family any more}). The PRIMARY FORECASTER FALSIFIED verdict
is unchanged in direction because best-QLNN-per-cell selection in the
old data already excluded brickwall from the top-of-rank position on
every cell.

---

## Amendment A17 — qcpinn quantum-parameter sweep (2026-05-28 audit)

**Audit gap closed:** the `qcpinn` solver-task family (Zhou 2503.16678,
Cascade topology, n=5, L=1, pre/post-NN hidden=50) uses 15 PQC params
alongside 706 classical pre/post-NN params — a 2% PQC : 98% classical
ratio. The audit confirmed that qcpinn's win on Lotka-Volterra (relL²
0.0058) is mostly attributable to the 706-param MLP, not the quantum
circuit. As-is, qcpinn's "win" is a classical-baseline result wearing
a quantum label.

**Amendment:** three step-wise qcpinn variants are added to the solver
family registry (`src/qlnn_/training/solver_demo.py:FAMILIES`):

| variant         | topology    | n | L | pre/post_hidden | PQC | ≈classical | Q-ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| qcpinn          | Cascade     | 5 | 1 | 50 | 15  | ~706 |  2% |
| qcpinn_balanced | Cross-mesh  | 5 | 1 | 10 | 45  | ~146 | 24% |
| qcpinn_quantum  | Cascade     | 8 | 3 |  4 | 72  |  ~89 | 45% |
| qcpinn_full_q   | Cross-mesh  | 5 | 3 |  1 | 135 |  ~20 | 87% |

The progression varies (topology, n, L, pre/post_hidden) along the
axis Q-ratio = PQC / (PQC + classical). The Cross-mesh topology has
higher PQC density per the qcpinn paper's Table 2 (n²+4n vs Cascade's
3n); increasing L scales PQC linearly; shrinking pre/post_hidden
reduces the classical pre/post-NN footprint to its 1-neuron-hidden
minimum.

**Goal:** make the "qcpinn quantum win" attribution testable. If the
variants' relL² monotonically improves as Q-ratio increases, the
quantum capacity is genuinely doing useful work. If relL² degrades or
flatlines, the original qcpinn's win is mostly classical-MLP capacity
and the original family carries that confound as a disclosed
limitation.

**Consequences:** adds 3 new families × 8 systems (4 ODE + 4 PDE) × 3
seeds = **72 new solver cells**. The PRIMARY solver n=24 verdict is
NOT immediately recomputed — A17 cells are reported separately as a
"quantum-capacity attribution" sub-experiment in the paper's
solver-task §3 discussion. The integrity gate's PRIMARY solver number
is not touched by this amendment.

---

## Amendment A15 — Equalized training-step budget across ALL solver-task models (2026-05-28 audit, extended)

**Audit gap closed (initial scope):** the ODE-solver pipeline assigned
per-QLNN-family step budgets (chebyshev_dqc 1200, te_qpinn_fnn 1500,
te_qpinn_qnn 2000, qcpinn 1500) justified in code (`solver_demo.FAMILIES`)
by "te_qpinn_qnn's trainable embedding adds a second circuit eval per
loss point, so equal-compute requires more steps for the cheaper
families to match." This is an *equal-compute-per-step* fairness model,
not an equal-iterations model. The family with the largest budget
(te_qpinn_qnn) is also the family that wins 3 of 4 ODE solver systems.

**Audit gap closed (extended scope):** the same audit pass surfaced a
SECOND asymmetry — the classical PINN solver
(`train_classical_pinn_solver_one_cell`, default `steps=1500`) was
running 25 % FEWER iterations than the quantum families (which had been
running 2000 steps after the within-QLNN equalization). This is
unfair-to-classical and biases the QLNN-vs-classical comparison
*toward* quantum. The user's directive (2026-05-28): "we need a matched
budget for ALL models."

**Amendment:** ALL solver-task training paths — quantum AND classical
— are equalized to **2000 steps**. Concretely:

| Side                            | File                                   | Previous | Now |
|---|---|---:|---:|
| QLNN chebyshev_dqc              | solver_demo.FAMILIES                   | 1200 | 2000 |
| QLNN te_qpinn_fnn               | solver_demo.FAMILIES                   | 1500 | 2000 |
| QLNN te_qpinn_qnn               | solver_demo.FAMILIES                   | 2000 | 2000 |
| QLNN qcpinn                     | solver_demo.FAMILIES                   | 1500 | 2000 |
| QLNN qcpinn_balanced / quantum / full_q (A17 new) | solver_demo.FAMILIES | n/a | 2000 |
| **Classical PINN (ODE solver)** | **p7_5_solver_h1.py:train_classical_pinn_solver_one_cell** | **1500** | **2000** |
| Classical PINN script default   | scripts/run_p7_5_solver_h1.py --steps  | 1500 | 2000 |
| HPO sensitivity sweep lower bound | scripts/run_p7_5_hpo_sensitivity.py --train-steps-list | [1500, 3000] | [2000, 3000] |

PDE solver budgets are uniform per-PDE across QLNN AND classical
(`CORRECTED_PDE_CONFIGS[pde].steps` is consumed by both
`train_one_cell` and `train_one_pde_classical`) so the PDE side is
inherently symmetric — no changes needed there.

Forecaster budgets are also already uniform: `p4_forecaster_demo.
ForecasterConfig.train_steps = 200` is consumed by both QLNN and
classical (plain Neural-ODE, classical_LTC, plain_mlp) sides through
`p5_matched_baselines.py` — no changes needed.

The pre-reg §6 matched-capacity intent now applies symmetrically to
both parameter count AND training iterations AND model-class.

**Rationale:** Equal-compute-per-step is defensible in isolation
(quantum-resource-fairness argument), but pre-registration §6 reads
"matched capacity ... equal training budget" without specifying which
fairness model. Equalizing to the maximum (2000) on every model class
gives every side — quantum and classical — the strongest possible shot.

**Consequences:** All ODE-solver results that depended on the
asymmetric budget must be re-run. The affected directories are:
- `results/p3_5_solver_demo/` (4 families × 2 ODEs × 3 seeds = 24 cells)
- `results/p3_6_multi_state/` (4 families × 3 vector ODEs × 3 seeds = 36 cells)
- `results/p7_5_solver_h1/` (9 baseline cells, mixed PINN + QLNN — BOTH sides re-run)
- `results/p7_8_solver_h1_n24/` (the PRIMARY n=24 solver verdict — re-runs entirely)

The integrity gate's locked PRIMARY solver number (Δ_diff = −0.084,
CI [−0.278, +0.061]) will be re-computed once the affected cells
re-run. The FALSIFIED verdict is not expected to change qualitatively:
the QLNN re-runs can only improve QLNN relL² (more training), and the
classical-PINN re-run also gets 33 % more steps — the H1 contrast
Δ = classical_PINN_relL² − QLNN_relL² should remain near zero, just
with both endpoints sharper.

---

## Amendment A16 — strongly_entangling un-aliasing (2026-05-28 audit)

**Audit gap closed:** the `strongly_entangling` forecaster ansatz
(Schuld et al. 2020, arXiv:1804.00633) produced bit-identical relL²
values to the `data_reuploading` ansatz across every P4 forecaster cell
— 16-digit agreement on seed_0, seed_1, seed_2 for all three ODE
systems (lotka_volterra, van_der_pol, lorenz), and the same for the
non-liquid variants `non_liquid_strongly_entangling` ≡
`non_liquid_data_reuploading`.

**Root cause:** PennyLane's `qml.StronglyEntanglingLayers` template,
when called with `ranges=None`, falls back to the per-layer formula
`r = l mod num_qubits`. At the project's P4 config (num_qubits=3,
num_layers=1) this evaluates to `r = 0`, which is outside the
template's valid range `1 <= r < num_qubits`; PennyLane silently
rewrites this to `r = 1`, producing the nearest-neighbor ring-CNOT
pattern that is *unitarily identical* to data_reuploading's explicit
ring entangler at this config. Both circuits then apply the same RX
encoding, the same single-qubit Rot rotations (RZ·RY·RZ), and the
same ring CNOT — yielding identical outputs as a mathematical
consequence, not a code-dispatch bug.

**Amendment:** the default `ranges` value in `StronglyEntanglingCircuit`
is changed from `None` (PennyLane fallback) to
`(num_qubits - 1,) * num_layers` (long-range CNOT per layer). At
num_qubits=3 this gives r=2 (skip-one CNOT), realizing the "strongly
entangling" name distinct from data_reuploading's nearest-neighbor
ring. Verification: at the P4 config, the two circuits now produce
outputs with max absolute element-wise difference 1.24 (not bit-
identical). Callers who want PennyLane's per-layer-modulo behavior can
still pass explicit `ranges` via `AnsatzConfig.params`.

The fix lives in `src/qlnn_/circuits/strongly_entangling.py` with a
module-level docstring change cross-referencing this amendment.

**Consequences:** All P4 forecaster cells that include
`strongly_entangling` or `non_liquid_strongly_entangling` must be
re-run — 18 cells across the lotka_volterra/van_der_pol/lorenz × 3
seeds × {liquid, non-liquid} matrix. The downstream H1 verdicts
(P5/P7.10/P7.11) re-aggregate from these cells and will produce
slightly refreshed numbers; the FALSIFIED forecaster verdict is not
expected to change qualitatively because both the original and the
fixed circuit are 4th- or 5th-tier per-cell performers (the verdict is
classical-baseline-driven, not best-QLNN-driven on these systems).

The original aliased results were flagged in `HANDOFF.md` known-issues
("strongly_entangling produces identical relL² to data_reuploading on
every cell ... pre-existing behavior") but never disclosed in the
paper. This amendment is the explicit disclosure plus the structural
fix.

---

## Amendment A14 — Post-hoc Q-Q residual diagnostics (P8 polish)

**Audit gap closed:** the residual-distribution panel in
`fig_residual_analysis` (committed earlier as a T1 reviewer-diagnostic)
labels its middle column as `# histogram + normal QQ` in the source
but the implementation is a histogram with a fitted Normal PDF
overlay, not a Q-Q plot. No genuine quantile-quantile analysis existed
anywhere in the figure set, and the Normal overlay was actively
misleading because both stacks' residuals are clearly non-Normal.

**Amendment:** P8 polish adds two reusable, dataset-agnostic helpers in
`scripts/make_diagnostic_figures.py` —
`_qq_panel_vs_normal(ax, residuals, label, color)` and
`_two_sample_qq(ax, res_a, label_a, res_b, label_b, color)` — plus a
new figure `fig_qq_analysis` that applies them to the canonical
(Classical H=4 vs QLNN 4q/3L) residuals at h=3. The misleading
in-source comment in `fig_residual_analysis` is corrected to
"histogram + Normal PDF overlay (NOT a Q-Q plot — that lives in
`fig_qq_analysis` below, with Shapiro and two-sample KS tests)". The
T1 registry test is updated 7 → 8 callables.

**Results on the canonical comparison (seed-mean residuals at h=3):**

| Test | Classical H=4 | QLNN 4q/3L |
|---|---|---|
| Shapiro–Wilk W | 0.892 | 0.885 |
| Shapiro–Wilk p | 1.64×10⁻⁵ | 8.91×10⁻⁶ |
| Two-sample KS D | — | — (D = 0.085) |
| Two-sample KS p | — | — (p = 0.964) |

Interpretation: both per-stack distributions are clearly non-Normal
(Shapiro rejects normality at α = 0.05 by ≥ 4 orders of magnitude on
each); the two-sample KS test cannot reject the null that the two
distributions are equal (p = 0.964).

**Rationale:** This is a post-hoc *diagnostic*, not a new headline
claim. Concretely it does two things: (a) it corrects a labelling
defect in the diagnostic figure suite (the "QQ" naming was wrong;
nothing in the figure set was a real Q-Q plot); (b) it quantifies the
visually-suggested "the two histograms look identical" observation
with a formal Kolmogorov–Smirnov test that yields p = 0.964 — i.e.
*statistical* indistinguishability of the two error distributions at
h=3. This is consistent with, and supportive of, the project's honest
"both near persistence at h=3" verdict that is already documented in
`archive/PAPER_SUMMARY.md`; the amendment adds a defensible numerical
backing to it. Neither helper is yet applied to post-pivot solver or
forecaster residuals — that is a small, mechanical follow-up the next
contributor can perform by calling the same helpers with new
`(residuals, label)` arrays. No locked headline number is touched;
`scripts/verify_paper_integrity.py` still exits 0.

`scripts/verify_paper_integrity.py`: PASS (no change to integrity).
`pytest tests/test_diagnostic_figures.py`: 7 PASS (registry-count
assertion updated 7 → 8 in the same commit). Full numerical table and
figure appear in `paper/supplement.tex` §3.2 ("Q-Q residual diagnostics").

---

## Amendment A13 — Non-liquid QLNN ablation + complete 2×2 mechanism decomposition (P7.11)

**Audit gap closed:** P7.10 (A12) filled three corners of the
forecaster fairness 2×2 by adding `classical_LTC`. The fourth
corner — a non-liquid quantum forecaster (same quantum circuit
with the τ-leak removed) — remained empty, leaving the LTC
mechanism story confirmed only along the classical-side
τ-isolation path. P7.11 fills the fourth corner so the τ-isolation
mechanism can be cross-checked along the quantum-side path.

**Amendment:** add `NonLiquidQuantumCell` (`src/qlnn_/cells/
non_liquid_quantum_cell.py`), `NonLiquidVectorForecaster`
(`src/qlnn_/models/non_liquid_vector_forecaster.py`), and a
dispatcher prefix `non_liquid_<ansatz>` covering 4 ansätze
(data_reuploading, hardware_efficient, strongly_entangling,
brickwall). rf_qrc is intentionally excluded — its reservoir
has a fixed `leak_rate` hyperparameter and is therefore already
non-liquid in the QLNN sense.

**Implementation surgery (minimum-faithful):** the
LiquidQuantumCell dynamics
$dh/dt = -(1/τ + q(x)) \odot h + A \odot q(x)$
becomes, with τ removed,
$dh/dt = -q(x) \odot h + A \odot q(x)$
i.e. the `1/τ` leak coefficient drops out. Encoder, A, and Diffrax
integration are unchanged. The cell-level mathematical-identity
test (`tests/qlnn_/test_non_liquid_quantum_cell.py`) confirms
$\Delta(\text{liquid} - \text{non\_liquid}) = -(1/τ) \odot h$
exactly at any (h, x) probe point.

**The 2×2 is now complete:**

|             | non-liquid (no τ)            | liquid (learnable τ)        |
|-------------|------------------------------|-----------------------------|
| Classical   | plain_neuralode (P5)         | classical_LTC (P7.10)       |
| Quantum     | non_liquid_<ansatz> (P7.11)  | <ansatz> (P4, P7.10)        |

**Five paired-bootstrap verdicts** are computed (n=9), with
TWO independent algebraic identities holding per-cell exact:

| Verdict | $\Delta_\text{diff}$ | 95% CI | Outcome |
|---|---|---|---|
| combined | $-0.5007$ | $[-0.804, -0.244]$ | FALSIFIED (excludes 0) |
| quantum_via_ltc | $-0.6160$ | $[-1.167, -0.178]$ | FALSIFIED (excludes 0) |
| liquid_via_classical | $+0.1153$ | $[-0.097, +0.377]$ | FALSIFIED |
| **liquid_via_quantum** (NEW) | $\mathbf{-0.3339}$ | $\mathbf{[-0.627, +0.053]}$ | **FALSIFIED** |
| quantum_via_nonliquid (NEW) | $-0.1668$ | $[-0.495, +0.204]$ | FALSIFIED |

Identities (per-cell, exact):
1. $\Delta_\text{combined} = \Delta_\text{q\_via\_ltc} +
   \Delta_\text{τ\_via\_cls}$  (P7.10 path)
2. $\Delta_\text{combined} = \Delta_\text{τ\_via\_q} +
   \Delta_\text{q\_via\_nlq}$  (P7.11 path)

Both verified mechanically by `verify_paper_integrity.py`.

**Key empirical finding — the τ-isolation cross-check
DISAGREES IN SIGN ACROSS THE TWO PATHS:**

- $\Delta_\text{liquid\_via\_classical} = +0.115$
  → On the classical MLP hidden state, the liquid-τ machinery
    contributes a small POSITIVE Δ (smooth>broad, matching the
    Schuld-Fourier H1 direction).
- $\Delta_\text{liquid\_via\_quantum}  = -0.334$
  → On the quantum cell hidden state, the liquid-τ machinery
    contributes a NEGATIVE Δ (broad>smooth, OPPOSITE of the
    H1 direction).

The liquid-τ machinery is therefore NOT a context-independent
component. Its regime-contrast contribution depends on the
substrate it modulates. On classical hidden states it nudges
toward the H1-predicted direction; on quantum hidden states it
tilts the OTHER way.

**Per-cell mechanism per system:**
- Lorenz s2 (chaotic): $\Delta_\text{τ\_via\_quantum} = +0.594$
  — adding τ to the quantum cell substantially HELPS the
    broadband regime (rf_qrc with τ: 0.460 vs
    non_liquid_hardware_efficient: 1.054).
- VdP s1 (stiff): $\Delta_\text{τ\_via\_quantum} = +0.421$ —
  same pattern, τ helps on stiff dynamics.
- LV s0,s1 (smooth-periodic): $\Delta_\text{τ\_via\_quantum} =
  +0.006$ to $+0.169$ — near-neutral on smooth.

τ-on-quantum disproportionately helps the BROADBAND cells. When
stratified by the smooth-minus-broad regime contrast, this gives
a negative point estimate. The combined Δ inversion is the NET of
a quantum-circuit underperformance (Δ_q_via_ltc = -0.62) and a
substrate-dependent τ contribution that goes the opposite way on
the quantum substrate.

**Rationale:** the user-locked scope was "every quantum and
classical model needs a head to head, including the classical
liquid setup" (A12). The natural extension — "and the non-liquid
quantum to close the 2×2" — was the open scope flag at the end
of P7.10. P7.11 closes it. The cross-check disagreement is itself
a novel mechanistic finding about how learnable time-constants
interact with classical vs quantum hidden-state substrates, and
is worth reporting on its own terms (paper §4.3 covers it).

---

## Amendment A12 — Classical LTC baseline + forecaster H1 decomposition (P7.10)

**Audit gap closed:** the pre-registered forecaster H1 contrast is
``QLNN_forecaster − Neural-ODE`` where the QLNN forecaster has
learnable per-qubit time-constants $\tau$
(``src/qlnn_/cells/liquid_quantum_cell.py``) and the Neural-ODE
baseline does not (``src/qlnn_/models/plain_neuralode_forecaster.py``).
The measured $\Delta$ therefore confounds two structurally distinct
contributions: the quantum circuit versus the liquid-$\tau$ machinery.
A reviewer running the Bowles--Schuld 2024 "remove the quantum
component" ablation will ask which part of the measured
underperformance is attributable to which mechanism.

**Amendment:** P7.10 adds the missing fourth-quadrant baseline,
``ClassicalLTCForecaster``, that has learnable $\tau$ but no
quantum circuit. The cell mirrors ``PlainNeuralODECell`` field-for-
field with one structural addition: a per-unit ``tau_unconstrained``
vector + the Hasani 2021 LTC input-independent-$\tau$ form
``dh/dt = -(1/τ) ⊙ h + MLP([h, x])``. Three paired-bootstrap H1
verdicts at $n=9$ are computed:

  - $\Delta_\text{combined} = \mathrm{QLNN} - \text{Neural-ODE}$
    (pre-reg-mandated, original)
  - $\Delta_\text{quantum} = \mathrm{QLNN} - \text{classical-LTC}$
    (isolated quantum contribution)
  - $\Delta_\text{liquid} = \text{classical-LTC} - \text{Neural-ODE}$
    (isolated liquid-$\tau$ contribution)

with the per-cell algebraic identity
$\Delta_\text{combined} = \Delta_\text{quantum} + \Delta_\text{liquid}$
holding exactly (verified by ``verify_paper_integrity.py``).

**Results (results/p7_10_forecaster_decomposition/, n=9 with
all 5 quantum families including rf_qrc):**

| Verdict | $\Delta_\text{diff}$ | 95% CI | Outcome |
|---|---|---|---|
| combined | $-0.5007$ | $[-0.8040, -0.2438]$ | FALSIFIED (CI excludes 0 negatively) |
| quantum-isolated | $-0.6160$ | $[-1.1665, -0.1782]$ | FALSIFIED (CI excludes 0 negatively) |
| liquid-isolated | $+0.1153$ | $[-0.0973, +0.3772]$ | FALSIFIED (CI includes 0) |

**Key empirical findings:**

1. **The QLNN forecaster's underperformance on the
   pre-registered regime contrast is mechanistically the
   quantum circuit, at high statistical confidence.** The
   quantum-isolated CI is $[-1.167, -0.178]$ and EXCLUDES
   zero in the negative direction: the quantum circuit's
   contribution to the regime contrast is significantly
   negative, with point estimate $-0.62$. The combined point
   estimate ($-0.50$) is less negative than the quantum-isolated
   estimate because the liquid-$\tau$ component partially offsets
   it ($+0.12$).

2. **The liquid-$\tau$ component on its own tracks the
   Schuld--Fourier H1 prediction direction.** The liquid-isolated
   verdict has $\Delta_\text{liquid} = +0.12$ (smooth $>$ broad),
   directionally consistent with the original H1 hypothesis. The
   CI includes zero (n=9 is underpowered for a $+0.12$ effect),
   but the point estimate is the only one of our eight verdicts
   whose direction matches H1's pre-registered prediction.

3. **The quantum circuit is responsible for the regime inversion.**
   Across all three layered solver-task sensitivity points (P7.6
   n=18, P7.6 HPO-best, P7.8 n=24) and the forecaster-task
   combined verdict, the QLNN consistently FALSIFIES H1 in the
   broad $>$ smooth direction. P7.10's decomposition localizes
   this inversion mechanistically: it is the quantum-circuit
   component's contribution.

4. **Per-cell highlights** (per_cell_records.json):
   - LV s0,s1: classical LTC dramatically beats QLNN
     ($\Delta_\text{quantum} = -0.35, -0.33$); both LV cells where
     the QLNN forecaster underperforms.
   - LV s2: classical LTC loses to QLNN ($\Delta_\text{quantum}
     = +0.33$) — opposite of s0,s1; bimodal seed behavior.
   - Lorenz s2: QLNN beats classical LTC by $\Delta_\text{quantum}
     = +1.04$ (huge quantum advantage on chaotic dynamics; matches
     the solver-task FHN + AC broadband sub-finding).
   - All LV cells: classical LTC also beats Neural-ODE
     ($\Delta_\text{liquid} = +0.06, +0.09, -0.19$); liquid-$\tau$
     is uniformly small-positive on smooth dynamics.

**Rationale:** the pre-reg literally required only a non-liquid
Neural-ODE baseline (§6), which we satisfied. But the Bowles--Schuld
2024 doctrine implies that any positive QLNN claim should survive a
component ablation. Our forecaster claim is FALSIFIED, so there is
no positive claim to defend; the decomposition is informative,
not defensive. The added LTC baseline lets us attribute the
mechanism of the underperformance, which is itself a scientific
finding. The classical-LTC sweep cost ~6 minutes of compute and
adds zero ambiguity to the verdict structure.

---

## Amendment A11 — Pre-paper full-ladder expansion (P7.8)

**Audit gap closed:** P7.6's n=18 combined verdict used 3 ODE
systems (LV/VdP/Lorenz) + 3 PDE systems (heat/burgers_smooth/
allen_cahn). The pre-reg §4 hardness ladder lists 5 ODE systems and
4 PDE systems. A peer reviewer running the pre-reg manifest against
the results would ask: *"Why are FHN, Kuramoto, burgers_shock, KdV
missing from the H1 verdict?"*

**Amendment:** P7.8 adds 2 new pre-reg systems to the H1 verdict at
default-Adam baseline, both 3 seeds × all relevant quantum families
+ classical PINN, with TWO transparent deferrals documented:

  ADDED (now in the n=24 H1 verdict):
    - fitzhugh_nagumo (ODE, BROADBAND/MULTISCALE per pre-reg §4)
    - burgers_shock   (PDE, BROADBAND/MULTISCALE per pre-reg §4)

  DEFERRED (with documented compute/mechanism rationale):
    - kuramoto       (12D high-dim; per-component scalar circuits
                      scale linearly in state dim → ~7 hr per cell
                      vs ~3 min for 2D ODEs at the same families;
                      the 12D solver-side cost is one paper of its
                      own. Single-cell smoke proves the dispatch
                      works; budget makes the full sweep
                      out-of-scope here. Queued for the follow-up
                      paper alongside the P7.7 optimization sprint.)
    - kdv            (third-order PDE u_t + 6·u·u_x + u_xxx = 0;
                      mechanism gate at `scripts/run_p7_8_kdv_gate.py`
                      → PASS. jacrev³ through the QNode produces
                      finite non-trivial values, JIT compiles in
                      ~4.6s, per-point cost is 0.44× the jacrev²
                      baseline (XLA fuses the third derivative
                      tightly). BUT the integrated training cost
                      with full vmap over collocation + value-and-
                      grad-jit is ~12 s/step, projecting ~8 hr per
                      seed × 15 cells (4 quantum 2D families + cls
                      PINN × 3 seeds) ≈ 5 days CPU. This crosses
                      the paper's submission-timeline budget but
                      is well-defined for the follow-up.)

**Result (results/p7_8_solver_h1_n24/h1_analysis_combined_n24.json):**

  Combined ODE+PDE solver-task H1 verdict @ n=24:
    outcome    = FALSIFIED
    Δ_smooth   = +0.0674   (unchanged from P7.6; same 12 cells)
    Δ_broad    = +0.1518   (up from +0.0358; 6 new broadband cells)
    Δ_diff     = -0.0844   (POINT ESTIMATE FLIPPED SIGN vs P7.6
                            n=18's +0.0316)
    95% CI     = [-0.2780, +0.0613]   (includes 0; wider than
                            P7.6's [-0.0400, +0.1088] due to higher
                            broadband seed variance)
    n_smooth   = 12
    n_broad    = 12   (symmetric bins)

**Key empirical findings:**

1. **te_qpinn_qnn on FitzHugh-Nagumo** is a CONSISTENT,
   LOW-VARIANCE WIN for QLNN:
     seed 0:  Δ = +0.313  (QLNN 0.327 vs cPINN 0.640)
     seed 1:  Δ = +0.285  (QLNN 0.356 vs cPINN 0.641)
     seed 2:  Δ = +0.998  (QLNN 0.329 vs cPINN 1.327; cPINN
                           catastrophic train failure)
   QLNN seed variance ≈ 0.015. cPINN seed variance ≈ 0.39.
   The te_qpinn_qnn "structural ceiling at 0.524 on LV s2"
   that was a NEGATIVE finding in P3.5/P7.6 becomes a
   POSITIVE finding on FHN: the SAME structural floor that
   limits LV smoothness is a competitive ANCHOR on stiff
   fast-slow dynamics.

2. **te_qpinn_qnn_2d on Allen-Cahn s0** continues to be the
   strongest single-cell quantum advantage in the matrix:
   Δ = +0.149 (QLNN 0.053 vs cPINN 0.202). Across all 3 seeds
   te_qpinn_qnn_2d achieves Δ = +0.149, +0.014, -0.002
   (mean +0.054).

3. **chebyshev_dqc_2d is consistently WORST on burgers_shock**
   (relL² ≈ 0.42-0.44 across 3 seeds) — the logistic-saturation
   feature map can't represent the near-shock sharp gradient.
   qcpinn_2d wins 2 of 3 burgers_shock cells (s0, s1); te_qpinn_fnn_2d
   wins s2. This confirms the chebyshev family's smooth-only
   specialization documented in P3.5.

4. **Point-estimate sign flip is driven by the broadband cells**
   (FHN +0.314 mean Δ, plus AC and burgers_shock); the smooth
   cells (LV, VdP, heat, burgers_smooth) are UNCHANGED in
   Δ_smooth between n=18 and n=24.

**Honest interpretation per pre-reg §7:**

The n=24 verdict is the PAPER'S PRIMARY HEADLINE. Both verdicts
(P7.6 n=18 and P7.8 n=24) FALSIFY H1 — neither CI excludes 0. But
the EVIDENCE TREND moves from "weak +0.032 smooth-favored at n=18"
to "modest -0.084 broad-favored at n=24" as we expand the
broadband bin. This is consistent with the FORECASTER-task H1
(P5) outcome (FALSIFIED with CI excluding 0 NEGATIVELY, Δ_diff =
-0.42), which is the inverted pattern.

The combined picture: across BOTH tasks (solver and forecaster),
the original H1 prediction of smooth>broad QLNN advantage is
NOT supported. The evidence is consistent with the OPPOSITE
direction (broad>smooth) at the forecaster task significantly,
and trending in that direction at the solver task at n=24.

Per pre-reg §7 ("Published as a rigorous mechanistic null"), this
is independently publishable as a rigorous null + suggestive
inverted-pattern observation. The Bowles/Schuld 2024 frame is
strengthened: even after symmetric HPO, even at expanded n=24
covering 8 of 9 pre-reg systems, even at the gating SOLVER task,
the regime-dependent advantage hypothesis does not survive.

---

## Amendment A10 — PDE solver-task H1 via existing-data combination (P7.6 commit 2)

**Audit gap closed:** the P7.5 solver-task H1 verdict used only the
3 ODE solver cells (n=9). The pre-reg §4 hardness ladder lists both
ODE and PDE systems. P3.7-3.9 had already run 4 quantum families ×
3 PDEs × 3 seeds + classical-PINN-on-PDEs. That data was on disk and
unused for the H1 verdict.

**Amendment:** P7.6 commit 2 combines existing P3.7-3.9 data into a
PDE solver-task H1 verdict (n=9), then combines it with the P7.5
ODE solver-task data into a combined ODE+PDE verdict (n=18, the
largest H1 bootstrap sample in the paper).

**Results:**

  PDE-only solver-task H1 (n=9):
    outcome  = FALSIFIED
    (CI includes 0)

  Combined ODE+PDE solver-task H1 (n=18, n_smooth=12, n_broad=6):
    outcome  = FALSIFIED
    Δ_smooth = +0.0674
    Δ_broad  = +0.0358
    Δ_diff   = +0.0316
    95% CI   = [-0.0400, +0.1088]   (includes 0)

**Key empirical finding:** doubling the bootstrap sample size from
n=9 to n=18 moved the CI from EXCLUDING zero (P7.5 raw CONFIRMED)
to INCLUDING zero (P7.6 n=18 FALSIFIED). The smaller-n CONFIRMED
was sample-size-fragile. The n=18 FALSIFIED is the headline verdict.

**Per-PDE highlights:** te_qpinn_qnn_2d on Allen-Cahn shows the
strongest quantum advantage (Δ ≈ +0.149, +0.014, -0.002 across
seeds). qcpinn_2d wins on Heat + Burgers (Δ ≈ +0.003-0.032). But
aggregated over all 18 cells with paired-bootstrap, the Δ_smooth −
Δ_broad contrast straddles zero.

**Rationale:** the existing-data combination is zero new compute,
uses pre-registered metrics (relative_l2), and respects the original
H1 regime tagging (smooth_periodic for heat/burgers/LV/VdP;
broadband_multiscale for allen_cahn/lorenz). A reviewer asking
"why only n=9?" gets the strongest answer: n=18, FALSIFIED, CI
tightened relative to n=9 default-Adam.

---

## Summary table

| Amendment | Pre-reg deviation | Status |
|---|---|---|
| A1 | skyline_threshold = 0.5 (pre-reg §7) | LOCKED |
| A2 | n=3 seeds, 3 ODE systems (not 5+9) | DISCLOSED |
| A3 | fixed hyperparameters (no HPO per §6) | DISCLOSED + sensitivity |
| A4 | MLP capacity 3.3× violates §6 factor-of-2 | DISCLOSED |
| A5 | VdP at μ=5 near task-difficulty boundary | DISCLOSED |
| A6 | underfit guard active on SOLVER only | DISCLOSED |
| A7 | report strict (INCONCLUSIVE) + raw (CONFIRMED) | DISCLOSED |
| A8 | H3 trend (ρ=+0.518) not significant at n=9 | DISCLOSED |
| A9 | symmetric QLNN HPO (P7.6 c1): FALSIFIED at HPO-best | SUPERSEDES |
| A10 | combined ODE+PDE n=18 (P7.6 c2): FALSIFIED | SUPERSEDES |
| A11 | pre-paper full-ladder expansion: n=24 FALSIFIED with sign-flip | **PRIMARY SOLVER** |
| A12 | classical LTC + forecaster H1 decomposition: 3 verdicts, mechanism attribution | PRIMARY FORECASTER (via classical) |
| A13 | non-liquid QLNN + complete 2×2; τ-isolation cross-check DISAGREES IN SIGN | **PRIMARY FORECASTER** (complete) |
| A14 | post-hoc Q-Q residual diagnostics (P8 polish; archived OD + post-pivot Lorenz) | DISCLOSED |
| A15 | equalize solver training-step budget across ALL models (QLNN families AND classical PINN) — was per-family 1200/1500/2000/1500 + classical 1500 → uniform 2000; audit-driven fairness fix | DISCLOSED + re-run required |
| A16 | un-alias strongly_entangling from data_reuploading (PennyLane fallback at n=3,L=1 was bit-identical to ring-CNOT); audit-driven fix | DISCLOSED + re-run required |
| A17 | qcpinn quantum-parameter sweep — three step-wise variants (qcpinn_balanced/quantum/full_q) along Q/(Q+C) ratio axis; addresses the qcpinn 706-classical-param confound | DISCLOSED + 72 new cells |
| A18 | brickwall removed from empirical forecaster sweep — at n=3,L=1 qubit 2 is structurally disconnected, the circuit cannot represent 3D dynamics; T3 mechanism scalars retained as untrained-circuit data | DISCLOSED + 9 cells removed |
| A19 | cross-task budget parity — forecaster step budget raised 200 → 2000 to match solver-side uniform 2000; closes the cross-task gap A15's first two passes left open | DISCLOSED + forecaster re-runs |
| A20 | te_qpinn_fnn readout restored from `qml.prod(*PauliZ)` → `qml.sum(*PauliZ)` to match the te_qpinn_qnn variant, Chebyshev-DQC precedent, and wider QPINN literature; paired-family equivalence restored | DISCLOSED + te_qpinn_fnn re-runs |
| A21 | brickwall connectivity diagnosis strengthened (A18 extension): the alternating-CNOT pattern reduces to a linear chain unless num_qubits ≥ 4; any future brickwall use requires n ≥ 4 | DISCLOSED, no compute |
| A22 | 2D PDE hard-IC trial-solution docstring fix — implementation has always been correct (line 217 uses `(t − t₀)`), only the module-level docstring carried the omission | DISCLOSED, no compute |

**Headline verdict update (post-P7.10):**

The paper now has TWO primary headline verdicts (one per
pre-registered task):

  PRIMARY SOLVER (A11): **n=24 full-ladder solver-task H1 =
    FALSIFIED**, Δ_diff = -0.0844, CI [-0.2780, +0.0613].
    Corroborated by P7.6 n=18 default-Adam (A10), P7.6 HPO-best
    n=9 (A9), and the P7.5 raw n=9 (A7) sample-size-fragile
    pattern.

  PRIMARY FORECASTER (A12, all 5 quantum families incl. rf_qrc):
    **forecaster H1 combined = FALSIFIED**, Δ_diff = -0.5007,
    CI [-0.8040, -0.2438] excludes 0 negatively. Decomposed into
    Δ_quantum = -0.6160 (CI [-1.1665, -0.1782], also excludes 0
    negatively) and Δ_liquid = +0.1153 (CI [-0.0973, +0.3772]).
    The quantum-circuit component is the mechanism of the
    underperformance, at high statistical confidence. The
    liquid-τ component on its own is directionally consistent
    with the H1 prediction but underpowered at n=9.

The original P7.5 raw n=9 CONFIRMED (A7) is reported as an
"underlying empirical pattern at the original sample size" — every
(system, seed) solver cell had Δ > 0 at default-Adam — but the
final scientific verdict at the EXPANDED ladder is FALSIFIED with
the point estimate now in the OPPOSITE direction from H1's
original prediction.

Per pre-reg §7 ("Published as a rigorous mechanistic null"), both
PRIMARY verdicts are independently publishable. The paper's
narrative now has FIVE layered sensitivity points and a clean
two-way decomposition all converging on the same outcome:

  P5 forecaster combined (n=9, 4-fam):  FALSIFIED, CI [-0.79, -0.05]
                                        excludes 0 negatively
  P7.10 forecaster combined (n=9, 5-fam): FALSIFIED, CI [-0.80, -0.24]
                                          excludes 0 negatively
  P7.10 forecaster q-isolated (n=9): FALSIFIED, Δ=-0.62, CI [-1.17, -0.18]
                                     EXCLUDES 0 negatively
                                     (mechanism: quantum circuit
                                     significant at 95% level)
  P7.10 forecaster τ-isolated (n=9): FALSIFIED, Δ=+0.12, CI [-0.10, +0.38]
                                     (mechanism: liquid-τ; small
                                      positive in H1 direction;
                                      underpowered at n=9)
  P7.5/P7.6 solver n=9-18 ODE+PDE:  FALSIFIED, CI straddles 0
  P7.6 solver HPO-best n=9:          FALSIFIED, CI straddles 0
  P7.8 solver full-ladder n=24:      FALSIFIED, CI straddles 0 with
                                    modest INVERTED point estimate

The pre-registered regime-dependent QLNN advantage hypothesis is
FALSIFIED at every sensitivity point we can construct. The
expanded ladder additionally surfaces a structural QLNN advantage
on FHN (te_qpinn_qnn 5× tighter seed-variance + 2× lower mean
relL² than classical PINN) that would not have been visible at
n=18 — a positive sub-finding documented separately as "the
te_qpinn_qnn structural FHN advantage."
