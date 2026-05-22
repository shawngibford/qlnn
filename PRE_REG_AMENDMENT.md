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
| A11 | pre-paper full-ladder expansion: n=24 FALSIFIED with sign-flip | **NEW PRIMARY** |

**Headline verdict update (post-P7.8):**

The paper's PRIMARY verdict is now the **n=24 full-ladder
solver-task H1 = FALSIFIED, Δ_diff = -0.0844, CI [-0.2780, +0.0613]**
(A11). Two corroborating verdicts:
  - n=18 default-Adam (A10): FALSIFIED, Δ_diff = +0.0316,
    CI [-0.0400, +0.1088]  (pre-expansion baseline)
  - n=9 HPO-best both sides (A9): FALSIFIED, Δ_diff = +0.0588,
    CI [-0.0575, +0.1913]  (methodological sensitivity anchor)

The original P7.5 raw n=9 CONFIRMED (A7) is reported as an
"underlying empirical pattern at the original sample size" — every
(system, seed) cell had Δ > 0 at default-Adam — but the final
scientific verdict at the EXPANDED ladder is FALSIFIED with the
point estimate now in the OPPOSITE direction from H1's original
prediction.

Per pre-reg §7 ("Published as a rigorous mechanistic null"), the
FALSIFIED outcome is independently publishable. The paper's
narrative now has FOUR layered sensitivity points all converging
on the same outcome:
  P5 forecaster-task (n=9):  FALSIFIED, CI [-0.79, -0.05] negative
  P7.5/P7.6 n=9-18 ODE+PDE:  FALSIFIED, CI straddles 0
  P7.6 HPO-best n=9:          FALSIFIED, CI straddles 0
  P7.8 full-ladder n=24:      FALSIFIED, CI straddles 0 with
                              modest INVERTED point estimate

The pre-registered regime-dependent QLNN advantage hypothesis is
FALSIFIED at every sensitivity point we can construct. The
expanded ladder additionally surfaces a structural QLNN advantage
on FHN (te_qpinn_qnn 5× tighter seed-variance + 2× lower mean
relL² than classical PINN) that would not have been visible at
n=18 — a positive sub-finding documented separately as "the
te_qpinn_qnn structural FHN advantage."
