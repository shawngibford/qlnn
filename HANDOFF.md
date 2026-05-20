# ⏯️ PICK UP HERE — MAJOR PIVOT APPROVED (next-chat handoff)

**The project has PIVOTED.** The bioreactor-OD work is a rigorous null
on an n=1 dataset (not publishable positive) and is **archived, not
deleted** (`verify_paper_integrity` stays green; `PAPER_SUMMARY.md` /
3-claims framing **superseded**). The new project is a **Quantum Liquid
NN ODE/PDE solver+forecaster** across an ODE→PDE hardness ladder.

**Authoritative spec: the approved plan at
`~/.claude/plans/i-want-to-automate-quiet-cerf.md` ("# PIVOT — QLNN
ODE/PDE solver/forecaster"). Read it first.** `PROJECT_DOSSIER.md`
describes the *old* (now-superseded) program; keep for archive only.

### PIVOT pick-up order — ⏩ RESUME AT P3.7

**Branch note (read first):** the pivot lives on the worktree branch
that was fast-forwarded onto the pivot base `1eabdc2` (it carries the
ansatz registry / circuit-search / unified-matrix / synthetic_ode / T3
infra the plan says to reuse). `refs/` is gitignored, so its **PDF
symlinks must be recreated per worktree** — re-`ln -sf` the 8
`/Users/shawngibford/dev/phd/qlnn/*.pdf` files into `refs/` if absent.
The committed P3a `.md` evidence trail (force-added) travels with git.

- ✅ **P0 setup** — `refs/` symlinks recreated; 10-ansatz roster +
  P3a gate locked in the plan.
- ✅ **P1 DONE** (commit `2646d74`) — `ODE_PDE_PRE_REG.md`: falsifiable
  H1 (Schuld-Fourier regime partition stated as QLNN−NeuralODE
  advantage gap, confirm/falsify rules), solver+forecaster task defs,
  rollout/relative-L2/VPT metric set (1-step MAE banned), mandatory
  non-liquid Neural-ODE baseline, underfit/skyline guards.
  `verify_paper_integrity` exits 0 (OD frozen).
- ✅ **P3a DONE** (commit `51bee95`) — all 7 literature ansätze
  PDF/arXiv-grounded + **independently dual-verified**;
  `refs/CIRCUIT_SPECS.md` is the binding manifest P3 consumes.
  Gate caught & corrected two plan errors: `te_qpinn_qnn` source =
  `2605.13892v1` (NOT Berger s41598 — Berger has no quantum
  trainable embedding); `qcpinn` real PDEs =
  Helmholtz/cavity-NS/wave/Klein-Gordon/conv-diff. `reuploading.py`
  confirmed Schuld-faithful (H1 mechanism real in-code) with 2
  non-blocking P3 caveats logged in CIRCUIT_SPECS.md.
- ✅ **P2 DONE** (commit `11fc134`) —
  `src/quantum_liquid_neuralode/data_processing/pde_systems.py`:
  Fourier-spectral + Cox-Matthews integrating-factor RK4 (numpy-only,
  deterministic) for **burgers_smooth / burgers_shock / allen_cahn /
  kdv**. Emits npz FIELD artifacts (`u[t,x]` + grids + IC + periodic
  BC + invariants + sha256 lock), NOT the CSV seam. H1 regime tags
  bound in code + asserted vs the pre-reg. 16 validation tests green
  (Burgers gradient catastrophe at inviscid t*≈1 vs none smooth +
  mass<1e-8; Allen-Cahn narrow-front RELAXES to √2·eps, stationary,
  G-L energy strictly decreasing Lyapunov; KdV soliton conserves
  mass+momentum <5e-3, amplitude c/2, speed c, rel-L2<0.10).
  `scripts/generate_pde_data.py` → `data/pde/*.npz` + manifest
  (gitignored; script committed). Full suite 162 green;
  `verify_paper_integrity` exit-0.
- ✅ **P3 strand-1 DONE** (commit `77009ce`) — solver path + the
  acceptance gate. `src/qlnn_/training/physics_residual_loss.py`:
  Chebyshev-tower DQC circuit faithful to `CIRCUIT_SPECS.md` §5
  (Kyriienko 2011.10395 — tower Eq.15, HEA Rz-Rx-Rz+ring-CNOT Fig.5a,
  Σ⟨Z⟩ readout §III.3); **Lagaris hard-IC trial solution**
  u=u0+(t−t0)·N(t) (IC structural — NOT a soft penalty at the
  Chebyshev-singular x=−1 endpoint); interior collocation excludes the
  inherently-degenerate bare ±1. **THE NESTED AUTODIFF WORKS** (Risk
  #2 retired): grad over the param pytree of a loss containing
  `jax.jacrev` w.r.t. the scalar coordinate of the PennyLane JAX
  QNode — finite, converges. Gate test (3, green): `u'=−u` solved by
  physics residual alone, recovers `e^{−t}` to interior MAE ≈0.003
  (seed0, deterministic), ≤0.0074 across seeds {0,1,2}. Full suite
  165 green; `verify_paper_integrity` exit-0.
- ✅ **P3 strand-2 DONE** (commits `4d28914` → `0bc44f7`, 5 atomic).
  5 of 6 SOTA literature families implemented faithfully; 1 deferred
  with rationale; 2 `reuploading.py` caveats cleaned. **Cumulative
  strand-2 tests: 45 green** (8 reuploading + 7 rf_qrc + 15 te_qpinn
  FNN+QNN + 15 qcpinn) on top of strand-1's 3 solver-gate tests.
  Status table + per-family homing recorded in
  `refs/CIRCUIT_SPECS.md` "Implementation binding (P3 STATUS)".
  Highlights:
  - **chebyshev_dqc** (solver) — already shipped in P3-1.
  - **rf_qrc** (forecaster, fixed reservoir + closed-form Tikhonov
    ridge) — its own train path, NOT a registry ansatz.
  - **te_qpinn_fnn** (solver, Berger 2025 — classical-FNN trainable
    embedding). Paper anchor 3·n·L=60 @ n=4,L=5 verified.
  - **te_qpinn_qnn** (solver, **2605.13892** corroborated by 2602.*
    — fully-quantum trainable embedding; P3a-corrected source
    attribution). Linearity-in-N_q·L scaling asserted in both axes.
  - **qcpinn** (solver, 4 topologies). Paper p.15 worked anchors
    Cascade(n=5,L=1)→(15,5,7) and Cross-mesh(n=5,L=1)→(45,20,24)
    both verified at the test level via pennylane tape inspection.
  - **lubasch_multicopy DEFERRED** with cited rationale (schematic
    source — would exceed what the PDF specifies, violating P3a).
    Documented as "context/baseline only"; revisit only if a
    P6 ablation explicitly requires it.

  Architecture: registry contract reserved for forecaster encoders;
  solver families live as solver-style builders (interchangeable
  inside the strand-1 `make_residual_loss`/`train_solver` via the
  shared `params["w"]` pytree convention).
  Per-family commits + tests are visible in `refs/CIRCUIT_SPECS.md`
  "Implementation binding (P3 STATUS)".
- ✅ **P3.5 DONE** — first visible empirical result (commits
  `a0f08d5` → `a62477c`, 3 atomic). 4-family head-to-head on
  `u'=−u` + logistic `u'=u(1−u)` across 3 seeds; the
  `{w, s, b}` pytree interop pattern that test_qnn_drop_in_interop
  asserts in theory works in practice for all 4 families. Interior
  MAE summary (mean over seeds {0,1,2}):

  | family         | expdecay | logistic | params       |
  |----------------|----------|----------|--------------|
  | chebyshev_dqc  | 0.0058   | 0.0102   | 60 pqc       |
  | te_qpinn_fnn   | 0.0003   | 0.0008   | 60 pqc + 100 |
  | te_qpinn_qnn   | 0.0583   | 0.0351   | 84 pqc       |
  | qcpinn         | 0.0002   | 0.0014   | 15 pqc + 706 |

  Real findings (not just smoke): te_qpinn_fnn and qcpinn dominate
  but qcpinn does it with 706 classical params (disclosed in fig);
  chebyshev's logistic is ~2× worse than expdecay (predicted weakness
  at sigmoid plateaus where the Chebyshev tower saturates);
  te_qpinn_qnn underperforms uniformly with near-zero seed variance —
  a genuine trainability finding for P7's T3 triangulation.
  Figure: `paper/figures/fig_p3_solver_demo.{png,pdf}`. Library +
  CLI + figure script committed; full suite green;
  `verify_paper_integrity.py` exit-0 (demo intentionally NOT in
  the paper-integrity contract).
- ✅ **P3.6 DONE** — multi-state ODE solver (commits `6633355` →
  `3fa251a`, 3 atomic). Extends P3.5 to vector-state ODEs via
  per-component scalar circuits (no AnsatzProtocol refactor; no
  quantum entanglement across components — minimum-faithful
  extension). 4 families × 3 H1-relevant systems × 3 seeds = 36 runs.
  Relative-L2 summary (mean across seeds, lower=better):

  | family         | LV (d=2) | VdP (d=2) | Lorenz (d=3) |
  |----------------|----------|-----------|--------------|
  | chebyshev_dqc  | 0.106    | 0.989     | 0.999        |
  | te_qpinn_fnn   | 0.123    | 0.835     | 0.995        |
  | te_qpinn_qnn   | 0.524    | 1.044     | 0.978        |
  | qcpinn         | 0.0058   | 2.315     | 0.995        |

  Key findings:
  1. **Lorenz universally fails (relL2≈1.0)** across all 4 families —
     first pre-baseline H1 datapoint supporting the predicted
     chaotic-broadband failure boundary.
  2. **qcpinn dominates LV (relL2 0.005)** but its 1412 classical
     params (706 × d=2) dwarf the 30 PQC — R1 confirmed.
     chebyshev_dqc at relL2 0.10 is the pure-quantum baseline.
  3. **Van der Pol stiffness defeats everyone** at μ=5 over 10 time
     units; qcpinn overshoots; a real solver-path gap for P6.
  4. **te_qpinn_qnn reproduces its P3.5 flat-line ceiling** on the
     vector tasks (1.4964/1.4998/1.4944 MAE on LV) — robust T3
     signal across scalar AND vector solver tasks.

  Figure: `paper/figures/fig_p3_6_multi_state.{png,pdf}`. Per-component
  dispatch validated; gradient mass flows independently into each
  component's weights. 15 smoke tests green (~3m20s).
- ⏩ **P3.7 — NEXT. PDE solver scaffolding + nested-autodiff gate.**
  Add (t, x) coordinate handling so we can train against the P2 PDE
  fields. Heat-equation gate (`u_t = ν u_xx`, exact `u = e^{−νt}sin(x)`)
  must converge to MAE < 0.05 to unblock; if it fails, that's a real
  Risk-#2-redux confirmation and PDE work stops. First real PDE
  target: `burgers_smooth` (P2 npz; H1 SMOOTH/PERIODIC). New module:
  `src/qlnn_/training/pde_residual_loss.py` (~250 LOC) +
  heat-equation gate test (~100 LOC). Sibling of
  `physics_residual_loss.py`, not an extension. Allen-Cahn and KdV
  deferred to P6.
- **P4 — Forecaster long-horizon autoregressive rollout.**
  Retask the data-driven forecaster from the persistence-trivial
  h-step MAE protocol to **autoregressive multi-step rollout on the
  P2 PDE fields + the existing 5 ODE systems** (ODE_PDE_PRE_REG.md
  §3.2 / §5). Reuse the existing Diffrax QLNN forecaster
  (`src/qlnn_/forecaster.py`) for the forecaster-registry ansätze
  (data_reuploading / hardware_efficient / strongly_entangling /
  brickwall) and the new `rf_qrc` for the SOTA forecaster path.
  Required:
  - **Rollout-eval path** (not a destructive change to
    `make_horizon_windows`): a new evaluation module that takes a
    trained forecaster + an initial history + a rollout horizon, and
    returns the field/state trajectory + the pre-registered metrics
    (relative-L2 primary, VPT/Lyapunov for chaotic, spectral error,
    invariant drift). **NO 1-step MAE as a headline** (banned in P1).
  - **Task-dispatch wiring** in `train_qlnn.py` / `train_baseline.py`
    so a config can pick {ODE forecaster, PDE field forecaster}.
- **P5 → P6 → P7 → P8** per the plan. P5 adds the matched baselines
  incl. the MANDATORY non-liquid Neural-ODE (the H1 contrast — see
  ODE_PDE_PRE_REG.md §6). P6 is the gated/system-grouped unified
  matrix v2 — `ODE_PDE_PRE_REG.md` is already committed before any
  P6 run; no >30-min sweep without user go-ahead. P7 = T3
  triangulation across all 9 implemented families. P8 = new dossier.

### 1. There is a DETACHED background training job — do NOT wait on it
*(OLD Option-B program — now superseded by the pivot, but let it finish
cleanly; its results feed the archived dossier only, not the new paper.)*

### 1. There is a DETACHED background training job — do NOT wait on it

`O-2` Option-B sweep resume is running detached (survives chat end).
Status at handoff: **9/12 configs done**, 3 missing:
`se_6q3l__{R1_weight_decay,R2_physics_prior,R3_smooth_convergence}`
(the slow 6-qubit configs, ~1h each).

- **Check progress:**
  `ls results/option_b/*/seeds_summary.json | wc -l`  (target = 12)
- **If it died / to re-resume the missing ones** (idempotent — skips
  any already having `seeds_summary.json`):
  ```bash
  cd <repo>; export PYTHONPATH=$PWD/src
  for s in se_6q3l__R1_weight_decay se_6q3l__R2_physics_prior se_6q3l__R3_smooth_convergence; do
    [ -f results/option_b/$s/seeds_summary.json ] && continue
    .venv/bin/python scripts/train_qlnn.py --config configs/option_b/$s.yaml \
      --output-dir results/option_b/$s --quiet
  done
  ```

### 2. ⚠️ HARD RULE: never remove the `data` symlink while jobs run

`data -> /Users/shawngibford/dev/phd/qlnn/data` (qZETA + synthetic CSVs;
`data/` is gitignored). An earlier chat ran `rm -f data` to avoid
staging it and **killed the O-2 sweep mid-run**. Instead: leave the
symlink in place; commit with **explicit `git add <paths>`, never
`git add -A`/`.`** so it is never staged. If `data` is absent, recreate:
`ln -sfn /Users/shawngibford/dev/phd/qlnn/data data`.

### 3. When O-2 hits 12/12 — the immediate sequence (user-gated)

```bash
export PYTHONPATH=$PWD/src
.venv/bin/python scripts/summarize_option_b.py            # 12-row penalized table
.venv/bin/python scripts/build_master_comparison.py       # all-vs-all
.venv/bin/python scripts/make_diagnostic_figures.py       # renders T2 + master
```
Then **present the table + `fig_master_comparison` + top-3 to the user
and PAUSE for the tier-1 go/no-go** (5-seed promotion of top-3). Do not
auto-run tier-1.

### 4. Gated pipeline order (each step is a user go/no-go; nothing
contends with another)

O-2 (finishing) → **tier-1** (top-3 → 5-seed, `run_circuit_search.sh`
pattern) → **tier-2** (G1+G2 survivors → 4 SE fractions) → **T3 exec**
(`analyze_quantum_trainability.py`, ~2-4h) → **unified matrix**
(`run_unified_matrix.sh ONLY=<dataset>`, one of 11 groups per gate,
multi-day) → **separate horizon phase**. The Option-B gate is
`scripts/check_circuit_regression.py`.

### 5. Still to BUILD (no compute — safe anytime, do these next)

- **E-2 expressivity architecture extensions** (the user's "circuits not
  expressive enough" concern): (a) richer measurements ⟨ZZ⟩/⟨XX⟩ —
  **this breaks the locked `output_dim == num_qubits` cell contract**,
  so it needs the SAME backward-compat + integrity-gate rigor as the
  O-1 plumbing (default-off PauliZ, every committed claim untouched);
  (b) de-bottlenecked encoder (the 7→4 `tanh` squash is the prime
  suspect); (c) high re-upload (8/12 layers) + 8-qubit axes. Mirror the
  ansatz-registry / O-1 pattern.
- **E-3**: fold the E-2 axes into a gated expanded search, interpreted
  with the T3 curves (measure-before-scale — naive scaling → barren
  plateaus).

### 6. Verification gate before any commit

`pytest` (full), `scripts/verify_paper_integrity.py` exits 0 (the 3
locked claims must never move), figure scripts regenerate. The dossier
header (`PROJECT_DOSSIER.md` snapshot line + §13 status board) must be
bumped when O-2 completes or a verdict lands.

---

# Handoff to the next coding agent

You're picking up a paper-ready research codebase. Empirical work for
the 3 pre-registered claims is **complete**; the expansion work above is
in-flight/gated.

**Read this file first. Then read `PAPER_SUMMARY.md` and `hypothesis.md`.**
Don't re-litigate decisions documented as locked.

---

## State summary

- **23 commits** on `master`. Linear history; no branches.
- **131 / 131 pytest passing.**
- **`scripts/verify_paper_integrity.py`** exits 0 — every paper headline number
  is verified against on-disk JSON.
- Project is **paper-ready**, not paper-written. `PAPER_SUMMARY.md` ends with
  a suggested §1–§7 section mapping.

```
HEAD = 46b3492  fix: address Step 5/6 fresh-review BLOCKERs + key HIGHs
       806dac5  docs: CLAUDE.md — final-state overview ...
       c4049f6  docs+chore: paper-prep — README refresh, figures, ...
       910653d  docs: PAPER_SUMMARY — Claim 3 final verdict ...
       ...
       c784d30  init: step 1 — classical Liquid-ODE baseline finalized
```

---

## Locked decisions — do NOT relitigate

These were debated, decided, committed, and documented. Reopening them costs
context and time.

1. **QWGAN-GP (Step 4) is dropped.** Single-run dataset can't support
   "synthetic data improves forecasting" without a held-out second
   fermentation run. Documented in `hypothesis.md` v2 §"Deviations from v1".
2. **Stack is hybrid by design.** PyTorch + torchdiffeq for the classical
   baseline; JAX + Equinox + Diffrax + PennyLane for the QLNN. Both share
   data preprocessing, metrics, and bootstrap modules. Don't try to
   unify them.
3. **Evaluation protocol is locked.** Train-only OD MinMax + physical clip
   at 3.8. Window=24, stride=1. h=3 is the discriminating regime. 5 seeds.
   95% t-CI + paired bootstrap. Documented in `README.md` and `hypothesis.md`.
4. **`PAPER_SUMMARY.md` is the single source of truth for paper numbers.**
   `spec.md` is historical (pre-QWGAN-drop). README cites both, with
   PAPER_SUMMARY explicitly named as authoritative.
5. **`jax_enable_x64` stays OFF in the JAX subpackage.** Enabling it breaks
   Diffrax dtype promotion in the QLNN forecaster. The empirical-Fisher
   accumulator deliberately uses numpy float64 instead. See
   `src/qlnn_/diagnostics/effective_dimension.py` for the in-line comment.
6. **`delta_scale` is learnable, not a hardcoded 0.1.** The legacy
   `delta_scale=...` kwarg is accepted for back-compat in
   `LiquidODForecaster.__init__`. Don't remove the legacy alias.
7. **The "+physics" loss is logistic-growth ONLY.** A "smoothness" term
   that algebraically reduced to MSE was removed in Phase A. Don't add
   it back without a real multi-step trajectory output.
8. **Step 5 monotonicity criterion was corrected post-hoc** in
   `STEP5_MONOTONICITY_NOTE.md`. The pre-registration's "monotonic
   increasing" criterion was mathematically wrong for rank-deficient
   trained-θ Fisher (the typical case). The corrected criterion is
   "monotonic in either direction with shrinking successive gaps,"
   which the empirical results satisfy.

---

## Open items (in priority order)

These came out of the latest code review (`REVIEW_step56.md`). The
BLOCKERs and most paper-affecting HIGHs were already fixed in `46b3492`.
What's left:

### MEDIUM — would improve robustness but not paper-blocking

**M1. Cross-run determinism risk in `verify_paper_integrity.py` (was H-03).**
The integrity check reads Claim 1's σ from `results/param_sweep/euler_h3_hidden4/`
but Claim 3's 100% cell from `results/sample_efficiency/classical_h4_h3_pct100/`.
These are two independent training runs of the same config; MPS/BLAS
nondeterminism can make them drift apart. **Fix:** either (a) drop one of the
duplicate 100% runs and have the verifier read both numbers from one
canonical run, or (b) explicitly document the drift tolerance and widen
the integrity tol. Currently the verifier reads from the param_sweep
run for the σ ratio; the sample_efficiency 100% run is essentially
redundant.

**M2. `run_sample_efficiency.sh` duplicates the 100% cell (was H-04).**
The 100% data classical run already exists at
`results/param_sweep/euler_h3_hidden4/` and the 100% QLNN run at
`results/qlnn_hybrid_h3/`. The sample-efficiency sweep re-runs both
(~4hr QLNN cost). **Fix:** skip the 100% cells in the sweep runner and
have `summarize_sample_efficiency.py` read them from the canonical
locations. Saves ~80 min of QLNN training on every reproduce.

**M3. `STEP5_MONOTONICITY_NOTE.md` Case 2 is idealized (was MEDIUM).**
The "asymptote = effective rank" reading assumes a clean step in the
eigenvalue spectrum (r large eigenvalues, D−r zero). Real empirical
Fisher matrices have a continuous spectrum with a soft transition.
**Fix:** add a paragraph hedging the language — "the asymptote is the
effective dimensionality of the support of the spectrum" rather than
"the rank." The Step 5 claim isn't invalidated; the explanation just
needs more care before going into the paper.

**M4. `monotonic_increasing` field is misnamed under v2 criterion.**
`effective_dimension_curve()` in
`src/qlnn_/diagnostics/effective_dimension.py` returns a key called
`monotonic_increasing` (legacy from the pre-correction criterion). Now
that the corrected criterion is "monotonic in either direction with
shrinking gaps," the field should be renamed `monotonic_with_shrinking_gaps`
(or similar). Mirror the rename on the PyTorch side
(`src/quantum_liquid_neuralode/diagnostics/effective_dimension.py`).
**Fix carefully:** this field is read by `scripts/run_effective_dimension.py`
and the JSON output is committed in `results/effective_dimension/`. A
clean rename requires updating the script, regenerating the JSON, and
sanity-checking that `verify_paper_integrity.py` still passes.

**M5. `test_trace_normalization_invariance_torch` scope is too narrow (was H-06).**
It only verifies F → αF (constant scaling); it does NOT verify that
d_norm is invariant under genuine parameter reparametrization (which is
the property the trace normalization is supposed to provide). **Fix:**
either rename the test (`..._under_constant_scaling`) or extend it to
actually exercise a reparametrization (e.g., pass through a Jacobian
of a smooth change of variables on θ).

**M6. Clip-helper code duplication.** `clip_predictions_norm()` is
inlined in both `scripts/train_baseline.py` and `scripts/train_qlnn.py`
with identical bodies and a TODO. **Fix:** lift into
`src/quantum_liquid_neuralode/evaluation/clipping.py` and import in
both scripts. Pure refactor, no behavior change. Make sure the
provenance and integrity checks still pass after.

### LOW — nice-to-haves

- **L1.** `make_paper_figures.py` doesn't have a unit test. Consider a
  smoke test that runs it against synthetic seed-summary JSONs in a
  tmp dir and confirms the output files appear with sane sizes.
- **L2.** `paper/figures/` PDFs are checked into git. Acceptable for a
  research repo but bloats the diff. Consider gitignore'ing the PDFs
  and only versioning the PNGs.
- **L3.** Three results from Step 5 have `delta_r2_raw ≈ -3` (heavily
  negative). The paper §5 discussion should explain what this means
  (the model captures essentially no signal on the OD-change quantity,
  even though raw R² is positive). Not a code issue but a writeup risk.

---

## Files the next agent should know about

### Truth-source documents (read in this order)

1. **`PAPER_SUMMARY.md`** — every paper-table number with verdicts.
2. **`hypothesis.md`** (v2) — pre-registration. Read §"Deviations from
   v1" to understand why QWGAN was dropped.
3. **`README.md`** — entry point for any reader of the repo. Cites
   `PAPER_SUMMARY` as authoritative.
4. **`CLAUDE.md`** — repo overview / commands.
5. **`STEP5_MONOTONICITY_NOTE.md`** — the one methodology correction.

### Audit trail (skim if you need to understand why something is the way it is)

- `REVIEW_step1_classical.md` — Phase A review (PyTorch baseline)
- `REVIEW_step23_quantum.md` — Phase A/B/C review (JAX subpackage)
- `REVIEW_methodology.md` — peer-review-style audit that drove the QWGAN drop
- `REVIEW_integration.md` — cross-stack integration audit
- `REVIEW_SYNTHESIS.md` — synthesis that prioritized Phase A/B/C work
- `REVIEW_step56.md` — fresh review of Steps 5/6 (the most recent one)

### Headline result directories

```
results/baseline_classical_table/baseline_table.md      # paper Table 1 candidate
results/horizon_sweep_table/horizon_sweep_table.md      # paper Table 2 candidate
results/param_sweep/param_sweep_table.md                # paper Table 3 candidate
results/sample_efficiency/sample_efficiency_table.md    # paper Table 4 candidate (THE headline)
results/effective_dimension/effective_dimension.md      # paper Table 5 candidate
paper/figures/fig_horizon_ablation.{png,pdf}            # paper Figure 1
paper/figures/fig_sample_efficiency.{png,pdf}           # paper Figure 2 (THE headline)
paper/figures/fig_reproducibility.{png,pdf}             # paper Figure 3
paper/figures/fig_quantum_circuit.{png,pdf}             # paper Figure 4 — locked PQC architecture
paper/figures/fig_dataset_overview.{png,pdf}            # paper Figure 5 — qZETA + splits
paper/figures/fig_baseline_metrics.{png,pdf}            # paper Figure 6 — 4 metrics × all baselines
paper/figures/fig_param_sweep.{png,pdf}                 # paper Figure 7 — params Pareto
paper/figures/fig_horizon_full_metrics.{png,pdf}        # paper Figure 8 — h sweep, all 4 metrics
paper/figures/fig_sample_efficiency_full.{png,pdf}      # paper Figure 9 — sample-eff, all 4 metrics
paper/figures/fig_effective_dimension.{png,pdf}         # paper Figure 10 — Claim 2 d_norm curves
```

NOTE: `fig_dataset_overview` is the only figure that needs the raw CSV at
generation time. It skips with a clear warning if `data/raw/qZETA_data_copy.csv`
is missing (e.g. running inside a worktree where `data/` is gitignored).
Symlink the main-repo `data/` into the worktree to regenerate it.

---

## Circuit search (new — Phase 1 plumbing landed, Phase 2/3 are user-gated)

The QLNN's parameterized quantum circuit is now **pluggable** via an ansatz
registry (`src/qlnn_/circuits/protocol.py`). Four ansätze ship registered:
`data_reuploading` (the legacy default), `hardware_efficient`,
`strongly_entangling`, `brickwall`. YAML configs select one via an optional
`model.ansatz: {name, params}` block — absent = legacy data_reuploading,
which is why every existing checkpoint still deserializes (verified by
`tests/qlnn_/test_qlnn_forecaster_ansatz_swap.py`).

The search itself runs in two phases — both are *user-gated* because of the
8h overnight budget locked in the plan:

```bash
# Phase 2 — per-axis ablation grid (12 configs × ~5 min single seed ≈ 1 h).
bash scripts/run_circuit_search.sh
.venv/bin/python scripts/summarize_circuit_search.py
# → results/circuit_search/circuit_search_table.{md,json,csv}

# Phase 3 — Optuna Bayesian search on the most-informative axes
.venv/bin/python scripts/circuit_search_optuna.py --n-trials 50
# → results/circuit_search_optuna/trial_*/seeds_summary.json
# → results/circuit_search_optuna/qlnn_circuit_search_v1_top.json
# (resumable across sessions via the SQLite study DB)

# Both phases run at SINGLE SEED (proxy budget). The top-K circuits get
# promoted to the full 5-seed locked protocol BY THE USER in a separate
# session before any number lands in PAPER_SUMMARY.md.
```

Generators / scripts:
- `scripts/generate_circuit_search_configs.py` — single source of truth for the
  per-axis grid. Edit the dicts at the top to extend the grid.
- `scripts/run_circuit_search.sh` — shell loop over configs/circuit_search/*.yaml.
  Symlinks the existing 5-seed `results/qlnn_hybrid_h3/seed_0/` as the
  reference cell unless `REUSE_REFERENCE=0` is set.
- `scripts/summarize_circuit_search.py` — emits the per-axis table + JSON + CSV.
- `scripts/circuit_search_optuna.py` — TPE-sampler Bayesian search. Requires
  `pip install -e ".[search]"` (Optuna is an opt-in extra).

Figures (auto-skip until search results land):
- `paper/figures/fig_ansatz_axis_effects.{png,pdf}` — paper Figure 11
- `paper/figures/fig_circuit_pareto.{png,pdf}` — paper Figure 12

Risks / gotchas the next agent should know:
- `scripts/run_effective_dimension.py:_rebuild_qlnn` was taught to read the
  ansatz block from the saved YAML — needed for non-default circuits to
  deserialize their checkpoints correctly.
- `jax_enable_x64` stays off (locked decision #5). Verified that none of
  the new ansätze trip that.
- The proxy-budget numbers from Phases 2/3 are NOT paper-grade. Promotion
  to the full 5-seed locked protocol is the user's gate before
  `PAPER_SUMMARY.md` is updated.

### Reproduce pipeline

```bash
bash scripts/reproduce_paper.sh                # ~8 hours unattended
.venv/bin/python scripts/verify_paper_integrity.py    # ~5 seconds; must exit 0
```

---

## Gotchas the previous agent learned the hard way

These will save you debug time:

1. **Diffrax uses `custom_vjp` internally.** `jax.jacfwd` will fail with
   "can't apply forward-mode autodiff to a custom_vjp function." Use
   `jax.jacrev` instead. Already in place in
   `src/qlnn_/diagnostics/effective_dimension.py` and the analysis script.

2. **`jax.config.update("jax_enable_x64", True)` poisons Diffrax.**
   Symptom: `RuntimeError: buffer.at[i].set with mismatched dtypes`
   when the QLNN forecaster runs. Don't enable global x64. The empirical
   Fisher accumulation does numpy-float64-on-the-side instead.

3. **PennyLane returns a tuple of expectations.** `qml.expval(...)` inside
   a QNode body returns a `MeasurementProcess` symbol, not a JAX array.
   The current code returns `tuple(qml.expval(qml.PauliZ(i)) for i in
   range(n))` and stacks AFTER the qnode call. Don't try to stack inside.

4. **Equinox checkpoint dtype must match the rebuilt skeleton's dtype.**
   `eqx.tree_deserialise_leaves` is strict. If you rebuild a skeleton
   under different JAX config than the original save, you'll see "leaf
   has changed dtype from float64 to float32." Cast the skeleton
   before deserializing — see
   `scripts/run_effective_dimension.py:_rebuild_qlnn` for the pattern.

5. **`HorizonWindows` is frozen.** Use `head(n)` for chronological
   truncation; don't try to mutate the arrays in-place.

6. **The QLNN's `seed_0` checkpoint can re-init differently after a code
   change.** If you change `QLNNForecaster.__init__` parameter order or
   add new fields, the `best_model.eqx` snapshots in
   `results/qlnn_hybrid_*/seed_*/` may fail to deserialize. Run
   `bash scripts/reproduce_paper.sh` to regenerate them.

7. **`scripts/train_qlnn.py` saves `best_model.eqx`, not `best_state.pt`.**
   `scripts/run_effective_dimension.py` knows about both — keep that
   asymmetry if you refactor.

8. **A `.claude/` directory accumulates in the repo as the tool runs.**
   It's git-ignored. If you see it in `git status`, leave it alone.

9. **The dataset has 778 rows but `load_qzeta` does a DATE sort that
   produces 777 valid rows.** Don't be alarmed if integrity reports
   say 777.

---

## What "done" looks like for the next pass

After you tweak the code, the following must all hold (gate before
committing):

```bash
.venv/bin/python -m pytest                              # 131 tests pass
.venv/bin/python scripts/verify_paper_integrity.py     # exits 0
.venv/bin/python scripts/make_paper_figures.py         # writes all 3 figures
```

If you change anything that touches `PAPER_SUMMARY.md` numbers, regenerate
the figures AND run `verify_paper_integrity.py` AND update PAPER_SUMMARY
in the same commit. Otherwise the truth-source claim breaks.

---

## What NOT to do without checking with the user

- Don't pick up paper writing autonomously — it needs the user's voice.
- Don't re-add QWGAN-GP code. That decision is locked.
- Don't enable `jax_enable_x64` globally. See gotcha #2.
- Don't widen the integrity-check tolerances. They were tightened in
  `46b3492` for a reason.
- Don't delete the legacy `delta_scale=...` kwarg on `LiquidODForecaster`.
  Configs and scripts still use it; the alias prevents back-compat
  breaks.
- Don't run any sweep that takes >30 min without the user's go-ahead.
  The QLNN side is slow.

---

## If you only have time for one thing

**Fix M4 (rename `monotonic_increasing`).** It's the most visible
documentation-vs-code drift left in the repo, and the rename is small
and bounded. After the rename, regenerate the effective-dimension JSON
and re-run `verify_paper_integrity.py`. Should take ~15 minutes.

If you have more time, M1 + M2 together remove a ~80-minute reproduce
inefficiency and harden the integrity check — high value for low cost.

Good luck. The paper is in good shape.
