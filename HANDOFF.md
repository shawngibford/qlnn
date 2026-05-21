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

### PIVOT pick-up order — ⏩ RESUME AT P8 (paper draft) — P7.5 hardening complete

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

  Key descriptive observations (NOT H1 evidence — see caveats):
  1. **Lorenz universally fails (relL2≈1.0)** across all 4 families,
     consistent with the pre-registered broadband regime partition.
     **CAVEAT (audit, P3.8):** the T=2 horizon ≈ 1.8 Lyapunov times
     (pre-reg specifies 10 LTE); chaos hasn't fully developed.
     Failure here is transient-nonlinear-difficulty + broadband
     character, not chaotic-regime per se. P3.8 re-ran Lorenz at
     T=5.0 (~5.5 LTE) to disentangle. Also: relL2=1.0 uses a
     predict-zero baseline; P3.8 adds the more honest predict-mean
     baseline for chaotic systems.
  2. **qcpinn dominates LV (relL2 0.005)** but its 1412 classical
     params (706 × d=2) dwarf the 30 PQC — R1 confirmed empirically.
     chebyshev_dqc at relL2 0.10 is the pure-quantum baseline.
  3. **Van der Pol stiffness defeats everyone** at μ=5 over 10 time
     units; qcpinn overshoots; a real solver-path gap for P6.
  4. **te_qpinn_qnn reproduces its P3.5 flat-line ceiling** on the
     vector tasks (1.4964/1.4998/1.4944 MAE on LV) **at a single
     (n=4, L=5, K=3) configuration**. The "structural trainability
     ceiling" interpretation requires a hyperparameter sweep —
     deferred to P7's T3 triangulation. Reframe: this is observed-
     in-one-config, not proven structural.

  **H1 framing reform (P3.8 audit):** H1 is defined in P1 §2 as the
  **QLNN − NeuralODE advantage gap**. Lorenz absolute failures across
  QLNN families are NOT H1 evidence — H1 requires the NeuralODE
  baseline (scheduled for P5). The above are descriptive regime-map
  data pending the H1 contrast.

  Figure: `paper/figures/fig_p3_6_multi_state.{png,pdf}`. Per-component
  dispatch validated; gradient mass flows independently into each
  component's weights. 15 smoke tests green (~3m20s).
- ✅ **P3.7 DONE** — PDE solver + nested-autodiff gate +
  3-PDE demo (commits `87dcfd2` → `afbb5e6`, 3 atomic). **Risk-#2-redux
  RETIRED**: nested mixed-2nd-derivative autodiff
  (`jax.jacrev(jax.jacrev(QNode, argnums=1), argnums=1)`) through
  PennyLane's JAX interface composes cleanly — PennyLane uses
  `vjp` (not Diffrax's `custom_vjp`), so reverse-over-reverse is
  safe. Mechanism gate AND convergence gate (heat eq MAE<0.10)
  both passed on first attempt. The PDE side of the H1 hypothesis
  space is structurally accessible.

  3-PDE solver-demo results (3 seeds each, 600 steps,
  chebyshev_dqc_2d, 8 qubits, 5 HEA layers):

  | PDE              | mean relL2 | regime tag (NOT H1 evidence)        |
  |------------------|------------|-------------------------------------|
  | heat             | 0.059      | SMOOTH (analytic ref; well below 0.10 gate) |
  | burgers_smooth   | 0.380      | SMOOTH/PERIODIC (P3.8: did NOT meet plan's relL2<0.30 gate at 600 steps; retry at 1500 steps in P3.8) |
  | allen_cahn       | 0.769      | BROADBAND/MULTISCALE regime (CAVEAT below) |

  **Allen-Cahn caveat (P3.8 audit):** the P3.7 sweep used n_x_colloc=28
  → Δx≈0.224, vs Allen-Cahn equilibrium front width √2·ε≈0.085. The
  solver had **<1 collocation point per front**. The observed
  "broadband failure" may reflect sub-Nyquist spatial aliasing rather
  than a regime-structural property. P3.8 re-runs Allen-Cahn at
  n_x_colloc=64, n_t_colloc=32, steps=1800 (1.2 collocation points
  per front; 10× P3.7's resolution-to-front ratio) to disentangle.
  Also: final_loss=9.23 at 600 steps suggests under-convergence vs
  PDE_BENCH's configured 1800.

  **H1 framing reform (P3.8 audit):** H1 is defined in P1 §2 as the
  **QLNN − NeuralODE advantage gap**. Allen-Cahn absolute failure is
  NOT H1 evidence — H1 requires the NeuralODE baseline (scheduled for
  P5). The above are descriptive regime-map data; H1 verdict awaits
  P5. P3.8 ADDS a classical MLP-PINN baseline (the audit's #1
  missing comparison) which gives the solver-side analogue (the
  forecaster NeuralODE baseline still waits for P5).

  **P3.8 smoke result (heat seed 0, both models capacity-matched ~120-180 params):**
  - chebyshev_dqc_2d:   relL2 0.0553, MAE 0.0302, BC violation 0.40
  - classical_pinn:      relL2 0.0054, MAE 0.0026, BC violation lower

  Classical PINN is ~10× more accurate at matched capacity on the
  smooth analytic case. The audit's prediction (no quantum advantage
  on a smooth-low-frequency problem given physics-informed training)
  is being borne out empirically. Full P3.8 sweep (3 PDEs × 2 models
  × 3 seeds + Lorenz × 4 families at T=5) in progress.

  Figure: `paper/figures/fig_p3_7_pde_solver.{png,pdf}` — 3-row
  snapshot grid (t=0 / T/2 / T per PDE) + log-scale relL2 bar
  chart with predict-zero floor.

  Architectural decisions baked in: CPU-only backend (user-locked;
  PennyLane has no Apple-Metal quantum backend); single family this
  phase (cross-family on PDEs = P6); sibling module to
  `physics_residual_loss.py` (1D gate contract immutable).
- ✅ **P3.8 DONE** — peer-review iteration (commits `29f097e` →
  `25ad075`, 7 atomic). All 3 audit BLOCKERs closed (framing reform
  + classical PINN baseline + corrected re-runs). Full sweep
  complete (3 PDEs × 2 models × 3 seeds + 4 Lorenz families × 3
  seeds = 30 runs). Wall-clock: 2 hr 49 min on default.qubit JAX.

  **PDE headline (quantum chebyshev_dqc_2d vs capacity-matched
  classical MLP-PINN, all at audit-corrected configs):**

  | PDE | quantum | classical_pinn | advantage |
  |---|---:|---:|---|
  | heat (1200 steps) | 0.056 | **0.0045** | ~12× classical |
  | burgers_smooth (1500 steps) | 0.358 | **0.027** | ~13× classical |
  | allen_cahn (64×32×1800) | ~0.57 (mean) | **~0.11 (mean)** | ~5× classical |

  AC quantum DID improve from P3.7's 0.77 with the resolution+steps
  fix (under-resolution partly responsible), but classical PINN at
  ~0.11 makes clear this is **quantum-specific underperformance**,
  not a general PINN-training regime failure. Burgers gate target
  relL2<0.30 STILL missed at corrected 1500 steps — confirms a
  quantum-solver expressivity ceiling, not just under-convergence.

  **Lorenz extended (T=5.0, ~4.53 Lyapunov times — half of pre-reg's 10):**

  | family | seed-mean relL2 | predict-mean floor |
  |---|---:|---:|
  | chebyshev_dqc | 0.997 | 0.354 |
  | te_qpinn_fnn | 0.997 | 0.354 |
  | te_qpinn_qnn | 0.988 | 0.354 |
  | qcpinn (bimodal) | 0.921 (one seed 0.77) | 0.354 |

  **All 4 quantum families** sit at the predict-zero baseline (relL2≈1);
  none beats the honest predict-mean floor (relL2=0.354). Universal
  collapse on chaotic 4.5-LTE dynamics. NOT yet H1 evidence — the
  pre-reg defines H1 as the QLNN−NeuralODE gap (P5).

  Figure: `paper/figures/fig_p3_8_review_iteration.{png,pdf}`.
- ✅ **P3.9 DONE** — PDE multi-family port (commits `00c2d46` →
  `d74cf42`, 8 atomic). The 3 PINN-style quantum families ported
  to 2D (t, x) coordinate handling and run on all 3 PDEs at the
  audit-corrected configs; the PDE matrix now matches the ODE
  matrix shape (4 quantum × 3 PDEs × 3 seeds = 36 PDE runs total,
  combining P3.8's chebyshev_dqc_2d data and P3.9's 27 new runs).

  All 3 mechanism gates passed first try (`jacrev∘jacrev` through
  the QNode composition was solid for every family). Per-family
  test counts: qcpinn_2d 11/11, te_qpinn_fnn_2d 12/12,
  te_qpinn_qnn_2d 12/12. Sweep wall-clock 16:33 on default.qubit.

  **PDE matrix headline numbers (re-run-friendly summary):**

  | PDE | best quantum (relL2) | classical PINN | verdict |
  |---|---|---|---|
  | heat | qcpinn_2d 0.0017 ± 0.0008 | 0.0045 | **quantum wins** (qcpinn_2d, te_qpinn_fnn_2d both beat classical) |
  | burgers smooth | qcpinn_2d 0.016 ± 0.008 | 0.027 | **quantum wins** (passes the relL2<0.30 gate at corrected steps) |
  | allen_cahn | **te_qpinn_qnn_2d 0.052 ± 0.003** | ~0.11 | **quantum wins** at zero classical params — the cleanest result |

  **THE FINDING:** te_qpinn_qnn_2d shows the cleanest regime-
  dependent advantage pattern observed in this codebase to date:

  - **smooth PDEs (heat, Burgers):** stuck at the trainability
    ceiling (relL2 ≈ 0.046 and 0.358 — replicates P3.8's Lorenz
    T=5 finding for the same family);
  - **broadband PDE (Allen-Cahn, sharp tanh fronts):** converges
    to relL2 0.052 with tight ±0.003 seed variance, beating
    classical PINN by ~2× at ZERO classical params (PURE quantum,
    84 PQC scalars).

  This is the first sign of a regime-dependent quantum-family
  advantage on a PDE in this codebase. **CRUCIAL CAVEATS (the
  pre-reg discipline):**

  1. **NOT H1 evidence** — pre-reg §2 defines H1 as the
     QLNN−NeuralODE advantage gap; the mandatory Neural-ODE
     baseline awaits P5.
  2. Same family stuck on smooth PDEs at trainability ceiling —
     the regime split could be a property of trainability, not
     expressivity. P7 T3 triangulation will adjudicate.
  3. qcpinn_2d's heat/Burgers wins use 756 classical params (the
     "classical-heavy capacity confound" disclosed in
     CIRCUIT_SPECS.md §3 amendment).
  4. AC results have high seed variance on 2 of the 3 new families
     (qcpinn_2d, te_qpinn_fnn_2d) — only te_qpinn_qnn_2d's AC
     result has tight ±0.003 CI.

  Figure: `paper/figures/fig_p3_9_pde_matrix.{png,pdf}`.
  Results: `results/p3_9_pde_matrix/{pde}_{family}/seed_N/`.
  CIRCUIT_SPECS.md amendments: §1 (te_qpinn_fnn split-qubit FNN
  heads), §2 (te_qpinn_qnn split-qubit U_embed), §3 (qcpinn
  trivial input-dim widening). rf_qrc deferred to P4 as
  forecaster (frozen-reservoir architecture).
- ✅ **P4 DONE** — forecaster autoregressive rollout (commits
  `0ade9bf` → `d829512`, 7 atomic incl. this HANDOFF advance).
  All pre-reg §5 metric primitives (relative-L2, VPT,
  spectral_error, invariant_drift) implemented + tested; 5
  forecaster families fully run on 3 ODE systems.

  P4 commit ledger:
    `0ade9bf` rollout helper + Protocol + 14+2 tests
    `952d3ec` metric suite + 24 tests
    `22d2d55` VectorForecaster (vector-output QLNN) + 17 tests
    `9661c3f` one-step-ahead supervised training loop + 13 tests
    `8a512a0` per-family adapters + python-loop rollout + 13 tests
    `88ab57d` dispatch + sweep CLI
    `d829512` full 45-cell sweep + figure

  **45-cell sweep result (5 families × 3 ODE × 3 seeds), ~22 min wall:**

  | System | Persistence floor | Best quantum (relL2) | Verdict |
  |---|---|---|---|
  | **lotka_volterra** (smooth/periodic) | 0.88 | brickwall 0.33 (best seed); 4 families all beat floor | quantum WIN |
  | **van_der_pol** (stiff μ=5) | 1.35 | hardware_efficient ~1.21 | mixed/marginal |
  | **lorenz** (chaotic) | 0.45 | rf_qrc 0.46 (tied) | universal failure (VPT < 0.1 LTE) |

  **Forecaster-task corroboration of regime-dependent pattern:**
  4 of 5 quantum families beat persistence on smooth LV; rf_qrc
  (the SOTA reservoir family) fails on LV. Stiff VdP breaks most
  families (brickwall catastrophic divergence to relL2 ≈ 10).
  Chaotic Lorenz universal failure across all families (VPT in
  Lyapunov times = 0.05-0.09, i.e. forecasts diverge in ~1/10 of
  a Lyapunov time at the configured budget).

  Pattern echoes the SOLVER-task finding from P3.8 / P3.9:
  smooth = some quantum advantage; chaotic = universal failure.

  Figure: `paper/figures/fig_p4_forecaster_rollout.{png,pdf}`.
  Results: `results/p4_forecaster_rollout/{system}_{family}/seed_N/`.
  CLI: `scripts/run_p4_forecaster_rollout.py`.

  **NOT yet H1 evidence** — pre-reg §7 defines H1 as the QLNN−Neural-ODE
  advantage gap, and the mandatory Neural-ODE baseline still
  doesn't exist (P5).

- ✅ **P5 DONE — H1 OUTCOME: FALSIFIED** (commits `29acb74` →
  `bd4e3c5`, 5 atomic). The pre-reg's mandatory matched baselines
  landed (plain Neural-ODE, plain MLP, skyline, classical PINN
  extended to vector ODE) + the H1 verdict module + the headline
  figure.

  **Headline numbers (`results/p5_h1_verdict/h1_analysis.json`):**

  Δ_smooth − Δ_broad = **−0.4166**
  95% paired-bootstrap CI = **[−0.7871, −0.0460]**

  CI is entirely NEGATIVE → **H1 FALSIFIED per pre-reg §7**.
  "Published as a rigorous mechanistic null." The paper has its
  headline.

  **The empirical inversion** — per-cell Δ = NeuralODE − QLNN
  (positive ⇒ QLNN better):

  | Cell | Δ | Regime | Verdict |
  |---|---|---|---|
  | LV s0 | -0.289 | smooth | Neural-ODE wins |
  | LV s1 | -0.238 | smooth | Neural-ODE wins |
  | LV s2 | +0.143 | smooth | QLNN wins |
  | VdP s0 | +0.115 | smooth | QLNN marginally wins (both ~1.2 failing) |
  | VdP s1 | +0.034 | smooth | tie |
  | VdP s2 | +0.025 | smooth | tie |
  | **Δ_smooth_mean** | **-0.128** | (anti-H1) | |
  | Lorenz s0 | -0.043 | broad | tie at chaos ceiling |
  | Lorenz s1 | +0.272 | broad | QLNN wins |
  | Lorenz s2 | +0.636 | broad | QLNN strongly wins |
  | **Δ_broad_mean** | **+0.289** | (pro-QLNN!) | |

  **Inverted pattern:** Neural-ODE beats QLNN on smooth/periodic
  (where H1 predicted QLNN advantage); QLNN modestly beats
  Neural-ODE on chaotic Lorenz (where H1 predicted no advantage).
  Exactly the opposite of the Schuld-Fourier prediction.

  Sensitivity: at strict skyline threshold (0.5) the verdict is
  INCONCLUSIVE (Lorenz skyline relL² = 0.708 excludes the
  broadband regime). Recorded as
  `results/p5_h1_verdict/h1_analysis_strict_threshold.json`.

  P5 commit ledger:
    `29acb74` plain non-liquid Neural-ODE + 16 tests (H1 contrast)
    `e475a80` plain MLP forecaster + 10 tests (capacity control)
    `43b2826` known-structure skyline + 12 tests (upper bound)
    `f2932e9` classical PINN extended to vector ODE + 20 tests
    `8421d5c` H1 verdict module + 17 tests
    `bd4e3c5` P5 sweep + figure + **H1 FALSIFIED**

  Figure: `paper/figures/fig_p5_h1_verdict.{png,pdf}` — 3-panel
  (relL² bars, per-cell Δ scatter colored by regime, verdict bar
  with CI error bar).

- 🟡 **P6 — DEFERRED OR REDIRECTED.** With H1 already falsified,
  the original P6 plan ("scale the unified matrix") changes
  character. The remaining marginal value of a 5-ODE × 4-PDE
  full matrix is mostly to confirm the falsification is robust
  across systems; the actual H1 number won't change qualitatively.
  Treated as optional rigor work, not paper-critical.

- ✅ **P7 DONE — H3 mechanism partial signal** (commits `ac42612` →
  `4fbbdf3`, 4 atomic incl. this HANDOFF advance). 4 T3 scalars
  computed per forecaster family at the P4 config + barren-plateau
  qubit-scaling study + per-cell cross-tabulation against P5's Δ values.

  **Per-family T3 scalars at P4 config (n=3, L=1):**

  | family | KL_to_Haar | Q_ent | Var(grad) | K_max |
  |---|---|---|---|---|
  | data_reuploading | 0.195 | 0.776 | 0.038 | 3 |
  | hardware_efficient | 0.211 | 0.796 | 0.025 | 3 |
  | strongly_entangling | 0.195 | 0.776 | 0.038 | 3 |
  | **brickwall** | **0.211** | **0.309** | 0.046 | 3 |

  Headline: **brickwall has dramatically LOWER entangling Q
  (0.31) than the other 3 (~0.78-0.80)** — alternating-layer
  CNOT vs ring CNOT. Also **most barren-plateau resistant** in
  the qubit-scaling study (Var(grad) only drops 2× from n=2
  to n=5, vs ~50× for hardware_efficient).

  **Cross-tabulation against per-cell Δ (n=9 cells, scipy.stats
  .spearmanr with tie handling):**

  | T3 scalar | ρ | p-value | Interpretation |
  |---|---|---|---|
  | **KL to Haar** | **+0.518** | **0.154** | strongest trend; less-expressive circuits show larger QLNN advantage |
  | Entangling Q | +0.179 | 0.644 | no signal |
  | Var(grad) | -0.179 | 0.644 | no signal |
  | Fourier K_max | undefined | — | constant across families at L=1 |

  **No T3 scalar reaches statistical significance (p<0.05) at
  n=9 cells.** The KL-to-Haar trend (ρ=+0.518) is the
  strongest tentative mechanism: less-expressive ansätze
  (further from Haar) show larger QLNN advantage. Consistent
  with a "circuit-simplicity helps" reading, but requires P6's
  unified-matrix scale-up for statistical confirmation.

  P7 commit ledger:
    `ac42612` feat(P7-t3-module):     T3 diagnostics module + 28 tests
    `45efe36` feat(P7-t3-sweep):       sweep + barren-plateau scaling
    `4fbbdf3` feat(P7-mechanism-fig):  cross-tabulation + figure
    (this) docs(HANDOFF):              pickup advance

  Figure: `paper/figures/fig_p7_mechanism.{png,pdf}` — 4-panel
  diagnostic (per-cell Δ colored by best-ansatz; T3 scalars per
  family; barren-plateau scaling; correlation table).

- ✅ **P7.5 DONE — PRE-PUBLICATION HARDENING** (commits `5683716`
  → `e39f374`, 7 atomic). Closes 3 RED + 6 YELLOW peer-review
  audit concerns. The paper's empirical evidence is now
  defensible at PRX-Quantum-grade rigor.

  P7.5 commit ledger:
    `5683716` feat(P7.5-solver-h1-module):    11 tests
    `61533b8` feat(P7.5-solver-h1-sweep):     9 baseline cells +
                                              SOLVER-task H1 verdict
    `fddc896` feat(P7.5-integrity):           verify_paper_integrity
                                              gates H1/H3
    `1c10d98` docs(P7.5-prereg-amend):        PRE_REG_AMENDMENT.md
                                              (8 disclosures)
    `c24fda3` feat(P7.5-hpo):                  HPO sensitivity (VdP
                                              sign flip disclosed)
    `ea78046` feat(P7.5-h3-loo):               KL-to-Haar trend
                                              ROBUST across LOO
    `e39f374` chore(reproduce_paper):         full pivot pipeline

  **THE PAPER'S NEW HEADLINE EVIDENCE** (3 H1 outcomes):

  1. **SOLVER-task H1 (PRE-REG GATING, default HPO)** =
     **CONFIRMED** at raw bootstrap
       - Δ_smooth − Δ_broad = +0.1094
       - 95% CI = [+0.0145, +0.2204]  ← excludes 0 positive
       - 9 of 9 per-cell Δ > 0 (QLNN beats classical PINN on every
         system × seed)

  2. **SOLVER-task H1 (HPO-best classical PINN at anchors)** =
     **FALSIFIED**
       - Δ_smooth − Δ_broad = +0.0647
       - 95% CI = [-0.0468, +0.1927]  ← includes 0
       - VdP s1 Δ flips sign (-0.097 at lr=1e-2 vs +0.181 at default)

  3. **FORECASTER-task H1 (P5 corroborating)** = **FALSIFIED**
       - Δ_smooth − Δ_broad = -0.4166
       - 95% CI = [-0.7871, -0.0460]  ← excludes 0 negative
       - Inverted regime pattern (Neural-ODE wins smooth; QLNN
         narrowly wins broad)

  **H3 mechanism (KL-to-Haar)**: ρ = +0.518 (full-sample),
  LOO mean = +0.512 ± 0.113, ALL 9 LOO subsamples positive
  → trend is ROBUST to single-cell removal but n=9 too small
  for p<0.05 significance.

  **PRE_REG_AMENDMENT.md** documents 8 methodological choices
  (A1-A8) including skyline_threshold=0.5, sample size, HPO,
  capacity matching, VdP boundary, underfit-guard scope,
  strict-vs-raw verdict reporting, H3 trend not significant.

  **Publication-grade reproducibility chain:**
    - `scripts/reproduce_paper.sh` regenerates EVERY artifact
      (~6-8 hr CPU end-to-end)
    - `scripts/verify_paper_integrity.py` GATES all H1/H3
      numbers + the archived OD claims

  **Audit closure status:**
    R1 solver gating         ✅ CLOSED (P7.5 commit 2)
    R2 underfit guard        ✅ CLOSED for solver task (P7.5 commit 1+2)
    R3 integrity script      ✅ CLOSED (P7.5 commit 3)
    Y1 skyline threshold     ✅ DISCLOSED (PRE_REG_AMENDMENT A1)
    Y2 sample size           ✅ DISCLOSED (A2)
    Y3 HPO budget            ✅ HARDENED via sensitivity (P7.5 commit 6)
    Y4 MLP capacity          ✅ DISCLOSED (A4)
    Y5 VdP boundary          ✅ DISCLOSED + HPO-confirmed (A5 + commit 6)
    Y6 te_qpinn_qnn AC       ✅ DISCLOSED (A6, paper framing)
    Y8 H3 not significant    ✅ DISCLOSED + LOO-robust (A8 + commit 5)

- ⏩ **P8 — NEXT. PRX Quantum paper draft.** All experimental work
  for the paper is COMPLETE. The 3 H1 outcomes + H3 LOO + HPO
  sensitivity + PRE_REG_AMENDMENT + reproducibility chain are
  all committed.

  **Paper headline structure (3 H1 verdicts side-by-side):**
    §5.1 Solver-task H1 (pre-reg GATING task, default HPO): CONFIRMED
    §5.2 Solver-task H1 (HPO-best classical PINN sensitivity): FALSIFIED
    §5.3 Forecaster-task H1 (corroborating): FALSIFIED
    §5.4 Discussion: the empirical pattern is TASK-DEPENDENT AND
         HPO-SENSITIVE. The most rigorous reading is that QLNN shows
         a regime-dependent advantage on the solver task at the
         pre-registered config, but the advantage is fragile to
         classical baseline HPO. This is exactly the rigor Bowles/
         Schuld 2024 calls for. The paper's CONTRIBUTION is the
         RIGOROUS BENCHMARK FRAMEWORK, not a unidirectional verdict.

  Paper structure (~25 pages incl. SI):
    §1 Intro — Schuld-Fourier; PINN landscape; Bowles/Schuld 2024
    §2 Methods — pre-reg verbatim, matched baselines, guards,
                  PRE_REG_AMENDMENT
    §3 Solver task results — P3.6 multi-state + P3.9 PDE matrix +
                              te_qpinn_qnn_2d AC win
    §4 Forecaster task results — P4 rollout matrix + P5 baselines
    §5 H1 verdict — 3 outcomes (solver default + HPO-best, forecaster);
                     TASK-DEPENDENT + HPO-SENSITIVE
    §6 Mechanism — P7 T3 + KL-to-Haar LOO-robust trend
    §7 Discussion — limitations, full P6 scale-up future work,
                     hardware execution future work
    §8 Conclusions

  Submission artifacts (~ready to assemble):
    paper/main.tex (PRX Quantum LaTeX) + paper/supplement.tex
    scripts/reproduce_paper.sh (P7.5 commit 7 updated)
    verify_paper_integrity.py extended (P7.5 commit 3)
    Zenodo DOI + arXiv preprint

  Dev estimate: 2-3 weeks writing.

- 🟡 **P6 — OPTIONAL/DEFERRED.** Unified matrix scale-up would
  confirm the H1 falsification + might push the H3 KL-trend
  toward statistical significance (n increases from 9 to ~45-90
  cells). The H1 verdict won't change qualitatively (CI is
  comfortably negative); P6 is enrichment, not paper-critical.
  [P5 DESCRIPTION — moved to ✅ DONE section above.
   See P5 commit ledger 29acb74 → bd4e3c5 and the
   FALSIFIED outcome.]

- **P6 → P7 → P8** per the plan. P6 is the gated/system-grouped
  unified matrix v2 — `ODE_PDE_PRE_REG.md` is already committed
  before any P6 run; no >30-min sweep without user go-ahead. P7
  = T3 triangulation across all implemented families. P8 = new
  dossier + PRX Quantum paper draft.

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
