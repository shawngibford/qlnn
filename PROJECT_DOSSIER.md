# PROJECT DOSSIER — Classical Liquid-ODE vs Quantum Liquid Neural Network forecasting

> **Snapshot:** generated at commit `c80c021`, Option-B sweep `O-2` at 8/12
> configs complete (resuming). Numbers in §2 and §5 are transcribed
> verbatim from `verify_paper_integrity`-checked sources — not recomputed.
> DONE / IN-FLIGHT / GATED is labelled explicitly throughout (see §13).
> This document is self-contained: a model with no repo access can design
> the paper from it alone.

---

## 1. Thesis (one paragraph)

A head-to-head comparison of a **classical Liquid Neural ODE** forecaster
and a **Quantum Liquid Neural Network (QLNN)** forecaster on a 778-row,
single-fermentation-run bioreactor optical-density (OD) dataset. The
stack is hybrid by design: PyTorch + torchdiffeq for the classical
baseline, JAX + Equinox + Diffrax + PennyLane for the QLNN, sharing one
data/eval/bootstrap pipeline so numbers are bit-comparable. The honest,
reframed finding (the original "QLNN beats classical" framing was too
clean): **the QLNN advantage is horizon-conditional and
reproducibility-flavored, not pure-accuracy.** At short horizons where
the classical model cannot beat persistence the QLNN can; at the
discriminating h=3 horizon a matched-parameter classical model is
slightly more accurate, but the QLNN is ~3.8× more reproducible across
seeds and more sample-efficient at low data — which matters in regulated
industrial settings. (Source: `PAPER_SUMMARY.md` §"Paper narrative
recommendation".)

---

## 2. The three pre-registered claims — final verdicts

Pre-registration: `hypothesis.md` v2 (committed before downstream steps
5/6). Verified end-to-end by `scripts/verify_paper_integrity.py` (exits
0). The original v1 QWGAN-GP synthetic-data-lift claim was **dropped**
after a peer-review-style audit — see §12.

| # | Claim | Pre-reg threshold | Result | Verdict |
|---|---|---|---|---|
| 1 | **Reproducibility** — QLNN test-MAE σ ≥ 2× tighter than classical at matched params (h=3) | ratio ≥ 2.0 | σ_classical_H4 = 0.016615, σ_QLNN = 0.004367 → **ratio 3.80×** (paper anchor cites 3.77×) | ✅ MET |
| 2 | **Expressivity** — QLNN effective dimension d_norm (Abbas et al. 2021 Eq. 4) exceeds classical by > 1.0 at matched params | Δd_norm > 1.0 | classical d_norm 8.029 ± 1.297, QLNN 9.514 ± 4.738 → **Δ = +1.485** | ✅ MET, caveated: QLNN d_norm seed-variance (σ 4.74) ≫ classical (1.30) |
| 3 | **Sample efficiency** — QLNN reaches the classical-100% accuracy target at less data | paired-bootstrap wins at low fractions | QLNN wins **10% (p=0.015)** and **25% (p=0.002)**, ties 50% (p=0.226), classical wins 100% (p=0.029) | ✅ MET, stronger than the pre-reg threshold required |

(Source: `PAPER_SUMMARY.md` §"The three claims", `results/baseline_lock.json`,
`results/effective_dimension/effective_dimension.md`,
`results/sample_efficiency/sample_efficiency_table.md`.)

---

## 3. Dataset + locked evaluation protocol

- **Dataset:** `data/raw/qZETA_data_copy.csv` — 778 rows (a DATE-sort
  yields 777 valid rows), one single fermentation run. Features: OD
  (target), PRE, TEMP_EXT, TEMP_CULTURE, PAR_LIGHT, PH, DO.
- **Splits:** train 70% / val 15% / test 15%, **chronological** (no
  shuffle — this is a time series).
- **Task:** 1-hour-ahead OD forecast from a 24-step history window,
  stride 1. Horizon ablation h ∈ {1, 3, 6, 12}; **h=3 is the
  discriminating regime** (at h=1 persistence is near-unbeatable —
  OD autocorrelation ≈ 0.99; at h ≥ 6 every model collapses).
- **Scaling:** train-only MinMax (a leak-fix); predictions clipped to
  [0, 3.8] raw OD at evaluation (strain physical max).
- **Metrics:** MAE_raw, RMSE_raw, R²_raw, MSE_norm, ΔOD-R²_raw.
  Persistence + linear-extrapolation baselines reported every run.
- **Seeds:** {0,1,2,3,4}; report mean ± ddof=1 std **and** 95%
  t-CI **and** paired-bootstrap p-values.
- **Selection:** best validation MSE_norm checkpoint.
- Any change to this protocol breaks comparability across all milestones.

(Source: `CLAUDE.md` "Evaluation protocol", `PAPER_SUMMARY.md`
"Locked methodology bindings".)

---

## 4. Architecture

**Classical — `LiquidODForecaster`** (`src/quantum_liquid_neuralode/`,
PyTorch + torchdiffeq). A Liquid-CT-RNN cell returns dh/dt; the caller
integrates over the history then the horizon. Residual forecast around
persistence: `OD(t+h) = OD(t) + tanh(delta_head(h)) · delta_scale`,
`delta_scale` **learnable** (softplus + floor; legacy fixed-0.1 kwarg
kept for back-compat). `ode_method ∈ {euler, rk4, dopri5}`.

**Quantum — `QLNNForecaster`** (`src/qlnn_/`, JAX + Equinox + Diffrax +
PennyLane). Pipeline per sample: linear encode of x[0] → h0; evolve
`dh/dt = cell(t,h,x_i)` over history then horizon via Diffrax (tsit5);
residual delta head. The cell's vector field routes the input through a
**quantum feature encoder**: `x → π·tanh(Wx+b) → PQC → ⟨Z⟩ ∈ [-1,1]^Q`.
Hidden width = num_qubits by construction. ~**114 trainable params** at
the 4-qubit / 3-layer reference (verified by rebuilding the checkpoint).

**Pluggable ansatz registry** (`src/qlnn_/circuits/`): every PQC is a
swappable factory honoring one contract — `(inputs:(Q,), weights) →
jnp.ndarray (Q,)` in [-1,1]. Four registered families:
`data_reuploading` (Pérez-Salinas 2020, the reference: 4q/3L, ring
entanglement, RX encode, Rot variational), `hardware_efficient`
(RY+RZ + CNOT), `strongly_entangling` (PennyLane template),
`brickwall` (even/odd-pair CNOT). `ansatz=None` ⇒ historical
data_reuploading, so every committed checkpoint deserializes unchanged.

**Differentiation (important, non-obvious):** device = `default.qubit`
with `interface="jax"` ⇒ effective `diff_method="backprop"` (JAX-native
autodiff through the statevector). **NOT** `lightning.qubit`/`adjoint` —
the QNode is nested inside a Diffrax `diffeqsolve` that is itself
reverse-mode differentiated (`jax.jacrev`) and JIT-compiled; only a
pure-JAX backprop QNode composes through that. At 4 qubits (16-amplitude
statevector) backprop-under-JIT is near-optimal; the cost driver is the
number of QNode evaluations per ODE trajectory, not the per-call
gradient method. This path is load-bearing for every committed result.

(Source: `CLAUDE.md` "Architecture", `src/qlnn_/circuits/`, this
session's device-rationale analysis.)

---

## 5. Headline result tables (verbatim)

### 5.1 Classical baseline — 1h-ahead OD (`results/baseline_classical_table/baseline_table.md`)

| Model | test MAE | test RMSE | test R² | seeds |
|---|---|---|---|---|
| Persistence (OD(t+h)=OD(t)) | 0.0934 | 0.1129 | 0.9052 | n/a |
| Linear extrapolation | 0.1276 | 0.1631 | 0.8024 | n/a |
| Liquid-ODE (Euler, train-only OD) | 0.0928 ± 0.0082 | 0.1128 ± 0.0097 | 0.9048 ± 0.0161 | 5 |
| Liquid-ODE (dopri5, train-only OD) | 0.0929 ± 0.0081 | 0.1129 ± 0.0097 | 0.9047 ± 0.0160 | 5 |
| Liquid-ODE +physics (train-only OD) | 0.0899 ± 0.0073 | 0.1094 ± 0.0089 | 0.9105 ± 0.0144 | 5 |
| Liquid-ODE (Euler, fixed [0,3.8] OD — leak sensitivity) | 0.0741 ± 0.0150 | 0.0892 ± 0.0196 | 0.9386 ± 0.0256 | 5 |

> Headline honesty: at h=1 the trained Liquid-ODE essentially **ties
> persistence** (0.0928 vs 0.0934 MAE). The "+physics" lift is
> logistic-growth-only (~3%); the fixed-OD row is a scaler-leak
> sensitivity comparator, not the headline.

### 5.2 Horizon ablation (`results/horizon_sweep_table/horizon_sweep_table.md`)

Window counts per split — train: h1 500 / h3 472 / h6 436 / h12 385;
test: h1 86 / h3 71 / h6 53 / h12 17 (h=12 has <30 test windows ⇒
supplementary only). Verified Test-R² anchors (from
`verify_paper_integrity.py`): h=1 persistence 0.9052 / LO-ODE 0.9048;
h=3 persistence **−0.0371** / LO-ODE **0.1108**; h=6 persistence
−9.7136 / LO-ODE −9.5004; h=12 persistence −977.02 / LO-ODE −999.90.
**h=3 is where a trained model first separates from persistence; h≥6
both collapse.**

### 5.3 Param-matched sweep at h=3 (`results/param_sweep/param_sweep_table.md`)

| Model | hidden | params | test MAE (mean ± std) | test R² (mean ± std) |
|---|---|---|---|---|
| classical | 2 | 42 | 0.2449 ± 0.0224 | 0.1564 ± 0.1545 |
| classical | 4 | 90 | 0.2594 ± 0.0166 | 0.0534 ± 0.1211 |
| classical | 8 | 210 | 0.2581 ± 0.0168 | 0.0579 ± 0.1207 |
| classical | 16 | 546 | 0.2564 ± 0.0204 | 0.0777 ± 0.1501 |
| classical | 32 | 1602 | 0.2491 ± 0.0250 | 0.1108 ± 0.1535 |
| **qlnn** (4q/3L, ~114 params) | — | — | **0.2655 ± 0.0044** | 0.0133 |

> At h=3 the matched-param classical (H=4, 90 params) is **more
> accurate** (0.2594 vs 0.2655) but **far less reproducible**
> (σ 0.0166 vs 0.0044 → the 3.80× Claim-1 ratio).

### 5.4 Effective dimension, n=472 (`results/effective_dimension/effective_dimension.md`)

| Model | D | mean d_norm | std | n_seeds |
|---|---|---|---|---|
| Classical Liquid-ODE (H=4) | 90 | 8.0290 | 1.2965 | 5 |
| QLNN (h=3) | 114 | 9.5144 | 4.7381 | 5 |

Δd_norm = **+1.4854** (>1.0 ⇒ Claim 2 MET). Caveat: QLNN per-seed
d_norm ranges 3.68–14.78 (σ 4.74) — high; the monotonicity sanity
criterion was corrected post-hoc (see §12, `STEP5_MONOTONICITY_NOTE.md`).

### 5.5 Sample efficiency at h=3 (`results/sample_efficiency/sample_efficiency_table.md`)

Test MAE, fraction of training windows truncated chronologically from
start, mean ± 95% CI:

| Stack | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| Classical H=4 | 0.2788 ± 0.0239 | 0.2546 ± 0.0288 | 0.2564 ± 0.0265 | 0.2594 ± 0.0206 |
| QLNN | 0.2686 ± 0.0080 | 0.2507 ± 0.0200 | 0.2633 ± 0.0074 | 0.2655 ± 0.0054 |

Window counts (train): 47 / 118 / 236 / 472. Target X = classical-100%
MAE = 0.2594. Paired-bootstrap verdicts: QLNN wins 10% (p=0.015) & 25%
(p=0.002); tie 50% (p=0.226); classical wins 100% (p=0.029).

---

## 6. Circuit-search exploration (post-hoc, OUTSIDE the pre-registration)

Exploratory science informing §7 discussion — **does not touch the
three pre-registered claims**. Three-phase protocol:

- **Phase 2 (axis ablation, ~1 h):** 12 single-seed configs varying one
  axis at a time (entanglement / variational gate / encoding / depth /
  qubits) + brickwall.
- **Phase 3 (Optuna TPE, ~3 h, 22 trials):** Bayesian search over the
  same discrete space. **Bug found + fixed mid-run:** the driver
  suggested an `entanglement` knob for `strongly_entangling`/`brickwall`,
  which ignore it — TPE overfit to a phantom hyperparameter (3 identical
  brickwall trials at val 0.0824). Fixed via a `FIXED_TOPOLOGY` skip set.
- **Phase 4 (promotion, ~6 h):** top-3 by proxy test-MAE re-run at the
  full 5-seed locked h=3 protocol.

**Key finding — the single-seed proxy ranking did NOT survive 5-seed
promotion:**

| Proxy rank | Circuit | Proxy MAE (1 seed) | Promoted MAE (5-seed, mean ± 95% CI) |
|---|---|---|---|
| 1 | data_reuploading 4q/5L | 0.2466 | 0.2638 ± 0.0155 (collapsed to ≈ reference) |
| 2 | hardware_efficient 4q/3L | 0.2610 | 0.2661 ± 0.0105 (≈ reference) |
| **3** | **strongly_entangling 6q/3L** | 0.2612 | **0.2555 ± 0.0314** (best, but high-variance) |

Interpretation: at this dataset size single-seed val noise scrambles
the test-MAE ranking of circuits within ~0.02 MAE of each other;
ansatz-topology variance ≈ within-circuit seed variance; deeper
re-uploading (L=5) did **not** pay off on 5 seeds. (Source:
`PAPER_SUMMARY.md` §"Circuit search",
`results/circuit_search/circuit_search_table.md`.)

---

## 7. Option-B — the "best-for-all" constrained problem  *(IN-FLIGHT, no verdict yet)*

After the circuit search, the user defined success as **beat classical
accuracy without losing the reproducibility advantage and without
regressing the sample-efficiency wins.** Encoded as 5 hard gates frozen
in `results/baseline_lock.json`:

| Gate | Requirement | Reference QLNN (dr 4q/3L) | Best promoted (se 6q/3L) |
|---|---|---|---|
| **G1 accuracy** | h=3 5-seed MAE < **0.2594** | 0.2655 ❌ | 0.2555 ✅ |
| **G2 reproducibility** | σ ≤ **0.00831** (½·σ_classical_H4; keeps Claim-1 ≥2×) | 0.0044 ✅ (3.80×) | ~0.0253 ❌ (0.66×) |
| **G3 no-regress 10%** | SE MAE < classical 0.2788 | 0.2686 ✅ | unknown |
| **G4 no-regress 25%** | SE MAE < classical 0.2546 | 0.2507 ✅ | unknown |
| **G5 no-regress 50/100%** | ≤ current QLNN reference | (is the reference) | unknown |

**Central tension:** every circuit found so far is *either*
accurate-but-unstable (strongly_entangling: G1 ✅ G2 ❌) *or*
stable-but-inaccurate (reference: G1 ❌ G2 ✅). **No circuit passes both
G1 and G2** — the "feasible box" is empty (visualized in
`fig_master_comparison`). **Hypothesis under test:** the
accuracy↔variance tradeoff is a *regularization-strength artifact*, not
fundamental — a sufficiently regularized expressive circuit could
collapse its variance under G2 while keeping its accuracy.

**Status:** Phase **O-1** (plumbing: pluggable `lr_schedule`
constant/cosine + `init_circuit_std` through the YAML) ✅ DONE, 162/162
tests, integrity green. Phase **O-2** (curated factorial: 3 circuits
{strongly_entangling-6q3L, data_reuploading-4q3L, hardware_efficient-4q3L}
× 4 regularization regimes {R0 control, R1 weight-decay, R2
physics-prior, R3 cosine+tight-clip+small-init}, 3-seed proxy,
penalized objective `MAE + 5·relu(σ−0.00831)`) 🔄 **IN-FLIGHT — 8/12
configs done** (resumed after an operator error removed the `data`
symlink mid-sweep; 7 completed cleanly, 5 re-running). **No Option-B
verdict exists yet.** Tier-1 (top-3 → 5-seed) and Tier-2 (survivors →
4 SE fractions) are GATED on user go/no-go. (Source:
`.planning/OPTION_B_SEARCH_DESIGN.md`, `results/baseline_lock.json`.)

---

## 8. Synthetic ODE battery  *(harness DONE+tested; runs GATED)*

**Why:** the single bioreactor OD signal is ~0.99 autocorrelated —
persistence is nearly unbeatable at h=3, so it cannot exercise the
continuous-time inductive bias these Neural-ODE models are built around.
A controlled ODE suite isolates the dynamical regime with unlimited
clean data and no seed-variance/scarcity confound, enabling *mechanistic*
claims ("what are QLNNs good at") instead of dataset-bound ones.

**Canonical 5** (the standard Neural-ODE benchmark set):
`lotka_volterra` (2D coupled, conserved quantity), `fitzhugh_nagumo`
(2D excitable relaxation oscillator), `van_der_pol` (2D **stiff**,
μ=5), `lorenz` (3D **chaotic**, sensitive dependence), `kuramoto`
(**12-D** coupled phase oscillators).

**Design notes:** pure-numpy fixed-step RK4 (zero new deps,
deterministic); `dt` governs integration accuracy while `sample_every`
decouples the sampled-row stride so each 4000-row series spans 30–180
cycles/Lyapunov-times (otherwise a stiff oscillator degenerates to a
persistence-trivial ramp); Kuramoto observes `sin(θ)` (a raw unwrapped
phase drifts ~linearly — the exact pathology the suite exists to
escape); CSVs emit the **exact qZETA schema** (DATE in DD/MM/YYYY so
`load_qzeta`'s `dayfirst=True` round-trips bit-exactly — ISO would be
silently misparsed; this bug was caught and fixed during the build) ⇒
`train_baseline.py`/`train_qlnn.py` consume synthetic data with **zero
trainer changes**; 1 row = 1 hour so `horizon_hours=3` = "3 steps
ahead", identical discrete-step semantics to qZETA. Status: 21/21 tests
pass; runs GATED behind the unified matrix. (Source:
`src/quantum_liquid_neuralode/data_processing/synthetic_ode.py`.)

---

## 9. Unified model × dataset matrix  *(generators+tests DONE; runs GATED)*

The paper's deepest question — **is the accuracy↔variance behavior a
property of the model or the dataset?** — is answerable only if the
*same model suite* is evaluated *identically* on every dataset.

- **48 models** (dataset-agnostic): 7 classical (capacity sweep H ∈
  {2,4,8,16,32} + dopri5 + classical-+physics, all at matched H=4 for
  the ablations) + 41 QLNN (4 ansatz families × 4 regularization regimes
  = 16, plus **25 dedup'd prior topologies** = the axis-ablation grid +
  20 unique Optuna specs + promoted runs, at native R0 regime).
- **11 datasets**: qZETA-OD + 5 ODE systems × {`m472` ≈ 778 rows →
  ~472 windows = exact qZETA parity / `full` 4000 rows = data-scaling
  ablation}.
- = **529 configs** (48 × 11 + 1 qZETA-only `fixed_od_clip`, which is a
  preprocessing variant, not a model — a [0,X] clip is undefined for
  signed ODE states). LOCKED protocol, 3-seed proxy.
- **Per-dataset baseline locks** (`build_dataset_baseline_locks.py`)
  derive each dataset's own G1/G2 from its own classical H-sweep, so
  "passes Option-B" means the same thing on Lorenz as on the bioreactor.
- The matched-472 variant is the data-volume-controlled head-to-head;
  the full variant is the data-scaling ablation.
- The horizon sweep (h ∈ {1,3,6,12}) is a 4× eval-axis multiplier and
  is **deferred to a separate gated phase** on a curated subset, not
  folded into this matrix.
- Status: 31/31 harness tests pass; model-suite-identity is
  test-enforced across datasets (with the one documented qZETA-only
  exception). Execution is GATED, dataset-grouped (one ~48-config group
  per go/no-go), sequenced strictly after Option-B tier-1 so nothing
  contends with the critical path. ~multi-day even gated. (Source:
  `.planning/UNIFIED_MATRIX_DESIGN.md`.)

---

## 10. Figure inventory (25 figures + a parametric search-space table)

**Headline / claims (`scripts/make_paper_figures.py`):**
- `fig_horizon_ablation` — R²+MAE vs h; argues h=3 is discriminating.
- `fig_sample_efficiency` — test MAE vs n_train; the Claim-3 crossover.
- `fig_reproducibility` — CI-width per stack/fraction; Claim 1.
- `fig_quantum_circuit` — the locked 4q/3L data-reuploading PQC.
- `fig_dataset_overview` — OD + 6 covariates + train/val/test split.
- `fig_baseline_metrics` — 4-panel MAE/RMSE/R²/MSE_norm all baselines.
- `fig_param_sweep` — params-vs-{MAE,R²,ΔR²}, QLNN overlaid.
- `fig_horizon_full_metrics` — h sweep, all 4 metrics.
- `fig_sample_efficiency_full` — SE, all 4 metrics, both stacks.
- `fig_effective_dimension` — d_norm curves + Claim-2 verdict bar.

**T1 reviewer-diagnostic (`scripts/make_diagnostic_figures.py`):**
- `fig_learning_curves` — train/val loss + 5-seed band (dynamic Claim 1).
- `fig_forecast_trajectory` — actual vs predicted OD + persistence.
- `fig_pred_vs_actual` — calibration scatter + R².
- `fig_residual_analysis` — residual-vs-time / hist / ACF.
- `fig_paired_bootstrap` — per-sample QLNN−classical diff vs null.
- `fig_seed_strip` — every seed's MAE/R² (the tight QLNN cluster IS Claim 1).
- `fig_all_circuit_diagrams` — all 4 ansätze, gate structure.

**T2 Option-B narrative + capstone:**
- `fig_ansatz_axis_effects` — per-axis effect on MAE/R² (skips until O-2).
- `fig_circuit_pareto` — params vs MAE by ansatz family vs classical.
- `fig_circuit_regime_heatmap` — circuit×regime penalized objective.
- `fig_regularization_arrows` — R0→{R1,R2,R3} vectors in (MAE,σ) space.
- `fig_master_comparison` — every config in (MAE,σ) space + the empty
  G1/G2 feasible box (the Option-B problem stated visually; 40 configs,
  0 in the box pre-Option-B).

**Circuit presentation (main text + supplement):**
- Main text = the 4-family templates (`fig_all_circuit_diagrams`) +
  `fig_quantum_circuit` (reference) + a **parametric circuit
  search-space table** (`results/circuit_search_space/search_space_table.md`,
  `scripts/build_circuit_search_space.py`) — axes ×
  levels + the 28-topology list. This is the standard QML way (not 28
  diagrams in the body).
- Supplement = **full gallery**, every distinct topology drawn via
  `qml.draw_mpl`, one figure per family:
  `fig_circuit_gallery_data_reuploading` (12),
  `fig_circuit_gallery_strongly_entangling` (6),
  `fig_circuit_gallery_hardware_efficient` (5),
  `fig_circuit_gallery_brickwall` (5) = **28 distinct topologies total**.
  Note: a regime (R0–R3) is a *training* knob, not a topology — the 48
  models / 529 configs collapse to 28 distinct circuit diagrams.

---

## 11. Methodology rigor & provenance

Pre-registration (`hypothesis.md`) committed before downstream steps;
ddof=1 std **and** 95% t-CI **and** paired-bootstrap p-values reported;
`scripts/verify_paper_integrity.py` checks every headline number against
on-disk JSON and **exits 0**; per-run `provenance.json` (git SHA +
data SHA-256 + package versions + platform); per-seed predictions saved
as `.npz` for retrospective bootstrap; `jax_enable_x64` is **off**
globally (enabling it breaks Diffrax dtype promotion — the
empirical-Fisher accumulator uses numpy-float64 on the side);
`jax.jacrev` (reverse-mode) is required through Diffrax's `custom_vjp`
(jacfwd fails). 162/162 (core) + 31/31 (matrix/ODE harness) tests pass.

---

## 12. Honest limitations & locked decisions

- **Single fermentation run is the binding constraint on every claim.**
  This is why the QWGAN-GP synthetic-data-lift claim (original v1
  headline) was **dropped** — it needs a held-out second run; rationale
  in `hypothesis.md` v2 "Deviations from v1".
- **Persistence dominates at h=1** (OD autocorr ≈ 0.99) — the headline
  must use h=3, not h=1.
- **QLNN d_norm has high seed variance** (σ 4.74 vs classical 1.30) —
  Claim 2 holds in the mean but the per-seed spread is reported honestly.
- **Step-5 monotonicity criterion was corrected post-hoc**
  (`STEP5_MONOTONICITY_NOTE.md`): the pre-reg "monotonic increasing"
  was mathematically wrong for rank-deficient trained-θ Fisher; the
  corrected criterion ("monotonic in either direction with shrinking
  gaps") is the one the data satisfies.
- **Proxy/full divergence** (§6): single-seed proxy rankings are not
  paper-grade; promotion to 5 seeds is mandatory before any number
  enters `PAPER_SUMMARY.md`.
- Locked decisions (do not relitigate): hybrid stack by design; eval
  protocol locked; `delta_scale` learnable; "+physics" is
  logistic-growth-only (a smoothness term that was algebraically MSE
  was removed). (Source: `HANDOFF.md` "Locked decisions".)

---

## 13. Status board

| Workstream | Status |
|---|---|
| Claim 1 (reproducibility, 3.80×) | ✅ DONE, verified |
| Claim 2 (expressivity, Δd_norm +1.49) | ✅ DONE, verified, caveated |
| Claim 3 (sample efficiency crossover) | ✅ DONE, verified |
| Classical + QLNN baselines (locked protocol) | ✅ DONE, frozen in `baseline_lock.json` |
| Circuit search (Phase 2/3/4) | ✅ DONE — proxy ranking did not survive promotion |
| 21 publication + diagnostic figures | ✅ DONE (Option-B figs skip until O-2) |
| Option-B O-1 plumbing | ✅ DONE, integrity green |
| Option-B O-2 proxy sweep | 🔄 IN-FLIGHT — 8/12 configs (resuming) |
| Option-B tier-1 / tier-2 | ⏸ GATED (no verdict yet) |
| Synthetic ODE harness | ✅ DONE+tested; runs ⏸ GATED |
| Unified 529-config matrix | ✅ generators+tests DONE; runs ⏸ GATED |
| Separate horizon phase | ⏸ GATED (deferred, curated subset) |
| T3 quantum-trainability (expressibility KL-to-Haar / Meyer-Wallach Q / barren-plateau scaling / Fisher) — **directly answers the "are circuits expressive enough?" concern** | ✅ analysis+figures BUILT, 7/7 smoke; execution ⏸ GATED after O-2 |
| Expressivity expansion E-2 (richer measurements ⟨ZZ⟩/⟨XX⟩, de-bottlenecked encoder, high re-upload, 8 qubits) | ⏸ NEXT BUILD (backward-compat, integrity-gated like O-1) |
| **Paper prose (intro/methods/results/discussion)** | ❌ **NOT STARTED** — needs the author's voice |

---

## 14. Artifact map (commit arc `9d0fdb3 … c80c021`)

Each claim/number traces to a `results/` path + generating script +
commit. Key commits (newest first):

```
c80c021 unified-matrix: fair-comparison expansion 21→48 models (529 configs)
dc8f438 unified-matrix: same model suite × every dataset (231 configs)
26093b6 ode-suite: synthetic ODE benchmark harness (canonical 5)
b3a73d4 figures: master all-vs-all comparison (T2 capstone)
3ed74cc figures: T2 Option-B narrative figures
a152eb3 figures: T1 reviewer-diagnostic figure set (7)
e8de9fa option-b: Phase O-2 scaffolding (12-config curated factorial)
1d430dd qlnn_: Phase O-1 plumbing (lr_schedule + init_circuit_std)
ade078a circuit-search: freeze baseline lock + Option-B regression gate
c1a4453 results: commit circuit_search + optuna + promoted data
d1c9b9d figures: publication-grade figure set expanded 3 → 13
7c4f201 scripts: circuit-search harness (Phase 2/3/4)
46945af qlnn_: pluggable ansatz registry + 4 circuit families
46b3492 fix: Step 5/6 fresh-review BLOCKERs
dbca3b7 feat: Step 6 (sample efficiency) + Step 5 monotonicity correction
df25283 feat: Step 5 (effective dimension) + QLNN +physics ablation
d7d6495 docs: refocus paper on LNN-vs-QLNN; drop QWGAN-GP
2b5c4b9 feat: Phase C — statistical rigor + param-matched + pre-registration
9d0fdb3 feat: Phase B — task hardening (train-only OD, learnable Δ-scale)
```

Number → evidence: Claim 1 → `results/param_sweep/euler_h3_hidden4/` +
`results/qlnn_hybrid_h3/` (via `verify_paper_integrity.py`); Claim 2 →
`results/effective_dimension/effective_dimension.json`; Claim 3 →
`results/sample_efficiency/`; circuit search →
`results/circuit_search{,_optuna,_promoted}/`; Option-B gates →
`results/baseline_lock.json`; figures → `paper/figures/`.

---

## 15. Suggested paper structures (for the external design model to choose)

`PAPER_SUMMARY.md` proposes a §1–§7 mapping: §1 intro (LNNs for low-data
bioprocess forecasting), §2 methods (locked protocol + both
architectures + stats), §3 classical baseline + horizon ablation, §4
head-to-head QLNN vs classical (h=1 vs h=3, the reproducibility result),
§5 expressivity (Claim 2), §6 sample efficiency (Claim 3), §7 discussion
(limitations, QWGAN-drop, d_norm variance). Circuit-search / Option-B /
ODE-battery extend §7 or become a new "what are QLNNs good at" methods
section.

**Three viable framings** (the design model should pick one and we
adapt):
1. **Reproducibility paper** (safest, fully supported now): "QLNNs
   trade pointwise accuracy for ~4× seed-reproducibility and
   sample-efficiency at low data on industrial forecasting." Stands
   entirely on the verified 3 claims; circuit-search/Option-B are §7.
2. **Pareto-region paper** (needs Option-B verdict): "QLNNs occupy a
   distinct accuracy×reproducibility Pareto region; we search for a
   circuit that dominates classical on both." Headline upgrade *iff*
   Option-B finds a feasible circuit; otherwise a rigorous negative
   result.
3. **Mechanistic paper** (needs ODE battery + unified matrix): "What
   dynamical regimes favor QLNNs?" — characterizes both stacks across
   stiff/chaotic/coupled ODE systems with a controlled data-volume axis.
   The strongest contribution but the most pending compute.

---

## 16. Open scientific questions

1. **Is the accuracy↔variance tradeoff a model property or a dataset
   property?** — the unified-matrix question; answerable once the
   529-config matrix runs.
2. **Does regularization cross the Option-B feasible box?** — the O-2
   hypothesis; verdict pending.
3. **Which dynamical regime (stiff / chaotic / high-D coupled) favors
   the QLNN?** — the ODE-battery question.
4. **Quantum-trainability** (barren plateaus, expressibility KL-to-Haar,
   Meyer–Wallach entangling capability, Fisher eigenspectrum) — T3,
   not yet built; a known QC-reviewer expectation.
5. **Why is QLNN d_norm seed-variance so high** (3.68–14.78)? — a
   trainability/landscape question that ties §5 to question 4.

---

*End of dossier. Regenerate the snapshot header + §13 status board when
Option-B O-2 completes and a tier-1 verdict exists.*
