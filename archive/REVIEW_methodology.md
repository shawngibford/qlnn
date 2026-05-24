# Methodology Review — Quantum-Liquid Neural ODE Bioreactor Paper

Reviewer: methodology-focused peer reviewer (Nature Comp. Sci. / NeurIPS / QST tier).
Scope: scientific methodology, claims, evidence, statistical rigor, fairness of
comparators. Not a code review.

Artifacts reviewed: `spec.md`, `README.md`, `CLAUDE.md`, all four `configs/*.yaml`,
`results/baseline_classical_{euler,dopri5,physics}/{seeds_summary,baselines,protocol,config}.json`,
representative `seed_*/metrics.json` and `seed_0/history.csv`, and the canonical
source files `src/quantum_liquid_neuralode/evaluation/metrics.py`,
`src/quantum_liquid_neuralode/models/{forecaster,liquid_cell}.py`,
`scripts/train_baseline.py`, `scripts/train_qlnn.py`.

---

## Executive summary

This project is well-organized and the evaluation contract (locked protocol,
frozen configs in `config.json`, deterministic baselines reported every run,
multi-seed aggregation) is far above average for a quantum-ML paper. As-is,
however, it is **not yet on track for a top-tier venue**. The single biggest
methodological risk, and the one that subsumes most of the others, is that **the
classical Liquid-ODE baseline is essentially a persistence model in disguise**:
three of the five physics seeds and at least one no-physics seed converge at
`best_epoch=1`, the architecture explicitly outputs `OD(t+h) = OD(t) +
tanh(delta)*0.1` (a hard clamp around persistence with `delta_scale=0.1`), and
the test R² difference between persistence (0.9052) and Liquid-ODE dopri5
(0.9339 ± 0.022) is only ~0.7σ. On a single 778-sample trajectory with
~0.99 autocorrelation at the 1 h horizon, "we beat persistence" is the only
honest classical claim, and right now even that claim is statistically
marginal. Until the task is hardened (longer/multi-horizon, log-phase-only
ablation, multi-run dataset) and the residual head is unclamped or the
non-persistence component is shown to carry signal, **any QLNN-vs-classical
gap is at risk of being noise**, killing claim 2 (expressivity) and claim 3
(sample efficiency) before they leave the gate. Claim 1 (synthetic data lift)
depends on a DTW comparison that is not pre-registered and is currently being
referenced via numbers from a different paper.

The protocol is fixable. The findings below are ordered by how much they
threaten publishability.

---

## Findings

### 1. [BLOCKER] The Liquid-ODE baseline is operating near the persistence floor; "model wins" is statistically marginal

**Issue.** Inspect `results/baseline_classical_dopri5/seed_0/history.csv`:
val MSE_norm at epoch 1 is `3.12e-4` and the model **never beats epoch 1
again across all 300 epochs** (epoch 10: `7.08e-4`; epoch 100: `1.10e-3`).
`metrics.json` confirms `best_epoch: 1`. The same pattern repeats in
`baseline_classical_physics/seed_{0,2,4}/metrics.json` (`best_epoch: 1` for
three of five seeds; seeds 1 and 3 converge at 160 and 260). The forecast head
in `forecaster.py:245` is hard-clamped to a `tanh(.) * delta_scale` (=0.1)
deviation from `OD(t)`, i.e. the model can shift the persistence forecast by
**at most ±0.1 raw OD units per step** in residual space. With persistence
already at test R² = 0.9052 and the headline model at 0.9339 ± 0.022, the gap
(`Δ = 0.029`) is `~1.3σ` if you take the std at face value — and as flagged in
finding 5, the std is reported with `ddof=0`, biasing it low; with `ddof=1` the
classical-vs-persistence gap drops further toward 1σ.

**Why a reviewer cares.** A residual-around-persistence architecture is fine —
but if the early-stopping consistently picks epoch 1 (essentially the
initialization with a small learned shift) and the val curve **diverges** with
further training (history.csv shows val MSE roughly tripling over 100 epochs),
the model is overfitting in a way the protocol pretends to handle but
doesn't. Any reviewer will ask: is the QLNN going to be compared against an
under-fit, under-trained classical, or against a properly tuned one? At 5
seeds and 1.3σ separation, you cannot reject the null that QLNN ≈ classical ≈
persistence.

**Concrete fix.**
1. Decouple the head from persistence: remove the `tanh * delta_scale` clamp
   (or treat `delta_scale` as a learnable scalar) and report what the
   unclamped baseline does. If unclamped Liquid-ODE substantially beats
   persistence on multi-step horizons, you have a real baseline; if not, your
   classical model is *just* a learned bias on persistence and the paper
   should say so up front.
2. Investigate the val-curve divergence. With `patience=10` and `eval_every=10`,
   the early-stopping logic at the current settings effectively gives up after
   ~100 epochs of no improvement. The fact that `best_epoch=1` is hit so
   often suggests either (a) the learning rate (2.24e-3) is too high for the
   bias-only regime, (b) the residual is so over-parameterized for the task
   that any non-trivial dh is noise, or (c) the architecture's effective
   capacity beyond persistence is genuinely tiny. Run a small LR sweep
   (`{2e-4, 5e-4, 1e-3, 2e-3}`) and report the best, not the first.
3. Report **paired** persistence vs. Liquid-ODE per-window error and run a
   paired t- or Wilcoxon test on the 86 test windows (this is the discriminating
   stat reviewers will want, not seed-averaged R²).

---

### 2. [BLOCKER] The 1 h-ahead horizon is not discriminating on this dataset

**Issue.** With 10-min sampling and ~0.99 OD autocorrelation at lag 6
(estimated from `spec.md`'s "0.95+ at 8h lag" — at 1h it must be ≥0.99),
predicting OD 1h ahead is dominated by `OD(t)`. Empirically:

| model         | test R² | test MAE_raw |
|---------------|---------|---------------|
| persistence   | 0.9052  | 0.0934        |
| linear extrap | 0.7231  | 0.1545        |  ← *worse* than persistence
| Liquid-ODE dopri5 | 0.9339 ± 0.022 | 0.0771 ± 0.0123 |
| Liquid-ODE +phys  | 0.9594 ± 0.0046 | 0.0615 ± 0.0035 |

The linear-extrapolation baseline being *worse* than persistence is a
diagnostic that the task is mostly "predict the current value." The headroom
above persistence is ~0.05 R²; reviewers will (correctly) say that any
methodologically interesting model claim requires a harder task. Persistence
R² = 0.905 is the floor a quantum/classical comparison sits on — both will
score in the high 0.9s and the differences will be in the noise.

**Why a reviewer cares.** Claims 2 (expressivity) and 3 (sample efficiency) are
inferences about what the model is *capable of*, not what it gets credit for.
A task where persistence wins 90% of the variance leaves almost no signal in
which to detect quantum advantage.

**Concrete fix (pick at least two).**
1. **Add longer horizons.** Report a horizon ablation: 1 h, 3 h, 6 h, 12 h.
   The 1 h number stays as a "sanity floor"; the 6–12 h numbers become the
   actual paper claim. This costs ~zero engineering — change `horizon_hours`
   in the config, re-run.
2. **Log-phase-only evaluation.** Restrict val/test to windows whose forecast
   target is in the exponential-growth segment (where `dOD/dt` is large and
   persistence is provably wrong). Report this as a separate row.
3. **Multi-step / trajectory forecast.** Replace the single-step head with an
   *N*-step rollout (e.g. predict OD over the next 6 hours conditioned on the
   last 4 hours). DTW between predicted and observed trajectories is the
   right metric here and lines up with the QWGAN evaluation already planned.
4. **Predict ΔOD instead of OD.** A model that predicts the *change* and adds
   it to `OD(t)` is mathematically equivalent to the current residual
   parameterization but reports a metric (R² on ΔOD) where persistence is by
   definition zero — making the model's contribution legible.

The paper will be far stronger if Table 1 reads "persistence collapses to
R²≈0 at 6 h horizon, classical Liquid-ODE keeps R²=0.X, QLNN keeps R²=0.Y."

---

### 3. [HIGH] The QLNN-vs-classical comparison is not parameter-matched

**Issue.** From `configs/qlnn_hybrid.yaml`: `num_qubits=4`, `num_layers=3`.
Assuming the standard reuploading PQC (3 single-qubit rotations per qubit
per layer), the quantum core has `4 × 3 × 3 = 36` trainable angles. Adding
the surrounding tau, encoder, delta_head (consistent with classical
scaffolding) puts the QLNN forecaster well under **100 params**.

The classical Liquid-ODE with `hidden_size=32`, `input_size=7` (from
`forecaster.py:106-108` + `liquid_cell.py:51-55`):
- encoder `Linear(7→32)`: 256
- `W_h` 32×32 (no bias): 1,024
- `W_x` 32×7 + bias: 256
- `tau_unconstrained`: 32
- `delta_head` `Linear(32→1)`: 33

Total **≈ 1,601 params** — roughly **20–40× the QLNN**.

**Why a reviewer cares.** The paper's claim 2 is literally "matched parameter
count." Comparing a 1.6k-param classical to a <100-param quantum is not
matched. Two failure modes:
- If the QLNN ties or loses, the result is uninterpretable (was the classical
  over-parameterized? was the quantum under-parameterized?).
- If the QLNN wins, the obvious reviewer response is "shrink the classical
  to 100 params and re-run."

**Concrete fix.**
1. Add a **classical-shrink ablation**: re-run the classical baseline with
   `hidden_size ∈ {2, 4, 8, 16, 32}`. Plot test MAE vs. param count.
2. Report parameter counts explicitly in the paper table — one column.
3. Choose the QLNN's expressivity claim against the *Pareto frontier* of
   classical models, not against the single 32-unit version. The honest
   claim is "at matched params, QLNN reaches X; the classical needs Y× more
   params to match." That is a defensible expressivity statement; the
   current setup is not.

---

### 4. [HIGH] 3 seeds for QLNN is below the floor for any "mean ± std" claim

**Issue.** `configs/qlnn_hybrid.yaml` line 54: `seeds: [0, 1, 2]`. The
classical seeds_summary shows the std on test R² for dopri5 is `0.0219`,
on physics is `0.0046`, on euler is `0.0220`. To detect a 0.029 effect (the
size of the classical-vs-persistence gap) with α=0.05 two-sided, power=0.8,
and a pooled std around 0.022, you need **roughly n=10 per group** — and
that's for the easier classical-vs-persistence gap. To detect the smaller
QLNN-vs-classical gaps the paper will be claiming, n=3 is hopeless.

With **n=3**, the t-distribution 95% CI half-width is `t_{0.025, 2} × std/√3
= 4.30 × std / 1.73 ≈ 2.49 × std`. If the QLNN test-R² std is similar to the
classical's 0.022, the 95% CI is **±0.055** — wider than the entire
classical-vs-persistence gap. No paired claim can survive that.

**Why a reviewer cares.** "Mean ± std" with n=3 is folklore, not evidence.
QST and NeurIPS reviewers will flag this immediately.

**Concrete fix.**
1. Run **at least 5, ideally 10 seeds** for the QLNN. The config-stated
   reason ("quantum training is slow") is real but not a methodology defense
   — say so in the paper, and report what 5 seeds buys you.
2. Report **95% CI using the t-distribution** (`t_{α/2, n-1} × std / √n`),
   not the bare std. With n=5: multiplier 2.78; with n=10: 2.26.
3. For the head-to-head, use a **paired bootstrap** over test windows (you
   have 86 of them) within each seed and pool across seeds. This is far
   more powerful than seed-averaged means.

---

### 5. [HIGH] `aggregate_seed_metrics` uses biased (population) std

**Issue.** `src/quantum_liquid_neuralode/evaluation/metrics.py:81`:
```python
"std": float(vals.std(ddof=0)),
```
With `ddof=0` and n=5, the std is multiplied by `√(4/5) ≈ 0.894` relative
to the unbiased estimator. Every "± std" in the paper table is therefore
**~10–11% too small**. For the dopri5 row that reads `0.9339 ± 0.0219`, the
unbiased std is closer to `0.0245`. This is small in absolute terms but it
*tightens every confidence interval the reader computes from the table*, in
the direction that favors the authors. A reviewer running through the math
will see it.

**Concrete fix.** Change to `ddof=1` and re-aggregate. Also report CIs
(see finding 4), not just std.

---

### 6. [HIGH] Fixed OD scaler bounds `[0.0, 3.8]` leak test-set information

**Issue.** `configs/*.yaml` lock the MinMax scaler to `od_min=0.0, od_max=3.8`
in **every split**. From `spec.md`: dataset OD range is `[0.47, 3.80]` and
the chronological split is 70/15/15. Cultivation curves grow monotonically
through log → stationary, so the **max OD is in the test segment** (the
stationary plateau). The training-segment OD max is materially below 3.8.
By fixing the scaler max at 3.8, you are telling the model the global max
**before training**. This is a soft form of test leakage: the normalized
target is bounded by what is, in part, a test-set statistic.

**Why a reviewer cares.** Standard practice for time-series with hold-out is
to fit scalers on **train only** (`fit_minmax(..., fit_end=split.train_end)`
is already doing this for non-OD features per `train_baseline.py:163`) and
clamp/clip test values on inference. The current code overrides this for OD
specifically using the `fixed_bounds` argument — a deliberate decision that
needs to be justified or undone.

**Concrete fix.**
1. **Option A (cleanest):** fit OD scaler on train only; clip predictions
   and targets to `[0, 1]` at inference; document the clipping rate.
2. **Option B:** keep fixed bounds but justify them as a *prior* (e.g. "3.8
   is the maximum realistic OD for this strain, established by literature
   reference X"), with a citation. Currently `[0.0, 3.8]` is presented as
   "from the data" — but it's from the **full** dataset including test.
3. Either way: run a sensitivity analysis. Re-fit OD on train (max ≈ 2.5 or
   3.0 depending on phase coverage), report the metric delta. If results
   are robust, finding is downgraded.

---

### 7. [HIGH] Chronological split + biological phase structure means test set is extrapolation

**Issue.** From `spec.md`: three growth phases — lag (early), log (middle),
stationary (late). With a 70/15/15 chronological split, the test set is
predominantly **stationary phase**. The classical Liquid-ODE numbers
(test R² > val R² across the board: dopri5 0.934 test vs 0.912 val; physics
0.959 test vs 0.941 val — see `seeds_summary.json`) confirm this: the test
set is *easier* than the val set, because stationary OD is nearly constant
and persistence dominates.

**Why a reviewer cares.** When test > val, that almost always means the
test distribution is structurally different from train. The paper currently
implies "we generalize"; the honest reading is "we generalize from
lag+log+early-stationary onto stationary, where the answer is roughly
'don't change.'" Claim 3 (sample efficiency) is especially vulnerable: if
the test set is mostly the easy phase, halving the training data may not
hurt much, and the result will not transfer to a fresh fermentation run.

**Concrete fix.**
1. Annotate each row with phase membership (lag / log / stationary; can be
   inferred from OD level + derivative). Report **metrics per phase** in
   supplementary, and pick log phase as the headline if you want a genuine
   generalization claim.
2. Run **expanding-window cross-validation** as a sensitivity check: 5
   folds where the train segment grows and val/test slide. Even with n=778,
   this is cheap (training is minutes per seed). Report mean ± CI across
   folds. If headline numbers survive this, the chronological-split claim is
   defensible; if they collapse, the paper needs that finding.
3. **Be explicit in the prose**: "we evaluate on a single fermentation run;
   our claims are about within-run extrapolation, not population-level
   generalization." Reviewers will accept honest scoping; they will not
   accept implicit overclaims.

---

### 8. [HIGH] The +physics ablation is one-sided — QLNN should get a +physics row too

**Issue.** The classical model gets a `+physics` row (test R² = 0.9594 ±
0.0046, the best in the table). The QLNN config (`qlnn_hybrid.yaml`) has no
physics regularizers. If the paper compares "QLNN" against "Liquid-ODE
dopri5" (R² 0.934), the QLNN may beat it but lose to "Liquid-ODE + physics"
(R² 0.959). A reviewer will read this and ask whether the comparison was
cherry-picked.

**Concrete fix.**
1. Run a **QLNN + physics** variant under the same regularizer weights
   (`lambda_logistic=0.1`, `lambda_smooth=0.05`). Report it in the paper
   table. The logistic and smoothness losses are model-agnostic and live in
   `training/losses.py`, so this is a config change.
2. If physics + classical genuinely dominates physics + QLNN, the honest
   paper claim shifts to "QLNN matches at fewer parameters" (claim 2)
   and/or "QLNN is more data-efficient" (claim 3) — both of which require
   parameter-matched and data-fraction ablations, not raw test R².

---

### 9. [MEDIUM] No git commit hash or data checksum in run artifacts

**Issue.** `config.json`, `protocol.json`, and `seeds_summary.json` capture
configs and split sizes, but no `git_commit` field and no SHA256 of the CSV.
If the dataset is silently edited (e.g., a TEMP_EXT column is fixed), the
old `seeds_summary.json` becomes uninterpretable.

**Why a reviewer cares.** Reproducibility is a Nature Computational Science
benchmark requirement. Same for top ML venues' artifact reviews.

**Concrete fix.** In `train_baseline.py` (around line 144) and
`train_qlnn.py` (around line 116), write a small `provenance.json`:
```python
{
  "git_commit": subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip(),
  "git_dirty": bool(subprocess.check_output(["git","status","--porcelain"]).decode().strip()),
  "data_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
  "torch_version": torch.__version__,
  "platform": platform.platform(),
}
```
This is two dozen lines and protects every future paper number.

---

### 10. [MEDIUM] Deterministic baselines reported once, but presented in seed-aware tables

**Issue.** `baselines.json` (e.g. `baseline_classical_dopri5/baselines.json`)
correctly reports persistence and linear extrapolation as single
deterministic numbers. But `baseline_table.md` lists them in the same table
as Liquid-ODE rows with seeds — they are **not** in `seeds_summary.json` and
do not carry a `± std`. A reader skimming the markdown may not notice that
persistence has no error bar while Liquid-ODE has one.

**Concrete fix.** In `baseline_table.md` (and the eventual paper), either
(a) put a small footnote `(deterministic, no seed dependence)` on the
persistence/linear rows, or (b) report `0.000` std explicitly. The latter
makes the contrast with the seeded models visually unmistakable.

---

### 11. [MEDIUM] Per-seed numbers not in a single supplementary table

**Issue.** Per-seed metrics live in `seed_{0..4}/metrics.json`. Aggregates
live in `seeds_summary.json`. Neither makes it trivial to reproduce a paired
analysis or for a reader to compute their own CIs. The paper should publish
the **full per-seed table** in supplementary (5 rows per model × 4 metrics).

**Concrete fix.** Extend `summarize_baselines.py` to also write
`per_seed_table.{md,csv}` with one row per (run, seed) and one column per
metric. A reviewer can then re-run any test.

---

### 12. [MEDIUM] Pre-register the QWGAN-GP success criterion now

**Issue.** README claim 1 is "synthetic data lift." `spec.md` references a
DTW score of 0.83 (quantum) vs. 1.95 (classical) from a related paper but
does not commit the QLNN paper to any DTW threshold. After-the-fact criteria
("our DTW was X, which we now declare success") will be flagged as p-hacking
by any reviewer who has been reading QML papers in 2024–2026.

**Concrete fix.** Before running the QWGAN-GP step, write a short
`hypothesis.md` (or update `spec.md`) committing to:
1. The **primary endpoint**: e.g. "synthetic augmentation must reduce test
   MAE on the 6 h horizon by ≥ 10% over no-augmentation, with paired-seed
   p<0.05."
2. **DTW thresholds** for "the synthetic trajectories are realistic":
   absolute (e.g. DTW < 0.5 on normalized OD) and relative (DTW_quantum <
   DTW_classical with a margin).
3. **Negative result handling**: what you will report if the lift is zero
   or negative. (A pre-registered null result is publishable; a post-hoc
   pivot is not.)

Commit this file to git **before** running step 4. The git timestamp is the
pre-registration.

---

### 13. [MEDIUM] Effective-dimension methodology should be locked to Abbas et al. 2021

**Issue.** Step 5 (expressivity via Fisher / effective dimension) is
mentioned but unspecified. The canonical reference for "normalized effective
dimension" in quantum ML is Abbas, Sutter, Zoufal, Lucchi, Figalli, Woerner,
"The power of quantum neural networks," *Nature Computational Science* 1,
403–409 (2021). At ~68–100 quantum params and ~1,600 classical params, the
Fisher information matrix is small and the eigenvalue spectrum is cheap.

**Concrete fix.** In the methods section, commit to:
1. **Definition.** Abbas et al.'s normalized effective dimension (Eq. 4 in
   that paper), computed over n samples and parameter volume V_Θ.
2. **Estimator.** Empirical Fisher (sample mean of `∇log p · ∇log p^T`).
   For a regression model with Gaussian output, this maps to the Jacobian
   of the model w.r.t. params. State it explicitly.
3. **Comparison.** Report ED_quantum / ED_classical at *matched parameter
   count* (per finding 3). Reporting at unmatched param counts is
   uninterpretable — a 1,600-param model trivially has higher dimension
   than a 68-param one.
4. **Sanity check.** ED should monotonically increase with sample size n
   for an over-parameterized model; verify and report.

---

### 14. [MEDIUM] Sample-efficiency claim needs a pre-specified ablation design

**Issue.** Claim 3 (sample efficiency) has no operational definition in
README/spec.md. Two distinct claims compete:
- (a) "Converges in fewer epochs" — epoch-to-target metric.
- (b) "Reaches target accuracy with less data" — data-fraction-to-target
  metric.

(b) is far more defensible because (a) is dominated by optimizer hyperparams.

**Concrete fix.** Pre-register a **data-fraction sweep**: train on
{10, 25, 50, 100}% of training windows (chronologically truncated from the
**start**, so the same model never sees later windows — keeps the temporal
split honest). Report test MAE at each fraction for QLNN vs. classical
(parameter-matched, per finding 3). Headline statistic: "QLNN reaches test
MAE ≤ X with k% of the data; classical needs m% (m > k)."

---

### 15. [LOW] Seed control is wired correctly but DataLoader shuffling and CUDA non-determinism not addressed

**Issue.** `train_baseline.py:268-269` seeds both `torch` and `numpy` *before*
the `LiquidODForecaster` is instantiated — good. However, there is no
`torch.backends.cudnn.deterministic = True` (relevant if anyone runs this on
CUDA), no `torch.use_deterministic_algorithms(True)`, and no explicit seeding
of the DataLoader generator. On MPS/CPU this is usually OK; under CUDA,
seeds 0..4 will produce slightly different results across machines.

**Concrete fix.** Wrap the seeding into a `set_seed(seed)` utility that:
1. Calls `torch.manual_seed`, `np.random.seed`, `random.seed`, and on CUDA
   `torch.cuda.manual_seed_all`.
2. Sets the cudnn flags above.
3. Returns a `torch.Generator` to pass to any DataLoader.

Document in the paper which determinism guarantees hold across which devices.

---

### 16. [LOW] `forecast_steps=4` is mismatched with the dopri5 sub-stepping story

**Issue.** `configs/baseline.yaml:32`: `forecast_steps: 4` is used for
Euler/RK4 sub-stepping but ignored by `dopri5` (per `forecaster.py:177-203`,
dopri5 uses adaptive stepping with `rtol=1e-3, atol=1e-4`). Reporting
"dopri5 with forecast_steps=4" in the paper would be confusing. The current
config carries the same value across solvers, which is fine, but the paper
text must not imply that dopri5 uses 4 sub-steps.

**Concrete fix.** In the methods section: "Fixed-step methods (Euler, RK4)
use `forecast_steps=4` sub-steps over the 1 h horizon (15-min sub-step);
adaptive dopri5 selects step sizes to maintain `rtol=1e-3, atol=1e-4`,
which empirically corresponds to N≈M steps per integration (report the
median from the solver)." Pull that median out of `torchdiffeq` and put it
in a footnote.

---

## Recommendations for the paper's evaluation section (top 5)

1. **Harden the task before the paper writes itself.** Add a horizon
   ablation (1 h / 3 h / 6 h / 12 h) and a log-phase-only row. The 1 h
   number on chronological-split test stays as a sanity floor, but the
   headline becomes a longer-horizon and/or phase-restricted task where
   persistence is provably insufficient. **Without this, claims 2 and 3
   are unfalsifiable noise.** (Findings 1, 2, 7.)

2. **Report parameter counts and add a parameter-matched ablation.** One
   column in the paper table: "params." Run the classical at
   `hidden_size ∈ {2, 4, 8, 16, 32}` and locate it on a param-vs-MAE
   Pareto curve. Place the QLNN on that curve. The expressivity claim
   becomes "to match QLNN-at-N-params, classical needs M-params, M > N."
   (Finding 3.)

3. **Fix the statistics layer.** Switch `aggregate_seed_metrics` to
   `ddof=1`. Report **95% CIs from the t-distribution**, not bare stds.
   Run **n ≥ 5** seeds for the QLNN; if compute genuinely prevents this,
   say so and report what 5 seeds buys you. Run a **paired bootstrap over
   test windows** for the QLNN-vs-classical head-to-head — this is the
   power-maximizing test and uses your 86 test windows, not just 3–5
   seed means. (Findings 4, 5.)

4. **Lock evaluation provenance.** Add `git_commit`, `data_sha256`, and a
   per-seed CSV (one row per seed × model × metric) to the artifact
   directory. Publish both in supplementary. (Findings 9, 11.)

5. **Pre-register downstream success criteria.** Before running step 4
   (QWGAN-GP) or step 5 (effective dimension), commit a `hypothesis.md`
   to the repo that specifies (a) the primary endpoint and acceptance
   threshold for synthetic-data lift, (b) the DTW thresholds for
   trajectory realism, (c) the Abbas et al. 2021 normalized effective
   dimension as the operational expressivity metric (with the empirical
   Fisher estimator and matched param counts), and (d) the data-fraction
   sweep design for the sample-efficiency claim. The git timestamp is the
   pre-registration. **Without pre-registration, the three headline
   claims are easy targets for reviewers.** (Findings 12, 13, 14.)

---

## Notes the authors will probably push back on, and the reviewer's reply

- *"The dataset is small; we cannot run 10 seeds for QLNN."*
  Reply: agreed compute is tight; report 5 seeds + paired-window bootstrap
  rather than 3 seeds + naive std. The bootstrap is the cheap power gain.
- *"Persistence is hard to beat on smooth bioprocess data; this is known."*
  Reply: agreed — which is why the paper should not use the 1 h horizon
  as its showcase. Show the regime where persistence fails (longer
  horizons, log phase, transient excursions) and place the model there.
- *"Fixed OD bounds [0, 3.8] is a domain prior, not leakage."*
  Reply: fine, but cite the source for 3.8 (literature, strain spec,
  bioreactor capacity), do not derive it from the data file. Otherwise
  it remains leakage.
- *"+physics dominates; that is the strongest classical."*
  Reply: then run +physics-QLNN, not no-physics-QLNN, against it.
  Otherwise the comparison is asymmetric and reviewers will catch it.
