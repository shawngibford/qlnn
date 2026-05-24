# Pre-registration of paper claims (qlnn) — v2

**Status:** PRE-REGISTRATION v2 — committed 2026-05-17 (replaces v1, which
included a QWGAN-GP synthetic-data-lift claim that we have explicitly
dropped after the Phase A/B/C audit; see §"Deviations from v1" at the bottom).

**Project:** Liquid Neural Networks vs Quantum Liquid Neural Networks for
bioreactor OD forecasting.

**Dataset:** qZETA bioreactor, 778 samples, single fermentation run
(SHA-256 recorded in every `results/*/provenance.json`).

---

## Scope of claims

This paper restricts its claims to **within-run extrapolation on a single
fermentation trajectory**. It does NOT claim population-level generalization
across fermentation runs, organisms, or process conditions. All claims are
relative to the locked evaluation protocol described in README.md.

---

## Locked evaluation protocol (already in effect)

- Dataset: data/raw/qZETA_data_copy.csv, chronological 70/15/15 split.
- Window: 24 steps history. Stride 1.
- OD scaling: train-only MinMax (R3 leak fix); predictions clipped at
  od_phys_max=3.8 (domain prior — strain literature max, NOT data-derived).
- Metrics: MAE_raw, RMSE_raw, R²_raw, MSE_norm, ΔOD_R²_raw.
- Seeds: {0, 1, 2, 3, 4} for both classical and QLNN.
- Statistical reporting: ddof=1 std AND 95% t-CI AND paired-bootstrap p-values.
- Selection: best val MSE_norm checkpoint.
- Statistical test for head-to-head: paired bootstrap over test windows
  (n_iter ≥ 10000, two-sided), Stouffer's Z-method to combine seed p-values.

---

## Three claims (binding, v2)

The original v1 claims were (1) synthetic-data lift via QWGAN-GP, (2)
expressivity, (3) sample efficiency. v2 drops claim (1) — the QWGAN-GP
investment is not justified on a single-run dataset (the generator can't
add information the model didn't already have). The three v2 claims are:

### Claim 1 — Reproducibility (the surprise finding from Phase C data)

**Pre-registered hypothesis:** At matched parameter count (within a factor
of 2), the QLNN exhibits MATERIALLY tighter seed variance than the classical
Liquid-ODE across every headline metric, by at least a factor of 2 on
test MAE.

**Primary endpoint:** ratio of seed std at h=3, test MAE:
  σ(classical H=4) / σ(QLNN) ≥ 2.0
on a 5-seed re-run; report both ratio and the underlying stds.

**Empirical anchor (Phase C):** σ_classical_H=4 = 0.0166, σ_QLNN = 0.0044
→ ratio 3.77×. The pre-registration threshold of ≥ 2× is conservative
relative to what was observed.

**Acceptance:** Threshold met OR explicitly failed. Both publishable.

**Why this matters:** Industrial regulators (e.g. FDA process-validation
guidance for biologics) care about *consistency* of model predictions
across re-trains, not just headline accuracy. A 3× reproducibility win at
no accuracy penalty is a defensible practical contribution.

### Claim 2 — Expressivity at matched parameter count

**Pre-registered hypothesis:** At MATCHED parameter count (within a factor
of 2), the QLNN achieves a HIGHER normalized effective dimension on the
locked training data than the classical Liquid-ODE.

**Primary endpoint:** Normalized effective dimension d_norm following
Abbas, Sutter, Zoufal, Lucchi, Figalli, Woerner, "The power of quantum
neural networks," *Nature Computational Science* 1, 403–409 (2021), Eq. (4).

with:
- N = 500 training samples (matches the locked window count at h=3 train).
- Empirical Fisher information matrix I_F = (1/n) Σ_i J_i^T J_i, where
  J_i = ∂ ŷ_i / ∂ θ is the per-sample parameter Jacobian evaluated at the
  trained parameters.
- Computed via `jax.jacfwd` (forward-mode for the small param count) for
  the QLNN; via `torch.autograd.functional.jacobian` for the classical.

**Acceptance threshold:** d_norm(QLNN) > d_norm(classical_H=4), with
the difference exceeding 1 unit (Abbas et al. visual convention).

**Param matching:** The QLNN has ≈ 100 trainable parameters
(encoder + cell + 2 heads + delta-scale). The matched classical is
configs/param_sweep/baseline_euler_h3_hidden4.yaml, **90 parameters**
(audited from a deserialized checkpoint). Ratio 100/90 = 1.11, well within
the 2× tolerance.

**Sanity check:** d_norm should increase monotonically with n for an
over-parameterized model. Verify on n ∈ {100, 200, 350, 500} and report
the curve; if NOT monotonic, the empirical Fisher is mis-estimated and the
finding is withdrawn pending fix.

**Null-result handling:** Reported. A null result on Claim 2 still allows
Claims 1 and 3 to carry the paper.

### Claim 3 — Sample efficiency

**Pre-registered hypothesis:** The QLNN reaches a target test MAE
(at h=3, the discriminating regime) with FEWER training windows than the
param-matched classical model.

**Primary endpoint:** Data-fraction sweep at {10, 25, 50, 100}% of training
windows (chronologically truncated from the START — the model never sees
later windows; preserves the temporal split). For each fraction, report
test MAE (mean ± 95% CI across 5 seeds). Headline statistic: the smallest
fraction at which QLNN reaches test MAE ≤ X, where X is the param-matched
classical model's test MAE at 100% data.

**Acceptance threshold:** QLNN reaches X at ≤ 50% of the data while
classical needs > 50%.

**Empirical anchor:** From Phase C, classical H=4 test MAE at h=3 with
100% data = 0.2594. The QLNN must reach ≤ 0.2594 at the 10%, 25%, or 50%
points.

**Null-result handling:** Reported.

---

## What we are NOT claiming (any more)

- We do not claim a QWGAN-GP synthetic-data lift. The QWGAN-GP step was
  dropped after the Phase A/B/C audit; the single-run dataset cannot
  support that claim with current evidence. The QWGAN-GP investment would
  require a second fermentation run (different organism, different
  conditions) to test the cross-run generalization claim that motivates
  it. **This is the right design but is out of scope for this paper.**
- We do not claim QLNN > classical at h=1. (Phase B/C finding: at h=1
  classical at H=4 collapses onto persistence, while QLNN does beat
  persistence with Stouffer p < 0.0001. This is in the paper's results
  section but is not pre-registered because it was discovered, not
  hypothesized.)
- We do not claim cross-fermentation-run generalization.
- We do not claim quantum advantage on quantum hardware. The QLNN is run
  on classical simulators (PennyLane default.qubit JAX) throughout.

---

## Methodology bindings

### Effective dimension (Claim 2)

- Definition: Abbas et al. (2021) Eq. (4), normalized over parameter volume V_Θ.
- Estimator: empirical Fisher I_F = (1/n) Σ_i (∇_θ log p_θ(y_i|x_i))(∇_θ log p_θ(y_i|x_i))^T,
  using the trained parameters.
- Gaussian-output regression: this reduces to (1/n) J^T J where J is the
  per-sample Jacobian of the model's prediction w.r.t. parameters.
- Sample size: n = 500 training windows.
- Param-matching: H=4 classical (90 params) vs QLNN (~100 params).
- Sanity check: monotonic increase in d_norm with n for an over-parameterized
  model; report and verify before claiming the headline.
- Implementation lives in `src/qlnn_/diagnostics/effective_dimension.py`
  (forthcoming) for the JAX side, mirrored in
  `src/quantum_liquid_neuralode/diagnostics/effective_dimension.py` for
  the PyTorch side.

### Paired bootstrap (Claims 1, 3 head-to-head)

- Implemented in `src/quantum_liquid_neuralode/evaluation/bootstrap.py`.
- n_iter = 10000, two-sided.
- Per-seed bootstrap; meta-analysis via Stouffer's Z-method
  (Whitlock 2005).

### Sample-efficiency sweep (Claim 3)

- Truncation: from the START of the training segment (chronological).
  This means the 10% fraction sees the lag phase, the 25% sees lag + early
  log, etc. — preserving the temporal-split discipline.
- Configs: `configs/sample_efficiency/{classical,qlnn}_h3_pct{10,25,50,100}.yaml`
  (forthcoming).
- All other hyperparameters (lr, batch_size, epochs, patience) held
  constant across fractions; only the training-window count changes.

---

## Provenance bindings

All numerical results in the paper must come from runs whose
`provenance.json` records:
- A git commit on the public repo.
- The qZETA CSV SHA-256.

Any deviation from this pre-registration found at paper-writing time will
be disclosed in a "deviations from pre-registration" section.

---

## Deviations from v1

This is v2; v1 was committed at 2026-05-17 earlier the same day. Deviations:

| v1 claim | v2 status | Reason |
|---|---|---|
| Claim 1: QWGAN-GP synthetic-data lift | DROPPED | Single-run dataset cannot meaningfully support a "synthetic data improves forecasting" claim without a held-out second run. The QWGAN-GP investment was multi-day for what would likely be a null or marginal result attributable to data augmentation rather than quantum generation. The paper redirects to LNN-vs-QLNN comparison as the core scientific question. |
| Claim 2: Expressivity (Abbas et al.) | UNCHANGED in v2 | This is now Claim 2 in the new numbering. Methodology and threshold preserved. |
| Claim 3: Sample efficiency | UNCHANGED in v2 | Now Claim 3 in the new numbering. |
| (new) Claim 1: Reproducibility | ADDED | Phase C revealed a 3.77× QLNN/classical std ratio on test MAE. The reproducibility advantage was not pre-registered in v1 because it was discovered, not hypothesized. It is pre-registered HERE in v2 at a conservative ≥ 2× threshold (well below the observed 3.77×) so that the next 5-seed re-run constitutes a real prediction. |

The git timestamp of v2 is the pre-registration of the reproducibility
claim. The expressivity and sample-efficiency claims remain pre-registered
as of v1 (earlier same day).

---

## Web-sourced references

- Abbas et al. — "The power of quantum neural networks" (Nature
  Computational Science, 2021): https://www.nature.com/articles/s43588-021-00084-1
- Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993 (paired
  bootstrap convention).
- Whitlock 2005, "Combining probability from independent tests: the
  weighted Z-method is superior to Fisher's approach", *J. Evol. Biol.* 18(5).
- COS pre-registration guidelines: https://www.cos.io/initiatives/prereg
- NeurIPS Pre-registration Workshop: https://preregister.science
