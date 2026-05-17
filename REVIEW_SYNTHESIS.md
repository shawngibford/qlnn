# Review synthesis — what to fix, in what order

Aggregated findings from 4 parallel reviewers:
- R1 — classical code (`REVIEW_step1_classical.md`)
- R2 — quantum code (`REVIEW_step23_quantum.md`)
- R3 — methodology / statistical rigor (`REVIEW_methodology.md`)
- R4 — cross-stack integration (`REVIEW_integration.md`)

Total: **5 BLOCKERs, 19 HIGHs**, plus mediums and lows.

---

## Tier 0 — Existential: the headline numbers are at risk of being noise

These are R3's two BLOCKERs and they subsume everything else.

### 0.1 The classical Liquid-ODE is near the persistence floor
- `best_epoch: 1` in 4 of 5 dopri5 seeds and 3 of 5 +physics seeds.
- val MSE *worsens* monotonically after epoch 1 (3.1e-4 → 1.1e-3 over 100 epochs).
- Architecture hard-clamps output to `OD(t) + tanh(.) * 0.1` — can deviate from persistence by at most ±0.1 OD.
- Test R² gap classical-vs-persistence = 0.029, std ≈ 0.022 → **~1.3σ**. Statistically marginal.

### 0.2 The 1h horizon is dominated by persistence
- Persistence test R² = 0.905; linear-extrap is *worse* (0.723) — diagnostic that the task is "predict the current value."
- Only ~5% of variance is left for the model to fight for at 1h.

**Implication: claims 2 (expressivity) and 3 (sample efficiency) are unfalsifiable as configured. Any QLNN-vs-classical gap is at risk of being noise.**

**Fix (paper-design, needs your decision):**
- Add horizon ablation: re-run at 1h / 3h / 6h / 12h.
- Add log-phase-only evaluation row.
- Optionally: unclamp `delta_scale` or make it learnable.
- Optionally: switch to multi-step trajectory forecast + DTW (lines up with QWGAN later).

---

## Tier 1 — Real bugs that change reported numbers

These three I will fix immediately. They will move all reported numbers (probably by 1-3% each).

### 1.1 [R1-B1] Off-by-one in `make_horizon_windows`
- `range(0, n - window_size, stride)` should be `range(0, n - window_size + 1, stride)`.
- Drops one window per segment. Same bug exists in `preprocessor.py` (legacy).

### 1.2 [R1-B2] The "smoothness" loss is algebraically MSE
- `excess = (yp - od_last) - (yb - od_last) = yp - yb`.
- The +physics ablation's lift (0.077 → 0.062 MAE) comes from the **logistic** term plus a small data-loss-reweight from `lambda_smooth=0.05`. The attribution would be misleading in the paper.
- Fix: remove the broken `lambda_smooth` branch and re-run +physics with only the logistic term. Update the YAML.
- (Real smoothness would require multi-step output, which is also recommended by R3 — defer that to the horizon-ablation work.)

### 1.3 [R1-B3] `aggregate_seed_metrics` uses `ddof=0` (biased)
- With n=5, std is ~10% too small (multiplied by √(4/5) ≈ 0.894).
- Switch to `ddof=1`. Re-aggregate; mean unchanged, std modestly wider.

---

## Tier 2 — HIGH: comparability and statistical defensibility

### 2.1 [R3-H3] Param count mismatch (1601 classical vs ~100 QLNN)
- The "matched parameter count" claim in the README is unsupported.
- Fix: add a column for param count in the paper table; run classical at `hidden_size ∈ {2, 4, 8, 16, 32}` for a Pareto-curve ablation.

### 2.2 [R3-H4, R4] QLNN n=3 seeds → CI half-width wider than effect size
- Push QLNN config to n=5 seeds.
- Add t-distribution 95% CI to reporting (not bare std).
- Add paired bootstrap over test windows for the head-to-head.

### 2.3 [R3-H6] Fixed OD bounds `[0, 3.8]` leak test-set max
- The peak OD sits in the stationary (test) phase.
- Fix options:
  - (A) Fit OD scaler on train-only, clip predictions at inference.
  - (B) Justify [0, 3.8] as a domain prior (literature / strain spec / bioreactor capacity) and cite.

### 2.4 [R3-H7] Test > val on every classical row
- Chronological split puts the easy stationary phase in test.
- Report per-phase metrics in supplementary; consider log-phase-only as headline.

### 2.5 [R3-H8] Asymmetric +physics ablation
- Classical gets +physics (its best variant); QLNN doesn't. Comparison can look cherry-picked.
- Fix: run a QLNN +physics variant (logistic-only after Tier 1.2 fix).

### 2.6 [R2-H1] `tau_min ≤ 1` guard missing in `LiquidQuantumCellConfig`
- If `tau_min > 1`, the leak coefficient `-(1/tau + q(x))` can flip sign for q(x) < -1/tau, turning the cell into an exponentially-growing ODE. Silent HPO landmine.
- Fix: validate `tau_min ≤ 1` in `__post_init__`.

### 2.7 [R2-H2] `dt ≤ 0` guard missing in QLNN forecaster's `_integrate`
- A duplicate or non-monotone timestamp would integrate over [0,0] or backwards. NaN propagates silently.
- Fix: assert `dt > 0` (or skip-and-warn at window construction time).

### 2.8 [R4-H] Training-config divergence (classical vs QLNN)
- batch_size, eval_every, patience, n_seeds differ deliberately.
- Either align or disclose in methods paragraph.

---

## Tier 3 — MEDIUM (testing gaps, reproducibility hygiene)

- [R1-H4] No test for best-checkpoint correctness.
- [R1-H5] No test for multi-seed determinism.
- [R2-H3] `test_solver_swap_runs_for_both_tsit5_and_dopri5` only checks finiteness — add `allclose`.
- [R2-H4] `test_encoder_batched_via_vmap` only tests identical inputs — add heterogeneous-input case.
- [R3-M9 / R4] Add `git_commit` + `data_sha256` to artifact directories (provenance.json).
- [R3-M11] Add `per_seed_table.{md,csv}` to `summarize_baselines.py`.
- [R3-M12] Pre-register `hypothesis.md` before step 4 (QWGAN) — primary endpoint, DTW thresholds, null-result handling.
- [R3-M13] Lock effective-dimension methodology to Abbas et al. 2021 (Eq. 4) before step 5.

---

## Tier 4 — LOW (nits, paper polish, documentation)

- [R1-H1] Document logistic loss is left-endpoint (forward-Euler) discretization.
- [R1-H2] Justify `μ_norm=0.4` choice (or sensitivity-sweep) — spec says μ=0.3.
- [R1-H3] Record segment-boundary window drops in `protocol.json`.
- [R2-M1] PennyLane circuit re-evaluated on every Diffrax VF call — perf, not correctness.
- [R2-M6] Pre-eval NaN silently returns untrained model.
- [R2-M7] No config-match check on checkpoint deserialization.
- [R3-L15] Add `set_seed(seed)` utility covering torch.manual_seed + cudnn.deterministic + np + random + DataLoader generator.

---

## Verdict

**The protocol is fixable.** The plumbing is correct (R4 confirms: byte-identical baselines.json across stacks). The implementation bugs are real but bounded. The deeper issue is that **the task as configured doesn't have enough headroom to discriminate models** — Tier 0 must be addressed before the QLNN can earn a paper claim.

---

## Recommended path forward (3 phases)

### Phase A — Code fixes (no paper-design decisions). ~1 hour.
1. Fix windowing off-by-one (R1-B1).
2. Fix `ddof=0` → `ddof=1` (R1-B3).
3. Remove broken `lambda_smooth` branch + update `baseline_physics.yaml` (R1-B2).
4. Add `tau_min ≤ 1` guard (R2-H1).
5. Add `dt > 0` guard in QLNN integrate (R2-H2).
6. Tighten test gaps (R1-H4/H5, R2-H3/H4).
7. Add `provenance.json` (git commit + data SHA) to both training scripts.
8. Add `per_seed_table.{md,csv}` to summarize_baselines.

After Phase A, **re-run all baselines + QLNN**. Numbers will shift slightly.

### Phase B — Task hardening (paper-design decisions, needs your call). ~half-day per decision.

Pick at least two of:
- B.1 — Horizon ablation: re-run at {1h, 3h, 6h, 12h}.
- B.2 — Log-phase-only evaluation row (annotate phase, filter windows).
- B.3 — Multi-step trajectory forecast (lines up with QWGAN).
- B.4 — Predict ΔOD directly (R²-on-ΔOD where persistence is by construction 0).
- B.5 — Unclamp or learn `delta_scale`.
- B.6 — Switch OD scaler to train-only (or cite [0, 3.8] as domain prior).

### Phase C — Comparator alignment + statistical rigor. ~half-day to a day.

- C.1 — Push QLNN to n=5 seeds.
- C.2 — Run QLNN + physics variant.
- C.3 — Param-matched classical ablation (`hidden_size ∈ {2, 4, 8, 16, 32}`).
- C.4 — Report 95% CIs (t-distribution) + paired bootstrap over test windows.
- C.5 — Pre-register `hypothesis.md` for step 4 (QWGAN-GP success criterion).

---

**My recommendation: Phase A immediately, then pause for your decision on Phase B before re-running anything large.**
