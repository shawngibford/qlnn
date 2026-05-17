# REVIEW — Step 1: Classical Liquid-ODE Baseline

**Reviewer:** Claude (adversarial code review for paper-readiness)
**Scope:** PyTorch package `src/quantum_liquid_neuralode/`, scripts, configs, tests
**Verdict:** **NOT READY for paper submission.** Two BLOCKER-class bugs invalidate specific paper claims (the physics ablation, and the last-window-dropped data-loss). One additional HIGH issue (ddof=0 std) systematically understates reported uncertainty. The rest is mostly defensible but has notable physics-loss approximation issues and test-coverage gaps that a reviewer will flag.

---

## Executive summary

The classical Liquid-ODE pipeline is, on the whole, carefully engineered: the residual-around-persistence formulation is correct, the dopri5 rescaling to `[0,1]` is mathematically sound, the train/val/test windowing avoids future leakage by construction, the OD MinMax uses fixed `[0, 3.8]` bounds (no train-only-bounds leak), and multi-seed parameter initialization is correctly re-seeded per seed via `torch.manual_seed(seed)` immediately before `LiquidODForecaster(...)` construction in `train_baseline.py`. However, three issues will materially affect the paper. (1) The "smoothness" branch in `_physics_loss_terms` is mathematically just MSE-against-target, not a smoothness penalty — so the `+physics` ablation row in the paper isn't actually testing smoothness regularization, it is testing "1.05× MSE." (2) `make_horizon_windows` (and the legacy `BioreactorDataPreprocessor.create_sequences`) drop the *last valid window* via an off-by-one in `range(0, n - window_size, stride)`. With stride=1 and window=24, every split loses exactly one valid window — small in absolute terms but biases against the end-of-segment regime. (3) `aggregate_seed_metrics` reports `std(ddof=0)` (population), not the conventional `ddof=1` for n=5 — std is understated by ~12%. Beyond these, the logistic-growth physics loss pairs `dOD/dt` with the *left endpoint* OD (forward-Euler), not the midpoint, which is fine but should be documented; and there is no test guarding multi-seed determinism, train/val window-boundary non-overlap, or best-checkpoint state correspondence. Fix the three numbered items above before submitting, address the documentation/test gaps if reviewers push back on rigor.

---

## BLOCKER

### B1. Off-by-one in `make_horizon_windows` drops the last valid window
- **File:** `src/quantum_liquid_neuralode/data_processing/windowing.py:143`
- **Also:** `src/quantum_liquid_neuralode/data_processing/preprocessor.py:83`
- **Symptom:** `for start in range(0, n - window_size, stride):`. The stop bound is exclusive, so `start = n - window_size` is never reached. That index corresponds to `end = n`, `end_idx = n-1` — the legitimate final window.
- **Why it matters:** With `stride=1, window_size=24`, every split loses exactly one window (the most recent one). On the *test* segment (n≈117 rows), this is ~1/93 ≈ 1% of windows missed, but they are systematically the most recent windows in the chronological evaluation — exactly the regime where the model has had no warm-up. Worse, the legacy `BioreactorDataPreprocessor.create_sequences` has the identical bug, and any HPO results produced through it are slightly biased. For paper-grade comparability the cutoff must be deterministic and symmetric across models. Inconsistency between `n - window_size` (drops last) and the obvious convention `n - window_size + 1` (includes last) is the kind of detail reviewers ding.
- **Fix:**
  ```python
  # windowing.py:143
  for start in range(0, n - window_size + 1, stride):
  # preprocessor.py:83
  for i in range(0, n_rows - window_size + 1, stride):
  ```
  Add a regression test:
  ```python
  def test_make_horizon_windows_includes_last_window():
      n = 30
      t = np.arange(n, dtype=np.float64) / 6.0
      od = np.linspace(0.1, 0.9, n).astype(np.float32)
      feats = od.reshape(-1, 1)
      win = make_horizon_windows(
          features=feats, od=od, time_hours=t,
          window_size=6, stride=1, horizon_hours=0.0 + 0.001,
          horizon_tolerance_hours=1.0,  # accept any
      )
      # Last window should end at end_idx = n - 1 (= 29).
      assert int(win.end_idx.max()) == n - 1
  ```

### B2. `lambda_smooth` physics branch is NOT a smoothness loss — it is MSE again
- **File:** `src/quantum_liquid_neuralode/training/trainer.py:137-144`
- **Symptom:**
  ```python
  delta_pred = yp - od_last
  delta_true = yb - od_last
  excess = delta_pred - delta_true     # == yp - yb
  total = total + cfg.lambda_smooth * excess.pow(2).mean()
  ```
  `excess = (yp - od_last) - (yb - od_last) = yp - yb`. So this term is mathematically identical to `lambda_smooth * MSE(yp, yb)` — i.e., it just up-weights the reconstruction loss by `lambda_smooth`. It is in no sense a smoothness penalty (no second differences, no curvature, no temporal regularization).
- **Why it matters:** `configs/baseline_physics.yaml` sets `lambda_smooth=0.05`. The paper's "physics-informed ablation" row will claim "+logistic + smoothness," but the smoothness term is non-functional. Reviewers checking the ablation table against the source will catch this. Worse, the docstring on line 110-114 explicitly markets the term as a smoothness penalty ("approximated as `|delta_pred|^2`" — but the code computes `|delta_pred - delta_true|^2`, not `|delta_pred|^2`, so even the docstring is wrong).
- **Fix:** Either (a) implement a real multi-step smoothness term by predicting an intermediate point or by penalizing model internal `d²h/dt²`, or (b) drop the term and remove `lambda_smooth` from the physics ablation config, and from the paper's claims. The safest minimum: penalize the *magnitude* of the predicted delta, which at least is a curvature prior on the forecast:
  ```python
  if cfg.lambda_smooth > 0.0:
      delta_pred = yp - od_last
      total = total + cfg.lambda_smooth * delta_pred.pow(2).mean()
  ```
  Mark the docstring honestly: "soft prior penalizing large predicted deltas (a forecast-magnitude regularizer, not a true second-difference smoothness)." If the paper needs a real smoothness loss, it must operate on a ≥3-point trajectory — which means the forecaster needs to emit intermediate samples (currently it only emits OD at horizon).

### B3. `aggregate_seed_metrics` uses ddof=0 — paper std is systematically understated
- **File:** `src/quantum_liquid_neuralode/evaluation/metrics.py:81`
- **Symptom:** `"std": float(vals.std(ddof=0))` with n=5 seeds. Population std vs. unbiased (sample) std differ by `sqrt(n/(n-1)) = sqrt(5/4) ≈ 1.118`. The paper currently underreports uncertainty by ~12% in every "mean ± std" cell.
- **Why it matters:** Standard practice in ML papers reporting "mean ± std over k seeds" is the unbiased estimator (`ddof=1`, equivalent to numpy's default in `np.std` is actually `ddof=0`, but `pandas.std` defaults to `ddof=1`; sklearn and most stats packages use `ddof=1`). With n=5, the difference is non-trivial. A reviewer running their own reproduction with `ddof=1` will get numbers that don't match the paper. At minimum this needs to be documented.
- **Fix:**
  ```python
  "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
  ```
  Also: emit both `std` and `sem = std / sqrt(n)` so the paper can use 95% CI directly. Add a regression test for the exact expected value with two seeds:
  ```python
  def test_aggregate_seed_metrics_std_is_sample_std():
      ms = [
          ForecastMetrics(mse_norm=0.01, mae_raw=0.1, rmse_raw=0.12, r2_raw=0.9),
          ForecastMetrics(mse_norm=0.03, mae_raw=0.2, rmse_raw=0.22, r2_raw=0.8),
      ]
      agg = aggregate_seed_metrics(ms)
      # sample std of {0.1, 0.2} == 0.05*sqrt(2) ≈ 0.0707
      assert agg["mae_raw"]["std"] == pytest.approx(0.05 * (2 ** 0.5), rel=1e-6)
  ```

---

## HIGH

### H1. Logistic-growth residual is a left-endpoint (forward-Euler) finite difference
- **File:** `src/quantum_liquid_neuralode/training/losses.py:62-68`; used in `trainer.py:128-134`.
- **Symptom:** The residual uses `od_mid = od_[..., :-1]` — the *left endpoint* of each interval — paired with the forward difference `dOD/dt ≈ (od[i+1] - od[i]) / dt`. The standard midpoint-rule centering would be `od_mid = 0.5*(od[i] + od[i+1])`.
- **Why it matters:** In `_physics_loss_terms` the trajectory is `[od_last, yp]`, length 2. So the loss is exactly `((yp - od_last)/h − mu * od_last * (1 − od_last/K))^2`. With `mu_norm=0.4` and `h=1`, this drives `yp − od_last ≈ 0.4 * od_last * (1 − od_last)`. For OD already near saturation (`od_last ≈ 1.0` normalized) the expected delta is ~0, which matches reality. For mid-growth (`od_last ≈ 0.5`) it pushes `yp − od_last ≈ 0.1`. The forward-Euler bias is in a known direction (overestimates growth in concave-down regions). Documented physics-informed-NN literature uses either midpoint or trapezoidal. With a 2-point trajectory you can't do better than forward-Euler — fine, but the paper should not claim "logistic-growth physics loss" without disclosing the discretization.
- **Fix:** Either (a) use the midpoint by replacing `od_mid = od_[..., :-1]` with `od_mid = 0.5*(od_[..., :-1] + od_[..., 1:])` in `logistic_growth_residual_loss` (semantically a midpoint Euler, slightly biased differently but more accurate for smooth dynamics), or (b) document the forward-Euler choice in the loss docstring and the paper methods section.

### H2. Logistic loss with `mu_norm=0.4` is applied in normalized space without dimensional analysis
- **File:** `src/quantum_liquid_neuralode/training/trainer.py:24-25, 130-134`
- **Symptom:** `PhysicsLossConfig.mu_norm: float = 0.4 # growth rate (1/h)` paired with `K_norm: float = 1.0`. The spec calls for `μ=0.3, K=3.8` in raw OD units. Since the logistic equation `dOD/dt = μ OD (1 − OD/K)` is invariant under the change of variables `u = OD/K` (giving `du/dt = μ u (1 − u)`), the SAME μ works in both raw and normalized space *as long as `K_norm = OD_max/OD_max = 1`*. With fixed bounds `[0.0, 3.8]`, that holds. So mathematically `mu_norm=mu_raw`. The 0.4 vs 0.3 discrepancy is a separate decision — fine, but not justified anywhere.
- **Why it matters:** A reviewer will check whether `μ` is set from the data. Currently it's hardcoded. The spec quotes 0.3 ("Estimated from data") but the config uses 0.4 without explanation. The physics ablation row's numbers depend on this choice and there is no sensitivity analysis.
- **Fix:** Either (a) fit μ from training data (linearize log(OD/(K−OD)) vs time, take slope), document the fit; or (b) cite the source of 0.4 and run a small sensitivity sweep (0.2, 0.3, 0.4, 0.5) for the paper appendix.

### H3. `make_horizon_windows` silently drops train windows whose target lies in val
- **File:** `scripts/train_baseline.py:71-96` (segment-windowing helper); `windowing.py:147-154`
- **Symptom:** Each split is windowed independently using the LOCAL `time_hours[start:end]` slice. A window whose `end_idx` is inside `train` but whose `target_idx` would land in `val` exceeds the local `n` and is dropped. Per-segment, this discards roughly `horizon_hours / dt = 6` windows at the train→val boundary, and another ~6 at val→test.
- **Why it matters:** It is the right thing to do for *leak avoidance* (no information from val flows into train training labels). However, the dropped windows are real labeled examples that no model in the paper sees. This is fine as a fixed protocol, but the `protocol.json` should record that ~12 valid windows per run are excluded by segment boundaries so the dataset size in the paper matches reality. Worse, the legacy `train_liquid_od_baseline.py` may not enforce the same convention — check.
- **Fix:** Document the choice in `protocol.json`:
  ```python
  protocol["windows_dropped_at_segment_boundaries"] = (
      # approximate; compute exactly by also windowing the union
  )
  ```
  Add a `make_horizon_windows` test that verifies a window whose horizon falls past the local segment is dropped (not silently included with a wrong target):
  ```python
  def test_make_horizon_windows_no_target_outside_segment():
      n = 20
      t = np.arange(n, dtype=np.float64) / 6.0
      od = np.linspace(0.1, 0.9, n).astype(np.float32)
      feats = od.reshape(-1, 1)
      win = make_horizon_windows(
          features=feats, od=od, time_hours=t,
          window_size=6, stride=1, horizon_hours=1.0, horizon_tolerance_hours=1e-6,
      )
      assert int(win.target_idx.max()) < n
  ```

### H4. No test that the best-checkpoint state matches the best val MSE
- **File:** `tests/test_trainer.py`
- **Symptom:** `train_one` records `best_state` inside the eval block, then reloads it at the end. There's no test confirming that, after training, `compute_metrics(model_with_loaded_best_state) == val_metrics_at_best_epoch`. With early stopping, this is the single most important determinant of paper numbers.
- **Why it matters:** A regression (e.g., a future refactor that re-shuffles state-dict cloning) could silently start reporting the *last* model rather than the *best* — and the paper would not catch it. This is exactly the kind of bug Sjöberg-style reviewers ask about.
- **Fix:** Add:
  ```python
  def test_train_one_returns_best_checkpoint_state():
      # Use a tiny model and verify that val_metrics returned matches metrics
      # computed by loading result.model_state into a fresh model.
      ...
      result = train_one(...)
      fresh = LiquidODForecaster(...same args...)
      fresh.load_state_dict(result.model_state)
      fresh.eval()
      # Recompute val MSE_norm with `fresh` and compare to result.val_metrics.mse_norm.
      assert recomputed_val_mse == pytest.approx(result.val_metrics.mse_norm, rel=1e-5)
  ```

### H5. No multi-seed determinism test
- **File:** `tests/test_trainer.py`
- **Symptom:** No assertion that running with `seed=0` twice yields the same final `val_metrics.mse_norm`. With MPS some ops are non-deterministic, but on CPU PyTorch + numpy with fixed seeds should be bit-reproducible.
- **Why it matters:** The paper makes a multi-seed-mean ± std claim. If two runs of seed=0 give different numbers, the seed contract is broken and the "± std" is mixing seed variation with non-determinism. Reviewers ask.
- **Fix:** Add a CPU-only test:
  ```python
  def test_train_one_is_deterministic_on_cpu():
      r1 = train_one(..., seed=0, device=torch.device("cpu"))
      r2 = train_one(..., seed=0, device=torch.device("cpu"))
      assert r1.val_metrics.mse_norm == pytest.approx(r2.val_metrics.mse_norm, rel=1e-6)
  ```
  Note that `_to_loader` uses default DataLoader which on CPU is deterministic given `torch.manual_seed(seed)` is set inside `train_one` before constructing loaders. If this test fails, file a separate bug.

---

## MEDIUM

### M1. Forecaster history-evolution uses the LEFT-endpoint input over each interval
- **File:** `src/quantum_liquid_neuralode/models/forecaster.py:228-232`
- **Symptom:** `for i in range(T - 1): dt = t[:, i+1] - t[:, i]; h = self._integrate(h=h, x=x[:, i, :], dt=dt, n_substeps=1)`. The input is held at `x[i]` (left endpoint) during the interval `[t_i, t_{i+1}]`. The LAST observation `x[T-1]` is then used only as the forecast-horizon constant input — never within the history.
- **Why it matters:** Each step's input is "stale" by one observation. For 10-min spacing this is minor, but the model is forced to map a 23-step input history (effectively) plus a final t-end value into the horizon. The asymmetric treatment is defensible — it preserves time-ordering — but the docstring at line 58-62 says "with the step's input held constant" without clarifying *which* step's input. For a reviewer who runs the equation through, this is ambiguous.
- **Fix:** Document the convention clearly in the docstring. Optionally offer a configurable "midpoint input" mode (`x_mid = 0.5*(x[i] + x[i+1])`) as an ablation. Not blocking.

### M2. `_integrate_euler` substep semantics for history vs horizon are inconsistent
- **File:** `src/quantum_liquid_neuralode/models/forecaster.py:158-163, 232, 236-241`
- **Symptom:** During history, `n_substeps=1` over each ~10-min interval (so Euler is at full data resolution). During horizon (1h), `n_substeps=forecast_steps=4` so sub-dt=0.25h. The history Euler step size (0.167h) is smaller than the horizon Euler step size (0.25h).
- **Why it matters:** For a unified ODE-solver claim, the integrator should have a *consistent* accuracy budget. With dt/tau ≈ 0.32 on horizon and ≈0.21 on history, Euler is OK but not great. The `forecast_steps=4` config value is from HPO — fine — but the paper should mention that the effective Euler sub-step over the horizon is 15 min, *larger* than the data resolution. dopri5 sidesteps this. Worth a one-line comment.
- **Fix:** None required if dopri5 is the headline. For the Euler-fast config, increase `forecast_steps` to 6 so sub-dt matches data resolution (~10 min).

### M3. `_Scaled` is redefined as an inner class on every dopri5 forward pass
- **File:** `src/quantum_liquid_neuralode/models/forecaster.py:189-198`
- **Symptom:** `class _Scaled(nn.Module): ...` is defined inside `_integrate_dopri5`. Every forward pass creates a new class object and a new instance.
- **Why it matters:** Class creation in CPython is cheap but non-zero; more importantly, the freshly-created `nn.Module` instance has an empty `_parameters` dict and just wraps a reference to `vf` and `scale`. There's no parameter registration bug (the underlying cell's params are reached via `self.base` → `self.base.cell`), but the pattern is fragile to future changes. Also: torchdiffeq introspects the module's parameters for adjoint differentiation in some modes; using `odeint` (not `odeint_adjoint`) this is fine, but is worth noting in case adjoint mode is enabled later.
- **Fix:** Hoist `_Scaled` to module scope, or use a closure-based callable:
  ```python
  def scaled_vf(t: Tensor, h_: Tensor) -> Tensor:
      return dt_t * vf(t, h_)
  sol = odeint(scaled_vf, h, u, method="dopri5", rtol=self.rtol, atol=self.atol)
  ```
  torchdiffeq accepts plain callables for `odeint` (not `odeint_adjoint`). Verify with a unit test if you change.

### M4. `load_qzeta` uses `format="mixed"` — pandas explicitly warns this can produce inconsistent parses
- **File:** `src/quantum_liquid_neuralode/data_processing/qzeta.py:39`
- **Symptom:** `pd.to_datetime(df["DATE"], format="mixed", dayfirst=True, errors="raise")`. The `format="mixed"` mode infers per-row, and pandas warns: "When using `format='mixed'`, please specify each format separately... using mixed can yield surprising results."
- **Why it matters:** If `qZETA_data_copy.csv` is ever updated and a row has an ambiguous date (e.g., `01/02/2024`), pandas may swap day/month silently. For a paper that locks evaluation to this exact CSV, the parse must be deterministic and verifiable.
- **Fix:** Inspect the CSV, identify the actual date format (likely `DD/MM/YYYY HH:MM:SS` or similar), and pin it:
  ```python
  dt = pd.to_datetime(df["DATE"], format="%d/%m/%Y %H:%M:%S", errors="raise")
  ```
  Add a test that loads a fixture CSV with a known timestamp and asserts the parsed value.

### M5. `_to_loader` does not pin its own generator — DataLoader shuffle inherits global RNG
- **File:** `src/quantum_liquid_neuralode/training/trainer.py:62-68`
- **Symptom:** `DataLoader(..., shuffle=True)` without a `generator` argument. PyTorch uses the global RNG for the sampler. `train_one` does call `torch.manual_seed(seed)` before constructing the loader, so the per-seed initial shuffle order is deterministic. But any RNG consumption between seeds (parameter init in train_baseline.py, etc.) won't affect the *first* shuffle of `train_one` — because `train_one` reseeds — but later epochs continue to draw from a global RNG that may be perturbed by ODE solver internals or other consumers.
- **Why it matters:** Determinism in this layout is correct *as long as* the model forward / backward consumes no random numbers. `LiquidODForecaster` uses no dropout, no random noise — OK. But this is brittle: a future addition of dropout would silently break seed determinism across epochs.
- **Fix:** Make the loader's RNG explicit:
  ```python
  g = torch.Generator()
  g.manual_seed(seed)
  return DataLoader(ds, batch_size=..., shuffle=shuffle, generator=g, drop_last=False)
  ```
  And pass `seed` into `_to_loader` (currently it doesn't take one). Combined with the M5-related test in H5, this locks reproducibility.

### M6. No test that train/val/test windows do not overlap in time
- **File:** `tests/test_windowing.py`
- **Symptom:** No test asserting that `w_train.target_idx.max() < w_val.end_idx.min()` and similar boundary checks.
- **Why it matters:** Paper claim of "chronological splits, no leakage." A regression in segment slicing (`_segment_windows`) could silently violate this and reviewers will ask "how do you know."
- **Fix:** Add an integration-style test that runs the train_baseline.py windowing path on a synthetic 100-row time series and asserts `train.target_idx < val.end_idx.min()` and `val.target_idx < test.end_idx.min()`.

### M7. `_physics_loss_terms` is invoked even when both lambdas are 0 if check above is bypassed
- **File:** `src/quantum_liquid_neuralode/training/trainer.py:220-228`
- **Symptom:** The guard is `if cfg.physics.lambda_logistic > 0.0 or cfg.physics.lambda_smooth > 0.0`. With both lambdas = 0 (baseline config), `_physics_loss_terms` is not called — good. But `_physics_loss_terms` itself adds 0 if a lambda is 0, so calling it would be a no-op. The guard is defensible but tightly coupled. If a third regularizer is added and the guard isn't updated, it'd be skipped. Low risk for now.
- **Fix:** Refactor to a list of (lambda, term_fn) so adding regularizers can't desync. Not blocking.

### M8. Smoothness loss requires T ≥ 3 but the API exposes that requirement only in the call site
- **File:** `src/quantum_liquid_neuralode/training/losses.py:98-99`
- **Symptom:** `smoothness_loss` raises if `T < 3`. In `_physics_loss_terms`, the comment explains that with only 2 points the function can't be used — but `smoothness_loss` is then never called. Dead path / misleading docstring.
- **Fix:** Either remove `smoothness_loss` from `trainer.py`'s import (it's unused there), or call it on a longer trajectory once the model emits intermediate forecasts.

---

## LOW

### L1. `LiquidCell.tau()` re-computes softplus on every call inside the integration inner loop
- **File:** `src/quantum_liquid_neuralode/models/liquid_cell.py:91`
- **Symptom:** For RK4 with `n_substeps=4`, that's 16 softplus calls per horizon integration (plus history). The tensor is small (hidden_size=32) so cost is negligible.
- **Fix:** None — keep clarity. Maybe `self._tau_cached` if profiling shows it matters.

### L2. `_integrate_dopri5` imports `torchdiffeq` inside the function
- **File:** `src/quantum_liquid_neuralode/models/forecaster.py:180`
- **Symptom:** Lazy import. Fine for keeping the package usable without `torchdiffeq` installed, but it adds per-call overhead on first use. Not a perf concern.
- **Fix:** None.

### L3. `_physics_loss_terms` docstring contains an inaccurate description
- **File:** `src/quantum_liquid_neuralode/training/trainer.py:111-114`
- **Symptom:** "here approximated as `|delta_pred|^2`" — the code actually computes `|delta_pred - delta_true|^2`. Tied to B2.
- **Fix:** Rewrite docstring after fixing B2.

### L4. `linear_extrapolation_forecast` returns float32 but uses NaN as a sentinel mid-computation
- **File:** `src/quantum_liquid_neuralode/evaluation/baselines.py:31-34`
- **Symptom:** `safe_dt = np.where(dt > 0, dt, np.nan)` promotes to float64, division produces NaN, then `np.where(np.isfinite(pred), pred, od_last)` masks them out. Cast to float32 at the end. Correct, but the NaN intermediate is a code smell — a future contributor might delete the `isfinite` mask thinking it's redundant.
- **Fix:**
  ```python
  pred = np.where(
      dt_last_hours > 0,
      od_last + (od_last - od_prev) * (horizon_hours / np.where(dt_last_hours > 0, dt_last_hours, 1.0)),
      od_last,
  )
  return pred.astype(np.float32)
  ```
  Or simply branch with two-step `np.where`.

### L5. `BioreactorDataPreprocessor` is "legacy" but still imported in tests
- **File:** `src/quantum_liquid_neuralode/data_processing/preprocessor.py`, `tests/test_preprocessor.py`
- **Symptom:** Tests exercise the legacy path. The legacy class shares the windowing off-by-one (B1) and uses non-temporal `MinMaxScaler` (whole-df fit, not train-only) — which is a leak if it ever feeds into a paper number. The class is exported from `data_processing/__init__.py`.
- **Fix:** Either delete (preferred) or stamp a `DeprecationWarning` and document that it is not used by `scripts/train_baseline.py`. Update tests accordingly.

### L6. `compute_metrics` returns NaN for R² on constant `y_true_raw`
- **File:** `src/quantum_liquid_neuralode/evaluation/metrics.py:53`
- **Symptom:** `sklearn.r2_score` returns 0.0 (and warns) when variance of `y_true` is 0; OK on real data where val/test variance is non-zero.
- **Fix:** None for now; tests cover non-pathological cases.

### L7. Configs duplicate the same training hyperparameters across three YAMLs
- **File:** `configs/baseline.yaml`, `configs/baseline_euler_fast.yaml`, `configs/baseline_physics.yaml`
- **Symptom:** lr, batch_size, weight_decay, patience, etc. are copy-pasted. A future tuning sweep needs to be applied in three places.
- **Fix:** Either accept this (research-code convention) or split into a `base.yaml` + override files. Not blocking.

### L8. `tests/test_losses.py:8` tests `logistic_growth_residual_loss` on all-zero OD
- **File:** `tests/test_losses.py:7-13`
- **Symptom:** Loss on `od = 0` is trivially zero because both the derivative and the expected term are zero. This doesn't actually exercise the residual logic — pretty weak test.
- **Fix:** Add a test that a logistic trajectory `OD(t) = K/(1 + exp(-mu*(t - t0)))` gives near-zero loss with appropriate sampling, and a non-logistic trajectory (e.g. linear) gives nonzero loss.

### L9. `train_baseline.py` writes `config.json` from the *post-CLI-override* config but does not separately save the original YAML
- **File:** `scripts/train_baseline.py:144`
- **Symptom:** If a reviewer asks "what config produced this run," the frozen `config.json` reflects CLI overrides. Generally fine, but the relationship to the source YAML is lost.
- **Fix:** Also `shutil.copy(args.config, output_dir / "config.source.yaml")` for traceability.

### L10. No assertion that `delta_scale` is large enough to learn the observed delta range
- **File:** `configs/*.yaml` (`delta_scale: 0.1`); `forecaster.py:245`
- **Symptom:** Output is `od_last + tanh(.) * 0.1`. With OD ranging in raw [0, 3.8] (normalized [0, 1]) and 1-hour growth being possibly ~0.3 normalized in log phase, the delta cap of 0.1 may be undersized. If the true delta exceeds 0.1, the model can NEVER produce a perfect forecast — it is structurally biased toward persistence.
- **Why it matters:** Paper claim "Liquid-ODE beats persistence" is in part determined by `delta_scale`. The HPO presumably selected this; document the selection.
- **Fix:** Add a one-line check at startup: `assert delta_scale > (estimated_max_true_delta * 1.5)` warning if not. Or include `delta_scale` in the HPO appendix with a sensitivity sweep.

---

## What is actually correct

For balance — these were checked and are sound:

- **Residual formulation** (`forecaster.py:243-246`): `od_last + tanh(delta_head(h)) * delta_scale`. ✓
- **dopri5 rescaling** (`forecaster.py:176-203`): u ∈ [0, 1], dh/du = dt · f(h, x). The cell ignores `t`, so the integrator's t-argument has no effect. Sign and broadcasting correct. ✓
- **Per-seed parameter init** (`train_baseline.py:268-282`): `torch.manual_seed(seed)` is called before `LiquidODForecaster(...)` construction. Each seed gets fresh, distinct parameter initialization. ✓
- **OD fixed-bounds MinMax** (`windowing.py:57-63`): When `fixed_bounds` is provided for OD, the scaler is fit on `[0, 3.8]` not on training data — no leak. ✓
- **Per-segment windowing** prevents future leakage: train windows whose horizon target falls in val are correctly dropped. ✓ (But see H3 — they're dropped silently.)
- **`linear_extrapolation_forecast`** correctly handles `dt == 0` via NaN-mask → fall back to persistence. ✓
- **Best-checkpoint state** is captured in `train_one:249` and reloaded at the end before computing val/test metrics. ✓ (But not regression-tested — see H4.)
- **History integration uses the real, data-driven `dt = t[i+1] - t[i]`**: asynchronous-sampling preserved. ✓

---

## Suggested fix order for paper submission

1. **B1** (one-line off-by-one) — re-run all seeds and update results.
2. **B2** (smoothness loss) — re-run `+physics` ablation with either a fixed term or remove the smoothness column from the paper.
3. **B3** (ddof=1) — update aggregation; re-emit `seeds_summary.json` (no retraining needed since per-seed metrics are unchanged).
4. **H1, H2** — document the discretization and the choice of `μ`.
5. **H4, H5** — add the regression tests; they take an hour and prevent embarrassment.
6. **M-series** — tighten on the way to paper revision.

---

_End of review._
