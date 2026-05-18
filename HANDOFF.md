# Handoff to the next coding agent

You're picking up a paper-ready research codebase. Empirical work is **complete**.
What's left is small code tweaks (this doc) and the paper draft itself
(out of scope for a coding agent — needs the user's voice).

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
```

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
