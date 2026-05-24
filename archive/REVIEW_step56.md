# Step 5 + Step 6 Adversarial Code Review

**Date:** 2026-05-18
**Scope:** Step 5 (effective dimension) and Step 6 (sample efficiency)
plus the paper-prep utilities (figures, reproduce script, integrity check)
and the four docs (`hypothesis.md`, `PAPER_SUMMARY.md`, `README.md`,
`CLAUDE.md`, `STEP5_MONOTONICITY_NOTE.md`).
**Posture:** adversarial — assume defects.

## Executive summary

The Step 5/6 implementation is, on the whole, mathematically careful — the
trace-normalized empirical Fisher formula is implemented correctly on both
the JAX and PyTorch sides, the JAX vs PyTorch equality test is non-trivial
(uses a real PSD matrix, not an identity), the `head()` chronological-
truncation logic is correct, and `n_train_windows` is recorded in
`protocol.json`. The corrected `STEP5_MONOTONICITY_NOTE.md` math is right
for the idealized rank-r model but glosses over the realistic
continuous-spectrum case. However there are real, paper-affecting defects:
(BLOCKER) a documentation inconsistency between CLAUDE.md/PAPER_SUMMARY.md
and the actual code state — CLAUDE.md still describes the pre-QWGAN-drop
roadmap (Phase 4 QWGAN-GP, Phase 5 forthcoming) directly contradicting
hypothesis.md v2 and PAPER_SUMMARY.md; (HIGH) the JAX-side
`empirical_fisher` declares `dtype=jnp.float64` but with `jax_enable_x64`
deliberately off this silently downgrades to float32, so the docstring
("float64") and code disagree (live in test code, not on the headline
path, but a future caller would be misled); (HIGH) `make_paper_figures.py`
hardcodes `n_train_classical = [47, 118, 236, 472]` instead of reading
`protocol.json` — a `train_fraction` change in any config silently
mis-labels the figure; (HIGH) `verify_paper_integrity.py` Claim 1 reads
`results/param_sweep/euler_h3_hidden4/` while Claim 3 100% reads
`results/sample_efficiency/classical_h4_h3_pct100/` — two independent
training runs of the SAME config, so the integrity check is silently
exposed to seed-determinism drift across the two runs; (HIGH)
`run_sample_efficiency.sh` re-trains the 100% cell as a duplicate of the
existing `qlnn_hybrid_h3` / `euler_h3_hidden4` artifacts at 4-hour cost,
without invocations that pin determinism; (HIGH) the `head()` method has
no unit test for boundary conditions; (HIGH)
`pre_registered_hypothesis_met = bool(threshold_met > 1.0)` uses strict
greater-than but the hypothesis writeup says "exceeding 1 unit" — fine,
but the verify script tolerance of 0.01 on Δd_norm could in principle flip
the flag while passing integrity, an inconsistency. The full list with
severity is below.

---

## BLOCKER

### B-01 — `CLAUDE.md` contradicts the v2 paper framing it should describe

**File:** `CLAUDE.md:88-92`
**What's wrong:** The "Intended pipeline (from spec.md)" section still
reads "Phase 4 (QWGAN-GP synthetic generator), and Phase 5 (Fisher /
effective-dimension expressivity diagnostics) are planned." Phase 5 is
DONE (Step 5 just shipped). Phase 4 (QWGAN-GP) was explicitly DROPPED
per `hypothesis.md` v2 §"Deviations from v1". The CLAUDE.md header
correctly states "experiments complete; paper writing is the remaining
work" but the architecture section says the opposite. Line 69
("DTW … once QWGAN lands") is also stale.
**Why it matters:** Reviewer who reads CLAUDE.md and hypothesis.md side by
side gets contradictory statements about the project's scope. The
"deviations from pre-registration" section in the paper must accurately
reflect what was dropped — having an actively-shipped doc that still
describes QWGAN as upcoming undermines the pre-registration discipline.
**Remediation:** Rewrite CLAUDE.md §"Intended pipeline" to describe the
v2 final state: Phase 1 classical (done), Phase 2/3 QLNN (done), Phase 5
effective dimension (done), Phase 6 sample efficiency (done), QWGAN-GP
explicitly dropped — reference hypothesis.md v2 §"Deviations from v1".
Remove the "once QWGAN lands" parenthetical on line 69.

### B-02 — `PAPER_SUMMARY.md` "What's blocked and what's next" still lists QWGAN-GP as Step 4

**File:** `PAPER_SUMMARY.md:149,156-170`
**What's wrong:** Line 149 says `**Synthetic data lift** (Step 4 / QWGAN): primary endpoint = test MAE at h=3, paired-bootstrap p < 0.05, K=472 1:1 augmentation.` — this is a v1 binding that's been retracted. Lines 166-170 still say "Step 4 (next): QWGAN-GP synthetic generator…", "Step 5: Effective dimension via empirical Fisher…", "Step 6: Sample efficiency via data-fraction sweep." This contradicts the header ("Phases A/B/C complete + scope refocused") and the in-line claim 3 table.
**Why it matters:** PAPER_SUMMARY.md is declared "single source of truth for paper numbers" in CLAUDE.md, but it tells two contradictory stories about what the paper actually covers. A reviewer searching the doc for the canonical claims list will find the v2 table at the top AND the v1 step list at the bottom.
**Remediation:** Delete or rewrite §"Locked methodology bindings" line 149 (QWGAN entry) and the entirety of §"What's blocked and what's next" to reflect that Steps 4 (QWGAN) has been DROPPED and Steps 5/6 are COMPLETE. The "Step 4 (next)" phrasing is the clearest contradiction; remove it.

---

## HIGH

### H-01 — JAX-side `empirical_fisher` claims float64 but silently runs float32

**File:** `src/qlnn_/diagnostics/effective_dimension.py:88-103`
**What's wrong:** With `jax_enable_x64` deliberately OFF (lines 51-56), the
statement
```python
fisher = jnp.zeros((D, D), dtype=jnp.float64)
...
g = g.astype(jnp.float64)
fisher = fisher + jnp.outer(g, g)
```
does NOT produce a float64 Fisher. JAX silently downgrades any
`jnp.float64` request to float32 when x64 is disabled, with a warning that
is easy to miss. The returned `fisher / float(n)` is float32. The
docstring is silent about this, but the symmetric numpy-based code in
`run_effective_dimension.py:_fisher_qlnn` (lines 210-216) does the
accumulation in numpy float64 explicitly to work around the issue, which
betrays that the author knew about the problem on the script side but
left the library API broken.
**Why it matters:** Any future caller (or a test that doesn't bypass the
library, like `test_empirical_fisher_jax_matches_finite_difference`) will
get float32 precision while the docstring promises float64. The current
headline result is unaffected because the run script does not use this
function — but Step 5 is explicitly designed to be REUSED for future
expressivity analyses, and the dead-code path will mislead. The
`test_jax_and_torch_normalized_effective_dimension_agree` test loosens
tolerance from 1e-8 to 1e-6 for exactly this reason; the agreement is
limited by the unused-internally float32 accumulation that nobody actually
exercises.
**Remediation:** Either (a) delete the unused JAX-side `empirical_fisher`
and document that the canonical accumulation path is the numpy-float64
one in the run script, or (b) replace its accumulation with `np.asarray`
+ float64 outer-product (mirroring what `_fisher_qlnn` does in the script
and what `_slogdet_psd` already does internally), and update the docstring
to say "Returns: (D, D) float64 numpy array (cast inside to avoid x64
JAX contagion)."

### H-02 — `make_paper_figures.py` hardcodes `n_train_classical` instead of reading `protocol.json`

**File:** `scripts/make_paper_figures.py:136`
**What's wrong:**
```python
n_train_classical = [47, 118, 236, 472]   # from protocol.json
```
The comment admits these came from protocol.json but the values are
hardcoded. If any of the four `classical_h4_h3_pct{10,25,50,100}.yaml`
configs has its `train_fraction` changed, or if the underlying split
changes by a row, the figure will silently mis-label the x-axis with stale
window counts that don't match the actually-trained model.
**Why it matters:** Paper figure 2 ("sample-efficiency crossover") is the
headline figure for Claim 3. An x-axis mis-label here is reviewer
bait. The information is already on disk in
`results/sample_efficiency/{classical_h4_h3_pct*,qlnn_h3_pct*}/protocol.json`,
and `summarize_sample_efficiency.py:58-64` already reads it.
**Remediation:**
```python
n_train_classical = []
for pct in fractions:
    p = _load(f"results/sample_efficiency/classical_h4_h3_pct{pct}/protocol.json")
    n_train_classical.append(int(p["n_train_windows"]))
```
Same for QLNN if the per-stack counts can differ (they shouldn't,
identical window-construction logic, but assert and use the classical
value).

### H-03 — Claim 1 verification source and Claim 3 100%-cell run on different result trees

**File:** `scripts/verify_paper_integrity.py:32-35` vs `59`
**What's wrong:** Claim 1 verification reads
`results/param_sweep/euler_h3_hidden4/seeds_summary.json` (the Phase-C
sweep). Claim 3 at pct=100 reads
`results/sample_efficiency/classical_h4_h3_pct100/seeds_summary.json`.
Both runs use the SAME config (apart from train_fraction=1 which is a
no-op), the same seeds, the same data — they should produce identical
numbers. But they are two independent training runs and any source of
non-determinism (MPS BLAS, torch float-precision differences across
PyTorch versions, atomic-add nondeterminism inside torchdiffeq) will make
them disagree.
**Why it matters:** A green integrity check should mean "the paper's
numbers are reproducible from the on-disk artifacts." If Claim 1 reads
the Phase-C tree and Claim 3 reads the Step-6 tree, the paper-summary
"3.77×" headline can be GREEN in one tree and RED in the other after a
clean re-run. Worse: the σ_classical ratio that goes into the abstract is
computed from results that aren't even in the same directory as the
Step-6 sample-efficiency table.
**Remediation:** Either (a) consolidate — make Claim 1 read from the
Step-6 pct100 tree (which is the canonical post-Step-6 artifact set), OR
(b) add an assertion that the two seeds_summary.json files agree
bit-for-bit on the headline MAE values before computing the ratio. Option
(a) is cleaner.

### H-04 — `run_sample_efficiency.sh` re-trains the 100% cell as a duplicate of existing artifacts

**File:** `scripts/run_sample_efficiency.sh:21-37`
**What's wrong:** The 100% pcts (`pct100` classical and QLNN) are
duplicates of the existing `results/param_sweep/euler_h3_hidden4/` and
`results/qlnn_hybrid_h3/`. The script comment acknowledges this and says
it re-runs them "to be self-contained and to write predictions.npz with
the (post-Phase-C) shape if the older runs are missing fields" — but the
shell script unconditionally runs all four pcts, so the 100% cell is
re-executed every time (4-hour wallclock at the QLNN end), and there's no
check that it agrees with the canonical run.
**Why it matters:** Two-fold: (a) `reproduce_paper.sh` is documented as
~8 hours; this duplicate wastes ~4hr of that, encouraging future runs to
skip it; (b) when researchers DO skip the duplicate, the integrity check
that depends on it (Claim 3 pct=100, line 64 of verify_paper_integrity.py)
silently uses stale numbers from an unrelated branch.
**Remediation:** Either (a) symlink / copy `results/qlnn_hybrid_h3` →
`results/sample_efficiency/qlnn_h3_pct100` and same for classical, with a
small `make-symlinks.sh` invoked by `reproduce_paper.sh` instead of
re-training; OR (b) keep the re-run but assert agreement with the
canonical run in `summarize_sample_efficiency.py`.

### H-05 — No unit test for `HorizonWindows.head()` boundary conditions

**File:** `tests/test_effective_dimension.py` (no Step 6 test file exists)
**What's wrong:** `head()` (the critical mechanism for Claim 3's
chronological truncation) has the following untested behaviors:
- `head(0)`: should raise (covered by `n <= 0` guard).
- `head(-1)`: should raise (also covered).
- `head(len(self))`: should return an equivalent (or self-like) window
  set.
- `head(len(self) + 1)`: should raise.
- The truncated `target_idx`, `end_idx` arrays should preserve the
  original (pre-truncation) global indices (the truncation is on the
  FIRST axis only). This invariant is what guarantees the temporal-split
  discipline.

None of these has a regression test. A typo like `self.x[:n]` →
`self.x[n:]` would pass the integration smoke test (training would still
proceed, just on a different subset) and silently invalidate the entire
sample-efficiency claim.
**Why it matters:** Claim 3 is one of the three paper claims. The
`head()` method is the single point of failure for "chronologically
truncated from the start." A reviewer asking "how do I know the model
didn't peek at later windows" needs to be answered by code that has a
unit test pinning the truncation semantics.
**Remediation:** Add `tests/test_windowing_head.py`:
```python
def test_head_n_too_large_raises(): ...
def test_head_zero_raises(): ...
def test_head_keeps_first_n_chronologically():
    # Build a known-order HorizonWindows, head(3), verify
    # that end_idx is monotonic and ends at the (n-1)th smallest end_idx.
def test_head_preserves_global_indices(): ...
def test_head_equal_to_len_is_equivalent(): ...
```

### H-06 — Trace-normalization test passes for the wrong reason

**File:** `tests/test_effective_dimension.py:195-206`
**What's wrong:** The test scales F by 123.4 and asserts d_norm is
unchanged. That's correct for the trace-normalization step. But the
test's docstring claims "If theta -> alpha*theta, the Fisher scales by
1/alpha^2 (since gradient scales by 1/alpha for chain rule)". That's
backwards — gradient w.r.t. theta scales by 1/alpha if you reparametrize
theta → alpha·theta (because df/d(theta_new) = (1/alpha) df/dtheta).
Fisher = J^T J would then scale by 1/alpha^2, not by an arbitrary scale.
The test doesn't actually exercise a reparametrization; it just multiplies
F by a constant.
**Why it matters:** The test demonstrates that the trace-normalization
gives scale-invariance (which it does — trivially, since F * c → trace
* c → F_norm unchanged). It does NOT demonstrate the actual property that
matters for the paper — that the Fisher's value is independent of the
parametrization choice (radians vs. turns example in the module
docstring). The test mis-described means a future reader will assume the
parameter-rescaling invariance has been verified when it hasn't.
**Remediation:** Either rename the test to
`test_d_norm_is_invariant_to_scalar_rescaling_of_Fisher` and rewrite the
docstring to drop the reparametrization claim, OR add a second test that
actually computes F for `f(theta)` and `f(alpha·theta)` from an autograd
chain and shows d_norm agrees.

### H-07 — Sample-efficiency pct100 configs use `train_fraction: 1` (int) — typo-risk

**File:** `configs/sample_efficiency/*_pct100.yaml:15-16` (classical and QLNN)
**What's wrong:** Both pct100 YAMLs write `train_fraction: 1` (parsed by
PyYAML as Python int). The wiring in `train_baseline.py:274` does
`float(win.get("train_fraction", 1.0))`, so `1` → `1.0` and the test
`train_fraction < 1.0` skips the head() call. Behavior is correct.
**Why it matters:** Cosmetic but high-friction: every other config in
this set has a decimal (`0.1`, `0.25`, `0.5`), and only the 100% case
uses an int. A reviewer parsing YAMLs by eye will read `1` as
"experimental knob is on at default value" rather than as "no-op". More
importantly, if anyone later changes the parser to use `yaml.safe_load`
with strict types and removes the `float()` cast, the comparison `1 <
1.0` is still True in Python so this happens to work — but `1 != 1.0`
in a stricter language. Brittle.
**Remediation:** Make all four configs use the same decimal form:
`train_fraction: 1.0`. Same for `qlnn_h3_pct100.yaml`.

### H-08 — `_check` tolerance of 0.1 on a 3.77× ratio is two sig figs of slack

**File:** `scripts/verify_paper_integrity.py:35`
**What's wrong:** `_check("ratio (paper: 3.77)", ratio, 3.77, tol=0.1)`
accepts 3.67 ≤ ratio ≤ 3.87. The pre-registration threshold is ratio ≥
2.0. So a regression that drops the ratio to 2.05 (still passing
pre-reg but no longer the headline finding) would silently pass the
integrity check.
**Why it matters:** The reproducibility headline (3.77×) is the FIRST
of the three claims; a 0.1 tolerance is large enough that a real
regression in QLNN-side variance (the actual surprise finding of the
paper) wouldn't be flagged. The verify script is the last line of
defense against the paper drifting away from its own numbers.
**Remediation:** Tighten to `tol=0.05` (one sig fig) or — better —
verify the underlying stds (c.std and q.std) match the paper's
0.0166 and 0.0044 individually at `tol=0.001` so the ratio is computed
fresh and you check it in addition to the components.

### H-09 — `_check` returns `bool` but the unicode tick mark may break in non-UTF locales

**File:** `scripts/verify_paper_integrity.py:23-25`
**What's wrong:** `print(f"  {status} {label}: ...")` where status is "✓"
or "✗". If the script is run with `LANG=C` or `PYTHONIOENCODING=ascii`
(both common in CI), this raises `UnicodeEncodeError` and the script
crashes with exit code 1 — which the harness interprets as "integrity
failed", an actionable false-negative.
**Why it matters:** Continuous-integration false negatives erode trust in
the integrity gate (people start ignoring it).
**Remediation:** Replace ✓/✗ with `[OK]`/`[FAIL]` or sys.stdout.buffer
with explicit utf-8 — `[OK]/[FAIL]` is simpler and equally clear.

### H-10 — `reproduce_paper.sh` pre-flight `pytest` step won't surface failures

**File:** `scripts/reproduce_paper.sh:18-19`
**What's wrong:**
```bash
.venv/bin/python -m pytest -q --no-header > /dev/null
echo "[pytest] $(\.venv/bin/python -m pytest --collect-only -q 2>/dev/null | tail -1 | awk '{print $1, $2}') passing"
```
The first call discards stdout but is the actual gate (set -e + non-zero
exit). The second call's exit code is captured by `$()` and is the
LAST command in the pipeline (`awk`), not pytest. If pytest fails on the
second call, the user only sees the awk-formatted line and the script
keeps going (because awk exits 0). Worse: the first call's `> /dev/null`
swallows the failure message, so when set -e DOES kill the script the
user has no idea what failed.
**Why it matters:** `reproduce_paper.sh` is an 8-hour run. Failing fast
with diagnostics is the whole point of putting pytest at the top. Silent
suppression defeats it.
**Remediation:**
```bash
echo "[pytest] running…"
.venv/bin/python -m pytest -q || { echo "PYTEST FAILED — aborting reproduce_paper.sh"; exit 1; }
```

---

## MEDIUM

### M-01 — `STEP5_MONOTONICITY_NOTE.md` derivation assumes idealized rank-r Fisher

**File:** `STEP5_MONOTONICITY_NOTE.md` §"Case 2"
**What's wrong:** The Case 2 argument (rank r < D, equal eigenvalues
D/r) gives a clean d̂ → r asymptote. Real empirical Fishers have a
CONTINUOUS spectrum: a handful of "large" eigenvalues, then a long tail
of small-but-nonzero values. For a general PSD F with trace D and
eigenvalues λ₁ ≥ … ≥ λ_D ≥ 0:

    d̂(n) = Σᵢ log(1 + κ λᵢ) / log κ

Each non-zero λᵢ contributes log κ + log λᵢ asymptotically, so the
limit is (# nonzero λᵢ) + Σ_{nonzero} log λᵢ / log κ. The note's clean
"approach r from above" requires Σ_{nonzero} log λᵢ > 0, which is
NOT a property of a real rank-deficient Fisher unless all non-zero
eigenvalues are larger than 1 (i.e. concentrated). In the QLNN's
case, where many small eigenvalues are nominally non-zero but tiny, you
get a Σ log λᵢ that's dominated by the small ones and goes very
negative — in which case d̂ approaches the FULL rank from below, but
slowly.
**Why it matters:** The note is the published methodological correction
for Claim 2. A reviewer will check the derivation; the gap between the
idealized argument and the actual empirical spectrum is exactly the kind
of thing a referee asks for. The observation that |d̂(n)−d̂(n−1)|
decreases with n is a robust necessary condition, but the "asymptote
encodes effective rank" claim needs to be hedged unless you actually
compute the rank threshold (e.g. eigenvalue cutoff at numerical floor)
and show the asymptote agrees.
**Remediation:** Add a paragraph: "For an empirical Fisher with a
continuous decreasing eigenvalue spectrum, the formal asymptote of d̂(n)
depends on the joint distribution of all nonzero eigenvalues, not just
the rank. Numerically (Step 5 data), d̂ at n=472 is well below D for
both stacks, and the per-seed |d̂(n+1)−d̂(n)| sequences are monotonically
shrinking, which is the necessary asymptote-approaching property. The
clean 'asymptote = effective rank' interpretation should be read as a
heuristic upper-bound argument rather than a tight equality." Alternative:
compute the numerical-floor effective rank (count eigenvalues > 1e-6 ·
λ_max) per checkpoint and report it alongside d̂ for direct comparison.

### M-02 — `grad_one` jit retraces for every Python-int sample index

**File:** `scripts/run_effective_dimension.py:204-216`
**What's wrong:**
```python
@jax.jit
def grad_one(theta: jnp.ndarray, idx: int) -> jnp.ndarray:
    return jax.jacrev(lambda th: forward_scalar(th, idx))(theta)

for k, i in enumerate(sample_indices):
    g = np.asarray(grad_one(theta_flat, i), dtype=np.float64)
```
Passing `i` as a Python int triggers a retrace per unique value of `i`.
For 500 samples that's 500 traces of a Diffrax-heavy graph — every trace
is multi-second. This is why Step 5 takes much longer than its "~5 min"
estimate in `reproduce_paper.sh` (line 47).
**Why it matters:** Slows reproduction. NOT a correctness issue. But the
"~5 min" comment in reproduce_paper.sh is wrong; users will think the
script hung.
**Remediation:** Either (a) pass `idx` through a closure rather than as a
traced argument:
```python
def grad_one_factory():
    @jax.jit
    def _g(theta, x_i, t_i):
        return jax.jacrev(lambda th: model_fn(th, x_i, t_i))(theta)
    return _g
```
indexing x_all/t_all outside the jit, OR (b) update the wallclock
estimate in `reproduce_paper.sh:47` to reflect reality.

### M-03 — `fig_sample_efficiency` annotations may collide with data points

**File:** `scripts/make_paper_figures.py:165-171`
**What's wrong:** The "QLNN wins (p=0.015)" / "Classical wins" annotations
are placed at `min/max(c_mae, q_mae) ± 0.015 or 0.020`. With error bars
of half-width 0.02-0.03 (Step 6 data), the annotation can sit on top of
the error bar cap, making the figure hard to read at print resolution.
There's no automatic collision avoidance.
**Why it matters:** Figure quality for paper submission. The intent is a
clear "verdict per fraction" callout, but a sloppy overlap is reviewer
bait.
**Remediation:** Render the figure on the real data; inspect; adjust the
0.015/0.020 offsets, or use `matplotlib.transforms.offset_copy` to place
the text in display coordinates (constant pixel offset regardless of data
range). Also consider placing all four verdicts in a single legend-style
box rather than annotations.

### M-04 — `summarize_sample_efficiency.py` does not write `sample_efficiency_curve.png` to a deterministic path on `ImportError`

**File:** `scripts/summarize_sample_efficiency.py:134-170`
**What's wrong:** The plot block is wrapped in `try: import matplotlib …
except ImportError`. If matplotlib is not available the function returns
silently with no PNG; `verify_paper_integrity.py` and `make_paper_figures.py`
don't depend on this PNG (they generate their own), so this is mostly
fine, but the documentation in the script docstring promises the PNG
unconditionally.
**Why it matters:** Low. matplotlib IS a dependency (used by
make_paper_figures.py), so the ImportError branch is effectively dead.
**Remediation:** Drop the try/except since matplotlib is a hard
dependency, or document it as optional in the docstring.

### M-05 — `ed_jax.effective_dimension_curve`'s `monotonic_increasing` field is misnamed

**File:** `src/qlnn_/diagnostics/effective_dimension.py:188-198` and the
PyTorch twin at `src/quantum_liquid_neuralode/diagnostics/effective_dimension.py:145`
**What's wrong:** The returned dict key is `monotonic_increasing` and the
test is `b - a > -1e-6` — which checks that d_norms is non-DECREASING.
That's the original-pre-reg criterion. Per
`STEP5_MONOTONICITY_NOTE.md`, the corrected criterion is "monotonic in
EITHER direction." The function is now inconsistent with the documented
methodology, and the bool it returns is interpreted by callers as a pass/
fail flag. Worse: it's hardcoded into the run script's
`monotone_summary` field (line 370 of `run_effective_dimension.py`,
inlined with the same direction check, AND used in the markdown table
caption — line 447 — as "Monotonic increasing across n for every seed").
**Why it matters:** The run script's markdown report will say "Monotonic
increasing across n: NO" — which under the pre-reg-v1 reading is a
finding withdrawal, and under the corrected v2 reading is the expected
behavior. The artifact's text directly contradicts the methodology note.
**Remediation:** Rename the field to `monotonic` and check
`abs(np.sign(np.diff(d_norms)).sum()) == len(d_norms)-1` (all diffs same
sign). Update the inline check in `run_effective_dimension.py:370` and
the markdown line 447 to "Monotonic across n (either direction)."

### M-06 — `_slogdet_psd` clip floor of 1e-30 is dead defense

**File:** `src/qlnn_/diagnostics/effective_dimension.py:106-121`
and the torch twin at `src/quantum_liquid_neuralode/diagnostics/effective_dimension.py:77-86`
**What's wrong:** `_slogdet_psd` is called only on `I + κ F_norm`, whose
eigenvalues are all ≥ 1 (since F_norm is PSD). The `np.clip(eigs, 1e-30,
None)` is a no-op. The docstring claims this guards against "tiny
negative eigenvalues which a true SPD matrix cannot have; they are pure
roundoff" — but adding I guarantees no eigenvalue is even close to
roundoff territory.
**Why it matters:** Cosmetic. The clip is defensible defense-in-depth,
but the docstring is misleading.
**Remediation:** Either remove the clip, or update the docstring: "Defense
in depth: we add I (so eigenvalues ≥ 1) before calling slogdet, and the
clip floor is below any conceivable roundoff. Belt-and-braces."

### M-07 — `train_baseline.py` and `train_qlnn.py` duplicate the entire `clip_predictions_norm` helper

**File:** `scripts/train_baseline.py:86-138` and
`scripts/train_qlnn.py:62-107`
**What's wrong:** Both files redefine `clip_predictions_norm` and
`_predict_norm_with_clip` / `_qlnn_predict_norm_clipped` from scratch. The
TODO comments acknowledge this. Each file's docstring says "Inlined here
rather than living in a shared utility because two concurrent training
scripts each own their own copy until a common helper module is
introduced."
**Why it matters:** Code duplication is a defect on its own; in a
reproducibility-focused codebase it's worse because a fix to the clip
logic must be applied in two places. The Step 6 sweep adds 8 more runs
that each use this code path — divergence between classical and QLNN
clip behavior would silently change the apparent sample-efficiency
crossover.
**Remediation:** Move `clip_predictions_norm` into
`quantum_liquid_neuralode/evaluation/` and import from both scripts.

### M-08 — `flatten_model_params` JAX side does not record `names`

**File:** `src/qlnn_/diagnostics/effective_dimension.py:206-231`
**What's wrong:** The PyTorch version returns `(theta_flat, names, shapes,
unflatten)` so the caller can audit which parameter is at which index. The
JAX version returns `(theta_flat, unflatten)` with no `names` analog. For
debugging / paper-writing ("we trained 114 parameters: a × b × c + …"),
the index → name map is useful.
**Why it matters:** No correctness issue, but inconsistency between the
two stacks is annoying.
**Remediation:** Add `names` analog (the Equinox tree path strings) to
the JAX flatten function for parity.

### M-09 — `make_paper_figures.py` `_ci` silently falls back to `std` and labels it "CI"

**File:** `scripts/make_paper_figures.py:55-57,154-178`
**What's wrong:**
```python
def _ci(s: dict) -> float:
    return s.get("ci95_half_width", s.get("std", 0.0))
```
The figure axis label says "Test MAE at h=3 (raw OD, mean ± 95% CI)" and
the legend caption likewise. If `seeds_summary.json` is missing
`ci95_half_width` (e.g., from an older run pre-Phase-C), the figure
plots `std` but still labels it 95% CI.
**Why it matters:** A 0.95 t-CI half-width for 5 seeds is roughly 1.24·
std/√5 ≈ 0.55·std. Falling back to plotting std silently OVERSTATES the
uncertainty by ~1.8×. Paper figure mislabeling is reviewer bait, AND
it's the wrong direction (showing more uncertainty than there really is
isn't ethically a problem but the figure becomes a lie about which
statistic is plotted).
**Remediation:**
```python
def _ci(s: dict) -> float:
    if "ci95_half_width" not in s:
        raise KeyError(
            f"seeds_summary entry missing ci95_half_width; "
            f"re-run with current train_baseline.py / train_qlnn.py"
        )
    return s["ci95_half_width"]
```

### M-10 — `verify_paper_integrity.py` does not assert the sample-efficiency QLNN 100% matches the canonical QLNN h3 run

**File:** `scripts/verify_paper_integrity.py:59-65`
**What's wrong:** The check reads only the `sample_efficiency/` tree's
100% cell. The canonical `qlnn_hybrid_h3` run is not cross-checked. Same
asymmetry as H-03.
**Remediation:** Add:
```python
canonical = _load("results/qlnn_hybrid_h3/seeds_summary.json")
step6_100 = _load("results/sample_efficiency/qlnn_h3_pct100/seeds_summary.json")
all_ok &= _check("qlnn 100% MAE matches canonical",
                 step6_100["test"]["mae_raw"]["mean"],
                 canonical["test"]["mae_raw"]["mean"],
                 tol=1e-4)
```

---

## LOW

### L-01 — `effective_dimension_curve` argument validation is partial

**File:** `src/qlnn_/diagnostics/effective_dimension.py:188-198`
**What's wrong:** Validates `n_values` is non-decreasing but not that all
elements are >= 2 (which `normalized_effective_dimension` requires).
Caller eventually hits the inner `n < 2` check, but the error message
points at the wrong function.
**Remediation:** Add `if any(int(v) < 2 for v in n_list): raise ValueError(...)`.

### L-02 — `_random_psd` in test uses a hardcoded D=10 / seed=7 — magic numbers

**File:** `tests/test_effective_dimension.py:179-181`
**What's wrong:** `F_np = _random_psd(D=10, seed=7)` — the choice of seed
matters only for reproducibility but D=10 doesn't probe edge cases
(D=1 or D=100). No additional test at boundary D.
**Remediation:** Parametrize over D ∈ {1, 5, 50}.

### L-03 — `fig_reproducibility` divide-by-zero on single-seed runs

**File:** `scripts/make_paper_figures.py:201`
**What's wrong:** `ratios = [c / q for c, q in zip(c_ci, q_ci)]` —
if `q_ci[i] == 0` (zero std, e.g. n_seeds=1) → ZeroDivisionError.
**Why it matters:** Unlikely in headline 5-seed runs but breaks the
figure if someone runs a quick smoke-test on 1 seed.
**Remediation:** `ratios = [c / q if q > 0 else float('nan') for ...]`
and skip NaN bars.

### L-04 — `run_effective_dimension.py` comment is wrong about jacfwd vs jacrev

**File:** `scripts/run_effective_dimension.py:202-207`
**What's wrong:** The comment says "for D ≈ 100 it's actually D
forward-mode JVPs but tiny vs the quantum forward cost so it doesn't
matter." `jacrev` does ONE reverse-mode pass for a scalar output, not D.
The author confused jacrev with jacfwd's per-parameter cost.
**Remediation:** Fix the comment: "jacrev is one reverse-mode pass for
the scalar prediction; complexity is O(model forward) per sample, not
O(D·forward) as jacfwd would be."

### L-05 — `STEP5_MONOTONICITY_NOTE.md` references `Σⱼ₌₁^r log(1)` term

**File:** `STEP5_MONOTONICITY_NOTE.md` §"Case 2" line 42
**What's wrong:** `(D−r)·log(1)` is `0`, which is correctly dropped, but
the line includes it for clarity — fine. However the line above writes
the sum as `Σⱼ₌₁^r log(1 + κ (D/r))` with parens around the (D/r) — minor
typography. The text reads correctly but the line `Σᵢ log(1 + κ λᵢ) = Σⱼ₌₁^r log(1 + κ (D/r))  + (D−r)·log(1) = r · log(1 + κ D/r)`
should probably be `r · log(1 + κ D/r) + 0`.
**Remediation:** Cosmetic; consider rendering in LaTeX in the paper.

### L-06 — `run_effective_dimension.py` recomputes seed 0 monotonicity for every seed

**File:** `scripts/run_effective_dimension.py:325-343`
**What's wrong:** The comment line 319 says "compute on a single seed each
— seed 0 by default — to keep wall-clock manageable", but the code
actually runs the monotonicity curve for EVERY seed (`for seed in
args.seeds: … if not args.skip_curve: for n_k in curve_ns: …`). So the
wall-clock cost is `5 seeds × 4 n_values × (classical + QLNN)` of Fisher
computations, not the "single seed each" the comment suggests.
**Why it matters:** Comment-code drift; wall-clock estimate in
reproduce_paper.sh is wrong (related to M-02).
**Remediation:** Either restrict to `if seed == args.seeds[0]:` for the
curve, or update the comment to "we run the curve for every seed because
the per-seed Fisher gradients are not actually reusable here."

### L-07 — Two configs declare `lambda_smooth: 0.0` although it was removed in Phase A

**File:** `configs/sample_efficiency/classical_h4_h3_pct{10,25,50,100}.yaml:39`
**What's wrong:** Each classical config has
```
physics:
  lambda_logistic: 0.0
  lambda_smooth: 0.0
```
`lambda_smooth` is documented in `train_baseline.py:364-366` as
"intentionally ignored". The configs still write it.
**Why it matters:** Cosmetic; misleading to a reader.
**Remediation:** Delete the `lambda_smooth: 0.0` line from all four
`classical_h4_h3_pct*.yaml`.

### L-08 — Step 5 docstring uses `2 *` notation for the integral that's not in the trained-theta path

**File:** `src/qlnn_/diagnostics/effective_dimension.py:7-10`
**What's wrong:** The module docstring shows Abbas Eq. (4) with the `2 *`
prefactor (i.e. the parameter-volume-averaged definition), but the
actual code computes the trained-theta single-θ specialization (Eq.
without the factor-of-2 / 2-log relation, just `log det / log κ`). The
mathematical statement in the docstring is correct but it would be
clearer to lead with the single-theta form that the code actually
computes.
**Remediation:** Re-order so the single-θ form (Eq. for d̂_{n,γ}) is the
primary, with a parenthetical note that the population definition has
the extra (1/V_Θ) integral and factor of 2.

### L-09 — `verify_paper_integrity.py` tolerances mix relative and absolute

**File:** `scripts/verify_paper_integrity.py:78-81`
**What's wrong:** `tol=max(0.1, abs(paper_pers) * 0.01)` is a mixed
absolute/relative tolerance; for h=12 paper_pers=-977 → tol = 9.77. So a
−986 result still passes. The intent ("1% relative or 0.1 absolute,
whichever is larger") is reasonable for R² which can have huge dynamic
range, but pump-and-shift it could be more transparently expressed via
`np.isclose(actual, expected, rtol=0.01, atol=0.1)`.
**Remediation:** Use `np.isclose` for readability.

### L-10 — JAX-side `flatten_model_params` calls `jax.tree_util.tree_flatten`/`unflatten` without filtering for floating leaves

**File:** `src/qlnn_/diagnostics/effective_dimension.py:213-228`
**What's wrong:** Uses `eqx.is_array` to partition trainable vs static
leaves. Equinox's `is_array` includes integer arrays (e.g. solver state
indices) as well as floating ones. If the model has any integer-typed
trainable-looking leaves (it doesn't currently, but could in future), the
flatten step would cast them through float, breaking the unflatten cycle.
**Remediation:** Use `eqx.is_inexact_array` instead of `eqx.is_array`.

---

## Summary table

| ID | Severity | File | Issue |
|----|----------|------|-------|
| B-01 | BLOCKER | CLAUDE.md:88-92 | CLAUDE.md still describes QWGAN as Phase 4 upcoming |
| B-02 | BLOCKER | PAPER_SUMMARY.md:149,166-170 | "Single source of truth" doc contradicts itself on QWGAN status |
| H-01 | HIGH | src/qlnn_/diagnostics/effective_dimension.py:88-103 | JAX `empirical_fisher` claims float64 but silently float32 |
| H-02 | HIGH | scripts/make_paper_figures.py:136 | `n_train_classical` hardcoded instead of read from protocol.json |
| H-03 | HIGH | scripts/verify_paper_integrity.py:32-65 | Claim 1 and Claim 3 read different result trees of the same config |
| H-04 | HIGH | scripts/run_sample_efficiency.sh:21-37 | 100% cell duplicates canonical run, no determinism assertion |
| H-05 | HIGH | tests/ | `head()` method has zero unit tests for boundary semantics |
| H-06 | HIGH | tests/test_effective_dimension.py:195-206 | Trace-norm test claims to exercise reparametrization but doesn't |
| H-07 | HIGH | configs/sample_efficiency/*_pct100.yaml | `train_fraction: 1` (int) instead of 1.0 (float) — convention drift |
| H-08 | HIGH | scripts/verify_paper_integrity.py:35 | tol=0.1 on 3.77× ratio masks regressions |
| H-09 | HIGH | scripts/verify_paper_integrity.py:23 | Unicode tick marks break in ascii locales |
| H-10 | HIGH | scripts/reproduce_paper.sh:18 | pytest pre-flight swallows error output |
| M-01 | MEDIUM | STEP5_MONOTONICITY_NOTE.md | Case 2 argument idealized; real Fisher has continuous spectrum |
| M-02 | MEDIUM | scripts/run_effective_dimension.py:204 | `grad_one` retraces per Python-int sample idx — slow |
| M-03 | MEDIUM | scripts/make_paper_figures.py:165-171 | Verdict annotations may collide with error-bar caps |
| M-04 | MEDIUM | scripts/summarize_sample_efficiency.py:134 | Dead ImportError branch for matplotlib |
| M-05 | MEDIUM | diagnostics/effective_dimension.py + run script | `monotonic_increasing` field is misnamed under v2 criterion |
| M-06 | MEDIUM | diagnostics/effective_dimension.py:106-121 | `_slogdet_psd` 1e-30 clip is dead defense; docstring misleading |
| M-07 | MEDIUM | scripts/train_baseline.py:86-138, train_qlnn.py:62-107 | Clip helpers duplicated between scripts |
| M-08 | MEDIUM | src/qlnn_/diagnostics/effective_dimension.py:206 | JAX `flatten_model_params` lacks `names` analog |
| M-09 | MEDIUM | scripts/make_paper_figures.py:55-57 | `_ci` silently falls back to std and labels as "CI" |
| M-10 | MEDIUM | scripts/verify_paper_integrity.py | No cross-check between Step 6 100% cell and canonical h3 run |
| L-01 | LOW | diagnostics/effective_dimension.py:188 | curve fn doesn't validate n>=2 up-front |
| L-02 | LOW | tests/test_effective_dimension.py:179 | D=10 only — no boundary parametrization |
| L-03 | LOW | scripts/make_paper_figures.py:201 | divide-by-zero possible on single-seed runs |
| L-04 | LOW | scripts/run_effective_dimension.py:202 | comment confuses jacrev with jacfwd cost |
| L-05 | LOW | STEP5_MONOTONICITY_NOTE.md:42 | typography polish for the sum derivation |
| L-06 | LOW | scripts/run_effective_dimension.py:325 | comment says "single seed each" but code runs all seeds |
| L-07 | LOW | configs/sample_efficiency/classical_*.yaml:39 | dead `lambda_smooth: 0.0` lines |
| L-08 | LOW | src/qlnn_/diagnostics/effective_dimension.py:7 | docstring leads with population def vs code's single-θ form |
| L-09 | LOW | scripts/verify_paper_integrity.py:78 | mixed abs/rel tolerance could use np.isclose |
| L-10 | LOW | src/qlnn_/diagnostics/effective_dimension.py:213 | `eqx.is_array` should be `eqx.is_inexact_array` |

---

## What I checked and confirmed CORRECT

- The Abbas-2021 trained-theta formula `d̂ = log det(I + κ F_norm) / log(n/(2π log n))` with `κ = γn/(2π log n)` and `F_norm = F · D / trace(F)` is implemented correctly on BOTH stacks (JAX side: `normalized_effective_dimension` in `src/qlnn_/diagnostics/effective_dimension.py:124-165`; PyTorch side: same name in `src/quantum_liquid_neuralode/diagnostics/effective_dimension.py:89-129`). The trace-normalization is applied BEFORE the log-det. Confirmed.
- `test_jax_and_torch_normalized_effective_dimension_agree` is NOT tautological — it constructs a real PSD matrix via `A A^T / D + ε I`, computes d_norm on the same matrix through both stacks, and asserts agreement within 1e-6. Real cross-stack equivalence check. Confirmed.
- `HorizonWindows.head(n)` correctly slices the FIRST n windows in chronological order; the `target_idx` / `end_idx` arrays remain the original (global) indices. Truncation is purely from the start. Confirmed.
- `train_fraction` validation correctly rejects 0.0 and ≥ 1.0 with the upper bound at exactly 1.0 (no-op). `n_keep = max(1, int(round(n*train_fraction)))` guarantees a minimum of 1 window even at tiny fractions. Confirmed.
- `n_train_windows` IS recorded in `protocol.json` (lines 288 of train_baseline.py and 234 of train_qlnn.py). Confirmed.
- val and test windows are NOT touched by `train_fraction` (only `w_train = w_train.head(n_keep)` is gated by the conditional). Confirmed.
- `STEP5_MONOTONICITY_NOTE.md` case 1 (full-rank, all λᵢ>0, det≤1) math is correct: log det ≤ 0 → d̂ approaches D from above for det=1, from below for det<1.
- Color palette `#D55E00` / `#0072B2` is from the Wong colorblind-safe set, correctly identified as vermilion + cool blue.
- `reproduce_paper.sh` does use `set -euo pipefail`. Ordering puts classical FIRST (fast, fail-early), then horizon sweep, then QLNN, then Step 5/6. Sensible.
- `verify_paper_integrity.py` exits 0 on success and 1 on failure via `sys.exit(main())`. Confirmed.
- `pre_registered_hypothesis_met = bool(threshold_met > 1.0)` correctly implements the v2 acceptance threshold "exceeding 1.0".

---

_Reviewed: 2026-05-18_
_Reviewer: Claude (gsd-code-reviewer, adversarial pass for paper submission)_
_Depth: deep — cross-file analysis of Step 5/6 + paper-prep utilities + doc consistency_
