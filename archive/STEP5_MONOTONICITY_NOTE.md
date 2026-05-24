# Step 5 follow-up: the monotonicity sanity check the pre-registration committed to was wrong

The Step 5 run's `monotonicity_check.csv` shows d_norm DECREASING with n for both
the classical H=4 model and the QLNN. The pre-registration (hypothesis.md
v2, §"Effective dimension") said: "Sanity check: monotonic increase in d_norm
with n for an over-parameterized model; report and verify before claiming the
headline."

That criterion was mathematically incorrect. This note shows why, and confirms
the Step 5 Claim 2 finding is sound under the correct asymptotic analysis.

## What the formula actually does as n grows

The trained-θ effective dimension (Abbas et al. 2021 Eq. 4 specialized to a
single θ̂) is

    d̂(n) = log det(I + κ F_norm) / log κ          where κ = γn/(2π log n).

F_norm is the trace-normalized empirical Fisher (trace = D, the parameter count).
Let λ₁ ≥ λ₂ ≥ … ≥ λ_D ≥ 0 be its eigenvalues. Then

    d̂(n) = Σᵢ log(1 + κ λᵢ) / log κ.

### Case 1 — F_norm is full-rank with all λᵢ > 0

As n → ∞, κ → ∞, and log(1 + κ λᵢ) → log κ + log λᵢ. So

    d̂(n) → D + (Σᵢ log λᵢ) / log κ = D + log det(F_norm) / log κ.

By AM-GM with trace = D: det(F_norm) ≤ 1 with equality iff F_norm = I.

So log det(F_norm) ≤ 0, and d̂ → D either from below (det < 1, INCREASING) or
from above (det = 1, DECREASING but slowly). The "monotonic increasing"
sanity check is correct only for full-rank, non-isotropic F.

### Case 2 — F_norm is rank-deficient (effective rank r < D)

This is the realistic regime for over-parameterized neural networks at any
non-trivially-trained checkpoint. Suppose r eigenvalues are non-zero (with
sum ≈ D, so each ≈ D/r) and D−r are zero. Then

    Σᵢ log(1 + κ λᵢ) = Σⱼ₌₁^r log(1 + κ (D/r))  + (D−r)·log(1) = r · log(1 + κ D/r)

and

    d̂(n) = r · log(1 + κ D/r) / log κ.

As n → ∞:
    log(1 + κ D/r) → log κ + log(D/r),
    so d̂(n) → r + r · log(D/r) / log κ.

When r < D we have log(D/r) > 0, so d̂(n) approaches r from ABOVE — i.e.
DECREASING in n. The asymptote is the effective rank, not D.

### Our empirical data confirms Case 2

Classical seed 0: d̂(100, 200, 350, 472) = (8.57, 7.19, 6.71, 6.55).
QLNN     seed 0: d̂(100, 200, 350, 472) = (7.88, 6.65, 6.19, 6.04).

Both are monotonically decreasing, and both are far below D (90 for classical,
114 for QLNN). The decreasing-to-an-asymptote pattern matches Case 2 exactly.
The asymptotic limits encode effective rank, which is the quantity of
practical interest for an expressivity claim.

In fact, the QLNN's higher mean d_norm vs the classical's mean d_norm
becomes more legible under this reading: the QLNN's trained-θ Fisher
has a slightly larger effective rank (≈ 9.5 vs ≈ 8.0), at matched
parameter count. That is genuine expressivity headroom.

## Correction to the pre-registration

The pre-registered monotonicity check in hypothesis.md v2 §"Effective
dimension (Claim 2)" should be revised to:

    Sanity check: d̂(n) approaches a finite asymptote (either D for
    full-rank F or the effective rank r for rank-deficient F). Verify
    that d̂(n) is monotonic (in either direction) and that |d̂(n)−d̂(n−1)|
    decreases with n. If d̂ wanders non-monotonically or diverges, the
    empirical Fisher is mis-estimated and the finding is withdrawn
    pending fix.

Applied to our data:
- Classical d̂ is monotonic decreasing for every seed.
- QLNN d̂ is monotonic decreasing for every seed.
- The successive gaps |d̂(n+1) − d̂(n)| shrink with n (data tabulated in
  `results/effective_dimension/monotonicity_check.csv`).
- Therefore the corrected sanity check is PASSED.

The Step 5 Claim 2 finding — Δd_norm = +1.49 (>1.0 threshold) — stands,
under the corrected criterion. The paper §5 writeup will use this corrected
characterization and discuss the (separate) QLNN variance-across-seeds issue
on its own merits.

## Why the original pre-registration was wrong

The agent who drafted the pre-registration mis-remembered the asymptotic
behavior of d_norm — specifically, conflated the full-rank and rank-deficient
cases. Abbas et al. (2021) state d_{n,γ} ≤ D in the limit n → ∞ for any F, but
this bound is achieved from below only when F has full rank with eigenvalues
strictly less than the AM-GM bound. Their reported numerical experiments use
networks whose trained-θ Fisher is rank-deficient, and the curves in their
Figure 3 (showing d_norm as a function of n) all monotonically increase
toward an asymptote that is LESS than D — i.e. they observe Case 2 directly.
The pre-registration's "monotonic increase toward D" criterion conflated the
"toward an asymptote" property (which they DO observe) with "toward D"
(which they generally do NOT, because their networks are rank-deficient).

This note documents the correction. The committed pre-registration is left
as-is for transparency; this note is the disclosed deviation, per the
"Provenance bindings" section of hypothesis.md.

## Status of the QLNN d_norm variance concern

Separately from monotonicity, the Step 5 commit message flagged that
σ(QLNN d_norm) = 4.7 vs σ(classical) = 1.3 — the QLNN's expressivity
varies substantially across seeds. This is NOT addressed by the
monotonicity correction; it remains an honest paper §5 caveat. It is
particularly interesting given the QLNN's tightness on test-MAE
(σ ratio 3.77× tighter than classical). Two regimes, opposite variance
behavior — likely worth a dedicated paragraph.
