# Option-B circuit search — design

## The problem, restated precisely

Find a QLNN setup that passes all five gates in `results/baseline_lock.json`:

| Gate | Threshold | Reference QLNN (dr 4q/3L) | Best promoted (se 6q/3L) |
|---|---|---|---|
| G1 accuracy | h=3 5-seed MAE < **0.2594** | 0.2655 ❌ | 0.2555 ✅ |
| G2 reproducibility | σ ≤ **0.00831** (≥2× vs classical) | 0.0044 ✅ (3.80×) | 0.0253 ❌ (0.66×) |
| G3 no-regress 10% | SE MAE < 0.2788 | 0.2686 ✅ | unknown |
| G4 no-regress 25% | SE MAE < 0.2546 | 0.2507 ✅ | unknown |
| G5 no-regress 50/100% | ≤ reference | (is the reference) | unknown |

**Key realization:** the prior search varied only PQC *topology* and found
the accuracy↔variance frontier but never crossed it — every circuit is
either accurate-but-unstable or stable-but-inaccurate. Topology alone
does not move a point off that frontier. **Regularization and
optimization dynamics do.** That is the unsearched axis.

Hypothesis (the paper-worthy claim if it holds): *an expressive circuit
(strongly_entangling-class accuracy) regularized hard enough collapses
its seed variance under the G2 gate without surrendering the accuracy
gain.* Equivalently: the accuracy/variance tradeoff is a
regularization-strength artifact, not a fundamental property of the
ansatz family.

## What the trainer already supports (verified)

`src/qlnn_/training/trainer.py`:
- `weight_decay > 0` → switches Adam→**AdamW** (`_build_optimizer`, L85-88). Classic variance reducer. ✅
- `grad_clip_norm` → `optax.clip_by_global_norm` (L83-84). ✅
- `physics.lambda_logistic > 0` → logistic-growth regularizer (L199). Domain prior; variance reducer. ✅
- **No LR schedule** — `optax.adam(cfg.lr)` is a scalar LR (L88). Cosine decay would need a one-line trainer change.

`init_circuit_std` (encoder PQC weight init, default 0.05) and
`tau_init` are **not currently plumbed from YAML** into
`QLNNForecasterConfig`. Small additions needed to search them.

## Design — staged funnel (compute-bounded)

A curated **factorial** beats Optuna here for three reasons: (1) we
already hit TPE convergence pathologies in the prior search; (2) a
designed experiment that crosses *circuit family* × *regularization
regime* directly tests the hypothesis and is far more publishable than a
TPE scatter; (3) it bounds compute deterministically.

### Phase O-1 — plumbing (no compute)

1. Add `lr_schedule: "constant" | "cosine"` to `QLNNTrainerConfig` +
   `_build_optimizer` (cosine via `optax.cosine_decay_schedule` over
   `epochs * steps_per_epoch`). Default `"constant"` ⇒ zero behavior
   change for every existing run / checkpoint / integrity check.
2. Plumb `init_circuit_std` and `tau_init` from the YAML `model:` block
   through `QLNNForecasterConfig` → `LiquidQuantumCellConfig` →
   `QuantumFeatureEncoderConfig` (all already accept the field; just
   wire `train_qlnn.py` to read it). Defaults preserve current behavior.
3. Unit tests: schedule builds & decays; YAML-absent ⇒ identical init.
4. Gate: full pytest + `verify_paper_integrity.py` still green
   (proves zero regression to the locked claims).

### Phase O-2 — variance-aware proxy (the search)

Curated grid = **circuit family × regularization regime**:

Circuits (3) — span the frontier:
- `strongly_entangling 6q/3L` — accuracy-strong, variance-broken (needs G2 fix)
- `data_reuploading 4q/3L` — variance-strong, accuracy-short (the reference; needs G1)
- `hardware_efficient 4q/3L` — middle of the frontier

Regularization regimes (4):
- **R0 control**: wd=0, lr=2e-3 const, clip=1.0, init_std=0.05, physics=0
- **R1 weight-decay**: wd=1e-3 (AdamW)
- **R2 physics-prior**: lambda_logistic=0.1
- **R3 smooth-convergence**: cosine LR + clip=0.5 + init_std=0.02

= **12 configs**. Proxy = **3 seeds** (the minimum to estimate σ at
all — G2 is a variance gate, 1-seed proxies are blind to it),
epochs unchanged (60, early-stop active).

Objective (logged, not auto-optimized — this is a designed experiment):
`penalized = mae_3seed_mean + 5.0 · relu(σ_3seed − 0.00831)`.
Lower is better; the penalty makes a variance-gate violation dominate.

Cost: 12 × 3 seeds × ~10 min ≈ **~6 h**.

### Phase O-3 — tier-1 promotion (G1 + G2)

Top-3 by penalized proxy score → full **5-seed h=3** locked protocol.
`scripts/check_circuit_regression.py --h3-run …` ⇒ G1+G2 verdict.
Cost: 3 × 5 seeds ≈ **~6 h**.

### Phase O-4 — tier-2 promotion (G3 + G4 + G5)

Only G1+G2 **survivors** get the expensive sample-efficiency treatment:
4 fractions × 5 seeds each (10/25/50/100%). SE at low fractions is
cheap (fewer windows); ≈ 6-8 h per surviving circuit. Run
`check_circuit_regression.py` with all `--se-pctNN` dirs ⇒ full
Option-B verdict. Cost: **~7 h per survivor** (expect 0-2 survivors).

### Total compute envelope

| Scenario | Wall-clock |
|---|---|
| Best case (0 survive tier-1) | ~12 h — and a publishable negative result |
| 1 survivor | ~19 h |
| 2 survivors | ~26 h |

All phases checkpoint to disk and are resumable. Tier-1/2 are
user-gated (no >30-min sweep without go-ahead, per project rule).

## Outcomes & paper value (both are wins)

- **A circuit passes all 5 gates** → headline upgrade: "QLNN is Pareto-
  dominant — more accurate *and* more reproducible than matched
  classical, with no sample-efficiency regression." Updates
  `PAPER_SUMMARY.md` §Circuit-search verdict + a new figure.
- **None pass** → rigorous negative result: "the accuracy/variance
  tradeoff is fundamental on this dataset and is *not* a
  regularization artifact — we tested 12 regularization regimes across
  3 circuit families." Strengthens §7 honesty; the three pre-registered
  claims are untouched either way.

## Risks

- 3-seed σ is itself noisy; a regime that looks G2-passing at 3 seeds
  may fail at 5. Mitigated by tier-1 re-running at the full 5 seeds
  before any verdict.
- Cosine schedule could destabilize Diffrax adjoint training. Caught by
  the Phase O-1 pytest gate before any sweep.
- `jax_enable_x64` stays off (locked decision #5) — no new ansatz or
  optimizer path may flip it.
