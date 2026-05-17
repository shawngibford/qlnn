# QLNN JAX subpackage — adversarial code review (Steps 2 + 3)

**Scope:** `src/qlnn_/**`, `scripts/train_qlnn.py`, `scripts/qlnn_smoke_encoder.py`, `configs/qlnn_hybrid.yaml`, `tests/qlnn_/*`.
**Stance:** peer-review skepticism. Findings tagged BLOCKER / HIGH / MEDIUM / LOW.

---

## Executive summary

The quantum stack is in solid shape architecturally: the data-reuploading PQC, the Liquid Quantum Cell, and the Diffrax-integrated forecaster all compose cleanly through Equinox PyTrees, and the trainer correctly reuses the project-canonical `compute_metrics` / `aggregate_seed_metrics` so QLNN rows are drop-in for `summarize_baselines.py`. Best-checkpoint selection and Optax wiring (clip-then-Adam, AdamW switch, `eqx.filter` on init/grad/update) are correct. The most serious finding is a **silent stability footgun in `LiquidQuantumCell`**: with the default `tau_min=0.1` the vector field is safe, but the config accepts any `tau_min > 0`, and for `tau_min > 1` the leak coefficient `(1/tau + q(x))` can become negative when `q(x) ≈ -1`, turning the ODE into an exponentially-growing system. This is not a bug in the equations but it is a config-level paper-credibility risk that should be guarded. Other notable gaps: (a) the "solver swap" test does not actually verify tsit5 ≈ dopri5 within tolerance — it only checks both run; (b) there are no guards against `dt ≤ 0` in the history loop; (c) the encoder's batched-vmap test only asserts identical inputs yield identical outputs (the PI explicitly flagged this as necessary-but-not-sufficient); the cell- and forecaster-level vmap tests do cover the differentiating-inputs case, so the underlying behavior is exercised, but the encoder-level coverage is misleadingly narrow. The Pérez-Salinas re-uploading ordering (RX → Rot → CNOT entangler) is a defensible variant — the last layer's entangler is parameter-wise wasted before local-Z measurement, but that is a design choice, not a bug. Diffrax integration, dt0 clipping, SaveAt(t1=True), and PIDController plumbing are all correct.

---

## BLOCKER

_None._ Every result-invalidating issue I traced through can be defended or has only narrow exposure under the shipped config.

---

## HIGH

### H1. `LiquidQuantumCell` admits configurations where the ODE is exponentially unstable
**File:** `src/qlnn_/cells/liquid_quantum_cell.py:72-86,144,172-178`

The vector field is
```
dh/dt = -(1/tau + q(x)) ⊙ h + A ⊙ q(x)
```
with `q(x) ∈ [-1, 1]` (per-qubit ⟨Z⟩) and `tau ≥ tau_min` by construction. The leak coefficient is `-(1/tau + q)`. For this to be **stably damping** we need `1/tau + q ≥ 0` ⇔ `tau ≤ 1/|q|`. In the worst case `q = -1` this requires `tau ≤ 1` ⇔ `tau_min ≤ 1` and `tau_init ≤ 1`. The shipped config (`tau_min=0.1`, `tau_init=1.0`) is safe (`tau ≥ 0.1`, so `1/tau ≥ 10 ≫ 1`). But `__post_init__` only validates `tau_min > 0` and `tau_init > tau_min`. A reasonable-looking config (`tau_min=2.0`, `tau_init=3.0` — "slower dynamics, longer memory") produces a vector field that is **antidamping** whenever `q(x) < -0.5`, and the integrator will return numerically growing trajectories (or NaN once `max_steps` is exceeded).

**Why it matters:** This is the kind of silent failure that gets caught in a paper revision when a reviewer asks "what happens if you sweep tau?" If you later run HPO over `tau_min`, the search can drift into the unstable region with no error — you'd get garbage metrics and no diagnostic. The cell's docstring (line 5) advertises the form as if it were unconditionally a Liquid CT-RNN, but the Hasani et al. derivation assumes the synaptic conductance is non-negative; here it can flip sign.

**Fix:**
1. In `LiquidQuantumCellConfig.__post_init__`, enforce `tau_min ≤ 1.0` (or more conservatively `≤ 0.5`) with a clear error message citing the q ∈ [-1, 1] constraint. Equivalently, guarantee `1.0/tau_min > 1.0`.
2. Or change the parameterization so the conductance contribution is non-negative, e.g. use `softplus(q)` or `|q|` in the leak term and keep the raw `q` only in the drive `A ⊙ q`. This is a substantive modeling decision — flag it for the PI but do not silently change it.
3. Add a regression test that constructs the cell at the edge (`tau_min=0.99`) and confirms `dh/dt` stays bounded over a longish integration.

---

### H2. `_integrate` does not guard against `dt ≤ 0`; clipped `dt0` collapses to zero
**File:** `src/qlnn_/models/qlnn_forecaster.py:136-162,190-192`

`dt = t_hours[i + 1] - t_hours[i]` is consumed as `t1=dt` in `diffeqsolve`. If the input window contains a non-monotone or duplicate timestamp (`dt == 0`), then `dt0 = jnp.minimum(cfg.dt0, dt * 0.5) = 0` and `diffeqsolve` is asked to integrate from `t0=0` to `t1=0` with `dt0=0`. Diffrax in current versions either errors with an opaque message or silently returns `y0`; under JIT either outcome is hard to diagnose. The classical PyTorch loader uses `make_horizon_windows` which drops windows whose realized horizon is out-of-tolerance, but it does NOT drop windows with internal `dt ≤ 0`.

**Why it matters:** qZETA data is mostly regularly sampled, but the dataset has known irregularities (sensor pauses, duplicate seconds). One bad window per epoch is enough to corrupt the training step (silent NaN on one sample propagates to the gradient via the mean).

**Fix:**
- Either: at the script level (`scripts/train_qlnn.py`), assert `np.all(np.diff(t_hours, axis=1) > 0)` for `w_train.t`, `w_val.t`, `w_test.t` before training, and raise with a clear message naming the offending window.
- Or: inside `_integrate`, replace `t1=dt` with `t1=jnp.maximum(dt, eps)` and clamp `dt0` to `max(dt0, eps)`. I prefer the script-level assert: silent eps-substitution makes a config bug look like good data.

---

### H3. Solver-swap test does not verify tsit5 ≈ dopri5 within tolerance
**File:** `tests/qlnn_/test_qlnn_forecaster.py:157-163`

The test runs both solvers and only asserts the outputs are finite. Both Tsit5 and Dopri5 are 5th-order embedded RK pairs targeting the same `rtol/atol`, so they should agree on a smooth, low-stiffness vector field to within (a small multiple of) `rtol`. The current test would pass even if Tsit5 silently produced wildly different trajectories from Dopri5 — defeating the purpose of having both.

**Why it matters:** If one solver becomes broken (e.g., a Diffrax version regression on tracing, or a wrong solver wired up via a string typo), the test is silent. The PI specifically called this out.

**Fix:** Add an assertion of the form
```python
y_tsit5 = _model(solver="tsit5")(x, t)
y_dopri5 = _model(solver="dopri5", rtol=1e-6, atol=1e-7)(x, t)  # tighten one to use as reference
assert jnp.allclose(y_tsit5, y_dopri5, atol=5e-3, rtol=5e-3)
```
The tolerance must be loose enough that the default `rtol=1e-3` regime still passes, but tight enough to catch real divergence.

---

### H4. Encoder `test_encoder_batched_via_vmap` only checks the homogeneous-input case
**File:** `tests/qlnn_/test_quantum_feature_encoder.py:26-32`

The test tiles a single feature vector 5× and asserts the 5 outputs are identical. That's the same thing a broadcast would do — it does NOT prove `jax.vmap` over the QNode produces correct per-sample outputs when the inputs differ. This is the exact failure mode the PI flagged ("identical inputs → identical outputs is necessary but not sufficient").

**Why it matters:** PennyLane's `default.qubit` + JAX has a track record of subtle vmap interactions (e.g., the older `default.qubit.jax` device had to be explicitly opted into; tape construction under vmap has historically been a source of bugs). Without a heterogeneous-input vmap test at the encoder level, a future PennyLane upgrade can silently break this without any test catching it. The cell test (`test_vmap_over_batch`, line 89-104) and forecaster test (`test_vmap_over_batch_of_samples`, line 119-127) DO exercise the heterogeneous case end-to-end, so the underlying integration is exercised — but the encoder is the lowest-level surface where this should be pinned.

**Fix:** Add to `test_quantum_feature_encoder.py`:
```python
def test_encoder_vmap_heterogeneous_inputs():
    enc = _enc()
    key = jax.random.PRNGKey(11)
    X = jax.random.normal(key, (8, 7))
    # Per-sample reference: call encoder one-by-one.
    Y_ref = jnp.stack([enc(X[i]) for i in range(X.shape[0])])
    Y_vmap = encoder_apply_batched(enc, X)
    assert jnp.allclose(Y_ref, Y_vmap, atol=1e-5)
```
This pins the actual contract: vmap == per-sample loop.

---

## MEDIUM

### M1. The `q(x)` term is computed once per ODE call but the integrator may evaluate the VF many times
**File:** `src/qlnn_/cells/liquid_quantum_cell.py:172`, `src/qlnn_/models/qlnn_forecaster.py:140-147`

`cell.__call__(t, h, x)` calls `self.encoder(x)` every time Diffrax evaluates the vector field. Since `x` is held constant over each sub-interval (zero-order hold), `q(x)` is a constant of the integration — but Tsit5/Dopri5 will evaluate `vf` 6+ times per accepted step plus rejections. Each evaluation re-runs the full PennyLane circuit. Under JIT the XLA optimizer might CSE this, but PennyLane QNode calls are typically not visible enough to XLA for that to be guaranteed.

**Why it matters:** Not a correctness bug — but for a paper claim about "quantum dynamics," it would be embarrassing to be asked at peer review "how many circuit evaluations per training step?" and discover it's 24 (history) × ~30 (step evals) × `num_qubits` measurements per epoch per sample, not 24 × `num_qubits`.

**Fix:** Refactor so the encoder fires once per sub-interval, before the solve, and the cell becomes `cell(t, h, q_const, A_const, tau_const)`:
```python
def _integrate(self, h, x_const, dt):
    q_const = self.cell.encoder(x_const)   # one PQC call
    inv_tau = 1.0 / self.cell.tau()
    A = self.cell.A
    def vf(t, y, args):
        q_, inv_tau_, A_ = args
        return -(inv_tau_ + q_) * y + A_ * q_
    sol = diffrax.diffeqsolve(..., args=(q_const, inv_tau, A), ...)
```
This is a 5–10× speedup on the dominant cost. It also makes the "input held constant" property explicit in the term signature.

---

### M2. `_predict_all` is not JIT-stable in shape — the trailing partial batch triggers retrace
**File:** `src/qlnn_/training/trainer.py:94-109, 144-158`

`predict_batch = eqx.filter_jit(jax.vmap(model))` is cached on the batch shape. The last mini-batch in val/test (`n % batch_size != 0`) has a different leading dim and forces XLA to recompile. This is a perennial JAX perf gotcha. Same applies inside the training loop for the trailing batch of every epoch.

**Why it matters:** Quantum training is slow; doubling recompiles per epoch is a real time cost. Not a correctness bug.

**Fix:** Pad the last batch to `batch_size` with replicates and slice off the extras after `predict_batch`. Or skip JIT for the trailing partial batch (a few sample evals on Python-driven path).

---

### M3. `float(loss)` host-syncs every batch
**File:** `src/qlnn_/training/trainer.py:187-189`

```python
train_se_sum += float(loss) * bs
```
`float(loss)` blocks on a device-to-host copy every batch. On a small CPU run this is invisible; on GPU it serializes the pipeline.

**Why it matters:** Minor perf. Listed because the trainer is supposed to be reusable for the Step 3+ hybrid where wall-clock starts mattering.

**Fix:** Accumulate `loss * bs` as a JAX array inside the loop and call `float(...)` once after the epoch. Or move accumulation into a `jax.lax.scan` over the epoch.

---

### M4. `dt0` clipping silently changes dtype
**File:** `src/qlnn_/models/qlnn_forecaster.py:147`

```python
dt0 = jnp.minimum(jnp.asarray(cfg.dt0, dtype=jnp.float32), dt * 0.5)
```
`cfg.dt0` is hardcoded to `jnp.float32`, but `dt` comes from `t_hours[i+1] - t_hours[i]` whose dtype depends on the caller. The script passes `t = time_hours[start:end].astype(np.float64)` into `make_horizon_windows`, so `w_train.t` is float64. Once converted to JAX (no `jax.config.update("jax_enable_x64", True)` anywhere), float64 is silently demoted to float32 — so in practice things line up. But if anyone ever turns on x64 mode (recommended for ODE work), `jnp.minimum(float32, float64)` will upcast `dt0` and the downstream solver state to float64 — which is fine, but `cfg.dt0`'s float32 asarray is then a useless step.

**Why it matters:** Hidden coupling to JAX's global float-precision setting. Hurts portability of the JAX subpackage.

**Fix:** Drop the explicit `dtype=jnp.float32` and let JAX promote:
```python
dt0 = jnp.minimum(jnp.asarray(cfg.dt0), dt * 0.5)
```
Or, better, promote to `dt.dtype`:
```python
dt0 = jnp.minimum(jnp.asarray(cfg.dt0, dtype=dt.dtype), dt * 0.5)
```

---

### M5. Missing test: gradient through the full forecaster (including Diffrax) is exercised but tolerances are not pinned
**File:** `tests/qlnn_/test_qlnn_forecaster.py:84-97`

`test_gradients_flow_to_all_leaves` asserts every leaf gets `> 0` gradient mass. Good. But it doesn't pin a *known* gradient (e.g., via finite-difference check on a tiny config) and it doesn't pin gradients with respect to **inputs `x`** (only with respect to params). If `eqx.filter_grad` were misconfigured to drop a subset of leaves (say, if someone made `cell` static by accident), the test would still pass as long as the *remaining* leaves get grad. A stronger test compares analytic vs FD on at least one parameter.

**Fix:** Add a small FD check on `cell.A[0]` (cheapest leaf to perturb):
```python
def test_grad_fd_check_on_amplitude():
    m = _model(num_qubits=2, num_layers=1)
    x, t = _sample()
    def f(model): return model(x, t)
    eps = 1e-3
    g_ana = eqx.filter_grad(lambda mdl: f(mdl))(m).cell.A
    m_plus = eqx.tree_at(lambda mdl: mdl.cell.A, m, m.cell.A.at[0].add(eps))
    m_minus = eqx.tree_at(lambda mdl: mdl.cell.A, m, m.cell.A.at[0].add(-eps))
    g_fd = (f(m_plus) - f(m_minus)) / (2*eps)
    assert jnp.allclose(g_ana[0], g_fd, atol=1e-2)
```

---

### M6. `best_model` defaults to the untrained input model if first eval is NaN
**File:** `src/qlnn_/training/trainer.py:162-208`

```python
best_val = float("inf")
best_model = model  # the untrained input
...
improved = val_m.mse_norm < best_val   # NaN < inf is False
```
If the first eval returns `mse_norm = NaN` (e.g., a config that blows up immediately — see H1), the loop will report no improvement, `best_model` stays as the untrained initial PyTree, and the returned val/test metrics will be computed on the initial untrained model. The training history will show NaN values and `best_epoch = 0` (initial) — silently bogus results that will land in `seeds_summary.json`.

**Fix:**
- Either: also accept the first eval unconditionally regardless of NaN (sentinel "first eval always wins"), then bail with an error if `best_val` is non-finite after training.
- Or: raise immediately if any `val_m.mse_norm` is `NaN`. Quietly returning a "trained model" that wasn't trained is the worst outcome.

---

### M7. Static field `circuit: DataReuploadingCircuit` relies on default object hashing
**File:** `src/qlnn_/encoders/quantum_feature_encoder.py:77, 92`

`DataReuploadingCircuit` is registered as an Equinox static field. Equinox uses the static portion of the PyTree as part of the JIT cache key, which requires hashability. `DataReuploadingCircuit` doesn't define `__eq__`/`__hash__`, so it inherits Python's default identity-based hash. This is fine when the *same* encoder instance is reused across JIT calls (the case in training), but if two encoders with identical configs are created and a function is jitted over both, they'll get different cache entries. More worryingly: `eqx.tree_serialise_leaves` (used in `scripts/train_qlnn.py:271`) only serializes the *leaves*. On deserialization the static config and circuit are taken from the freshly-constructed shell, which should match — but if the shell is built with a different `init_circuit_std` or `ring_entanglement`, the loaded model will silently use the new (mismatched) circuit topology with the old weights.

**Why it matters:** Reproducibility of saved checkpoints depends on the caller building the right shell. There's no checksum/config check on load.

**Fix:** Save `config.json` alongside `best_model.eqx` (already done in `train_qlnn.py:116`), AND when loading require the new shell's `config` to match. Add a small `load_qlnn_forecaster` helper that takes the directory and rebuilds the matching shell from `config.json`.

---

### M8. `tests/qlnn_/test_qlnn_trainer.py::test_best_checkpoint_is_returned` doesn't prove the model parameters are best-epoch
**File:** `tests/qlnn_/test_qlnn_trainer.py:180-206`

The test checks `res.val_metrics.mse_norm == min(history.val_mse_norm)`. But `val_metrics` is re-computed from `final_model = best_model` after the training loop, so this is largely tautological: if `best_model` were *any* model with val MSE equal to the minimum in history, the assertion passes. To prove it's the actual best-epoch model, you'd need to compare model parameters (e.g., serialize `best_model` and confirm it differs from the latest `model`).

**Fix:** Strengthen the test:
```python
# After training: build a model from history with the final params and confirm it does NOT match the returned model.
latest_pred = jax.vmap(res.model_at_end_if_we_had_one)(...)
best_pred = jax.vmap(res.model)(...)
assert not jnp.allclose(latest_pred, best_pred)
```
Easier: store the model at the end of training (no best-selection) in a separate test, then with best-selection on, assert the leaves differ in at least one position. Or use `eqx.tree_equal` against an obvious non-best epoch.

---

## LOW

### L1. Last-layer entangler is wasted before local-Z measurement
**File:** `src/qlnn_/circuits/reuploading.py:67-75`

After the final layer's CNOT block, the circuit immediately measures `qml.PauliZ(i)` per wire. Entanglement created by that final CNOT is partially lost on the local-Z measurement — the per-wire ⟨Z⟩ values are unchanged by `CNOT` followed by `Z⊗I` on the control wire's view only if the post-CNOT state's wire-i reduced density matrix is unchanged. In general it is not. The variational power is fine; the *parameter count* is fine. This is a known design choice in the Pérez-Salinas family (some variants drop it, some keep it). Not a bug.

**Fix:** None required. If the PI wants to be especially defensible at peer review, add a config flag `last_layer_no_entangle: bool = False` and note in the spec which variant is used.

### L2. `ring_entanglement` flag wording in docstring is misleading
**File:** `src/qlnn_/circuits/reuploading.py:29-31, 71-75`

The dataclass docstring says "ring close" only fires when `num_qubits > 2`. The PI's review prompt asked whether a 2-qubit "ring" would be a no-op — actually, with `num_qubits=2`, the linear chain has CNOT(0,1) and the ring would add CNOT(1,0), which is *not* a no-op and *not* a duplicate of CNOT(0,1) (they don't commute). So the gating is defensive against a *redundant* operation rather than a no-op. The comment at line 30 is fine; the inline gate condition is correct. Pure nit: the comment around lines 31 says "double-entangle" — that's not quite the right description. Tighten the comment.

**Fix:** Adjust line ~30 docstring to: "Skipped when num_qubits ≤ 2 because the wrap-around CNOT would either be redundant (1 qubit: no-op) or add a second CNOT that does not generate meaningfully new entanglement on a 2-qubit chain."

### L3. `num_layers=1` is allowed but silently disables re-uploading
**File:** `src/qlnn_/circuits/reuploading.py:36-40`

With `num_layers=1`, the circuit applies data → Rot → entangle exactly once. By definition this is **not** data re-uploading (re-uploading needs ≥ 2 data injections to gain the Fourier expressivity Schuld et al. analyze). The config accepts it without a warning.

**Fix:** Either tighten to `num_layers >= 2` (mild paper credibility win — re-uploading is literally in the class name) or emit a `UserWarning` from `__post_init__` when `num_layers == 1`. Don't fail silently.

### L4. `init_w_std = 0.1` produces near-zero embedding angles at init; `init_circuit_std = 0.05` produces near-identity circuit
**File:** `src/qlnn_/encoders/quantum_feature_encoder.py:44-47, 109-119`

With `init_w_std=0.1` and typical normalized inputs `x ∈ [0, 1]^F`, `pre = Wx + b ~ N(0, F * 0.01)`. For F=7 that's `N(0, 0.07)` → `tanh ≈ pre` (tiny) → angle ≈ 0. With circuit weights also near zero, the PQC at init is near-identity and outputs `q ≈ +1` on every qubit. This is benign for gradient flow (the encoder gradient test at line 35-47 passes) but it means **all hidden states at init are clipped near the "all +1" corner of [-1,1]^Q**, so the cell's vector field starts in a degenerate region (`leak = (1/tau + 1) h`, full damping; `drive = A`, constant). Training has to climb out of this. Not a bug, but worth documenting in the spec so reviewers don't ask "what's the initial latent geometry?"

**Fix:** Either bump `init_w_std` to ~0.3 (still inside the tanh's linear regime: `tanh(0.3*sqrt(7)*1) ≈ tanh(0.79) ≈ 0.66`) so the initial angles span more of `[-π, π]`, or document the choice in the encoder docstring.

### L5. `_loss_fn`, `_build_optimizer` are module-private but imported by tests
**File:** `tests/qlnn_/test_qlnn_trainer.py:24`

```python
from qlnn_.training.trainer import _build_optimizer
```
Reaching into a `_`-prefixed name from tests is fine but ties test stability to internal refactor. Consider exposing a public `build_optimizer` (no underscore) since it's part of the testable contract.

### L6. `delta_head_W` has shape `(num_qubits, 1)` and `.squeeze(-1)` afterwards
**File:** `src/qlnn_/models/qlnn_forecaster.py:128-131, 198`

A scalar-output head is more idiomatically `delta_head_W : (num_qubits,)` and the result is already scalar — no squeeze needed. Pure nit; the current form is the deliberate analog of the PyTorch `nn.Linear(Q, 1)`.

### L7. `seeds_summary.json` shape consistency assumed but not asserted
**File:** `scripts/train_qlnn.py:282-288`

The script asserts in the docstring that `seeds_summary.json` matches `scripts/train_baseline.py`'s shape. It does (both use `aggregate_seed_metrics`), but there's no test pinning this. If `train_baseline.py` adds a field, the QLNN summary will silently diverge.

**Fix:** Add a snapshot test that loads both summaries (after a tiny 1-seed run of each) and asserts the keys match. Lives in `tests/integration/` rather than `tests/qlnn_/`.

### L8. `Any` import unused
**File:** `src/qlnn_/encoders/quantum_feature_encoder.py:27`, `src/qlnn_/cells/liquid_quantum_cell.py:31`

Both files import `Any` but use it only as the type of `t` in the cell's `__call__`. The encoder import is fully unused. Minor.

### L9. Tests don't cover `od_index != 0`
**File:** `tests/qlnn_/test_qlnn_forecaster.py:23-46`

Every test uses `OD_INDEX = 0`. The config validates `0 <= od_index < input_dim`, but no test confirms the residual head correctly anchors to `x[-1, od_index]` for non-zero `od_index`. If a future refactor accidentally hardcodes `x[-1, 0]`, the tests pass.

**Fix:** Parametrize `test_residual_anchors_to_od_last_at_init` over `od_index ∈ {0, 1, INPUT_DIM-1}`.

### L10. `time_hours` array dtype assumption
**File:** `src/qlnn_/training/trainer.py:134-136`

`jnp.asarray(t_train)` with `t_train: np.ndarray` of dtype `float64` becomes float32 under default JAX. This is mentioned in M4 but is also a latent issue here: integration accuracy at 1e-3 / 1e-4 tolerance is fine in float32, but a paper reviewer might ask. Worth a one-line comment in the trainer's docstring naming the precision contract.

---

## Cross-stack consistency — verified OK

These were specifically flagged in the prompt; I traced them and they're correct:

- `compute_metrics` is imported from `quantum_liquid_neuralode.evaluation.metrics` (`trainer.py:23`), not shadowed. Same function as classical.
- `aggregate_seed_metrics` is imported from the same module (`train_qlnn.py:49`).
- `seeds_summary.json` shape matches: `{n_seeds, seeds, val: {mse_norm: {mean,std,min,max,n_seeds}, mae_raw:..., rmse_raw:..., r2_raw:...}, test:...}`. Identical to what `train_baseline.py` produces.
- Per-seed `metrics.json` shape (`{best_epoch, val: ForecastMetrics.to_dict(), test: ForecastMetrics.to_dict()}`) matches.
- `summarize_baselines.py` will pick up QLNN runs without modification.
- Best-checkpoint semantics: `best_model = model` after a JAX-functional `apply_updates` is safe — each update returns a new PyTree, the rebound name does not mutate prior captures. Confirmed correct.
- Optimizer chain `optax.chain(clip, opt)` applies clip BEFORE adam. Correct (Optax runs left-to-right). When `grad_clip_norm == 0` the clip is omitted; when `weight_decay > 0` AdamW is used; otherwise Adam.
- `opt.init(eqx.filter(model, eqx.is_array))` filters to array leaves only — the static `config` and `circuit` (PennyLane QNode) are not in `opt_state`. Verified.
- `eqx.filter_value_and_grad(_loss_fn)` filters to array leaves on `model` (first arg). Correct.
- `SaveAt(t1=True)`, `PIDController(rtol, atol)`, `max_steps` all wired correctly.
- Diffrax closure over `self.cell`: re-bound on every `__call__` invocation; under outer JIT the cell's leaves are tracers carried via the model argument, not stale Python-level captures. Standard Equinox+Diffrax pattern.
- `tau` inverse-softplus init `log(expm1(tau_init - tau_min))`: numerically stable for the configured `delta = 0.9`. `expm1` (rather than `exp(x) - 1`) is the right choice. The post-init guard `tau_init > tau_min` (strict) correctly rejects equality. Verified.
- `ring_entanglement` gate `num_qubits > 2` is correct (see L2 for nit about wording).
- Re-uploading order RX → Rot → CNOT per layer is a defensible Pérez-Salinas variant; `weights[layer, i, :]` is correctly indexed. Verified.

---

## Recommended action order

1. **Fix H1** before any HPO over `tau_min` is run. (Add config guard + test.)
2. **Fix H2** before training on the full qZETA dataset (add the script-level `dt > 0` assert).
3. **Fix M6** in the same patch as H1 — they share root cause (silent garbage when the model fails to train).
4. **Fix H3 + H4** in a tests-only patch — these are pure coverage gaps, no code change needed.
5. **Fix M1** before the Step 4 / 5 work — the 5–10× wall-clock win on circuit evaluations compounds across seeds and epochs.
6. **M7** — add a `load_qlnn_forecaster` helper that reads `config.json` and rebuilds the shell — before any inference-only / paper-figure workflow that depends on serialized checkpoints.

Everything in **LOW** can be deferred to a "cleanup" PR; none of it affects paper claims.
