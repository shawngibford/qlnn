# P3.9 — PDE multi-family port (te_qpinn_fnn + te_qpinn_qnn + qcpinn → 2D)

**Purpose:** close the audit gap raised after P3.8 — the PDE side
currently has only `chebyshev_dqc_2d` as a quantum family, while the
ODE side has 4. P3.9 ports the three remaining PINN-style solver
families to (t, x) coordinate handling so the PDE matrix matches the
ODE matrix shape (4 quantum × 3 PDEs × 3 seeds = 36 PDE runs).

**Out of scope for P3.9:** rf_qrc — its frozen-reservoir closed-form
ridge architecture is fundamentally different from PINN-style residual
training (no autodiff through the reservoir; the "training" is a
linear solve against ground-truth windows, not a physics residual). A
faithful 2D port either reformulates rf_qrc as a forecaster (its
intended use; that's P4 territory) or rebuilds it as a PINN-style
trainable reservoir, which would no longer be the paper's family.
Deferred to P4 as a forecaster; not represented in the P3.9 PDE matrix.

## What stays unchanged

- `src/qlnn_/training/pde_residual_loss.py` — `make_pde_residual_loss`
  and `train_pde_solver` are already family-agnostic. They accept any
  `circuit(t_chev, x_chev, weights) → scalar` callable. P3.9 adds
  three new callables; the loss + train loop stay frozen.
- The 1D solver builders (`te_qpinn.py`, `qcpinn.py`, `reuploading.py`)
  — P3.9 produces **sibling 2D builders**, not modifications.
- The P3.7 PDE gate test for chebyshev_dqc_2d — must keep passing.
- The 1D ODE gate test (`physics_residual_loss.py`) — must keep passing.
- The `{w, s, b}` pytree wrapping pattern for Lagaris hard-IC trial
  solutions — every 2D port produces a scalar field circuit that
  plugs into the same wrapping.

## The three ports

### 1. `qcpinn_2d` — trivial port

Source: Farea et al. 2025 (arXiv:2503.16678v6), `CIRCUIT_SPECS.md` §3.

The paper's QCPINN architecture is:
```
x → pre-NN → angle embedding → HEA → ⟨Z⟩ per qubit → post-NN → u_θ(x)
```
where the pre-NN already has a configurable `input_dim`. To extend to
PDE coordinates, **the only change** is to set `input_dim = 2` on the
config. Everything downstream — the angle embedding, the per-topology
PQC, the post-NN — stays bit-identical to the 1D version.

**File:** `src/qlnn_/circuits/qcpinn.py` already has the structural
ability via `QCPINNConfig.input_dim`. P3.9 just exposes a `build_qcpinn_2d`
wrapper that calls `build_qcpinn(QCPINNConfig(input_dim=2, ...))` and
adapts the pipeline signature to `(t, x, weights) → scalar` (current
1D is `(x_scalar, weights) → scalar`).

**Faithfulness:** the paper's per-topology Table 2 param-count formulas
[4(n−1)L for Alternate, 3nL for Cascade, (n²+4n)L for Cross-mesh,
4nL for Layered] hold unchanged because they depend on n and L, not
on the pre-NN input dim. Unit-test hook stays the same. Pre-NN param
count grows by `pre_hidden` (one extra column in the first weight
matrix) — disclosed in the test.

**Estimated dev:** 1-2 hr (mostly tests).

### 2. `te_qpinn_fnn_2d` — small natural extension

Source: Berger et al. 2025 (s41598-025-02959-z), `CIRCUIT_SPECS.md` §1.

The paper's architecture:
```
x → rescale x̃ → FNN(x̃; θ_emb) = φ ∈ ℝⁿ → ⊗_k R_y(φ_k · x̃) → HEA → ⟨⊗_k Z_k⟩
```
The FNN's purpose is to generate **per-qubit angle weights** as a
function of the input. For a 2D coordinate, the natural extension is:
- FNN input dim becomes 2 → FNN output stays ℝⁿ (one angle weight per
  qubit, same as 1D).
- The embedding layer per qubit becomes `R_y(φ_k · t̃) · R_y(ψ_k · x̃)`
  where ψ is a second FNN-generated angle vector — **OR** the simpler
  split-qubit option matching the chebyshev_dqc_2d convention:
  - first n_t qubits get `R_y(φ_t,k · t̃)` from `FNN_t(t̃, x̃)`,
  - last n_x qubits get `R_y(φ_x,k · x̃)` from `FNN_x(t̃, x̃)`.

**Locked design choice (P3.9):** split-qubit version. Reasons:
1. Symmetric with chebyshev_dqc_2d's split-qubit layout (apples-to-apples
   comparison: same qubit count, same per-coordinate qubit budget).
2. The FNN-generates-per-qubit-weights pattern from the paper extends
   cleanly (one FNN per coordinate, each producing per-qubit angles).
3. Per-coordinate FNN heads are functionally analogous to the paper's
   single FNN, just doubled. Disclose as a declared design choice.

**Faithfulness hooks:**
- PQC rotation count remains 3·n·L (paper Eq. 12 unchanged; HEA depth-L
  on n qubits).
- FNN param count grows by 2× (two heads instead of one), with each
  head's input dim 2 instead of 1.
- Test hook: assert N_rot = 3·n·L unchanged; assert two FNN heads in the
  pytree (`fnn_t_*` and `fnn_x_*` keys).

**Estimated dev:** 3-4 hr.

### 3. `te_qpinn_qnn_2d` — design choice required

Source: 2605.13892v1 (QPINN lid-driven cavity), `CIRCUIT_SPECS.md` §2.

The paper's architecture:
```
x → x̃ → U_embed(θ_Q, x̃) → α_k = π·⟨Z_k⟩
                        → ⊗_k R_y(α_k) → HEA U_var(θ_var) → Σ_j ⟨Z_j⟩
```
U_embed takes the input as an angle parameter in its embedding layers
(`R_z(θ_emb · x̃)` in our P3 implementation). The 2D extension question:
how do both coordinates feed U_embed?

**Locked design choice (P3.9):** split-qubit U_embed.
- The U_embed PQC operates on n_total = n_t + n_x qubits.
- Each embedding-layer iteration applies trainable `R_y(θ_emb_t)` on
  the t-qubits, input-modulated `R_z(θ_t · t̃)` on the t-qubits, then
  the same for x-qubits with separate trainable params, then nn-CNOT
  chain across all n_total qubits (so the entanglement mixes t and x).
- Output: α_k = π·⟨Z_k⟩ for all k ∈ [n_total]. Both t-encoded and
  x-encoded qubits contribute to α.
- Re-encode α via R_y(α_k) into the variational HEA across n_total
  qubits; readout Σ_j ⟨Z_j⟩.

**Why split-qubit (not interleaved):** the paper's text leaves the
multi-input embedding schematic. Split-qubit preserves per-coordinate
expressivity equivalence with the other two ports and chebyshev_dqc_2d.
Worth disclosing as a declared design choice in the spec card.

**Faithfulness hooks:**
- Trained-param linearity in n·(K_embed + L_var) remains (the paper's
  scaling hook). With n = n_t + n_x and K_embed and L_var unchanged,
  the linear-in-n·(K+L) test passes by construction.
- Disclose split-qubit U_embed as DECLARED DESIGN CHOICE in
  CIRCUIT_SPECS.md (a §2 amendment).

**Estimated dev:** 4-5 hr.

## New files

| File | Action | LOC | Purpose |
|---|---|---|---|
| `src/qlnn_/circuits/pde_2d/__init__.py` | NEW | ~10 | Public re-exports for the three new 2D builders. |
| `src/qlnn_/circuits/pde_2d/qcpinn_2d.py` | NEW | ~80 | `build_qcpinn_2d` thin wrapper over `build_qcpinn` with `input_dim=2`. |
| `src/qlnn_/circuits/pde_2d/te_qpinn_fnn_2d.py` | NEW | ~180 | Split-qubit FNN-embedding port. |
| `src/qlnn_/circuits/pde_2d/te_qpinn_qnn_2d.py` | NEW | ~220 | Split-qubit U_embed port. |
| `tests/qlnn_/test_pde_2d_ports.py` | NEW | ~250 | Per-family: (a) signature + shape; (b) param-count hook still holds; (c) mechanism gate — `jacrev(jacrev)` through each circuit returns finite, non-trivial values at random init; (d) heat-equation convergence-mini gate per family (seed 0, looser MAE < 0.20 since not all families are Chebyshev-towers). |
| `src/qlnn_/training/pde_demo.py` | EDIT | ~50 | Add per-family dispatch alongside chebyshev_dqc_2d. New `FAMILIES` dict keyed by 2D family name → builder + default config. |
| `scripts/run_p3_9_pde_matrix.py` | NEW | ~80 | CLI: 3 new quantum families × 3 PDEs × 3 seeds = 27 runs (chebyshev_dqc_2d's 9 already in P3.8 — combined matrix figure references both). |
| `scripts/make_p3_9_pde_matrix_figure.py` | NEW | ~250 | 4 quantum × 3 PDE bar grid (relL2, log scale, classical PINN floor). Per-PDE loss-trajectory panel. Per-family BC violation panel. |
| `refs/CIRCUIT_SPECS.md` | EDIT | ~30 | Add 2D-port design-choice disclosures for each of the 3 ports (§1, §2, §3 each get a "P3.9 2D-port amendment" subsection). |

## Acceptance criteria

1. All 3 new builders produce a callable `(t_chev, x_chev, weights) →
   scalar` compatible with `make_pde_residual_loss`.
2. **Mechanism gate (per family):** `jax.jacrev(jax.jacrev(circuit,
   argnums=1), argnums=1)` returns finite, non-trivial values at random
   init. (Same gate as chebyshev_dqc_2d's mechanism test, replayed
   for each family.) BLOCKS the phase if any family fails.
3. **Convergence mini-gate (per family):** heat-equation seed 0,
   interior MAE < 0.20 after 1200 steps. Looser than chebyshev_dqc_2d's
   0.10 because non-Chebyshev families don't get the Chebyshev-tower's
   spectral advantage on the heat-equation's exponential decay.
4. Full P3.9 sweep: 3 new families × 3 PDEs × 3 seeds = 27 runs at
   audit-corrected configs (heat 1200, burgers 1500, AC 64×32×1800).
   Plus 3 seeds of classical PINN as the baseline floor.
5. `paper/figures/fig_p3_9_pde_matrix.{png,pdf}` rendered: 4-quantum-
   family × 3-PDE bar grid plus diagnostic panels.
6. Full pytest suite green; `verify_paper_integrity.py` exit-0.
7. CIRCUIT_SPECS.md amended with P3.9 design-choice disclosures.

## Atomic commit sequence

1. `feat(P3.9-port-qcpinn): qcpinn_2d.py + tests` — trivial port first
   to lock the 2D-port test pattern.
2. `feat(P3.9-port-te_fnn): te_qpinn_fnn_2d.py + split-qubit FNN heads + tests`
3. `feat(P3.9-port-te_qnn): te_qpinn_qnn_2d.py + split-qubit U_embed + tests`
4. `feat(P3.9-sweep): pde_demo.FAMILIES + run_p3_9_pde_matrix.py + sweep results`
5. `feat(P3.9-fig): make_p3_9_pde_matrix_figure.py + rendered figure`
6. `docs(P3.9-specs): CIRCUIT_SPECS.md 2D-port design-choice amendments`
7. `docs(HANDOFF): P3.9 done — PDE family coverage 4× quantum; resume at P4`

## Estimated compute (sweep)

- Heat 1200 steps × 3 families × 3 seeds = 9 runs: ~30 min (assuming
  3-5 min per run; qcpinn_2d's pre/post-NN is fast; te_qpinn variants
  similar to chebyshev_dqc_2d).
- Burgers 1500 steps × 3 families × 3 seeds = 9 runs: ~45 min.
- Allen-Cahn 64×32×1800 × 3 families × 3 seeds = 9 runs: ~3 hr.
- **Total: ~4-5 hr CPU.**

## Estimated dev

~3 days: ports (1.5 days), tests (0.5 day), sweep + figure + HANDOFF
(1 day). Bulk is the te_qpinn_qnn_2d port (it carries the most design
risk because U_embed's gate-by-gate schedule was "schematic only" in
the source paper — same risk profile as 1D).

## Risk: do the 2D ports each clear the mechanism gate?

Most likely yes: chebyshev_dqc_2d's mechanism gate passed first try,
and that gate exercises the *same* `jacrev(jacrev(QNode))` pattern.
The three new families differ in their **circuit content**, not in
their **autodiff topology** — every family is `(2-scalar-input,
weights) → scalar` through a PennyLane JAX QNode, and the autodiff
mechanism doesn't see the gate structure.

Possible failure mode: if a family has a discrete-output observable
or a piecewise non-differentiable component (e.g. PauliZ tensor
product over many qubits → exponentially small gradient at random
init). All three families avoid this — the readouts are linear in
single-qubit ⟨Z⟩ values (qcpinn: per-qubit ⟨Z⟩ then linear post-NN;
te_qpinn_fnn: tensor-product ⟨⊗ Z⟩ which COULD be exponentially small
at high n but stays manageable at n=4-8). Worth verifying at the
mechanism gate before committing the sweep.

## When P3.9 starts

After:
1. P3.8 deferred compute finishes (AC + Lorenz, ~5 hr running tonight).
2. P3.8 figure re-rendered with full data.
3. P3.8 HANDOFF advance commit lands.

Then P3.9 begins clean. Estimated start: same session if sweep
completes in time; next session otherwise.
