# CIRCUIT_SPECS.md — P3a faithfulness manifest (dual-verified)

**Status:** P3a gate SATISFIED for all 7 literature ansatz families.
Each family's circuit was (1) extracted from its source by a primary
agent reading the PDF via the Read tool in explicit page ranges (or
arXiv full-text HTML for the 3 foundational anchors), and (2)
**independently re-derived** by a second agent from the same source
without sight of the primary card. Primary and check agree on every
circuit's gate structure, scaling, readout, and derivative method. No
family is blocked.

**Rule (from the plan, enforced here):** no family enters the P6
unified search until its spec is PDF/source-grounded and
dual-cross-checked. That condition is now met for all 7.

Detailed gate-by-gate cards live alongside this file and are committed
as the evidence trail:

| Family | Primary card | Independent check |
|---|---|---|
| te_qpinn_fnn | `_speccard_te_qpinn.md` | `_check_te_qpinn.md` |
| te_qpinn_qnn | `_speccard_te_qpinn_qnn.md` | `_check_te_qpinn.md` (FAM2) |
| qcpinn | `_speccard_qcpinn.md` | `_check_qcpinn.md` |
| rf_qrc | `_speccard_rf_qrc.md` | `_check_rf_qrc.md` |
| chebyshev_dqc / lubasch_multicopy / data_reuploading | `_speccard_foundational.md` | `_check_foundational.md` |

---

## P3a corrections to the locked plan (faithfulness gate findings)

The gate did its job — it caught two non-faithful assumptions carried
in `~/.claude/plans/i-want-to-automate-quiet-cerf.md`. The 10-ansatz
roster and the science are unchanged; only factual source attributions
are corrected:

1. **`te_qpinn_qnn` source re-attributed.** The plan's roster table
   said `s41598-025-02959-z.pdf` (Berger, Nat Sci Rep 2025) yields
   *both* `te_qpinn_fnn` and `te_qpinn_qnn`. **It does not.** Berger
   defines only the classical-FNN trainable embedding; its only
   non-FNN embeddings are *fixed, non-trainable* Chebyshev baselines
   from prior work. A faithful fully-quantum trainable embedding **does
   exist** in the sibling PDFs: `2605.13892v1` (QPINN lid-driven
   cavity) is the chosen source for `te_qpinn_qnn`, independently
   corroborated by `2602.14596v1` (parabolic) and `2602.09291v1`
   (reaction–diffusion), which contain the same QNN-trainable-embedding
   construction. Roster intact (still 10 families); user-confirmed
   resolution path ("check variant PDFs first").

2. **`qcpinn` benchmark PDEs corrected.** The plan described QCPINN as
   "benchmarked on Poisson, Burgers, Navier–Stokes." The actual
   Appendix-A residual operators are **Helmholtz (A.1), 2-D
   lid-driven-cavity / incompressible Navier–Stokes (A.2), 1-D wave
   (A.3), nonlinear Klein–Gordon (A.4), 2-D convection–diffusion
   (A.5)** — recovered identically by both the App-A continuation pass
   and the independent dual-check. Use this list, not the plan's loose
   one, when wiring QCPINN's physics loss.

These corrections are recorded here (not silently absorbed) per the
pre-registration's deviation discipline.

---

## Per-family verified spec (the binding summary P3 consumes)

For each family the **gate structure below is dual-confirmed**; items
marked **[DECLARED DESIGN CHOICE]** are genuine source gaps that *both*
agents independently hit (the source is schematic/qualitative there) —
they are NOT extraction errors; the implementation fixes them with a
documented, cited choice and the choice is logged in the ansatz
docstring + unit test.

### 1. `te_qpinn_fnn` — Berger et al., Sci. Rep. 2025 (`s41598-025-02959-z.pdf`)
- **Embedding:** rescale `x → x̃`; classical FNN (TanH) produces
  per-qubit `φᵢ(x̃)`; embed `⊗ᵢ R_y(φᵢ(x̃)·x̃)` (Eq. 11).
- **Ansatz:** `L` HEA layers, each = per-qubit `{R_x,R_y,R_z}` +
  nearest-neighbour CNOT chain (Eq. 12, Fig. 3).
- **Readout:** `O = ⊗ᵢ Zᵢ` (Eq. 13).
- **Loss/derivative:** physics residual; input-coordinate derivative =
  parameter-shift on the PQC × FNN backprop (chain/product rule,
  Eq. 17). Optimiser L-BFGS.
- **Unit-test hook:** rotation-param count `N_rot = 3·n·L`; weight
  tensor shape `(L, n, 3)`. Cross-validated by the paper's stated
  "60 PQC params @ 4 qubits / 5 layers" (3·4·5 = 60).
- **[DECLARED DESIGN CHOICE]** exact per-layer CNOT pattern — the
  paper explicitly says it "varies in literature"; pick the
  nearest-neighbour chain and document it.

### 2. `te_qpinn_qnn` — QPINN lid-driven cavity (`2605.13892v1.pdf`)
- **Embedding (fully quantum, trainable):** affine-normalise coords to
  `[-1,1]`; trainable PQC `U_embed(θ_Q)` = alternating
  input-dependent gate layers + trainable rotation layers +
  nearest-neighbour CNOTs; measure `αᵢ = π·⟨Zᵢ⟩`; re-encode
  `⊗ᵢ R_y(αᵢ)` into the downstream solver (Eqs. 10–18, 25–26, Fig. 1,
  Algorithm 1).
- **Ansatz:** HEA variational block `U_var(θ_var)`, `L` layers of
  `{R_x,R_y,R_z}` + nn-CNOT.
- **Readout:** `O = Σⱼ Zⱼ`.
- **Loss/derivative:** physics residual; separate p/ψ circuits
  (2 evals/point); input derivative = parameter-shift × QNN-embedding
  backprop chain (Eq. 25). Optimiser L-BFGS.
- **Unit-test hook:** trainable params scale **linearly** in
  `N_q · L` (p. 6); anchor magnitude ≈360 params (paper's comparison
  point vs 608 FNN / 6594 classical PINN) — assert linearity, not the
  exact constant.
- **[DECLARED DESIGN CHOICE]** the per-gate generator schedule of
  `U_embed` and the exact rotations-per-layer multiplicity are
  schematic in *all three* sibling sources (Fig. 3 only) — fix a
  concrete schedule, document it, cite the schematic.

### 3. `qcpinn` — QCPINN (`2503.16678v6.pdf`)
- **Pipeline:** classical preprocessor NN (50-neuron TanH) → angle (or
  amplitude) embedding on `n=5` qubits → HEA
  `U(ψ)=∏ₖ Uₖ(ψₖ)Wₖ` → per-qubit Pauli-Z → classical postprocessor NN.
- **Topologies (pick per config; both agents derived identical
  formulas, verified vs the p. 15 worked example n=5,L=1):**

  | Topology | depth | params | 2-qubit gates |
  |---|---|---|---|
  | Alternate | `6L` | `4(n−1)L` | `(n−1)L` |
  | Cascade | `(n+2)L` | `3nL` | `nL` |
  | Cross-mesh | `(n²−n+4)L` | `(n²+4n)L` | `(n²−n)L` |
  | Layered | `6L` | `4nL` | `(n−1)L` |

- **Param counting:** embedding rotations are **NOT** in these totals
  (Table 2 / p. 15 are ansatz-layer only); trainable weights also live
  in the classical pre/post nets.
- **Loss/derivative:** physics residual (loss *structure* fully
  specified, Eq. 2); derivatives via **autodiff** (`shots=None`),
  parameter-shift explicitly avoided (§5.2/5.5).
- **Appendix-A PDE residuals:** Helmholtz (A.1, Eq. 5), 2-D
  lid-driven-cavity NS (A.2, Eq. 6), 1-D wave (A.3, Eq. 7), nonlinear
  Klein–Gordon (A.4, Eq. 8), 2-D convection–diffusion (A.5, Eq. 9).
- **Unit-test hook:** per-topology (params, 2q-gates, depth) formulas
  above; assert Cascade n=5,L=1 → (15, 5, 7); Cross-mesh → (45, 20, 24).
- **[UNSPECIFIED IN SOURCE]** exact data→`RX(θ)` embedding-angle
  formula; exact Cross-mesh gate ordering.

### 4. `rf_qrc` — RF-QRC, Phys. Rev. Research 6, 043082 (2024) (`2405.03390v2.pdf`)
- **Config:** QRC-C4, the row Table I labels "(RF-QRC)".
- **Reservoir (fixed/random — NOTHING here is trained):** on
  `|0⟩^⊗n`, recurrence operator `P = identity` (recurrence removed —
  the "RF" contribution; no state feedback); input feature map `Φ`
  (H + `R_y(angle)` + CNOT cascade + `R_y`), data rescaled to
  `[0,2π]`, applied **twice (×2)**; then a fixed random
  fully-entangled-symmetric `V(α)`, `α ~ U[0,4π]`, seeded and frozen
  per realisation.
- **Readout (the only trained object):** per-qubit Pauli-Z features
  `r`; closed-form Tikhonov ridge
  `(R Rᵀ + β I) W_out = R U_dᵀ` (Eq. 3); predict
  `u_p(t_{i+1}) = r(t_{i+1})ᵀ W_out` (Eq. 4); `β ∈ {1e-6,1e-9,1e-12}`.
- **Unit-test hook:** trainable-param count == `size(W_out)` only;
  reservoir states deterministic for fixed seed+input and **invariant
  to the readout fit**; circuit depth independent of `n` (Fig. 7).
- **[DECLARED DESIGN CHOICE]** exact `n>4` CNOT wiring for the
  "fully entangled" / "fully entangled symmetric" blocks — Figs. 22–25
  use a `⋮` ellipsis past `q₃`; *both* agents confirmed this is a real
  source gap. Fix a concrete generalisation, document + cite the
  schematic. Also unspecified: exact raw→`[0,2π]` rescale; shot count
  (paper uses noise-free statevector).

### 5. `chebyshev_dqc` — Kyriienko et al., PRA 103 052416 (2021), arXiv:2011.10395
- **Feature map (Chebyshev tower):** `⊗ⱼ R̂_{y,j}(2j·arccos x)` —
  angle `= 2j·arccos x`, tower degree linear in qubit index (Eq. 15).
- **Ansatz:** HEA `R_z–R_x–R_z` + CNON layers (Fig. 5a).
- **Readout:** total magnetisation `Ĉ = Σⱼ Ẑⱼ` (§III.3).
- **Derivative/loss (DQC):** circuit derivative via shift rule
  `d⟨Ĉ⟩/dx = ¼·(dφ/dx)·(⟨Ĉ⟩⁺ − ⟨Ĉ⟩⁻)` (Eqs. 9–10); differential-eq
  residual loss (Eqs. 19–22).
- **Unit-test hook:** at `x→0` the feature map must recover `T_n(0)`
  per the Chebyshev identity (Eq. 12).
- **[UNSPECIFIED IN SOURCE]** the "sparse" Chebyshev map variant has
  no numbered formula; generic HEA two-qubit gate left abstract.

### 6. `lubasch_multicopy` — Lubasch et al., PRA 101 010301(R) (2020), arXiv:1907.09032
- **Encoding:** amplitude encoding `|ψ⟩ = Σ_k ψ_k|k⟩` (Eq. 5) prepared
  by depth-`d` generic two-qubit `Û(λ)` (Fig. 1b).
- **Nonlinearity:** multiple identical copies fed through an
  ancilla Hadamard-test QNPU (Figs. 1a/2a) realising
  `F = f^{(1)*} ∏ⱼ (Oⱼ f^{(j)})`; nonlinear terms (e.g. `Σ|ψ_k|⁴`)
  read from ancilla `⟨σ̂_z⟩`.
- **Unit-test hook:** at copy-count `r=1` the QNPU reduces to a plain
  state overlap and ancilla `⟨σ_z⟩ = 1`.
- **[UNSPECIFIED IN SOURCE]** the two-qubit gate decomposition is
  Fig. 1b schematic only. Context/baseline circuit.

### 7. `data_reuploading` — Schuld, Sweke, Meyer, PRA 103 032430 (2021), arXiv:2008.08605
- **Construction:** `U(x) = W^{(L+1)} S(x) W^{(L)} … S(x) W^{(1)}`
  (Eq. 4); identical encoding blocks `S(x)`; Pauli generator ⇒
  integer-spaced accessible spectrum `Ω = {−L,…,L}` (Eq. 10/11,
  §II.2). This is the **truncated-Fourier backbone of hypothesis H1**.
- **Unit-test hook:** a single-qubit `L`-layer model is band-limited
  to `|ω| ≤ L`.
- **EXISTING-CODE FAITHFULNESS** (`src/qlnn_/circuits/reuploading.py`):
  Both agents confirm the code is **structurally Schuld-faithful** —
  per-layer identical `RX(inputs[i])` → trainable `Rot` → entangler
  reproduces the alternating `S(x)·W` of Eq. 4, and the `RX` generator
  yields the integer-spaced `Ω` that H1 depends on (so **H1's
  mechanism is real in-code**). Two recorded caveats for P3 (NOT
  P3a-blocking; spectrum unchanged):
  1. The code ends each layer on an entangler — it **omits the
     terminal trainable block `W^{(L+1)}`** of canonical Eq. 4.
     Strictly a *reduced* form (narrower variational family, identical
     frequency support). P3 may add the trailing `W^{(L+1)}` for full
     canonical faithfulness; if so, regenerate any affected baseline
     lock.
  2. The docstring **miscredits Pérez-Salinas (1907.02085)** for the
     architecture; the Fourier-expressivity claim correctly cites
     Schuld. P3 should fix the architecture citation.

---

## Implementation binding (P3 STATUS)

Per the architecture decision recorded in HANDOFF.md (commit
`7f29f03`), the registry `(inputs:(Q,)) → (Q,) PauliZ` contract is the
**forecaster encoder** interface. Solver-native families do NOT live
in the registry — forcing them in would manufacture the infidelity
P3a guards. The unified search at P6 crosses families × {solver |
forecaster} **as applicable** per the homing column below.

| Family             | Status | Homing      | Module                              | Tests                              |
|--------------------|--------|-------------|-------------------------------------|------------------------------------|
| data_reuploading   | ✅ pre-existing + P3-2a cleanup | forecaster (registry) | `src/qlnn_/circuits/reuploading.py` | `tests/qlnn_/test_reuploading_circuit.py` (8) |
| hardware_efficient | ✅ pre-existing | forecaster (registry) | `src/qlnn_/circuits/hardware_efficient.py` | existing |
| strongly_entangling| ✅ pre-existing | forecaster (registry) | existing | existing |
| brickwall          | ✅ pre-existing | forecaster (registry) | existing | existing |
| **chebyshev_dqc**  | ✅ P3-1 (`77009ce`) | solver builder | `src/qlnn_/training/physics_residual_loss.py` | `tests/qlnn_/test_physics_residual_solver.py` (3) |
| **rf_qrc**         | ✅ P3-2b (`5014eac`) | forecaster (own train path — fixed reservoir + closed-form ridge readout) | `src/qlnn_/circuits/rf_qrc.py` | `tests/qlnn_/test_rf_qrc.py` (7) |
| **te_qpinn_fnn**   | ✅ P3-2c (`28ff61a`) | solver builder | `src/qlnn_/circuits/te_qpinn.py` | `tests/qlnn_/test_te_qpinn.py` (7) |
| **te_qpinn_qnn**   | ✅ P3-2e (`0bc44f7`) | solver builder | `src/qlnn_/circuits/te_qpinn.py` | `tests/qlnn_/test_te_qpinn.py` (+8 = 15) |
| **qcpinn**         | ✅ P3-2d (`a7db628`) | solver builder | `src/qlnn_/circuits/qcpinn.py` | `tests/qlnn_/test_qcpinn.py` (15) |
| lubasch_multicopy  | ⏸ **DEFERRED (P3-2f)** — see rationale below | n/a | — | — |

### Lubasch (`lubasch_multicopy`) — explicit deferral rationale

Source (Lubasch et al., PRA 101 010301(R), 2020 / arXiv:1907.09032) is
*explicitly* schematic at the gate level: §6 of this manifest records
"the two-qubit gate decomposition is Fig. 1b schematic only" — i.e.
the construction names but does not gate-specify the "generic two-qubit
gate `Û(λ)`" that prepares the amplitude encoding, and the multi-copy
QNPU layout (Fig. 1a/2a) is also figure-only at fine grain. Per the
P3a discipline ("no family enters P6 until its spec is PDF-grounded
and dual-cross-verified"), implementing lubasch now would necessarily
exceed what the source specifies — manufacturing exactly the kind of
infidelity P3a exists to prevent.

This manifest's §6 already classifies lubasch as **"Context/baseline
circuit"** (NOT a primary first-class family). The P6 regime map's
scientific value sits on the SOTA-solver lineage
(`chebyshev_dqc` → `te_qpinn_fnn` → `te_qpinn_qnn` → `qcpinn`) and the
SOTA-forecaster path (`rf_qrc`); lubasch is bibliographic anchor.

**Deferral condition:** revisit only if a P6 result-driven need
emerges (e.g. an ablation explicitly requires lubasch's multi-copy
nonlinearity for a comparison). If revisited, the implementation must
either (a) be sourced from a less-schematic later paper, or (b)
include the chosen `Û(λ)` decomposition + QNPU layout as explicit
**[DECLARED DESIGN CHOICEs]** cited to Fig. 1b/2a, with a faithfulness
unit test exercising the `r=1` plain-overlap reduction hook.

### Strand-2 totals

5 of 6 literature families implemented faithfully, ALL P3a-grounded
+ each shipped with a unit-test-hook faithfulness assertion + every
`[DECLARED DESIGN CHOICE]` documented + cited inline. 1 deferred with
rationale. The two `reuploading.py` caveats were cleaned in P3-2a
(commit `4d28914`).

Cumulative P3 strand-2 test coverage (faithfulness-only, not counting
prior infra): 8 reuploading + 7 rf_qrc + 15 te_qpinn (FNN+QNN) + 15
qcpinn = **45 P3-strand-2 tests, all green**, on top of the 3
P3-strand-1 solver-gate tests.
</content>
