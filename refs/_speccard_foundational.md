# Foundational Ansatz Spec Cards (P3a faithfulness extraction)

Every line cites a source location (section / equation / figure) from the
arXiv full-text HTML. Claims with no source location are marked
**UNSPECIFIED IN SOURCE**. No model priors / training knowledge used.

Sources read:
- https://ar5iv.labs.arxiv.org/html/2011.10395 (full text) + https://arxiv.org/abs/2011.10395 (title/authors/year confirm)
- https://ar5iv.labs.arxiv.org/html/1907.09032 (full text) + https://arxiv.org/abs/1907.09032 (title/authors/year confirm)
- https://ar5iv.labs.arxiv.org/html/2008.08605 (full text) + https://arxiv.org/abs/2008.08605 (title/authors/year confirm)
- Local impl read: src/qlnn_/circuits/reuploading.py

---

## CARD 1 — `chebyshev_dqc`

**Source:** Kyriienko, Paine, Elfving, "Solving nonlinear differential
equations with differentiable quantum circuits", Phys. Rev. A 103, 052416
(2021); arXiv:2011.10395 (submitted Nov 2020). [arXiv:2011.10395 abstract page]

### Data / feature map (Chebyshev tower specifically)
- General feature map encodes x via the cost-function relation
  f(x) = ⟨f_{φ,θ}(x)|Ĉ|f_{φ,θ}(x)⟩ — Eq. (1), Sec. II. [2011.10395 §II Eq. 1]
- Per-qubit rotation decomposes through Chebyshev polynomials:
  R̂_{y,j}(φ[x]) = cos(n·arccos x)·𝟙_j + √(1−x²)·U_{n−1}(x)·X̂_jẐ_j,
  with dT_n/dx = n·U_{n−1}(x). — Eq. (12), Sec. III.1. [2011.10395 §III.1 Eq. 12]
- **Sparse Chebyshev feature map:** 𝒰̂_φ(x) = ⊗_{j=1}^N R̂_{y,j}(2·arccos x)
  — Eq. (14), Sec. III.1. [2011.10395 §III.1 Eq. 14]
- **Chebyshev tower feature map:** 𝒰̂_φ(x) = ⊗_{j=1}^N R̂_{y,j}(2j·arccos x)
  (rotation angle scales linearly with qubit index j → multi-frequency tower)
  — Eq. (15), Sec. III.1. [2011.10395 §III.1 Eq. 15]
- Product feature map variant: 𝒰̂_φ(x) = ⊗_{j=1}^N exp(−i·(arcsin x)/2·Ŷ_j)
  — Eq. (6), Sec. III.1. [2011.10395 §III.1 Eq. 6]
- Evolution-enhanced feature map: product rotations + exp(−iĤτ)
  — Fig. 3c, Sec. III.1. [2011.10395 §III.1 Fig. 3c]

### Variational ansatz (gate-by-gate)
- Hardware-Efficient Ansatz (HEA): alternating parametrized single-qubit
  rotation layers (R̂_z–R̂_x–R̂_z with independent angles θ) and
  nearest-neighbor CNOT entanglers, linear connectivity, repeated d times
  — Fig. 5a, Sec. III.2. [2011.10395 §III.2 Fig. 5a]
- Alternating Blocks Ansatz (ABA): checkerboard hardware-efficient
  subblocks of width N_b, depth b, repeated d_layers times
  — Fig. 5b, Sec. III.2. [2011.10395 §III.2 Fig. 5b]
- Generator constraint: non-commuting generators Ĝ_j with [Ĉ,Ĝ_j] ≠ 0
  — Sec. III.2. [2011.10395 §III.2]

### Depth / width scaling
- Register N = 6 qubits; primary depth d = 5; depths tested
  d ∈ {3,6,12,24}; >400 variational params at d=24 without reported
  barren plateau — Sec. IV (Results). [2011.10395 §IV]
- Training grid: 20 equidistant points (Eq. 30 example); 100 points
  (Eq. 32) — Sec. IV. [2011.10395 §IV Eqs. 30, 32]

### Measurement / readout
- Cost: f(x) = ⟨f_{φ,θ}(x)|Ĉ|f_{φ,θ}(x)⟩ — Eq. (1), Sec. II. [2011.10395 §II Eq. 1]
- Default cost operator: total magnetization Ĉ = Σ_j Ẑ_j — Sec. III.3
  (used in all examples). [2011.10395 §III.3]
- General cost: Ising Hamiltonian (inhomogeneous J_{j,j+1}, fields h_j^{z,x})
  — Eq. (16), Sec. III.3; weighted sum Ĉ = Σ_ℓ α_ℓ Ĉ_ℓ — Eq. (18),
  Sec. III.3. [2011.10395 §III.3 Eqs. 16, 18]

### Derivative / loss (the DQC circuit-derivative residual)
- Feature-map derivative via product rule:
  d𝒰̂_φ/dx = (1/2)(dφ/dx) Σ_{j'} ⊗_j R̂_{y,j}(φ[x] + π·δ_{j,j'})
  — Eq. (8), Sec. II. [2011.10395 §II Eq. 8]
- Parameter-shift circuit derivative:
  d⟨Ĉ⟩/dx = (1/4)(dφ/dx)(⟨Ĉ⟩⁺ − ⟨Ĉ⟩⁻), shifted rotations
  R̂_{y,j}(φ[x] ± (π/2)δ_{j,j'}) — Eqs. (9)–(10), Sec. III.1.
  [2011.10395 §III.1 Eqs. 9, 10]
- 2nd-order derivative: parameter shift applied twice → four shifted
  terms per generator — Sec. II. [2011.10395 §II]
- Differential equation form F[{d^m f_n/dx^m},{f_n(x)},x] = 0 — Eq. (3),
  Sec. II. [2011.10395 §II Eq. 3]
- Differential loss:
  ℒ_θ^(diff) = (1/M) Σ_{i=1}^M L(F[d_x f(x_i), f(x_i), x_i], 0)
  — Eq. (20), Sec. III.4. [2011.10395 §III.4 Eq. 20]
- Boundary loss: ℒ_θ^(boundary) = η·L(f(x_0), u_0), η > 1 — Eq. (21),
  Sec. III.4. [2011.10395 §III.4 Eq. 21]
- Total loss: ℒ_θ = ℒ_θ^(diff) + ℒ_θ^(boundary) — Eq. (19), Sec. III.4;
  optimization θ_opt = argmin_θ ℒ_θ — Eq. (4), Sec. II.
  [2011.10395 §III.4 Eq. 19; §II Eq. 4]
- Distance L: MSE L(a,b)=(a−b)² — Eq. (22), Sec. III.4; also MAE / KL
  variants — Sec. III.4. [2011.10395 §III.4 Eq. 22]

### Known-limit / unit-test hook
- Chebyshev decomposition T_n, U_n with dT_n/dx = n·U_{n−1} — Eq. (12),
  Sec. III.1: a unit test can verify the sparse feature-map state on N=1
  qubit reproduces ⟨Ẑ⟩ = T_1(x) = x for R̂_y(2·arccos x), and the
  parameter-shift derivative equals the analytic dT_n/dx.
  [2011.10395 §III.1 Eq. 12]

### Gaps
- Exact native gate set / gate times: **UNSPECIFIED IN SOURCE** (silicon-
  agnostic; Sec. IV simulations noiseless).
- Classical post-processing algorithm details: **UNSPECIFIED IN SOURCE**
  (Fig. 2 labels "classically post-processed" only).
- Regularization weight ζ(n_j) schedule hyperparameter selection: partly
  given (Eq. 29) but selection criteria **UNSPECIFIED IN SOURCE**.
- Multi-variable explicit gate construction: **UNSPECIFIED IN SOURCE**
  (stated generalizable, Sec. II, not detailed).

---

## CARD 2 — `lubasch_multicopy`

**Source:** Lubasch, Joo, Moinier, Kiffner, Jaksch, "Variational quantum
algorithms for nonlinear problems", Phys. Rev. A 101, 010301(R) (2020);
arXiv:1907.09032 (submitted 2019). [arXiv:1907.09032 abstract page]

### Data / state encoding
- Solution encoded directly as amplitudes:
  |ψ⟩ = Σ_{k=0}^{N−1} ψ_k |binary(k)⟩, N = 2ⁿ grid points on n qubits
  — Eq. (5) and surrounding text. [1907.09032 Eq. 5]
- Discrete PDE solution values {ψ_k} are the quantum amplitudes — Sec.
  text around Eq. (4)/(5). [1907.09032 §near Eq. 4]

### Variational ansatz (gate-by-gate)
- Parametrized unitary: |ψ(λ)⟩ = Û(λ)|0⟩ — Fig. 1(b) caption + text
  after Eq. (5). [1907.09032 Fig. 1b]
- Structure: network Û(λ) of depth d=5, n=6; λ={λ₁,λ₂,…} parametrize
  generic two-qubit gates — Fig. 1(b). [1907.09032 Fig. 1b]
- Explicit gate-by-gate decomposition beyond "generic two-qubit gates":
  **UNSPECIFIED IN SOURCE**.

### Multi-copy nonlinear construction
- Multiple identical copies created by choosing Û_i(λ)=Û_j(λ) so the
  same f^(i)=f^(j) appears multiple times, fed to a Quantum Nonlinear
  Processing Unit (QNPU) — Fig. 1(a), Fig. 2, QNPU text. [1907.09032
  Fig. 1a, Fig. 2]
- Ancilla / Hadamard-test scheme: top-line ancilla undergoes Hadamard
  gates Ĥ, controls operations, then is measured — Fig. 1(a).
  [1907.09032 Fig. 1a]
- |ψ|⁴ interaction term: controlled-NOT operations between copy pairs;
  ancilla measurement yields Σ_k |ψ_k|⁴ — Fig. 2(a), Eq. (4c).
  [1907.09032 Fig. 2a, Eq. 4c]
- Potential term V|ψ|²: point-wise multiplication via controlled-NOT
  gates → Σ_k Ṽ_k|ψ_k|² — Fig. 2(b), Eq. (4b). [1907.09032 Fig. 2b,
  Eq. 4b]
- Number of copies: |ψ|⁴ term requires four copies of |ψ(λ)⟩ — Fig.
  2(a). [1907.09032 Fig. 2a]

### Depth / width scaling
- Potential V̂ Fourier-expanded to order J → MPS bond dimension χ=J;
  circuit depth bound d ≤ 9n[(23/48)(2χ)² + 4/3], polynomial in n, χ
  — tensor-network expansion text. [1907.09032 §V̂-unitary text]
- Example (Eq. 6, quasi-periodic potential): d = 5(n−2)+1, far below the
  upper bound. [1907.09032 Eq. 6]
- QNPU's own depth formula: **UNSPECIFIED IN SOURCE**.

### Measurement / readout
- Cost: C = ⟨K⟩_c + ⟨P⟩_c + ⟨I⟩_c (kinetic + potential + interaction)
  — Eqs. (2),(4a–c). [1907.09032 Eqs. 2, 4a–c]
- Kinetic: ⟨K⟩ = (1 − ⟨σ̂_u⟩^K_anc)/h_n²; Potential: ⟨P⟩ =
  α⟨σ̂_u⟩^P_anc; Interaction: ⟨I⟩ = g⟨σ̂_u⟩^I_anc/(2h_n) — all via
  ancilla σ̂_u measurement — ancilla-measurement text. [1907.09032
  §ancilla measurement]

### Known-limit / unit-test hook
- Nonlinear Schrödinger equation (Eq. 1) with quasi-periodic potential
  (Eq. 6); IBM Q experimental data for harmonic potential with single
  parameter λ agrees with exact solution within ~10% — Fig. 1(c–d).
  [1907.09032 Eq. 1, Eq. 6, Fig. 1c–d]. Unit-test hook: reproduce the
  single-λ harmonic-potential cost-function curve vs. exact solution.

### Gaps
- Exact two-qubit gate sequence: **UNSPECIFIED IN SOURCE** ("generic").
- Full SWAP/swap-test construction detail: **UNSPECIFIED IN SOURCE**.
- Explicit QNPU depth for nonlinear terms: **UNSPECIFIED IN SOURCE**.
- Sample count M vs. problem size: **UNSPECIFIED IN SOURCE**.
- Burgers-equation construction: in supplement, not in fetched main
  full text → **UNSPECIFIED IN SOURCE (main text)**.

---

## CARD 3 — `data_reuploading` (Fourier-form verification)

**Source:** Schuld, Sweke, Meyer, "The effect of data encoding on the
expressive power of variational quantum machine learning models", Phys.
Rev. A 103, 032430 (2021); arXiv:2008.08605 (submitted Aug 2020).
[arXiv:2008.08605 abstract page]

### Quantum model → truncated Fourier series
- Model form: f_θ(x) = ⟨0|U†(x,θ) M U(x,θ)|0⟩ — Eq. (3), Sec. I.
  [2008.08605 §I Eq. 3]
- Decomposition: f(x) = Σ_{ω∈Ω} c_ω e^{iωx} — Eqs. (11)–(12), Sec. I.
  [2008.08605 §I Eqs. 11–12]
- Two properties: frequency spectrum Ω (from encoding) and coefficient
  expressivity {c_ω} (from full circuit) — Sec. I. [2008.08605 §I]

### Data-encoding gate structure → accessible frequency spectrum
- Encoding gate: S(x) = e^{−ixH}, H arbitrary Hamiltonian — Sec. I,
  after Eq. (4). [2008.08605 §I post-Eq. 4]
- Diagonalize H = V†ΣV (eigenvalues λ₁..λ_d); state amplitude carries
  e^{−i(λ_{j₁}+…+λ_{j_L})x} — Eq. (6), Sec. I. [2008.08605 §I Eq. 6]
- Frequency spectrum Ω = {Λ_k − Λ_j : k,j ∈ [d]^L}, Λ_𝐣 = λ_{j₁}+…+λ_{j_L}
  — Eq. (10), Sec. I. Ω solely determined by encoding-Hamiltonian
  eigenvalues; circuit controls c_ω. [2008.08605 §I Eq. 10]
- Single Pauli example: eigenvalues {−1,+1} → Ω = {−2,0,2} → A·sin(2γx+B)+C
  — Sec. II.1. [2008.08605 §II.1]

### Data-reuploading layout
- U(x) = W^(L+1) S(x) W^(L) … W^(2) S(x) W^(1): L layers, each a data-
  encoding block S(x) and trainable block W(θ); S(x) identical every
  layer — Eq. (4), Sec. I; Fig. 1. [2008.08605 §I Eq. 4, Fig. 1]
- Parallel vs. reuploading layouts — Fig. 2. [2008.08605 Fig. 2]

### Effect of repeating encoding
- r parallel single-qubit Pauli rotations → Ω_par = {−r,…,0,…,r}
  — Eqs. (22)–(25), Sec. II.2. [2008.08605 §II.2 Eqs. 22–25]
- Sequential (layer-wise) repetition gives Ω_seq = Ω_par; repeating r
  times raises truncated-Fourier degree to r — Sec. II.2; Fig. 4.
  [2008.08605 §II.2, Fig. 4]
- Universal Hamiltonian families: ℤ_K ⊆ Ω asymptotically → universal
  Fourier approximator — Sec. III (theorem). [2008.08605 §III]

### Measurement / readout
- Observable M in f_θ(x)=⟨0|U†MU|0⟩ — Eq. (3), Sec. I; specific M
  beyond "some observable": **UNSPECIFIED IN SOURCE** (general). [2008.08605 §I Eq. 3]

### Known-limit / unit-test hook
- Single Pauli-rotation encoding (eigenvalues ±1) must yield exactly
  Ω={−2,0,2}, model = A·sin(2γx+B)+C — Sec. II.1: unit test asserts the
  fitted model has only frequencies {0,2} (degree-1 truncated Fourier).
  [2008.08605 §II.1]

### Verification verdict: does `reuploading.py` match Schuld's construction?

`src/qlnn_/circuits/reuploading.py`:
- Layout (lines 120–132): for each of `num_layers`, applies a data block
  (per-qubit `qml.RX(inputs[i])`), then a trainable block (per-qubit
  `qml.Rot` = RZ∘RY∘RZ), then a CNOT entangler. This is exactly the
  alternating S(x)·W(θ) layered, layer-wise-repeated reuploading form of
  Eq. (4) / Fig. 1, with S(x) identical every layer. **MATCHES** the
  Schuld layout. [2008.08605 §I Eq. 4, Fig. 1; reuploading.py L120–132]
- Encoding generator: `RX(x)` = e^{−i(x/2)X}, generator X/2, eigenvalues
  {−1/2,+1/2}. By Eq. (10) a single such gate gives spectrum
  Ω = {−1,0,+1}; repeating layer-wise r=`num_layers` times gives
  Ω = {−r,…,r} (integer, evenly spaced). This is precisely the
  sequential-reuploading frequency-enlargement mechanism of Sec. II.2
  (Eqs. 22–25). **MATCHES** the truncated-Fourier construction:
  encoding generator → evenly-spaced integer frequency spectrum whose
  max degree = number of reuploading layers. [2008.08605 §II.2
  Eqs. 22–25; reuploading.py L125]
- NOTE (factual, not a mismatch): the paper's worked single-Pauli
  example (Sec. II.1) uses generator eigenvalues ±1 → Ω={−2,0,2}; `RX`'s
  ±1/2 eigenvalues give Ω={−1,0,1} per layer (frequencies scaled by the
  1/2 in the Pauli-rotation convention). Spectrum *structure* (integer,
  symmetric, evenly spaced, degree = #layers) is identical; only the
  frequency unit differs by the rotation-angle convention. This is
  consistent with Sec. II.2's generator-eigenvalue → spectrum rule, not
  a deviation from the construction. [2008.08605 §II.1 vs §II.2;
  reuploading.py L125]

**Overall: reuploading.py's gate structure MATCHES Schuld et al.'s
truncated-Fourier data-reuploading construction** (alternating
encoding/trainable blocks, identical encoding per layer, single-Pauli
encoding generator producing an evenly-spaced integer frequency spectrum
whose degree grows with the number of reuploading layers).

### Gaps
- Exact trainable-block depth requirement for full coefficient
  expressivity: **UNSPECIFIED IN SOURCE** (open, Sec. III).
- Generalization / VC-dimension bounds: **UNSPECIFIED IN SOURCE**.
