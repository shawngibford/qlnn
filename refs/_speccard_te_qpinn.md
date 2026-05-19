# TE-QPINN Circuit Spec Card — PDF-Grounded Faithfulness Extract

**Source:** Berger, Hosters, Möller, "Trainable embedding quantum physics
informed neural networks for solving nonlinear PDEs", *Scientific Reports*
15:18823 (2025). DOI 10.1038/s41598-025-02959-z.
**Pages read:** 1-6, 7-12, 13-16 (PDF is 14 pages total; all pages read in full).

---

## ⚠️ CRITICAL SOURCE FINDING — re: the two requested families

The task brief asked for two families: `te_qpinn_fnn` (classical NN →
trainable feature map → PQC) and `te_qpinn_qnn` (fully-quantum trainable
embedding).

**The PDF specifies ONLY ONE trainable-embedding construction: an
FNN-based one.** The paper's entire method is "we propose a method where
the quantum feature map is a trained FNN" (p.4, paragraph after Fig. 4
list; "We propose a method where the quantum feature map is a trained
FNN."). The *Embedding* subsection (p.5) defines the embedding solely via
an FNN Φ. There is **no fully-quantum / QNN trainable-embedding variant
defined anywhere in this source.**

The only non-FNN embeddings discussed are the **Chebyshev** and
**Tower-Chebyshev** feature maps — but those are cited *baselines from
prior work* (Kyrienko et al., ref. 11; p.3 "TE-QPINN ... is an evolution
of the differentiable circuits introduced by Kyrienko et al."; p.8 "The
first test case originates from the differential circuit paper11. Here,
the Chebyshev embedding and the Tower-Chebyshev embedding are now compared
to the TE-QPINN."). They are **fixed (non-trainable) angle maps**, not a
"te_qpinn_qnn" trainable-embedding family.

Therefore:
- `te_qpinn_fnn` → **fully specified below** (this is *the* TE-QPINN).
- `te_qpinn_qnn` → **UNSPECIFIED IN SOURCE.** The PDF does not define a
  fully-quantum trainable embedding. Documented below as a non-existent
  family with the closest in-paper baseline (fixed Tower-Chebyshev) noted
  only as context, NOT as a faithful reconstruction of a "qnn" family.

---

## FAMILY 1 — `te_qpinn_fnn` (= the paper's TE-QPINN)

### Feature map / data embedding
- Trainable embedding = a classical **feedforward neural network (FNN)**
  Φ that maps the rescaled input to a per-qubit scaling factor (p.4–5,
  *Embedding* subsection; Eq. 10). "The FNN is used as the quantum feature
  map" (p.5).
- Step 1: rescale every input feature to `[x_min, x_max]` via Eq. (9):
  `x̃ = (x − x_min)/(x_max − x_min)`, s.t. `x_min ≤ x̃ ≤ x_max` (Eq. 9,
  p.5). In experiments the rescale range is `[-0.95, 0.95]` (p.8 ODE
  example; p.9 Poisson; p.10 Burgers — "rescaled to the range [-0.95,
  0.95]"). A scaling factor (e.g. 10) is applied to the IC/boundaries
  (p.8 "initial condition ... given a scaling factor of 10"; p.9, p.11).
- Step 2: evaluate the FNN at `x̃`, producing
  `Φ(x̃) = (φ₁(x̃), φ₂(x̃), …, φₙ(x̃))ᵀ`, n = number of qubits (Eq. 10,
  p.5). FNN output layer width = number of qubits (p.8 Results: "an output
  layer generates a vector with dimensions equal to the number of
  qubits").
- Step 3: **angle embedding, rotation about the y-axis only** (p.5: "for
  this work we choose the y-axis"). Final per-qubit angle is the
  **product** `φᵢ(x̃) · x̃` (p.5: "The final rotation angle is given by
  φᵢ(x) · x. During testing, we found that φᵢ(x) · x performed better than
  just φᵢ(x).").
- Embedding unitary (Eq. 11, p.5):
  `U_emb(x) = ⊗_{i=0}^{n} R_y( φᵢ(x̃) · x̃ )`.
  (Tensor product over qubits; one R_y per qubit.)
- Multi-dimensional inputs: components of x are cycled through qubits via
  modulo `i % n` (p.5: "an iterative cycle through the components of x
  using the modulus operation i%n is applied. The number of qubits should
  be divisible by the number of dimensions. For example, with 4 qubits and
  x ∈ ℝ², the transformations ... (φ₁x₁, φ₂x₂, φ₃x₁, φ₄x₂) can be
  applied.").
- Trainable parameters of the embedding = the FNN weights and biases,
  denoted **Ξ** (Eq. 8, p.5: "U_emb(x,Ξ) ... Ξ are the weights and biases
  of the underlying FNN model").
- FNN architecture: input layer width = problem dimension; one or more
  hidden layers; TanH activation after input and each hidden layer; output
  width = #qubits (p.8 Results paragraph). Hidden-layer sizes are
  per-experiment (e.g. ODE: 1 hidden layer, 5 neurons — p.8; Poisson: 2
  hidden layers, 10 neurons each — p.9; Burgers: 2 hidden layers, 10
  neurons each — p.10; lid-driven cavity: 3 hidden layers, 10 neurons each
  — p.11; PINN-comparison: 2 hidden layers, 10 neurons — p.12).

### Gate-by-gate variational ansatz
- Ansatz = **Hardware-Efficient Ansatz (HEA)**, the one shown in Fig. 3
  (p.5 *Variational ansatz*: "Here, the HEA was already shown in Fig. 3
  will be utilized.").
- Per-layer structure (Fig. 3, p.4 caption + Eq. 12, p.5): each layer =
  single-qubit rotation gates on every qubit, **followed by a chain of
  CNOT operations** (p.5: "A layer consists of rotation gates followed by
  a chain of CNOT operations.").
- Rotation gate set used in this work: **R_x, R_y, R_z** (p.5, after
  Eq. 12: "The rotation gates used in this work are Rx, Ry, and Rz.").
- Per-qubit per-layer order from Fig. 3: `R_x(θ·1) → R_y(θ·2) →
  R_z(θ·3)` on each qubit, then CNOT entangling chain (Fig. 3, p.4:
  rotations indexed θ_{l q g}, e.g. layer 1 qubit 1 = R_x(θ₁₁₁),
  R_y(θ₁₁₂), R_z(θ₁₁₃)).
- Entanglement topology: CNOTs entangle "each qubit with its lower and
  upper neighbor" (Fig. 3 caption, p.4). Fig. 3 explicitly shows a
  nearest-neighbor CNOT chain on 4 qubits. Paper notes "The arrangement
  of CNOT gates and number of rotational gates varies in literature."
  (Fig. 3 caption, p.4) — i.e. the exact CNOT pattern beyond
  "neighbor chain" is **partially UNSPECIFIED IN SOURCE** (only the
  4-qubit Fig. 3 instance is concretely drawn).
- Formal ansatz (Eq. 12, p.5):
  `U_var(θ) = U_L(θ_L) ⋯ U₂(θ₂) U₁(θ₁)`, with
  `U_l(θ_l) = ∏_m e^{−i θ_{l,m} H_m} W_m`,
  where `W_m` is an unparameterized unitary and `H_m` a Hermitian
  generator; each `U_l` is one layer (the product over m = all gates in
  the layer) (Eq. 12 + surrounding text, p.5).
- Variational parameters denoted **θ**; optimized with **L-BFGS**
  ("LBFGS optimizer15", p.5; reaffirmed p.8 "L-BFGS optimizer as
  implemented in PyTorch using the default parameter with the strong Wolfe
  conditions").

### Depth / width scaling rule
- Width: #qubits n is a free hyperparameter; FNN output width must equal n
  (p.8). For x ∈ ℝ^d, "The number of qubits should be divisible by the
  number of dimensions" (p.5).
- Depth: number of HEA layers L is a free hyperparameter; "one can easily
  add more layers to make the circuit more expressive and keep the count
  of needed gates to a minimum" (p.5). No closed-form scaling law given;
  scaling is **empirical** via the (Layers × Qubits) sweeps in Table 1
  (Tower-Chebyshev) and Table 2 (TE-QPINN) (p.8). Example operating
  points: ODE 4 qubits / 3 layers (p.8), Fig. 6 uses 4 qubits / 5
  variational layers, PINN-comparison 4 qubits / 5 layers giving "60
  variational parameters in the PQC" (p.12).
- No explicit formula relating L, n to parameter count is stated →
  **scaling rule is empirical, closed form UNSPECIFIED IN SOURCE.**

### Measurement / readout
- Observable (Eq. 13, p.5): `O = ⊗_{i=1}^{n} Z_i` (tensor product of
  Pauli-Z on every qubit). "It measures the state of every qubit along
  the computational basis and sums the resulting measurements" (p.5).
- Circuit output / field approximation (Eq. 8, p.5):
  `ū(x; θ, Ξ) = C(x,θ,Ξ) = ⟨0| U_emb† U_var† O U_var U_emb |0⟩` — the
  expectation value of O is the scalar approximation of the unknown field
  u.
- On a state simulator the expectation value is computed exactly; on
  hardware the circuit is sampled multiple times to estimate it (p.6:
  "When working with a quantum state simulator, the expectation value of
  the observable can directly be computed. This is not possible on quantum
  hardware, there we need to evaluate the quantum circuit multiple times
  to get an estimate of the expectation value."). Implemented with
  PennyLane exact-expectation simulator (p.8: "implemented using the
  PennyLane framework17 ... computes the exact expectation value of the
  PCQ").
- Multi-output problems (e.g. ψ and p in lid-driven cavity) use **two
  separate TE-QPINN circuits**, one per field (p.11: "the two fields ψ
  and p are unknown, therefore two TE-QPINNs are utilized").

### Loss & derivative method
- Loss is identical to classical PINN loss (p.6 *Loss function*).
  Total (Eq. 14): `ℒ(θ,Ξ) = ℒ_PDE(θ,Ξ) + Σ_k λ_k ℒ_{BC,k}(θ,Ξ)`.
- PDE residual (Eq. 15): `ℒ_PDE = Σ_{xʲ∈𝒮_PDE} ( 𝓕(ū(xʲ);γ) − f(xʲ) )²`.
- Boundary residual (Eq. 16): `ℒ_{BC,k} = Σ_{xʲ∈𝒮_{BC,k}}
  ( 𝓑_k(ū(xʲ)) − g(xʲ) )²`. Each boundary individually weighted by scalar
  λ_j; initial conditions treated as temporal boundary values (p.6).
- **Two kinds of derivatives** (p.6, paragraph "For the TE-QPINN, two
  types of derivatives are required"):
  1. Derivative of the cost C w.r.t. **input x** (needed for the PDE
     residual at collocation points).
  2. Derivative of loss w.r.t. **parameters θ and Ξ** (needed for
     optimization).
- Derivative method = **hybrid: parameter-shift rule combined with
  classical backpropagation, glued by the chain/product rule** (p.6).
  - Quantum-circuit derivative (w.r.t. its rotation angles) via the
    **parameter-shift rule** (refs 12,14; p.6, and labeled underbraces in
    Eq. 17, 23, 28).
  - FNN (Ξ) derivative via **backpropagation** on classical hardware
    (p.4 point 3; p.6).
  - Input-coordinate derivative (Eq. 17, p.6):
    `∂ū/∂x = ∂C/∂x = Σ_{i=1}^{n} [ (∂C/∂R_yⁱ · ∂R_yⁱ/∂(φᵢ(x)x))
    · (1 + (∂φᵢ(x)/∂x)·x) ]`,
    with the first factor by parameter-shift and the
    `∂φᵢ(x)/∂x` term by backpropagation. The `(1 + (∂φᵢ/∂x)·x)` factor
    arises because the embedded angle is the *product* φᵢ(x)·x, requiring
    the product rule (p.6: "Since the qubits are rotated by φᵢ(x)·xᵢ, the
    product rule to get the accurate derivatives has to be utilized.").
  - Loss-vs-Ξ gradient (Eqs. 18–28, p.6–7) decomposes via chain rule into
    `∂𝓕/∂ū · ∂ū/∂Ξ`, with `∂ū/∂Ξ` again = parameter-shift × backprop
    (Eq. 23, Eq. 28).
- Efficiency claim: the hybrid scheme reduces required circuit
  evaluations to **twice the number of qubits, independent of FNN size**,
  because the quantum partial derivative is constant and reusable while
  the FNN partials are computed classically (p.6: "This hybrid approach
  reduces the required circuit evaluations to twice the number of qubits
  in the quantum circuit, independent of the FNN size.").
- Training loop = hybrid classical/quantum loop (Fig. 5, p.7): per
  collocation point evaluate QFM (FNN) → quantum circuit (regular for
  value, parameter-shifted for derivatives) → compute loss → L-BFGS
  update of both θ and Ξ (p.7 *Hybrid training loop*; Fig. 5).

### Difference between te_qpinn_fnn and te_qpinn_qnn
- **`te_qpinn_fnn` is the only trainable-embedding family the PDF
  defines** (see CRITICAL SOURCE FINDING). The embedding is a classical
  FNN Φ producing per-qubit R_y angles φᵢ(x̃)·x̃ (Eq. 10–11, p.5).
- **`te_qpinn_qnn` (fully-quantum trainable embedding): UNSPECIFIED IN
  SOURCE.** The paper contains no fully-quantum trainable embedding. The
  non-FNN embeddings it mentions (Chebyshev, Tower-Chebyshev; p.3, p.8,
  Fig. 8) are **fixed, non-trainable** prior-work baselines (Kyrienko et
  al., ref. 11) used purely for comparison — Fig. 8 caption (p.10): the
  Tower-Chebyshev rotation angles "stay the same, independent of the
  problem", confirming they are not trainable. They are NOT a
  TE-QPINN trainable-embedding variant. Any "te_qpinn_qnn" implementation
  cannot be faithfully derived from this PDF.

### Known-limit / unit-test hook
- **Embedding gate-count formula (PDF-grounded, Eq. 11, p.5):** the
  embedding `U_emb` contains **exactly n single-qubit R_y gates** (one per
  qubit), zero entangling gates. An implementation test can assert:
  `count(R_y in U_emb) == n_qubits` and `count(any 2-qubit gate in
  U_emb) == 0`.
- Complementary HEA per-layer hook (Fig. 3, p.4 + p.5): each variational
  layer contains exactly **3 single-qubit rotations per qubit**
  (`R_x, R_y, R_z`) followed by a neighbor CNOT chain ⇒ per-layer
  parameterized-rotation count `= 3·n_qubits`; total variational params
  `= 3·n_qubits·L` for the Fig. 3 scheme. Sanity check against the
  paper's own stated operating point: 4 qubits, 5 layers → 3·4·5 = 60,
  which **matches the paper's "60 variational parameters in the PQC"
  (p.12)** — a directly checkable consistency assertion.
  (Note: the paper says the exact rotation/CNOT arrangement "varies in
  literature" (Fig. 3 caption), so a strict test should pin the
  Fig.3/p.12 convention: `n_var_params == 3 * n_qubits * n_layers`.)

---

## FAMILY 2 — `te_qpinn_qnn` (fully-quantum trainable embedding)

**STATUS: UNSPECIFIED IN SOURCE — this family does not exist in the PDF.**

- Feature map / data embedding: **UNSPECIFIED IN SOURCE.** No
  fully-quantum trainable embedding is defined. The trainable embedding in
  this paper is *exclusively* the classical FNN of Eq. 10–11 (p.5).
- Gate-by-gate variational ansatz: would presumably reuse the same HEA
  (Fig. 3) — but the existence of a separate "qnn" family is **UNSPECIFIED
  IN SOURCE**; do not implement on assumption.
- Depth/width scaling: **UNSPECIFIED IN SOURCE.**
- Measurement/readout: **UNSPECIFIED IN SOURCE** for any "qnn" family
  (the paper's single readout `O = ⊗ Zᵢ`, Eq. 13, applies only to the
  FNN-embedding TE-QPINN).
- Loss & derivative method: **UNSPECIFIED IN SOURCE** for a "qnn" family.
- Closest in-paper *baseline* (NOT a faithful "qnn" reconstruction):
  fixed **Tower-Chebyshev** angle embedding from Kyrienko et al. ref. 11
  — fixed angles `2·arccos(x̃)` / `4·arccos(x̃)` shown in Fig. 8 (p.10),
  problem-independent and non-trainable (Fig. 8 caption). This is a
  comparison baseline, explicitly *not* a trainable embedding, and must
  not be relabeled as `te_qpinn_qnn`.
- Known-limit / unit-test hook: **UNSPECIFIED IN SOURCE** — none can be
  derived without fabricating content the PDF does not contain.

---

## Items explicitly UNSPECIFIED IN SOURCE (summary)
- The `te_qpinn_qnn` (fully-quantum trainable embedding) family entirely.
- Closed-form depth/width → parameter scaling law (only empirical sweeps,
  Tables 1–2).
- Exact CNOT entanglement pattern beyond "neighbor chain" / the 4-qubit
  Fig. 3 instance ("arrangement ... varies in literature", Fig. 3
  caption, p.4).
- FNN hidden-layer sizing rule (given only per-experiment, not as a
  general rule).
- Whether R_x/R_y/R_z all appear in *every* layer for *every* experiment
  vs. only the Fig. 3 illustration (Fig. 3 is captioned as one possible
  arrangement).
