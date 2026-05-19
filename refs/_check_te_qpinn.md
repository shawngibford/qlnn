# Independent P3a Dual-Check — TE-QPINN circuit spec re-derivation

**Pages read:**
- FAMILY 1 (te_qpinn_fnn): `refs/s41598-025-02959-z.pdf` — pp. 1–14 (full PDF; Berger, Hosters & Möller, *Sci Rep* 15:18823, 2025).
- FAMILY 2 (te_qpinn_qnn): `refs/2605.13892v1.pdf` — pp. 1–11 (full PDF; Dehaghani, Tran, Mengel, Wisniewski, Aguiar, "A QPINN Framework with Quantum Trainable Embeddings for the Lid-Driven Cavity Problem", arXiv:2605.13892v1).

Derived independently from source PDFs only. No spec-card files read. Every line cites a section/equation/figure.

---

## FAMILY 1 — te_qpinn_fnn (Berger et al., classical-FNN trainable embedding)

### Feature map / data embedding (exact gates + trainable params)
- The embedding is **angle embedding via single-qubit y-rotations**: `U_emb(x) = ⊗_{i=0}^{n} R_y^i( φ_i(x̄) · x̄ )` — p.5, Eq. (11). Tensor product over all `n` qubits; only `R_y` used ("for this work we choose the y-axis", p.5, *Embedding* paragraph).
- Per-qubit rotation angle = `φ_i(x̄) · x̄`, i.e. a scaling factor `φ_i` (output of the FNN) **multiplied by** the rescaled coordinate `x̄`. Authors note `φ_i(x)·x` performed better than `φ_i(x)` alone (p.5).
- Trainable parameters of the embedding are the **FNN weights and biases Ξ** that produce `Φ(x̄) = (φ₁(x̄), …, φ_n(x̄))ᵀ` — p.5, Eq. (10); Ξ defined under Eq. (8). The FNN is the *only* trainable part of the embedding (the embedding gates themselves carry no independent free parameters beyond Ξ via φ).
- Data rescaled first: `x̄ = (x − x_min)/(x_max − x_min)` to `[x_min, x_max]` — p.5, Eq. (9). Examples rescale input to `[−0.95, 0.95]` (ODE, p.8 Eq. (29); Poisson/Burgers p.9–10).
- The FNN ("quantum feature map"): input layer = dimension of computational domain, several hidden layers, **TanH activation after input and each hidden layer**, output layer dimension = number of qubits — p.8, *Results* paragraph. Default examples: 1 hidden layer × 5 neurons (ODE, p.8); 2 hidden layers × 10 neurons (Poisson, Burgers, p.9–10); 3 hidden layers × 10 neurons (Navier–Stokes lid cavity, p.11).
- Multi-dimensional inputs: cycle through components of `x` using `i % n` (modulus); qubit count should be divisible by number of dimensions; e.g. 4 qubits & x∈ℝ² → angles (φ₁x₁, φ₂x₂, φ₃x₁, φ₄x₂) — p.5, *Variational ansatz* preceding paragraph.

### Gate-by-gate variational ansatz
- Hardware-Efficient Ansatz (HEA), shown for 4 qubits / 2 layers in **Fig. 3** (p.4). Each HEA layer per qubit: `R_x(θ_{·1}) → R_y(θ_{·2}) → R_z(θ_{·3})`, then a **chain of CNOTs entangling each qubit with its lower and upper neighbor** (Fig. 3 caption, p.4).
- Formal form: `U_var(θ) = U_L(θ_L) ⋯ U_2(θ_2) U_1(θ_1)`, with `U_l(θ_l) = ∏_m e^{−i θ_{l,m} H_m} W_m` — p.5, Eq. (12). `W_m` = unparameterized unitary; `H_m` = Hermitian generator. A layer = rotation gates followed by a chain of CNOTs; **rotation gates used = R_x, R_y, R_z** (p.5, *Variational ansatz*).
- Fig. 4 (p.4) shows the generic structure: Embedding block then 3 "Variational Layer" blocks then measurement.

### Depth / width scaling rule
- **Width** = number of qubits `n` = FNN output dimension (p.8). **Depth** = number of HEA layers `L` (Eq. (12)), freely chosen; "one can easily add more layers to make the circuit more expressive and keep the count of needed gates to a minimum" (p.5).
- Rotations per HEA layer per qubit = 3 (R_x, R_y, R_z) → **3·n·L parameterized rotation gates** for an n-qubit, L-layer HEA. Tables 1–2 (p.8) sweep Layers ∈ {1,3,5,7,10} × Qubits ∈ {2,4,6,8}. Authors note number/arrangement of CNOTs "varies in literature" (Fig. 3 caption) — exact CNOT count rule **UNSPECIFIED IN SOURCE** beyond "chain to lower & upper neighbor".

### Measurement / readout
- Observable `O = ⊗_{i=1}^{n} Z_i` (tensor product of Pauli-Z on every qubit) — p.5, Eq. (13). Cost/circuit output `ū(x;θ,Ξ) = C(x,θ,Ξ) = ⟨0| U_emb† U_var† O U_var U_emb |0⟩` — p.5, Eq. (8). Expectation value computed exactly on a state-vector simulator (PennyLane) — p.5–6 and p.8 *Results*.

### Loss & input-coordinate-derivative method
- Loss identical to classical PINN: `L(θ,Ξ) = L_PDE + Σ_k λ_k L_BC,k` — p.6, Eq. (14); `L_PDE = Σ (F(ū(xʲ);γ) − f(xʲ))²` Eq. (15); `L_BC,k = Σ (B_k(ū(xʲ)) − g(xʲ))²` Eq. (16). Initial conditions = temporal boundary; each boundary weighted by scalar λ_j.
- **Input-coordinate derivative** `∂ū/∂x = ∂C/∂x` computed by **parameter-shift rule on the quantum circuit combined with backpropagation through the FNN, via the product/chain rule** because the rotation angle is `φ_i(x)·x`: `∂C/∂x = Σ_i (∂C/∂R_y^i)·(∂R_y^i/∂(φ_i(x)x)) · (1 + (∂φ_i(x)/∂x)·x)` — p.6, Eq. (17). Parameter-shift term + backprop term explicitly bracketed.
- Parameter (Ξ) derivatives also hybrid: parameter-shift (quantum) × backprop (FNN); the quantum circuit's partial derivative is constant and reused, reducing circuit evals to **twice the number of qubits, independent of FNN size** — p.6, text before Eq. (17); Eqs. (18)–(28). Optimizer = **L-BFGS** (PyTorch, default params, strong Wolfe conditions) — p.5 (LBFGS) & p.8 *Results*.

### Known-limit / unit-test hook
- **Parameterized-rotation count formula:** `N_rot = 3 · n · L` (R_x,R_y,R_z per qubit per HEA layer) for n qubits, L variational layers — derived from Eq. (12) + Fig. 3. → variational θ weight-shape `(L, n, 3)`.
- **Quantum-derivative circuit-eval bound:** number of circuit evaluations for all FNN-parameter gradients = `2n`, independent of FNN parameter count — p.6 (hybrid backprop + parameter-shift), explicit claim.
- Concrete instance (lid cavity): "60 variational parameters in the PQC and 164 FNN parameters, total 244" for 4 qubits & 5 variational layers — p.12. Check: 3·4·5 = 60 ✓ (matches the 3·n·L formula).
- Limiting behavior: if FNN output φ ≡ const (no learning), embedding reduces to fixed angle embedding `R_y(c·x̄)` — consistent with Eq. (11) at frozen Ξ.

---

## FAMILY 2 — te_qpinn_qnn (Dehaghani et al., fully-quantum trainable embedding)

### Feature map / data embedding (exact gates + trainable params)
- **Two-stage quantum trainable embedding.** Stage A — a *trainable embedding network* (itself a quantum circuit, "QNN-based"): `U_embed(x̃,ỹ;θ_Q)` acting on `|0⟩^{⊗N_q}` produces `|ψ_embed⟩` — p.3 (Sec. III-A), Eq. (11). `U_embed` = "alternating layers of input-dependent gates … and trainable rotation layers parameterized by θ_Q, along with a predefined pattern of entangling gates such as nearest-neighbor CNOT gates" — p.3, end of Sec. III-A col.2 / p.4 col.1.
- Stage A readout: a prescribed set of Hermitian observables `{O_i}_{i=1}^{N_q}` is measured on `|ψ_embed⟩` giving `Γ(x̃,ỹ) = [α₁,…,α_{N_q}]ᵀ`, with `α_i = g_i(⟨ψ_embed|O_i|ψ_embed⟩)`, `O_i = Z_i` (Pauli-Z on qubit i), post-processing `g_i(s) = π s` (linear scaling into [−π,π]) — p.4, Eqs. (12).
- Stage B — standard data-encoding unitary: `U_enc(x̃,ỹ;θ_Q) = ⊗_{i=1}^{N_q} R_y(α_i(x̃,ỹ))`, `R_y(·) = exp(−i(·)σ_y/2)`, producing `|ψ_enc⟩ = U_enc|0⟩^{⊗N_q}` — p.4, Eqs. (13)–(14). So the angles fed to the encoding y-rotations are the trainable-network outputs α_i.
- Coordinates pre-normalized by affine map `x̃ = N(x), ỹ = N(y)` to `[−1,1]` — p.3, Eq. (10).
- **Trainable params of embedding = θ_Q** (parameters of the QNN embedding network), optimized jointly with variational params — p.4, Eqs. (12)–(14) and p.4 col.1. Illustrative QNN embedding circuit: 4 qubits, 2 layers — **Fig. 3** (p.7). Embedding circuit uses same qubit count as the VQC (p.8). Exact per-layer gate list of the QNN embedding **UNSPECIFIED IN SOURCE** (only described as alternating input-dependent gates + trainable rotation layers + nearest-neighbor CNOTs; Fig. 3 not gate-legible from text).

### Gate-by-gate variational ansatz
- Variational unitary `U_var(θ_var) = U_L(θ_L) ⋯ U_2(θ_2) U_1(θ_1)`; each layer `U_ℓ(θ_ℓ) = ∏_m exp(−i θ_{ℓ,m} H_m) W_m` — p.4, Eq. (15). `H_m` = Pauli-basis generators (rotations about Bloch axes), `W_m` = fixed non-parameterized unitaries implementing nearest-neighbor CNOT entangling — p.4 col.2.
- Each layer = learnable single-qubit rotations followed by a fixed entangling pattern (chain of CNOTs) — p.4 col.2. Hardware-efficient VQC uses single-qubit rotation gates **(R_x, R_y, R_z)** and CNOT entangling — p.8 (*Quantum Circuit Architectures*); illustrative 4-qubit / 3-layer VQC in **Fig. 2** (p.8). Same `U_var(θ_var)` applied uniformly to all spatial inputs (parameter sharing) — p.4 col.1.
- Encoded state then variational state: `|ψ_var(x,y;Θ)⟩ = U_var(θ_var) U_enc(x̃,ỹ;θ_Q)|0⟩^{⊗N_q}`; `Θ = (θ_Q, θ_var)` — p.4, Eq. (16).
- Per-output-field separate circuit: pressure `p` and stream function `ψ` each have their own identical-architecture VQC with independent params — p.8 col.1 (*Observable and Solver Configuration*); two circuit evals per collocation point.

### Depth / width scaling rule
- Width = `N_q` qubits; depth = `L` variational layers (Eq. (15)) + embedding-circuit depth. "Total number of quantum parameters typically grows linearly with `N_q L`" — p.6 (Sec. III-F, *Computational Cost and Parameter Scaling*). Training cost ≈ `N_col · N_q · L` (N_col = collocation points) — p.6. Parameter count grows polynomially with `N_q`, `L`, and embedding-circuit depth — p.6.
- Experiments: QPINN models use **10 variational layers, 5 embedding layers** (Table I caption, p.10); Reynolds sweep uses **6 qubits** (Table II, p.10); best config = 4 qubits, 10 variational layers (Conclusion, p.11). Exact rotations-per-layer count rule **UNSPECIFIED IN SOURCE** (gates listed as R_x,R_y,R_z but per-layer multiplicity/order not stated in text; only Fig. 2/3 schematics).

### Measurement / readout
- Per output field `i ∈ {p, ψ}`, readout operator `O_i = Σ_{j=1}^{N_q} Z_j` (sum of single-qubit Pauli-Z over all qubits) — p.5 col.1, Sec. III-C. Predicted fields = expectation values: `p̃(x,y) = ⟨ψ_var|O_p|ψ_var⟩`, `ψ̃(x,y) = ⟨ψ_var|O_ψ|ψ_var⟩` — p.5, Eqs. (17)–(18). One scalar expectation per circuit eval; two evals/point (pressure + stream-function solvers, independent params) — p.8 col.1.
- Velocity recovered classically via `u = ∂ψ̃/∂y`, `v = −∂ψ̃/∂x` — p.4 Eq. (3) / p.5 Eq. (19)–(20).

### Loss & input-coordinate-derivative method
- Physics-informed loss `L = L_PDE + λ_B (L_wall + L_lid + L_ref)` — p.5, Sec. III-D, Eq. (their total-loss eqn after Eq. (24)). Components: `L_PDE = (1/|Ω_int|) Σ (R_x² + R_y²)` Eq. (21) with momentum residuals `R_x = u u_x + v u_y + p_x − (1/Re)(u_xx+u_yy)` Eq. (19), `R_y` Eq. (20); `L_wall = (1/|∂Ω_wall|) Σ (u²+v²)` Eq. (22); `L_lid = (1/|∂Ω_lid|) Σ ((u−1)²+v²)` Eq. (23); reference pressure `L_ref = p(0,0)²` Eq. (24).
- **Input-coordinate derivative** via chain rule splitting quantum-expectation sensitivity × classical backprop of embedding angles: `∂f̃/∂x = Σ_{m=1}^{N_q} (∂f̃/∂α_m)(∂α_m/∂x)`, `f ∈ {p,ψ}` — p.5, Eq. (25). `∂f̃/∂α_m` via **parameter-shift rule** (quantum); `∂α_m/∂x` via **classical backpropagation through the QNN embedding network** — p.6 col.1. Second-order derivatives (u_xx, u_yy, v_xx, v_yy) via repeated/nested automatic differentiation — p.6 col.1.
- Parameter gradients: `∂L/∂θ` for variational/embedding params combine autograd over PDE operators with quantum parameter-shift — p.6, Eq. (26); Algorithm 1 lines 25–28: `∇_{θ_Q}L` via classical backprop through embedding + parameter-shift for embedding gates; `∇_{θ_var}L` via quantum parameter-shift rule. Optimizer = **classical L-BFGS** with hybrid gradients — Fig. 1 caption (p.3), Algorithm 1 line 7 & line 28 (p.6).

### Known-limit / unit-test hook
- **Parameter-scaling law:** total trainable quantum params ∝ `N_q · L` (linear in qubits × layers) — p.6, Sec. III-F (explicit). Training-cost ∝ `N_col · N_q · L` — p.6.
- **Concrete parameter-count anchor:** QNN-TE-QPINN ≈ **360 trainable parameters** vs FNN-TE-QPINN ≈ 608 vs classical PINN ≈ 6,594 (PINN: 4 hidden layers × 32 neurons) — p.2 (Abstract/Intro) & p.8 col.2. Best config: 4 qubits, 10 variational layers (Conclusion). Note: the exact (N_q, L)→param map that yields 360 is **UNSPECIFIED IN SOURCE** (no explicit rotations-per-layer formula given to reproduce 360 from 4 qubits/10 layers + 5 embedding layers).
- **Readout limiting behavior:** observable `Σ_j Z_j` is bounded → expectation ∈ [−N_q, N_q]; with g(s)=πs the embedding angles α ∈ [−π,π] (Eq. (12)) — bounded-spectrum unit-test invariant (p.5 col.1, p.4 Eq. (12)).
- Two independent solver circuits (p & ψ) ⇒ exactly **2 circuit evaluations per collocation point** — p.8 col.1 (testable invariant).

---

## Candidate discrepancies (points NOT pinnable from source)

1. **FAM1 CNOT entanglement exact pattern/count** — Fig. 3 caption explicitly says "the arrangement of CNOT gates and number of rotational gates varies in literature"; only "lower and upper neighbor" stated. Exact ring-vs-linear and CNOT count per layer **UNSPECIFIED IN SOURCE**.
2. **FAM2 QNN-embedding gate-by-gate sequence** — described only as "alternating input-dependent gates + trainable rotation layers + nearest-neighbor CNOTs" (Fig. 3, 4q/2layers schematic). Per-layer gate list/order **UNSPECIFIED IN SOURCE** from text.
3. **FAM2 rotations-per-variational-layer multiplicity** — text lists R_x,R_y,R_z and CNOTs but never states how many of each per layer; cannot reproduce the "≈360 params @ 4 qubits/10 var layers/5 embed layers" from a closed formula. **UNSPECIFIED IN SOURCE.**
4. **FAM2 embedding-circuit depth contribution to param count** — "5 embedding layers" (Table I) but the per-embedding-layer trainable-param count not given. **UNSPECIFIED IN SOURCE.**
5. **FAM1 input-component cycling for ≥2D** — given by `i % n` example but general rule for arbitrary dim/qubit ratios only sketched (must be divisible). Borderline-specified.
