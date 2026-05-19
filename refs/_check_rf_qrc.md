# Independent Dual-Check: RF-QRC (arXiv:2405.03390v2, Phys. Rev. Research 6, 043082 (2024))

Pages read: 1-19 (full PDF: title/abstract p1; intro p1-2; CRC background p3-4; qubits p4; Alg.1/Alg.2 + hybrid QRC p5-6; RF-QRC Sec.III p6; numerical results Lorenz-63/96/MFE p6-15; conclusion p15; Appendix A circuits p16-17; Appendix B memory p16-17; refs p17-19).

Derived solely from the source PDF. NO _speccard file was read.

---

## 1. Input feature map / data embedding (exact gates, scaling)

- Input time-series is encoded into the circuit via **single-qubit rotational encoding** (Alg.2 step 1, p5; Fig.4). "The input time-series is encoded in the quantum circuit using single-qubit rotation angles" (Sec.II C, p5, around Eq.9).
- Classical data is **rescaled to the interval [0, 2π]** and used as rotation angles X_Θ (Appendix A, p16: "The rotation angles X_Θ represent the mapped classical data rescaled to the interval [0, 2π]"; restated p8 caption Fig.6 and Sec.IV B re V(α) sampled from [0,4π]).
- Concrete gates (Fig.6 inset p8, Fig.22-25 Appendix A p16-17): each qubit q_j initialized |0⟩, a **Hadamard H**, then a **R_Y(X_j)** rotation. For the product feature map (Fig.25, p17) encoding uses **R_Z(X_j)** followed by entangling + **R_Y(X_a × X_b)** product-angle rotations.
- Encoding is the unitary Φ(u_in(t_i)) applied to the state (Eq.9, p5): |ψ(t_{i+1})⟩ = V(α) Φ(u_in(t_i)) P(r(t_i)) |0⟩^{⊗n}.
- **UNSPECIFIED IN SOURCE**: exact closed-form of the "product" angle combination X_a × X_b (only schematic R_Y(X_0×X_1) etc. shown, Fig.25); exact rescaling formula from raw data to [0,2π].

## 2. Reservoir circuit gate-by-gate; FIXED/random vs TRAINED

- The full per-time-step circuit (Eq.9, p5; Fig.6, p8): three unitary blocks acting on |0⟩^{⊗n}:
  1. **P(r(t_i))** — previous-reservoir-state encoding unitary (single-qubit rotations encoding classical r(t_i)).
  2. **Φ(u_in(t_i))** — input feature map (Sec.1 above).
  3. **V(α)** — a **random parameterized circuit with n parameters α** (Sec.II C p5: "a third unitary operator V(α) given by a random parameterized circuit with n parameters α is applied"; Alg.2 step 3 p5: "Apply an additional unitary V(α) sampling from a uniform random distribution to provide randomization").
- Entanglement topology (Appendix A, Fig.22-25, p16-17): four feature maps —
  - (a) linearly entangled (Fig.22): H + R_Y per qubit, nearest-neighbour CNOT chain (control q_j → target q_{j+1}).
  - (b/c) fully entangling (Fig.23): H + R_Y, then CNOTs entangling pairs of qubits (denser CNOT pattern).
  - (d) fully entangling **symmetric** (Fig.24): same as (c) plus an additional R_Y data-encoding layer after the CNOTs.
  - product states (Fig.25): R_Z encoding + entangling + R_Y(product-angle) layer.
- **FIXED/RANDOM (NOT trained):** V(α) — α "sampled from a uniform distribution interval [0, 4π] with a predefined seed, kept fixed throughout training and prediction of a particular realization" (Sec.IV B, p8). Analogous to classical fixed W_in/W. All quantum gate parameters (encoding angles are data-driven, not learned; α random fixed). "no tuneable hyperparameter ρ exists" since U is norm-preserving (Appendix B, p16).
- **TRAINED:** only the classical linear readout matrix W_out (Sec.II C p6; Eq.3). This is RF-QRC's defining property.

## 3. Recommended / main RF-QRC configuration

- **Table I (p7)** lists five ansätze QRC-C1…QRC-C5 by (Reservoir states P / Input Φ / Variation V):
  - **QRC-C4 is explicitly labeled "(RF-QRC)"** in Table I, p7.
  - QRC-C4 row: P = "-" (no reservoir-state unitary → recurrence removed), Φ = **Fully entangled (×2)** (feature map applied twice), V = **Fully entangled symmetric**.
- QRC-C3 (P="-", Φ=Linearly entangled ×2, V=Linearly entangled) and QRC-C5 (P="-", Φ=Product states ×2, V=Linearly entangled) are the other recurrence-free configs but the paper's recommended/best RF-QRC is **QRC-C4**: "the proposed QRC-C4 architecture outperforms all other architectures … best prediction capabilities while requiring a smaller circuit depth" (p9); used for Lorenz-96 (Tab.III), MFE (Tab.IV: "Configuration C4 (RF-QRC)").
- Construction: P = identity (removed); Φ applied **twice** (×2) with fully-entangling topology; V = fully-entangled symmetric random fixed unitary. (Table I p7; Sec.III p6 "by setting P to the identity, and by applying Φ twice").

## 4. Reservoir-size / depth scaling

- Reservoir size N = state-vector dimension = 2^n, scales **exponentially with qubit number n** (Sec.IV A/B, p6,8; Fig.8 top axis n=5..11 ↔ N=32..2048).
- Recurrence-free choice yields **circuit depth independent of reservoir size**: "Removing the recurrence … yield circuits depths that are independent of the reservoir size" (Sec.III, p6); confirmed Fig.7 (p9) circuit depth vs n=8..12 — QRC-C3/C4/C5 flat/low vs QRC-C1/C2 exponentially growing.
- **No closed-form gate-count formula given** — only the qualitative claim "circuits depths that are independent of the reservoir size" and the empirical Fig.7 bar chart. Recurrent configs (C1,C2) stated to have "exponentially growing circuit depths" (Sec.III p6; p11). **UNSPECIFIED IN SOURCE**: explicit gate-count / depth as a closed function of n.

## 5. Measurement observables + linear readout training

- **Observable: Pauli-Z basis measurement** on each qubit (Alg.2 step 5, p5: "Measure the quantum circuit in a Pauli-Z basis to convert quantum reservoir state |ψ(t_{i+1})⟩ to classical reservoir state r(t_{i+1})"). Pre-processed reservoir vector r̂(t_{i+1}) ∈ ℝ^{2^n} derived from measured probabilities/expectation values in the computational basis (Sec.II C, p5, after Eq.9; Fig.6 r̂(t_{i+1}) ∈ ℝ^{2^n}).
- A constant output bias of 1 is appended: r → [r, 1] (Sec.II A, p4).
- **Readout training = closed-form linear ridge / Tikhonov regression.** Normal-equation form (Eq.3, p4):

  **(R Rᵀ + β I) W_out = R U_dᵀ**

  where R ∈ ℝ^{N_r × (N_tr − N_w)} is the concatenated post-processed reservoir-state matrix, U_dᵀ ∈ ℝ^{N_u × N_tr} the horizontal concatenation of target output data, β the user-defined Tikhonov regularization parameter, I the identity.
- Prediction (Eq.4, p4): **u_p(t_{i+1}) = [r(t_{i+1})]ᵀ W_out** (closed-loop autonomous when fed back).
- β values used: {1×10⁻⁶, 1×10⁻⁹, 1×10⁻¹²} (Tables II/III/IV, p10/11/14).

## 6. How recurrence is removed vs standard QRC (the "RF" contribution)

- Standard hybrid QRC (Sec.II C, p5-6; Fig.5/6 dotted feedback loop): each reservoir update r(t_i) is re-encoded into the circuit via unitary P(r(t_i)) and fed back at every time step → an active quantum-classical feedback loop (Eq.9 with P present; Alg.2 step 5 "Re-encode the measured r(t_{i+1}) to P unitary for the next time-step").
- **RF-QRC removes the recurrence by setting P to the identity** (Sec.III, p6: "Removing the recurrence, i.e. by setting P to the identity, and by applying Φ twice for stimulating the reservoir dynamics") — Table I QRC-C4 has P = "-". No reservoir-state re-encoding, no classical feedback loop (Fig.5/6 captions p7-8: "In RF-QRC, we remove this recurrent feedback layer / the feedback loop and the first unitary P").
- Lost recurrence memory is compensated by applying the feature map Φ **twice (×2)** and a richer fully-entangling feature map (Sec.III p6; p11: "fully connected input feature map that enriches the reservoir dynamics"). Distinguished from QELM and stated to differ because the leaky-integrator post-processing Eq.2 (linear memory + nonlinearity) is retained (Sec.III, p6).
- Benefit: parallelizable training (no feedback loop), circuit depth independent of reservoir size, suitable for NISQ (Sec.II C end p6; Sec.III p6; conclusion p15).

## 7. Known-limit / unit-test-hook property

- **Trained-parameter count = the readout matrix W_out only.** No quantum circuit parameter is optimized — encoding angles are data-driven and V(α) is random-but-fixed per realization (Sec.IV B p8; Sec.II C p6). Unit-test hook: assert number of trainable parameters == size of W_out (∈ ℝ^{N_r × N_u} incl. the +1 bias row), and that the quantum reservoir output for fixed seed + fixed input is **deterministic / independent of any "training" step** (i.e. reservoir states identical before and after readout fit). Additionally: RF-QRC reservoir update must NOT depend on previous reservoir state through the circuit (P = identity) — only through the classical leaky post-processing Eq.2.
- Corollary hook: circuit depth must be **invariant to n** for QRC-C4 (Fig.7, p9) — testable as depth(n=8) == depth(n=12) for the RF-QRC ansatz.

## UNSPECIFIED IN SOURCE (genuine gaps)

- Exact closed-form of the rescaling map raw-data → [0,2π] rotation angles.
- Exact functional form of product-feature-map combined angles X_a × X_b (Fig.25 schematic only).
- **Multi-qubit entanglement circuit figures use an ellipsis (vertical dots ⋮) for qubits q_3…q_n** — Fig.22, Fig.23, Fig.24, Fig.25 (p16-17) all draw only q_0..q_3 (or q_2) explicitly and represent the n>4 entanglement pattern with "⋮". **This is a real source gap: the exact CNOT wiring for the general n-qubit "fully entangled" and "fully entangled symmetric" topologies is NOT fully specified — only the ≤4-qubit pattern is drawn.**
- No closed-form gate-count / circuit-depth formula as a function of n (only qualitative "independent of reservoir size" + empirical Fig.7).
- Exact number of measurement shots used (paper states 10⁴–10⁶ is "common" but uses Qiskit noise-free statevector; explicit shot count for the reported results not given — Sec.IV p6).
- Precise definition of how r̂ ∈ ℝ^{2^n} is constructed from measured Pauli-Z probabilities (sampling described qualitatively, Sec.II C p5).
