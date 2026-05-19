# Independent Dual-Check: QCPINN (arXiv 2503.16678v6)

Pages read: 1-27 (full document incl. Appendix A/B/C). Derived solely from source PDF; no speccard consulted.

## Scope note
QCPINN defines BOTH a DV-Circuit (qubit) family and a CV-Circuit (photonic) family. The qcpinn-of-interest is the DV-Circuit (best performer; Sec 6, Conclusion p16). Both documented below.

## 1. Feature map / data embedding
- Architecture sandwich: classical preprocessor (2-layer NN) → QNN → classical postprocessor (Sec 4 p5; Fig 1 p6). Preprocessor: 1 hidden layer, 50 neurons, Tanh (Sec 5.2 p9 for DV; Sec 5.1 p8 for CV).
- DV embeddings: two schemes — **Amplitude** and **Angle** (Table 1 p6; Sec 4.2 p7). Angle maps each classical feature to a single-qubit rotation angle via parameterized gate e.g. RX(θ) (Sec "Angle vs. Amplitude Embedding" p15: "Angle embedding directly maps each classical feature to a rotation angle on a single qubit through parameterized gates such as RX(θ)").
- Embedding trainable? Fig 2 caption (p7) shows an "AngleEmbedding" block preceding parameterized rotation boxes; the embedding block is depicted as a fixed encoding stage, NOT a parameterized box (square boxes = parameterized rotations). Angle embedding described as "normalization-free feature mapping in which each input independently controls one qubit rotation" (p15) — input-controlled, not trainable weights inside the embedding gate itself. **Embedding rotations are NOT counted in the paper's parameter totals**: Table 2 (p7) parameter formulas (Alternate 4(n−1)L, Cascade 3nL, Cross-mesh (n²+4n)L, Layered 4nL) are stated as "Number of Parameters" for the ansatz layers L only; embedding is the separate "Embedding" stage of Fig 2 and is not added to these counts. The trainable embedding-like weights live in the classical preprocessor (θ₂), counted in classical Nₚ. (Caveat: paper never gives an explicit sentence "embedding angles are not trainable parameters" — inferred from Fig 2 box convention + Table 2 formulas being layer-only. Marked partially **UNSPECIFIED IN SOURCE** as an explicit statement.)
- CV embedding: Amplitude or Angle of input encoded via Displacement D(input[i],0), real amplitudes only (Algorithm 1 line 5-7, p25).

## 2. Gate-by-gate variational ansätze (DV, Fig 2 p7; Table 2 p7; HEA Eq. 4 p8)
General HEA form: U(ψ)=∏ₖ Uₖ(ψₖ)Wₖ (Eq. 4 p8), per-layer = single-qubit parameterized rotations then entangler.
- **Alternate** (Fig 2a): per qubit RX then RZ then RX parameterized rotations; nearest-neighbor CNOT entanglers in alternating pattern. Connectivity: nearest-neighbor. Depth 6L.
- **Cascade** (Fig 2b): parameterized single-qubit rotations + **CRX** (controlled-RX) entanglers in a ring topology. "Incorporating controlled rotation gates (CRX) improves entanglement" (Sec 4.2 p8). Depth (n+2)L.
- **Cross-mesh** (Fig 2c): parameterized rotations + all-to-all entangling (CRX/CRZ controlled rotations); densest. Depth (n²−n+4)L.
- **Layered** (Fig 2d): each qubit RX then RZ rotations; nearest-neighbor CNOT entangling layers, alternating rotation/entangling layers. Depth 6L.
- Allowed parameterized gates across all: RX, RY, RZ, CRX, CRZ (Sec 4.2 p7).

## 3. Depth/width + parameter & 2-qubit-gate formulas (Table 2 p7; worked example Sec "Feasibility" p15)
n = #qubits, L = #layers. Per Table 2:
| Topology | Depth | #Params | #2-qubit gates |
|---|---|---|---|
| Alternate | 6L | 4(n−1)L | (n−1)L |
| Cascade | (n+2)L | 3nL | nL |
| Cross-mesh | (n²−n+4)L | (n²+4n)L | (n²−n)L |
| Layered | 6L | 4nL | (n−1)L |
Worked sizing (p15, n=5, L=1): Cascade depth (5+2)·1=7, "about five entangling gates and roughly 15 trainable parameters" (3·5·1=15, 5·1=5 ✓). Cross-mesh depth (25−5+4)·1=24, "approximately 20 entangling gates and 45 trainable parameters" (5²+4·5=45, 25−5=20 ✓). DV experiments used 5 qubits, L=1 (Sec "Feasibility" p15; Sec 5 p9).

## 4. Measurement / readout; classical pre/post split
- DV readout: **Pauli-Z expectation per qubit** (Sec 5.2 p9: "Pauli-Z expectation measurements are performed on each qubit"). shots=None → analytic gradients via backprop (Sec 5.2 p9).
- CV readout: position quadrature ⟨q̂ᵢ⟩ OR photon-number ⟨n̂ᵢ⟩ (Sec 4.1 p6; Algorithm 1 line 15).
- Classical split: preprocessor θ₂ (linear dim-change + Tanh, 1 hidden 50-neuron layer) before QNN; postprocessor θ₃ mirrors it after QNN (Sec 4 p5; Fig 1 p6). QNN θ₁ in between.

## 5. Physics-residual loss + derivative method + Appendix A operators
- Loss: weighted MSE of PDE residual + BC + IC, soft imposition, fixed empirical weights λₖ (Eq. 2 p4; Sec 3.1 p4).
- Derivative method: **automatic differentiation** (PyTorch autograd), NOT parameter-shift. "We used PyTorch's automatic differentiation to compute the gradients for both classical and quantum parameters" (Sec 5.5 p9); shots=None to keep AD valid; parameter-shift explicitly avoided as too expensive (Sec 5.2 p9).
- Per-PDE residual operators (Appendix A):
  - **Helmholtz** — Eq. (5), residual u_xx+u_yy+αu (α=(a₁π)²+(a₂π)²), A.1 p21.
  - **2D lid-driven Cavity (unsteady incompressible NS)** — Eq. (6), residuals 𝓛_ru, 𝓛_rv (momentum), 𝓛_rc (continuity u_x+v_y), A.2 p21-22.
  - **1D Wave** — Eq. (7), residual u_tt − 4u_xx (c=2), A.3 p22.
  - **Klein-Gordon** (nonlinear, α=1,β=0,γ=1,k=3) — Eq. (8), residual u_tt − u_xx + u³ − f, A.4 p23.
  - **Convection-diffusion (2D)** — Eq. (9), residual u_t + c₁u_x + c₂u_y − DΔu − f (c₁=c₂=1, D=0.01), A.5 p23-24.

## 6. Known-limit / unit-test hook
- Table 2 closed-form invariants (testable): for n=5,L=1 — Cascade params=15, 2q-gates=5, depth=7; Cross-mesh params=45, 2q-gates=20, depth=24 (Sec p15). Any implementation must reproduce these exactly.
- Algorithm 1 phase-free flag: φ_s[l]=0, φ_d[l]=0, φ_bs[n]=0 when phase-free (CV, lines 10/12/22) — magnitude-only parameterization halves CV rotation params.

## UNSPECIFIED IN SOURCE
- Exact per-gate ordering within Cross-mesh all-to-all block (Fig 2c is dense/unlabeled at this resolution; gate sequence not enumerated in text).
- Explicit sentence stating embedding angles are excluded from parameter totals (inferred from Table 2 + Fig 2 box convention, never stated verbatim).
- Preprocessor exact output width (= #qubits? ) not stated numerically beyond "manages input-output dimensions" (Sec 4 p5).
- CV cutoff for DV n/a; DV n fixed at 5 but no n-sweep reported.
