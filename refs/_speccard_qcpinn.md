# Spec Card — `qcpinn` (DV-Circuit QCPINN ansatz family)

Source: arXiv:2503.16678v6 "QCPINN: Quantum-Classical Physics-Informed Neural Networks for Solving PDEs", Farea, Khan, Celebi.
Pages read: 1–19 (title/abstract p.1; Intro p.1–3; Background §3 p.4–5; §4 QCPINN p.5–6; Fig.1 p.6; §4.2 DV-Circuit + Fig.2 + Table 2 p.7; HEA Eq.4 + topology prose p.8; §5 Implementation/Training p.8–9; Table 3 results p.10; §6 results p.10–13; §7 Discussion incl. p.15 sizing example; §8 Conclusion p.16; References p.16–19). Appendix A (PDE-specific residual/loss formulas) not extracted — see UNSPECIFIED notes.

Scope note: This card covers the **DV-Circuit (qubit) QCPINN** family, which §6/§8 identify as the best-performing variant. The CV-circuit variant (§3.2, §4.1, §5.1, Eq.3) is a distinct family and is only cross-referenced where contrastive.

---

## 1. Feature map / data embedding
- The quantum layer is sandwiched between a classical preprocessor and postprocessor; the QNN replaces the middle NN (§4, p.5; Fig.1, p.6).
- Preprocessor = a **two-layer classical NN**: layer 1 linear transform to modify input dimension, then a nonlinear activation; layer 2 maps transformed inputs to the QNN layer (§4, p.5). Implemented as **one hidden layer, 50 neurons, Tanh activation** (§5.2, p.9; §5.1 states same for CV; §5.2 confirms "Similar to the CV-Circuit ... one hidden layer with 50 neurons and the Tanh activation function").
- Two embedding schemes evaluated: **Amplitude** and **Angle** (Table 1, p.6; §4.2, p.7).
- **Angle embedding** maps each classical feature to a single-qubit rotation angle via parameterized gates such as RX(θ); local, differentiable, normalization-free; encodes one feature per qubit (Angle-vs-Amplitude discussion, p.15). Figs.2 panels show an "Angle embedding" block as the first stage on all 5 qubits (Fig.2 caption, p.7).
- **Amplitude embedding** encodes data as state amplitudes under Σ|aᵢ|²=1; scales as 2ⁿ amplitudes for n features, may need zero-padding (p.15).
- Classical↔quantum interface: preprocessor output feeds embedding gates; quantum output (measurements) feeds postprocessor (§4, p.5; Fig.1, p.6).
- Trainable params in embedding: square boxes in Fig.2 (incl. the embedding-adjacent rotations) "denote parameterized single-qubit rotation gates ... which are optimized during training" (§4.2, p.7). The exact analytic embedding-angle formula (e.g., scaling factor) is **UNSPECIFIED IN SOURCE** (only "RX(θ)" named as example, p.15).

## 2. Gate-by-gate variational ansatz
General form = hardware-efficient ansatz (HEA): `U(ψ) = Π_k U_k(ψ_k) W_k`, each layer k alternates parameterized single-qubit rotations U_k(ψ_k) with an entangling operation W_k (Eq.4, p.8).
Parameterized single-qubit / controlled-rotation gates used across topologies: **RX, RY, RZ, CRX, CRZ** (§4.2, p.7).
Four named topologies (Fig.2, p.7; Table 2, p.7; prose p.8):
- **Alternate**: alternating layers of parameterized single-qubit rotations followed by nearest-neighbor CNOT gates (p.8). Fig.2(a) shows per-qubit RX–RZ–RX with CNOT entanglers.
- **Cascade**: ring topology; uses controlled-rotation (CRX) gates instead of CNOT; "alternating rotation and entangling" with ring connectivity (p.8).
- **Cross-mesh**: all-to-all connectivity; parameterized controlled rotations; strongest/global entanglement, largest depth (p.8).
- **Layered**: each qubit gets RX and RZ single-qubit rotations, entanglement via nearest-neighbor CNOT; alternating rotation and entangling layers (p.8). Fig.2(d) shows RZ–RX rotations then CNOT ladder then RX.
Exact per-gate ordering beyond what Fig.2 panels depict (e.g., precise CRX target/control list for Cascade ring, Cross-mesh pairing order) is only given pictorially in Fig.2; no closed-form gate list in text — treat fine ordering as **figure-derived, text-UNSPECIFIED**.

## 3. Depth / width scaling rule
From Table 2 (p.7), with n = number of qubits, L = number of layers:

| Topology | Circuit depth | Connectivity | # parameters | # two-qubit gates |
|---|---|---|---|---|
| Alternate | 6L | nearest-neighbor | 4(n−1)L | (n−1)L |
| Cascade | (n+2)L | ring | 3nL | nL |
| Cross-mesh | (n²−n+4)L | all-to-all | (n²+4n)L | (n²−n)L |
| Layered | 6L | nearest-neighbor | 4nL | (n−1)L |

Best configs implemented with **n = 5 qubits, single quantum layer L = 1** (§ Feasibility, p.15; "implemented with five qubits and a single quantum layer (L=1)").
Worked numbers (p.15): Cascade depth (n+2)L = 7, ≈5 entangling gates, ≈15 trainable params; Cross-mesh depth (n²−n+4)L = 24, ≈20 entangling gates, ≈45 trainable params.
(Note: a separate `(2n+3)L` depth / `O(n²L)` param statement on p.7 refers to the **CV-circuit**, not DV — do not apply to qcpinn DV.)

## 4. Measurement / readout
- DV-Circuit QCPINN: **Pauli-Z expectation measurement on each qubit** to extract features (§5.2, p.9: "Pauli-Z expectation measurements are performed on each qubit").
- Measurement vector → classical **postprocessor** (mirrors preprocessor structure: one hidden layer, 50 neurons, Tanh) producing final PDE-solution outputs (§4, p.5; §5.2/§5.3, p.9; Fig.1, p.6).
- `shots = None` → analytic/deterministic simulation enabling exact backprop gradients (§5.2, p.9).
- (CV variant instead uses quadrature ⟨q̂ᵢ⟩ or Fock-number ⟨n̂ᵢ⟩ — §4.1, p.6 — not applicable to qcpinn DV.)

## 5. Loss & derivative method
- PINN composite loss: `L(θ)=argmin Σ_k λ_k L_k`, with a PDE-residual term `λ₁L₁(D[u_θ(x);α]−f(x))` plus boundary/initial terms `Σ λ_k L_k(B[u_θ(x)]−g_k(x))` (Eq.2, p.4). Soft imposition of BC/IC with fixed empirical weights (§3.1, p.4).
- Physics block in Fig.1 (p.6) shows BC term Σ(yᵢ−ŷᵢ), IC term Σ(yᵢ(0)−ŷᵢ(0)), and a "Phy" residual using derivatives d_{x₁}, d_{x₂}, d_{x₃}… of the network output w.r.t. input coordinates.
- **Derivative method**: input-coordinate derivatives for the physics residual are obtained by **automatic differentiation**. "We used PyTorch's automatic differentiation to compute the gradients for both classical and quantum parameters" (§5.5, p.9). `shots=None` activates deterministic simulation so continuous gradients flow through the quantum circuit via backprop, explicitly avoiding parameter-shift (§5.2, p.9). Architecture "ensur[es] the higher-order differentiability necessary for residual loss" (§4, p.5).
- Optimizer: **Adam, lr = 0.005**, ReduceLROnPlateau (factor 0.9, patience 1000 epochs, for DV), gradient clipping to unit norm, batch size 64, max 20,000 epochs (§5.2 & §5.5, p.9).
- Exact per-PDE residual operators D[·] and forcing f(x) for Helmholtz/Cavity/Wave/Klein-Gordon/Convection-diffusion: defer to Appendix A (§5.4, p.9) — **UNSPECIFIED IN SOURCE (Appendix A not extracted)**.

## 6. Quantum vs classical split
- **Classical**: preprocessor NN (1 hidden layer, 50 neurons, Tanh) before the circuit, and postprocessor NN (same shape) after measurement (§4, p.5; §5.2, p.9; Fig.1).
- **Quantum (PQC)**: the QNN layer = embedding gates + HEA variational ansatz (one of Alternate/Cascade/Cross-mesh/Layered) + Pauli-Z readout (§4.2, p.7; Fig.1/Fig.2).
- Classical baseline PINN replaces the QNN with classical NN: Model-1 = two 50-neuron Tanh layers; Model-2 = preprocessor+postprocessor only, no middle NN (§5.3, p.9).
- Parameter-count motivation: inserting a QNN between two classical layers reduces O(hᵢhᵢ₊₁) to O(hᵢnᵢ + q_params + nᵢhᵢ₊₁) (p.8).

## 7. Known-limit / unit-test hook
**Assertable closed-form (Table 2, p.7), strongest as a hook because it is corroborated by an independent worked example on p.15:**

For the **Cascade** topology with n qubits, L layers:
- trainable quantum parameter count = `3·n·L`
- two-qubit (CRX) gate count = `n·L`
- circuit depth = `(n+2)·L`

Independent check (p.15) at n=5, L=1: depth = (5+2)·1 = 7 ✓; entangling gates ≈ 5 = n·L ✓; trainable params ≈ 15 = 3·n·L ✓.

A test can therefore assert, for a built Cascade circuit at (n,L): `num_trainable == 3*n*L`, `num_two_qubit_gates == n*L`, `circuit_depth == (n+2)*L`. (Cross-mesh analogue, also p.15-corroborated at n=5,L=1: params = (n²+4n)L → 45 ✓, two-qubit = (n²−n)L → 20 ✓, depth = (n²−n+4)L → 24 ✓.)

---

### Items explicitly UNSPECIFIED IN SOURCE
- Exact analytic embedding angle formula / input scaling (only "RX(θ)" named as example, p.15).
- Exact per-gate parameter assignment and intra-layer gate ordering beyond the Fig.2 pictures (no closed-form gate list in text).
- Per-PDE residual operator D[·] and forcing terms (Appendix A; not extracted).
- Whether embedding rotation gates are themselves trainable in the parameter-count formulas of Table 2, vs only the ansatz rotations (text says Fig.2 square boxes are "optimized during training", p.7, but does not partition the count).

---

## Appendix A — per-PDE physics residual operators (from PDF Appendix A)

Source: PDF "A Loss Function Design for PDEs" (pp.21–24). Eq. numbers below are the PDF's own (5)–(9). Notation: `u_θ` = network output; subscripts = partial derivatives; Ω = interior, Γ = boundary, Γ₀ = initial.

### A.1 Helmholtz (PDF §A.1, Eq. (5))
- **Governing PDE**: Δu(x,y) + k²u(x,y) = f(x,y) on Ω; u = h(x,y) on Γ₀ (Eq. (5)). Δ = ∂²/∂x² + ∂²/∂y².
- **Domain/params**: Ω = [−1,1]², wavenumber k=1, exact u(x,y)=sin(a₁πx)sin(a₂πy), a₁=1, a₂=4 (PDF §A.1).
- **PDE residual** (Eq. (5) loss expansion): `r = u_{θ,xx}(x,y) + u_{θ,yy}(x,y) + α·u_θ(x,y)`, with α=(a₁π)²+(a₂π)²; L_phy = MSE(r) over Ω.
- **Forcing/source**: f(x,y) = u(x,y)·[k² − (a₁π)² − (a₂π)²] (PDF §A.1).
- **BC penalty**: L_bc = MSE( u(x,y) − u_θ(x,y) ) on Γ₀ (Dirichlet) (Eq. (5) expansion).
- **IC penalty**: none (time-independent). **Loss weights**: λ₁=1.0, λ₂=10.0 (PDF §A.1).

### A.2 Time-dependent 2D lid-driven cavity / incompressible Navier–Stokes (PDF §A.2, Eq. (6))
- **Governing PDE**: ρ(∂u/∂t + u·∇u) = −∇p + μ∇²u; ∇·u = 0; u(0,x)=0; u(t,x₀)=0 on Γ₀ (walls); u(t,x_l)=1 on Γ₁ (lid) (Eq. (6)). Ω=(0,1)², grid (100,100), t∈[0,10] s, Δt=0.01 s, ρ=1056 kg/m³, μ=1/Re=0.01, U=1 m/s.
- **PDE residual** L_phy = L_{r_u} + L_{r_v} + L_{r_c} (PDF §A.2):
  - x-momentum: `L_{r_u} = MSE[ u_{θ_t} + u_θ u_{θ_x} + v_θ u_{θ_y} + (1/ρ) p_{θ_x} − μ(u_{θ_xx} + u_{θ_yy}) ]`
  - y-momentum: `L_{r_v} = MSE[ v_{θ_t} + u_θ v_{θ_x} + v_θ v_{θ_y} + (1/ρ) p_{θ_y} − μ(v_{θ_xx} + v_{θ_yy}) ]`
  - continuity: `L_{r_c} = MSE[ u_{θ_x} + v_{θ_y} ]`
- **BC penalties** (PDF §A.2): moving lid `L_up = MSE[(1.0 − û) + v̂]` on Γ₁; no-slip walls `L_bc1 = L_{bottom,right,left} = MSE[û + v̂]` on Γ₀.
- **IC penalty**: `L_u0 = MSE[û + v̂ + p̂]` (initial condition, all fields zero).
- **Forcing/source**: no explicit body force (RHS source = 0; pressure-gradient term internal). **Loss weights**: λ₁=0.1 (L_phy), λ₂=2.0 (L_up), λ₂=2.0 (L_bc1), λ₃=4.0 (L_u0) (PDF §A.2).

### A.3 1D wave equation (PDF §A.3, Eq. (7))
- **Governing PDE**: u_tt(t,x) − c²u_xx(t,x) = 0; u(t,x₀)=f₁ on Γ₀, u(t,x₁)=f₂ on Γ₁, u(0,x)=g(t,x), u_t(0,x)=h(t,x) (Eq. (7)). c=2, a=0.5, f₁=f₂=0; exact u(t,x)=sin(πx)cos(cπt)+0.5 sin(2πx)cos(4cπt); domain (t,x)∈[0,1]².
- **Concrete BVP** (PDF §A.3): u_tt − 4u_xx = 0; u(t,0)=u(t,1)=0; u(0,x)=sin(πx)+0.5 sin(4πx); u_t(0,x)=0.
- **PDE residual**: `r = u_{θ_tt}(t,x) − 4 u_{θ_xx}(t,x)`; L_phy = MSE(r) over Ω (Eq. (7) expansion).
- **BC/IC penalty**: `MSE[ u_θ(t,0) + u_θ(t,1) + u_θ(0,x) − sin(πx) − 0.5 sin(4πx) ]` on Γ₀∪Γ₁ (Eq. (7) expansion).
- **Initial-velocity penalty**: `MSE[ u_{θ_t}(0,x) ]` on ∂Ω (Eq. (7) expansion).
- **Forcing/source**: f = 0 (homogeneous). **Loss weights**: λ₁=0.1, λ₂=10.0, λ₃=0.1 (PDF §A.3).

### A.4 1D nonlinear Klein–Gordon (PDF §A.4, Eq. (8))
- **Governing PDE**: u_tt − αu_xx + βu + γu^k = f(t,x); u(t,x)=g₁ on Γ₀, u_t(t,x)=g₂ on Γ₁, u(0,x)=h (Eq. (8)). α=1, β=0, γ=1, k=3; exact u(t,x)=x cos(5πt)+(tx)³; domain [0,1]².
- **Concrete BVP** (PDF §A.4): u_tt − u_xx + u³ = x(−25π²cos(5πt) − 6t³) + 6tx³ + x³(cos(5πt)+t³x²)³; u(t,0)=0; u(t,1)=cos(5πt)+t³; u(0,x)=x; u_t(0,x)=0.
- **PDE residual**: `r = u_{θ_tt}(t,x) − u_{θ_xx}(t,x) + u_θ³(t,x) − f(t,x)`; L_phy = MSE(r) over Ω (Eq. (8) expansion).
- **BC/IC penalty**: `MSE[ u_θ(t,0) + u_θ(t,1) − cos(5πt) + t³ + u_θ(0,x) − x ]` on Γ₁ (Eq. (8) expansion).
- **Initial-velocity penalty**: `MSE[ u_{θ_t}(0,x) ]` on ∂Ω (Eq. (8) expansion).
- **Forcing/source**: f(t,x) = x(−25π²cos(5πt) − 6t³) + 6tx³ + x³(cos(5πt) + t³x²)³ (PDF §A.4).
- **Loss weights**: λ₁=1.0, λ₂=10.0, λ₃=1.0 (PDF §A.4).

### A.5 2D convection–diffusion (PDF §A.5, Eq. (9))
- **Governing PDE**: u_t + c₁u_x + c₂u_y − DΔu(t,x) = f(t,x); u(t,x₀)=g₀ on Γ₀, u(t,x₁)=g₁ on Γ₁, u(0,x)=h(0,x) (Eq. (9)). Δ = ∂²/∂x² + ∂²/∂y². c₁=c₂=1.0, D=0.01; exact u(t,x,y)=exp(−100((x−0.5)²+(y−0.5)²))·exp(−t); IC h(0,x,y)=exp(−100((x−0.5)²+(y−0.5)²)); domain [0,1]³.
- **Concrete IBVP** (PDF p.24): u_t + u_x + u_y − 0.01(u_xx + u_yy) = f(x,y); u(t,0,y)=exp(−25 − 100(y−0.5)² − t); u(t,1,y)=exp(−25 − 100(y−0.5)² − t); u(0,x,y)=h(x,y).
- **PDE residual**: `r = u_{θ_t} + u_{θ_x} + u_{θ_y} − 0.01(u_{θ_xx} + u_{θ_yy}) − f`; L_phy = MIN/MSE(r) over Ω (Eq. (9) expansion; note PDF uses "min" of the residual norm in the displayed loss).
- **BC penalties**: `‖ u_θ(t,0,y) − exp(−25 − 100(y−0.5)² − t) ‖` on Γ₀ and `‖ u_θ(t,1,y) − exp(−25 − 100(y−0.5)² − t) ‖` on Γ₁ (PDF p.24 expansion).
- **IC penalty**: `‖ u_θ(0,x,y) − h(x,y) ‖` on Ω₀ (PDF p.24 expansion).
- **Forcing/source**: f(x,y) = exp(−100((x−0.5)²+(y−0.5)²) − t)·[3 − 200(x−0.5) − 200(y−0.5) − 400((x−0.5)²+(y−0.5)²)] (PDF p.24).
- **Loss weights**: λ₁=1.0, λ₂=10.0, λ₃=10.0 (PDF p.24).

### Open-question resolution: are data-embedding rotations counted in the parameter totals?
**Answer: NO — embedding rotations are NOT included in Table 2 / p.15 parameter totals; only the variational-ansatz rotations are counted.** Citation: PDF p.15 ("DV-Circuit QCPINN … Angle-Cascade and Angle-Cross-mesh … five qubits and a single quantum layer (L=1)"). The Angle-Cascade is stated to have depth `(n+2)L = 7` and "roughly 15 trainable parameters" (= 3·n·L = 3·5·1) and "about five entangling gates" (= n·L); Angle-Cross-mesh "depth `(n²−n+4)L = 24`" with "approximately 45 trainable parameters" and "≈20 entangling gates". These counts match the Table-2 closed forms `3nL` / `(n²+4n)L`, which enumerate only the parameterized Rot/CRX gates of the ansatz. The angle embedding is described (PDF p.15, "Angle vs. Amplitude Embedding") as mapping "each classical feature to a rotation angle on a single qubit through parameterized gates such as RX(θ)" where θ is the **data-dependent input**, not a trained weight — hence excluded from the trainable-parameter totals. This **corrects** the prior spec-card "UNSPECIFIED" item.

### Still UNSPECIFIED IN SOURCE
- Exact analytic embedding angle/scaling formula for the data→RX(θ) map (only named, not given) — unchanged from prior pass.
