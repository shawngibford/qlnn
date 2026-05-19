# Spec Card — te_qpinn_qnn (fully-quantum trainable embedding)

**Source PDF:** /Users/shawngibford/dev/phd/qlnn/.claude/worktrees/upbeat-elbakyan-68b56a/refs/2605.13892v1.pdf
("A QPINN Framework with Quantum Trainable Embeddings for the Lid-Driven Cavity Problem")
**Pages read:** 1–6

Chosen over siblings 2602.14596v1 (QNN-TE-QPINN described but less fully
equationed) and 2602.09291v1 (QNN-TE-QPINN, Eq. 3 only) because this paper
gives the complete QNN embedding equation set (Eqs. 10–18), gradient
decomposition (Eqs. 25–26), and an explicit training algorithm
(Algorithm 1).

## 1. Feature map (trainable quantum embedding)

- Inputs: spatial collocation coords (x, y) for steady 2-D lid-driven
  cavity (Sec. III, p.3–4).
- Affine pre-normalization to [-1,1]^2: x̃ = N(x), ỹ = N(y) (Eq. 10, p.4).
- **Trainable quantum embedding network**: normalized (x̃, ỹ) processed by
  a parameterized quantum circuit with trainable parameters θ_Q, acting on
  |0⟩^⊗N_q, producing the embedded state
  |ψ_embed(x̃,ỹ;θ_Q)⟩ = U_embed(x̃,ỹ;θ_Q)|0⟩^⊗N_q (Eq. 11, p.4).
- U_embed = "alternating layers of input-dependent gates ... and trainable
  rotation layers parameterized by θ_Q, along with a predefined pattern of
  entangling gates, such as nearest-neighbor CNOT gates" (text after
  Eq. 11, p.4). → Embedding itself is a PQC with trainable θ_Q
  → **FULLY-QUANTUM-TRAINABLE**.
- Readout-to-angle: prescribed Pauli-Z observables {O_i} on the embedded
  state give Γ(x̃,ỹ) = [α_1,…,α_{N_q}]^T, with
  α_i = g_i(⟨ψ_embed|O_i|ψ_embed⟩), O_i = Z_i, g_i(s) = πs (Eq. 12, p.5).
- Re-encoding into solver circuit: U_enc(x̃,ỹ;θ_Q) = ⊗_{i=1}^{N_q}
  R_y(α_i(x̃,ỹ)), R_y(·) = exp(-i(·)σ_y/2);
  |ψ_enc⟩ = U_enc|0⟩^⊗N_q (Eqs. 13–14, p.5).

## 2. Gate-by-gate ansatz

- Embedding circuit U_embed: alternating input-dependent gate layers +
  trainable θ_Q rotation layers + nearest-neighbor CNOT entanglers
  (text after Eq. 11, p.4). **Per-gate generator/angle schedule
  UNSPECIFIED IN SOURCE** (described only qualitatively; circuit diagram
  Fig. 1 schematic, p.4).
- Variational solver circuit U_var(θ_var) = U_L(θ_L)···U_2(θ_2)U_1(θ_1),
  layer block U_ℓ(θ_ℓ) = ∏_m exp(-i θ_{ℓ,m} H_m) W_m, H_m ∈ Pauli basis
  (x/y/z Bloch-axis rotations, applied per qubit), W_m = fixed
  nearest-neighbor CNOT chain (Eq. 15, p.5).
- Full state: |ψ_var(x,y;Θ)⟩ = U_var(θ_var) U_enc(x̃,ỹ;θ_Q)|0⟩^⊗N_q
  (Eq. 16, p.5). Θ = (θ_Q, θ_var).

## 3. Depth / width scaling

- Quantum cost scales with N_q and circuit depth L; trainable-param count
  ~ N_q·L; total training cost ~ N_col·N_q·L (Sec. III-F, p.6).
- QNN embedding adds cost proportional to embedding-circuit depth
  (Sec. III-F, p.6).
- Reported instance: ≈360 trainable params for QNN-TE-QPINN vs ≈6,600 for
  classical PINN baseline (Sec. I, p.2). Exact N_q and L for the reported
  run **UNSPECIFIED IN SOURCE** (pages 1–6).

## 4. Measurement

- Embedding readout: Pauli-Z, O_i = Z_i, expectation in [-1,1] scaled by
  π → encoding angles (Eq. 12, p.5).
- Output fields: for i ∈ {p, ψ}, O_i = Σ_{j=1}^{N_q} Z_j;
  p̃(x,y) = ⟨ψ_var|O_p|ψ_var⟩, ψ̃(x,y) = ⟨ψ_var|O_ψ|ψ_var⟩
  (Eqs. 17–18, p.5). Velocity recovered as u = ∂ψ/∂y, v = -∂ψ/∂x
  (Sec. III-D, p.5).

## 5. Loss & derivative method

- Physics-informed loss L = L_PDE + λ_B(L_wall + L_lid + L_ref)
  (Sec. III-D, Eqs. 19–24, p.5).
- L_PDE = mean-sq momentum residuals R_x, R_y (Eqs. 19–21, p.5);
  L_wall no-slip (Eq. 22), L_lid driven boundary (Eq. 23),
  L_ref = p(0,0)^2 (Eq. 24), p.5.
- Spatial derivatives via chain rule: ∂f̃/∂x = Σ_m (∂f̃/∂α_m)(∂α_m/∂x),
  f ∈ {p,ψ} (Eq. 25, p.5); ∂f̃/∂α_m from quantum parameter-shift,
  ∂α_m/∂x from classical backprop through the QNN embedding network
  (text after Eq. 25 / Sec. III-E, p.6).
- Total-loss gradient ∂L/∂θ over Θ via Eq. 26, p.6: autodiff for PDE
  components + parameter-shift for circuit params (embedding gates and
  variational gates), classical backprop through embedding network
  (Algorithm 1, steps 25–26, p.6).
- Optimizer: classical L-BFGS with hybrid gradients (Fig. 1, p.4;
  Algorithm 1 step 28, p.6).

## 6. Known-limit unit-test hook

- Boundary-condition limits: u(x,1)=1, v(x,1)=0 (moving lid, Eq. 7,
  p.3); u=v=0 on ∂Ω_wall (Eq. 8, p.3); p(0,0)=0 (Eq. 9, p.3) — usable
  as exact-constraint regression checks on a trained model.
- Divergence-free identity: stream-function form makes ∂u/∂x+∂v/∂y ≡ 0
  by construction (Eq. 4, p.2) — unit-test that recovered (u,v) from
  ψ̃ satisfies incompressibility to autodiff tolerance.
- Pauli-Z expectation boundedness: α_i = πs with s ∈ [-1,1] ⇒ encoding
  angles ∈ [-π,π] (Eq. 12, p.5) — assert numeric range as a sanity hook.
- Closed-form analytic reference for lid-driven cavity:
  **UNSPECIFIED IN SOURCE** (paper notes no closed-form solution exists,
  Sec. I, p.1).
