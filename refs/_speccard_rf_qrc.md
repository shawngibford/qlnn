# Spec Card — `rf_qrc` (Recurrence-Free Quantum Reservoir Computing)

**Primary source:** Ahmed, Tennie, Magri, *"Prediction of chaotic dynamics and extreme events: a recurrence-free quantum reservoir computing approach"*, Phys. Rev. Research **6**, 043082 (2024) — arXiv:2405.03390v2 (`refs/2405.03390v2.pdf`).

**Pages read:** 1–19 (primary PDF, all pages). No optional/corroborating sources were consulted — every detail below is from the primary PDF. No model priors used.

**Identity of `rf_qrc`:** RF-QRC is exactly configuration **QRC-C4** in Table I (p. 7), explicitly labelled "QRC-C4 (RF-QRC)". This grounds every architectural choice below in Table I + Sec. III + Fig. 6 + Appendix A.

---

## 1. Architecture overview (Sec. III, p. 6; Fig. 6, p. 8; Eq. 9, p. 5)

General gate-based QRC reservoir update (Eq. 9, p. 5):
`|ψ(t_{i+1})⟩ = V(α) Φ(u_in(t_i)) P(r(t_i)) |0⟩^{⊗n}`
- `P(r(t_i))` = unitary encoding the *previous reservoir state* (the recurrence) — Fig. 6 caption (p. 8), Sec. II C (p. 5).
- `Φ(u_in(t_i))` = the data-encoding **quantum feature map** — Algorithm 2 step 2 (p. 5).
- `V(α)` = a fixed random parameterized unitary providing randomization — Algorithm 2 step 3 (p. 5); Sec. IV B 1 (p. 8).

**RF ("recurrence-free") contribution (Sec. III, p. 6; Sec. IV B 1, p. 8; Fig. 5 caption p. 7; Conclusion p. 15):**
- Recurrence is removed by **setting `P` to the identity** ("Removing the recurrence, i.e. by setting `P` to the identity, and by applying `Φ` twice", p. 6).
- To compensate the lost reservoir stimulation, the data feature map `Φ` is **applied twice** ("(x2)" in Table I, QRC-C4 Input column, p. 7; text p. 6).
- Net effect: no classical feedback loop at each time step (Fig. 5 caption p. 7: "In RF-QRC architecture, we remove this recurrent feedback layer"; Fig. 6 caption p. 8: "In RF-QRC, we remove the feedback loop and the first unitary `P`"). Circuit depth becomes **independent of reservoir size** (p. 6: "yield circuits depths that are independent of the reservoir size"; Conclusion p. 15: "small circuit depth that does not scale with the number of qubits").
- Distinct from Quantum Extreme Learning Machines because RF-QRC still uses the leaky-integral linear combination of memory + nonlinearity (Eq. 2), p. 6 ("Recurrence-free QRC is different than Quantum Extreme Learning Machines [49], because of the use of ... Eq. (2)").

So for `rf_qrc` the per-step circuit reduces to (P = I):
`|ψ(t_{i+1})⟩ = V(α) · Φ(u_in(t_i)) · Φ(u_in(t_i)) · |0⟩^{⊗n}`
(Eq. 9 with P=I and Φ applied twice — p. 6 text + Table I QRC-C4, p. 7.)

---

## 2. Input feature map / data embedding (Table I p. 7; Appendix A p. 16, Fig. 23)

- QRC-C4 Input (Φ) = **"Fully entangled (x2)"** (Table I, p. 7). The "Fully entangled qubits" circuit is **Fig. 23** (p. 16), referenced from Appendix A (p. 16: "Figs. 22–25 represent the quantum circuits for different ansätze presented in Tab. I").
- Per-qubit, the feature map (Fig. 23, p. 16) applies, on each qubit `q_k`, k = 0..n:
  1. a **Hadamard `H`** gate, then
  2. a **`R_Y(X_k)`** rotation, where the rotation angle `X_k` is the rescaled classical data component (Appendix A, p. 16: "The rotation angles `X_Θ` represent the mapped classical data rescaled to the interval [0, 2π]"; Fig. 6 inset shows `R_Y(X_0..X_Θ)`).
  3. then a block of **CNOT entangling gates** ("Fully entangling feature map, in which CNOT gates entangle pairs of qubits", Sec. III p. 6) connecting qubits, followed by a **second column of `R_Y(X_k)`** rotations (Fig. 23 shows H, R_Y, CNOT cascade, R_Y — two rotation layers around the entangler).
- **Data scaling:** classical input rescaled to `[0, 2π]` and used as rotation angles (Appendix A caption note, p. 16). Note Table II/III/IV list input scaling `σ_in` as `-` (dash) for Quantum RC — i.e. classical-style input scaling is **not a QRC hyperparameter**; scaling is the fixed `[0,2π]` angle map (Tables II p. 10, III p. 11, IV p. 14).
- "(x2)": the entire `Φ` (Fig. 23) circuit is applied **twice in sequence** (Table I QRC-C4, p. 7; text p. 6 "applying `Φ` twice").
- Exact CNOT pairing/topology of the "fully entangled" map: Fig. 23 (p. 16) shows a cascade of controls from each upper qubit onto lower qubits (a fully-connected CNOT pattern). The precise per-pair control/target ordering for arbitrary `n` beyond what Fig. 23 depicts is **partially shown only schematically (q_0..q_3, q_n with ellipsis)** — exact generalization rule to arbitrary n is **UNSPECIFIED IN SOURCE** (Fig. 23 uses "⋮" for qubits 4..n−1).

---

## 3. Fixed (untrained) reservoir circuit — which params are fixed vs trained

- **`P` (recurrence unitary): REMOVED (= identity)** for `rf_qrc` — not present, not trained (Sec. III p. 6; Fig. 6 caption p. 8; Table I QRC-C4 "Reservoir states (P)" column = `-`, p. 7).
- **`Φ` (feature map): FIXED structure, data-dependent, NOT trained.** Its rotation angles are deterministic functions of the (rescaled) input data, not free parameters (Appendix A p. 16; Algorithm 2 step 2 p. 5).
- **`V(α)` (variation unitary): random, FIXED, NOT trained.** QRC-C4 Variation (V) = "Fully entangled symmetric" (Table I, p. 7; this is the Fig. 24 "fully entangled symmetric" circuit, Appendix A p. 16). The `n` parameters `α` are **sampled once from a uniform distribution on `[0, 4π]` with a predefined seed and kept fixed throughout training and prediction** (Sec. IV B 1, p. 8: "we sample `V(α) ∈ ℝ^n` from a uniform distribution interval `[0, 4π]` with a predefined seed, which we keep fixed throughout the training and prediction ... This is similar to the classical input `W_in` and reservoir weight matrices `W`, which are also pseudo-randomly generated and fixed").
- **Only the linear readout `W_out` is trained** (Sec. III p. 6; Sec. II C p. 5; Algorithm 2 steps 5–6 p. 5; Fig. 5(a) p. 7).

Summary: the *entire quantum circuit is fixed/random* (no trained quantum gate parameters); the only trainable object is the classical output matrix `W_out`.

---

## 4. Measurement / readout observables & linear-readout training

- **Measurement:** measure the quantum reservoir state in the **Pauli-Z basis** to obtain a classical reservoir vector (Algorithm 2 step 5, p. 5: "Measure the quantum circuit in a Pauli-Z basis to convert quantum reservoir state `|ψ(t_{i+1})⟩` to classical reservoir state `r(t_{i+1})`"). Pre-processed reservoir vector `r̂(t_{i+1}) ∈ ℝ^{2^n}` derived from measured probabilities of `|ψ(t_{i+1})⟩` in the computational basis (Sec. II C, p. 5: "we derive a pre-processed reservoir state vector `r̂(t_{i+1})` from measured probabilities ... in the computational basis. This requires sampling and measuring the state multiple times"). Reservoir vector dimension is `2^n` (Fig. 6, p. 8: `r̂(t_{i+1}) ∈ ℝ^{2^n}`).
- **Post-processing:** the leaky-integrator update Eq. (2), `r(t_{i+1}) = (1−ε) r(t_i) + ε r̂(t_{i+1})`, with leak rate `ε` (Eq. 2 p. 3; applied to QRC per Sec. II C p. 5; "also known as the leaky-integral reservoir computing approach", p. 6). Reservoir state matrix `R` formed by concatenating reservoir vectors over training steps (Sec. II C, p. 5).
- **Symmetry-break bias:** a constant output bias of 1 is appended, `r → [r, 1]` (Sec. II A / II C, p. 4: "we add a constant output bias of 1 ... effectively replacing `r → [r, 1]`").
- **Readout training = closed-form ridge (Tikhonov) regression.** Solve the linear system (Eq. 3, p. 4):
  `(R Rᵀ + β I) W_out = R U_dᵀ`,
  where `β` is the user-defined Tikhonov regularization parameter, `I` the identity, `U_dᵀ ∈ ℝ^{N_u × N_tr}` the horizontal concatenation of target output data (Eq. 3, p. 4). This is an explicit closed-form ridge-regression solve, **not** iterative least squares / backprop (Sec. II A p. 3 "minimized through a simple linear ridge regression"; Sec. III/IV).
- **Prediction:** `u_p(t_{i+1}) = [r(t_{i+1})]ᵀ W_out` (Eq. 4, p. 4). Open-loop for one-step-ahead / training; closed-loop (recursive) for autonomous forecasting (Eq. 4 p. 4; Algorithm 2 step 6 p. 5; Fig. 5 p. 7).
- `β` values used: `1e-6, 1e-9, 1e-12` (Tables II p. 10, III p. 11, IV p. 14, "Tikhonov regularization" row, Quantum RC column).

---

## 5. Depth / width / reservoir-size scaling rule

- **Width:** reservoir state-vector dimension = `2^n` for `n` qubits (Sec. II C p. 5; Fig. 6 p. 8 `ℝ^{2^n}`). "The reservoir size corresponds to the state-vector dimension, which scales exponentially with the number of qubits (n)" (Sec. IV B 1, p. 9). Reservoir sizes N studied: Lorenz-63 N=32..2048 (n=5..11) (Fig. 8 p. 9); Lorenz-96 N=256..4096 (n=8..12) (Fig. 12 p. 11); MFE n=8..11 (p. 13).
- **Depth scaling rule (the RF-QRC headline):** because `P` is removed, **circuit depth is independent of reservoir size / qubit count** (Sec. III p. 6: depths "independent of the reservoir size"; Fig. 7 p. 9 shows QRC-C4 depth flat vs n=8..12 while recurrent ansätze C1/C2 grow exponentially; Conclusion p. 15: "small circuit depth that does not scale with the number of qubits"). Recurrent configs (QRC-C1, C2) instead require exponentially higher depth with reservoir size (Sec. IV C 1, p. 11).
- Exact closed-form depth formula (gate count as function of n) for the Fig. 23 fully-entangled (x2) map: **UNSPECIFIED IN SOURCE** (only the qualitative "independent of reservoir size" claim and the Fig. 7 bar chart are given; no algebraic depth expression).

---

## 6. Known-limit / unit-test hooks (checkable assertions, all PDF-grounded)

1. **Trained-parameter count = readout only.** The number of trained parameters equals `size(W_out)` only; *zero* quantum gate parameters are trained (Sec. III p. 6; Sec. IV B 1 p. 8; Algorithm 2 p. 5). With output bias, `W_out` has shape `((2^n + 1) × N_u)` — input vector is `[r, 1]` with `r ∈ ℝ^{2^n}` (Eq. 3/4 p. 4; Sec. II C p. 4–5; Fig. 6 `ℝ^{2^n}` p. 8). A test can assert: number of trainable params == `(2^n + 1) * N_u` and the quantum-circuit parameter set is fixed/non-differentiated.
2. **`V(α)` shape & fixedness.** `α ∈ ℝ^n` (one parameter per qubit), drawn uniform on `[0, 4π]` from a fixed seed, identical across training and prediction (Sec. IV B 1, p. 8). Test: `len(alpha) == n_qubits`, `0 ≤ α_i ≤ 4π`, and α is byte-identical between train and predict phases.
3. **Recurrence absent.** For `rf_qrc`, the `P` unitary must be the identity (no previous-reservoir-state encoding gates) and there must be **no feedback of `r(t_i)` into the circuit** (Table I QRC-C4 P-column = `-`, p. 7; Fig. 6 caption p. 8). Test: circuit applied at step i depends only on `u_in(t_i)` (and fixed α), not on `r(t_{i-1})`.
4. **Feature map applied exactly twice.** The `Φ` (Fig. 23) sub-circuit appears exactly 2× per time step ("(x2)", Table I p. 7; p. 6). Test: count of feature-map blocks per step == 2.
5. **Closed-form ridge solve.** `W_out` satisfies `(R Rᵀ + β I) W_out = R U_dᵀ` exactly (Eq. 3, p. 4) — test residual of the normal equation ≈ 0 for the produced `W_out`.
6. **Encoding-angle range.** Data-encoding `R_Y` angles lie in `[0, 2π]` (Appendix A, p. 16). Test: all feature-map rotation angles ∈ [0, 2π].

---

## Unspecified items (explicitly NOT guessed)

- Exact CNOT control/target wiring of the "fully entangled" map (Fig. 23) generalized to arbitrary `n` — **UNSPECIFIED IN SOURCE** (figure uses ellipsis for qubits 4..n−1).
- Closed-form circuit-depth / gate-count formula as a function of `n` — **UNSPECIFIED IN SOURCE** (only qualitative "independent of reservoir size" + Fig. 7 bars).
- Exact gate sequence of the `V(α)` "fully entangled symmetric" block (QRC-C4 V = "Fully entangled symmetric", Table I p. 7; corresponds to Fig. 24) beyond "fully entangled symmetric ... differs from (c) by an additional data encoding layer following the CNOT gates" (Sec. III, p. 6) — finer per-gate ordering for arbitrary n is **schematic only (Fig. 24)**.
- Number of measurement shots used in the emulated (noise-free Qiskit) results: paper states physical implementations would need `10^4–10^6` shots but the reported results use *noise-free* Qiskit statevector emulation (Sec. IV p. 6, Sec. III D p. 6) — exact emulation shot count is **not applicable / UNSPECIFIED** for the reported numbers.

---
*Every line above cites a specific PDF location (section / equation / figure / table / page). No abstract-only or memory-derived claims. No model training priors used.*
