# P3a Dual-Check — Foundational Families (independent re-derivation)

Sources read (ar5iv full-text HTML, not abstracts):
- chebyshev_dqc: https://ar5iv.labs.arxiv.org/html/2011.10395 (Kyriienko et al., PRA 103 052416, 2021)
- lubasch_multicopy: https://ar5iv.labs.arxiv.org/html/1907.09032 (Lubasch et al., PRA 101 010301(R), 2020)
- data_reuploading: https://ar5iv.labs.arxiv.org/html/2008.08605 (Schuld, Sweke, Meyer, PRA 103 032430, 2021)

NOTE: No refs/_speccard_*.md file was read. All claims cited to section/eq/figure.

---

## 1. chebyshev_dqc (arXiv:2011.10395)

- Feature map (plain Chebyshev, Eq. 14): `Û_φ(x) = ⊗_{j=1}^N R̂_{y,j}(2 arccos x)` — homogeneous degree-1 angle.
- Chebyshev *tower* feature map (Eq. 15): `Û_φ(x) = ⊗_{j=1}^N R̂_{y,j}(2j·arccos x)` — angle = `2j·arccos(x)`, qubit j carries Chebyshev degree n=j; tower degree scales **linearly with qubit index** (Methods III.1). Euler/Chebyshev expansion (Eq. 12): `R̂_y(2n arccos x) = T_n(x)·𝟙 + √(1-x²)·U_{n-1}(x)·X̂Ẑ`.
- Variational ansatz: hardware-efficient ansatz (Fig. 5a, Methods III.2), d repeated blocks of per-qubit `R_z–R_x–R_z` + nearest-neighbour CNOT entangling layer.
- Depth/width: N=6 main; ansatz depth d∈{3,6,12,24}; feature map = single rotation layer.
- Readout (Eq. 1): `f(x) = ⟨f_{φ,θ}(x)|Ĉ|f_{φ,θ}(x)⟩`, Ĉ = Σ_j Ẑ_j (total Z magnetization).
- Derivative/loss: feature-variable derivative via shift rule (Eq. 9): `d/dx ⟨Ĉ⟩ = ¼·(dφ/dx)·(⟨Ĉ⟩⁺ − ⟨Ĉ⟩⁻)`. DE residual (Eq. 3) `F[{d^m f_n/dx^m}, {f_n}, x]=0`; loss (Eq. 19–20) `ℒ = ℒ^(diff)+ℒ^(boundary)`, `ℒ^(diff)=(1/M)Σ_i L(F[...],0)`; boundary (Eq. 21) `ℒ^(boundary)=η·L(f(x₀),u₀)`.
- Unit-test hook: feed x→0 into tower map with degree-2 qubit; `R_y(2·2·arccos 0)=R_y(2π)` ⇒ T_2(0)=−1, U_1(0)=0 — check ⟨Z⟩ recovers T_n(x) per Eq. 12.

## 2. lubasch_multicopy (arXiv:1907.09032)

- Feature map: amplitude encoding (Eq. 5) `|ψ⟩ = Σ_{k=0}^{N-1} ψ_k |binary(k)⟩`, N=2^n.
- Ansatz: parametrized U(λ) (Fig. 1b), depth d=5, n=6; layered two-qubit gates, params λ set gate rotations; `|ψ(λ)⟩=Û(λ)|0⟩`.
- Depth/width: d∝poly(n), #params ∝ n·d; QNPU depth bound `d_> ≤ 9n[(23/48)(2χ)²+4/3]`.
- Readout: ancilla Pauli-z expectation; nonlinear `⟨I⟩=g⟨σ̂_z⟩^I/2h_N`, kinetic `⟨K⟩=(1−⟨σ̂_z⟩^K)/h_N²`, potential `⟨P⟩=α⟨σ̂_z⟩^P`.
- Multi-copy nonlinear term (QNPU, Fig. 1a/2a): r identical copies of |ψ(λ)⟩ (some via Û*) coupled by CNOTs giving point-wise products `F = f^{(1)*} ∏_{j=1}^r (O_j f^{(j)})`; Hadamard-test ancilla extracts `Σ_k Re{F_k}` (e.g. Σ_k|ψ_k|⁴) in one circuit.
- Unit-test hook: r=1 (no extra copy, identity O) ⇒ QNPU reduces to a normal overlap/Hadamard test; ancilla ⟨σ_z⟩ must equal Re⟨ψ|ψ⟩=1 for normalized state.

## 3. data_reuploading (arXiv:2008.08605)

- Encoding (Sec. I/II.1): `𝒢(x)=e^{-ixH}`, Pauli case `H=½σ`.
- L-layer model (Eq. 4): `U(x)=W^{(L+1)} S(x) W^{(L)} … W^{(2)} S(x) W^{(1)}` — **S(x) identical every layer**, W^{(i)} arbitrary trainable.
- Output (Eq. 3): `f_θ(x)=⟨0|U†(x,θ) M U(x,θ)|0⟩`, M arbitrary observable.
- Fourier result (Eq. 11): `f(x)=Σ_{ω∈Ω} c_ω e^{iωx}`; spectrum (Eq. 10) `Ω={Λ_k−Λ_j}`, Λ = sums of encoding-Hamiltonian eigenvalues. L repeats of one Pauli encoding (Sec. II.2) ⇒ integer spectrum `Ω={−L,…,−1,0,1,…,L}` (degree-L truncated Fourier series, ~Eq. 26).
- Unit-test hook: single qubit, L layers, Pauli encoding ⇒ model must be exactly band-limited to |ω|≤L; FFT of f(x) over [0,2π] should show zero power at |ω|>L.

---

## reuploading.py vs Schuld construction — independent verdict

**Verdict: YES (structurally matches Schuld Eq. 4, with one frequency-spectrum caveat).**

- Lines 120–132: per-layer loop emits encoding `qml.RX(inputs[i])` (lines 124–125) → trainable `qml.Rot` (128–129) → entangler (132). This is exactly the alternating `S(x)·W` block pattern of **Eq. 4** (`U(x)=W^{(L+1)}S(x)…S(x)W^{(1)}`).
- Encoding is **identical at every layer**: same `qml.RX(inputs[i])`, no per-layer scaling (lines 124–125) — satisfies Schuld's requirement that S(x) be repeated unchanged (Eq. 4).
- Generator = Pauli-X (`RX` ⇒ H=½σ_x), eigenvalues ±½ ⇒ per-feature integer-spaced spectrum `Ω={−L,…,L}` (Sec. II.2). Matches the integer-frequency Fourier construction.
- Caveat (not a mismatch): the trailing entangling CNOT block sits *after* the last layer's variational gates but there is **no final W^{(L+1)} trainable block after the last S(x)**; Eq. 4 places a trailing W^{(L+1)}. The code's structure is `[S W E]×L`, i.e. the last block ends in an entangler, not a trainable rotation. This narrows expressivity slightly vs the canonical Eq. 4 but does not change the accessible spectrum Ω. The docstring (lines 3–5) miscredits Pérez-Salinas 1907.02085 for the *architecture*; the integer-Fourier expressivity claim it makes (lines 7–9) is the Schuld result and is correctly attributed.

## UNSPECIFIED / discrepancies

- lubasch_multicopy: the explicit two-qubit gate decomposition of U(λ) is **UNSPECIFIED IN SOURCE** beyond "values λ determine the two-qubit gates" (Fig. 1b); no gate-level rotation formula given.
- chebyshev_dqc: "sparse Chebyshev" map referenced near Eq. 14 but **no separately numbered formula** — UNSPECIFIED IN SOURCE.
- data_reuploading: exact internal gate decomposition of trainable W^{(i)} is **UNSPECIFIED IN SOURCE** (W^{(i)} stated only as "arbitrary unitary").
- reuploading.py: missing trailing W^{(L+1)} block vs Schuld Eq. 4 (see caveat above).
