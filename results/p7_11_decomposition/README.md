# results/p7_11_decomposition/

P7.11 commit 3 — forecaster H1 with COMPLETE 2×2 mechanism decomposition.

Five paired-bootstrap H1 verdicts at n=9:
  - h1_combined.json              : QLNN − Neural-ODE  (pre-reg)
  - h1_quantum_via_ltc.json       : QLNN − classical_LTC
  - h1_liquid_via_classical.json  : classical_LTC − Neural-ODE
  - h1_liquid_via_quantum.json    : QLNN − non_liquid_QLNN  (NEW)
  - h1_quantum_via_nonliquid.json : non_liquid_QLNN − Neural-ODE (NEW)

Two algebraic identities (per-cell, exact):
  Δ_combined = Δ_quantum_via_ltc + Δ_liquid_via_classical
  Δ_combined = Δ_liquid_via_quantum + Δ_quantum_via_nonliquid

τ-isolation cross-check: do Δ_liquid_via_classical and
Δ_liquid_via_quantum agree in sign and magnitude? If yes, the
liquid-τ attribution is robust to which side of the 2×2 we
isolate it from. If not, that's itself a paper-worthy finding.
