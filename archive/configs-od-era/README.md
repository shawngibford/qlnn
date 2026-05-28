# OD-era configs (archived 2026-05-28)

The full pre-pivot `configs/` directory, archived as a single block:
**585 YAML files** for the OD bioreactor program (baseline configs,
horizon ablation, parameter sweeps, sample-efficiency cells, circuit
search, option-B search, the 530-file unified-matrix design).

None of these configs are referenced by any active script under
`scripts/` or any module under `src/qlnn_/` / `src/quantum_liquid_neuralode/`.
Preserved for reproducibility of the OD-era results in
`archive/results/` and for the historical record.

The current ODE/PDE benchmark does not consume YAML configs — per-cell
choices live in code (e.g.
`src/qlnn_/training/solver_demo.FAMILIES`,
`src/qlnn_/training/p3_8_review_demo.CORRECTED_PDE_CONFIGS`,
`src/qlnn_/training/p4_forecaster_demo.P4SweepConfig`).
