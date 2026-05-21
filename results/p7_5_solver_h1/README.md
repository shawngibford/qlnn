# results/p7_5_solver_h1/

**P7.5 PRIMARY H1 VERDICT** — pre-reg §7 gating task.

Pipeline:
  - QLNN best-ansatz solver relL²: read from results/p3_6_multi_state
    (4 families × 3 systems × 3 seeds = 36 cells already on disk)
  - Classical PINN solver baseline: trained here (physics-residual
    Lagaris hard-IC, capacity-matched MLP)
  - H1 verdict: paired-bootstrap CI of (Δ_smooth − Δ_broad) per
    pre-reg §7

Outputs:
  - h1_analysis_solver_task.json (PRIMARY at skyline_threshold=0.5)
  - h1_analysis_solver_task_sensitivity.json (at 0.75)
  - per_cell_records.json (9 (system, seed) Δ records)
  - {system}_classical_pinn/seed_N/ — per-cell baseline metrics

The forecaster-task H1 from results/p5_h1_verdict is now
corroborating evidence per pre-reg §7 gating rule.
