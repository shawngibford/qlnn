# results/p7_6_pde_solver_h1/

P7.6 PDE solver-task H1 verdict (zero new compute — all data
already on disk from P3.7-3.9 + P3.8). Plus the **combined
ODE + PDE** H1 verdict at n=18 cells.

Data sources (all read-only):
  - QLNN: results/p3_9_pde_matrix/ + results/p3_8_review/
  - classical PINN: results/p3_8_review/
  - ODE side: results/p7_5_solver_h1/per_cell_records.json

Outputs:
  - h1_analysis_pde_solver.json     (PDE-only, n=9)
  - h1_analysis_combined_solver.json (ODE+PDE, n=18)
  - per_cell_records.json           (all 18 cells)
