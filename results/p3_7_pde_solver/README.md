# results/p3_7_pde_solver/

P3.7 PDE solver demo output (three PDEs via the Chebyshev-DQC 2D
solver). **NOT a paper claim. NOT H1 evidence.**

Per the P3.8 peer-review audit, four caveats:

1. **H1 is QLNN−NeuralODE gap.** The "Allen-Cahn collapse confirms
   H1 PDE-side prediction" framing in earlier commits is INCORRECT.
   No NeuralODE baseline exists yet (scheduled for P5). Treat these
   as descriptive regime-map data pending the H1 contrast.

2. **Allen-Cahn was spatially under-resolved.** n_x_colloc=28 →
   Δx≈0.224 vs equilibrium front width √2·ε≈0.085. The solver had
   <1 collocation point per front — sub-Nyquist spatial aliasing.
   The "broadband failure" may be a resolution artifact, NOT a
   regime-structural property. P3.8 re-runs at n_x_colloc=64,
   n_t_colloc=32, steps=1800 (>10× the resolution-to-front ratio).

3. **Burgers smooth missed the convergence gate.** The plan's
   relL2<0.30 acceptance threshold was not met (P3.7 sweep landed at
   0.38). The "2.6× better than predict-zero floor" framing is true
   but doesn't excuse the gate miss. P3.8 re-runs at steps=1500
   (PDE_BENCH's configured budget; P3.7 sweep used 600).

4. **No classical PINN baseline.** Heat MAE=0.03 at 600 steps is
   mediocre by classical-PINN standards (typical: 1e-3 to 1e-5).
   The P3.8 smoke shows classical-PINN at matched capacity achieves
   relL2≈0.005 on heat, ~10× more accurate than the quantum solver.

Figure: `paper/figures/fig_p3_7_pde_solver.{png,pdf}` (rendered by
`scripts/make_pde_solver_figure.py`). The corrected version with
classical baseline + audit-corrected configs is at
`paper/figures/fig_p3_8_review_iteration.{png,pdf}`.
