# results/p3_9_pde_matrix/

P3.9 PDE multi-family matrix output. Adds qcpinn_2d,
te_qpinn_fnn_2d, te_qpinn_qnn_2d (the 3 PINN-style 2D ports)
alongside chebyshev_dqc_2d (already in results/p3_8_review/).
All at the audit-corrected configs (heat 1200, Burgers 1500,
AC 64×32×1800).

Figure: `paper/figures/fig_p3_9_pde_matrix.{png,pdf}`.
Closes the P3.8 audit coverage gap (PDE side was single-family).
NOT a paper claim — H1 verdict still requires P5's Neural-ODE.
