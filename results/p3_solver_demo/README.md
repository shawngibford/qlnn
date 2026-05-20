# results/p3_solver_demo/

P3.5 visible-first-results sprint output. NOT a paper claim.
Numbers are seed-dependent CPU JAX runs; not pinned by
`scripts/verify_paper_integrity.py`. The figure is rendered by
`scripts/make_solver_demo_figure.py` and writes to
`paper/figures/fig_p3_solver_demo.{png,pdf}`. Per-(family, ode)
`seeds_summary.json` mirrors the project's standard schema.
Per-seed `curves.npz` holds `t_eval`, `u_pred`, `exact`, and
`loss_history` for plotting.
