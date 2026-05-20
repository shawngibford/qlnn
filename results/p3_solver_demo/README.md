# results/p3_solver_demo/

P3.5 visible-first-results sprint output. **NOT a paper claim. NOT H1
evidence.**

Per the P3.8 peer-review audit, three caveats apply to these numbers:

1. **No classical baseline.** The te_qpinn_fnn MAE≈0.0003 on logistic
   and qcpinn MAE≈0.0002 on expdecay are QLNN-vs-QLNN comparisons; we
   don't know if these are quantum-specific or just physics-informed-
   training benefits. P3.8 adds the missing classical MLP-PINN
   baseline (see `results/p3_8_review/`).

2. **qcpinn architecture disclosure.** qcpinn carries 706 classical
   pre/post-NN params alongside its 15 PQC. The figure shows them as a
   single quantum family, but they are a quantum-bottleneck-with-
   classical-wrapper architecture, not pure quantum. Compare apples-
   to-apples with the classical PINN in P3.8.

3. **te_qpinn_qnn "structural trainability ceiling" was observed in
   ONE (n=4, L=5, K=3) configuration.** Hyperparameter sweep deferred
   to P7's T3 triangulation. Reframe: observed-in-one-config, not
   proven structural.

n=3 seeds with t-critical 4.303 gives wide t-CI; the project's
canonical n=5 is reserved for P5/P6 published claims.

Figure: `paper/figures/fig_p3_solver_demo.{png,pdf}` (rendered by
`scripts/make_solver_demo_figure.py`).
