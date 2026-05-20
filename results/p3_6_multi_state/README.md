# results/p3_6_multi_state/

P3.6 multi-state ODE solver demo output. **NOT a paper claim. NOT H1
evidence.**

Per the P3.8 peer-review audit, three caveats:

1. **The Lorenz "universal failure" framing is misleading.** T=2 ≈ 1.8
   Lyapunov times (pre-reg specifies 10 LTE); chaos hasn't fully
   developed. relL2≈1.0 used the predict-zero floor, which is a
   strawman for chaotic attractors far from origin. P3.8 re-runs
   Lorenz at T=5.0 (~5.5 LTE) and adds the predict-mean baseline
   (see `results/p3_8_review/lorenz_*/`).

2. **H1 is QLNN−NeuralODE gap, not QLNN absolute performance.** No
   NeuralODE baseline exists in this phase. Treat these as descriptive
   regime-map data pending P5.

3. **Per-component scalar circuits = no quantum entanglement across
   state components.** For Lotka-Volterra (prey-predator coupled via
   rhs), the two component circuits are entirely independent. This
   is classical decomposition with quantum-evaluated components, not
   quantum-coupled multi-state dynamics — minimum-faithful extension
   per the plan, but the "quantum" claim is structurally weaker than
   the figure narrative might suggest.

Figure: `paper/figures/fig_p3_6_multi_state.{png,pdf}` (rendered by
`scripts/make_multi_state_figure.py`).
