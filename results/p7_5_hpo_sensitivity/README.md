# results/p7_5_hpo_sensitivity/

P7.5 HPO sensitivity sweep. Closes audit concern Y3 (HPO
budget unfixed). Classical PINN baseline retrained at 3 LRs ×
2 train_steps at 3 anchor cells (LV s2, VdP s1, Lorenz s2),
while QLNN side stays at the fixed P3.6 multi_state config.

Verdict: if `overall_sign_stability == all_positive_across_all_cells`,
the SOLVER-task H1 CONFIRMED outcome is HPO-invariant — the
QLNN advantage holds at every (LR, steps) combination tested.
