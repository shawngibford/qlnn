# results/p5_matched_baselines/

P5 mandatory baselines per pre-reg §6:
  - plain_neuralode (MANDATORY H1 contrast)
  - plain_mlp (capacity-matched classical control)
  - skyline (structural upper bound)

Combined with `results/p4_forecaster_rollout/` (QLNN data),
the H1 verdict module computes Δ_smooth − Δ_broad CI per
pre-reg §7. Output: `results/p5_h1_verdict/h1_analysis.json`.
