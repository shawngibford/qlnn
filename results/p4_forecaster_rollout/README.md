# results/p4_forecaster_rollout/

P4 forecaster autoregressive rollout output. 5 forecaster
families × 3 ODE systems × 3 seeds = 45 cells.

Per-cell metrics (pre-reg §5):
  - relative_l2 over rollout horizon (PRIMARY endpoint)
  - VPT (in Lyapunov times for Lorenz; physical-time for
    LV/VdP)
  - spectral_error (FFT PSD L2)
  - invariant_drift (LV only; others have no invariant)
  - persistence_floor_relative_l2 (context, NOT a win)

Figure: `paper/figures/fig_p4_forecaster_rollout.{png,pdf}`.
NOT yet H1 evidence — H1 is defined as the QLNN−NeuralODE
advantage gap (pre-reg §2 / §7); the mandatory Neural-ODE
baseline awaits P5.
