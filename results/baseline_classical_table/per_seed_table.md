# Per-seed results (supplementary)

Full per-seed metrics for every run aggregated by `baseline_table.md`. Use this for reviewer-side CIs and paired analyses.

| model | seed | best_epoch | val_mse_norm | val_mae_raw | val_rmse_raw | val_r2_raw | test_mse_norm | test_mae_raw | test_rmse_raw | test_r2_raw |
|---|---|---|---|---|---|---|---|---|---|---|
| Liquid-ODE (Euler, train-only OD) | 0 | 1 | 0.009924 | 0.070300 | 0.090654 | 0.911351 | 0.012387 | 0.083193 | 0.101281 | 0.923768 |
| Liquid-ODE (Euler, train-only OD) | 1 | 1 | 0.013769 | 0.084450 | 0.106780 | 0.877005 | 0.016878 | 0.096797 | 0.118223 | 0.896130 |
| Liquid-ODE (Euler, train-only OD) | 2 | 1 | 0.012394 | 0.080303 | 0.101309 | 0.889287 | 0.018177 | 0.101623 | 0.122690 | 0.888133 |
| Liquid-ODE (Euler, train-only OD) | 3 | 1 | 0.014114 | 0.085419 | 0.108111 | 0.873921 | 0.016943 | 0.097453 | 0.118451 | 0.895729 |
| Liquid-ODE (Euler, train-only OD) | 4 | 1 | 0.010374 | 0.072151 | 0.092686 | 0.907332 | 0.012931 | 0.084966 | 0.103482 | 0.920419 |
| Liquid-ODE (dopri5, train-only OD) | 0 | 1 | 0.009933 | 0.070325 | 0.090695 | 0.911270 | 0.012399 | 0.083256 | 0.101331 | 0.923692 |
| Liquid-ODE (dopri5, train-only OD) | 1 | 1 | 0.013723 | 0.084285 | 0.106601 | 0.877419 | 0.016852 | 0.096693 | 0.118131 | 0.896292 |
| Liquid-ODE (dopri5, train-only OD) | 2 | 1 | 0.012361 | 0.080179 | 0.101175 | 0.889580 | 0.018172 | 0.101568 | 0.122672 | 0.888166 |
| Liquid-ODE (dopri5, train-only OD) | 3 | 1 | 0.014112 | 0.085413 | 0.108102 | 0.873941 | 0.016998 | 0.097609 | 0.118644 | 0.895389 |
| Liquid-ODE (dopri5, train-only OD) | 4 | 1 | 0.010391 | 0.072219 | 0.092760 | 0.907183 | 0.012987 | 0.085169 | 0.103703 | 0.920077 |
| Liquid-ODE +physics (train-only OD) | 0 | 1 | 0.009509 | 0.068411 | 0.088738 | 0.915057 | 0.011760 | 0.081286 | 0.098682 | 0.927630 |
| Liquid-ODE +physics (train-only OD) | 1 | 1 | 0.012110 | 0.078421 | 0.100143 | 0.891821 | 0.015071 | 0.091165 | 0.111716 | 0.907249 |
| Liquid-ODE +physics (train-only OD) | 2 | 1 | 0.011306 | 0.075777 | 0.096760 | 0.899007 | 0.016809 | 0.097136 | 0.117980 | 0.896557 |
| Liquid-ODE +physics (train-only OD) | 3 | 1 | 0.013556 | 0.083536 | 0.105951 | 0.878909 | 0.016621 | 0.096460 | 0.117320 | 0.897710 |
| Liquid-ODE +physics (train-only OD) | 4 | 1 | 0.009939 | 0.070390 | 0.090724 | 0.911214 | 0.012425 | 0.083491 | 0.101434 | 0.923537 |
| Liquid-ODE (Euler, fixed [0,3.8] OD — leak sensitivity) | 0 | 1 | 0.000311 | 0.050898 | 0.067063 | 0.951485 | 0.000378 | 0.062215 | 0.073885 | 0.959431 |
| Liquid-ODE (Euler, fixed [0,3.8] OD — leak sensitivity) | 1 | 1 | 0.000840 | 0.087574 | 0.110148 | 0.869126 | 0.000699 | 0.081856 | 0.100499 | 0.924940 |
| Liquid-ODE (Euler, fixed [0,3.8] OD — leak sensitivity) | 2 | 1 | 0.000583 | 0.071810 | 0.091761 | 0.909171 | 0.000872 | 0.092814 | 0.112218 | 0.906413 |
| Liquid-ODE (Euler, fixed [0,3.8] OD — leak sensitivity) | 3 | 90 | 0.000842 | 0.087594 | 0.110237 | 0.868914 | 0.000622 | 0.077750 | 0.094805 | 0.933205 |
| Liquid-ODE (Euler, fixed [0,3.8] OD — leak sensitivity) | 4 | 1 | 0.000253 | 0.046186 | 0.060448 | 0.960584 | 0.000288 | 0.055752 | 0.064536 | 0.969048 |
