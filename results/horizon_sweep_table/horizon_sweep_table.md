# Horizon ablation — Liquid-ODE and friends

Per-horizon comparison of trained models against persistence and linear extrapolation. Dataset: qZETA bioreactor (778 rows); splits 70/15/15 (chronological); window_size=24, stride=1.

Persistence and linear-extrapolation rows are deterministic (no seeds). Model rows report mean ± std across the seeds listed in each `seeds_summary.json`. Cells marked `—` mean that horizon has no run for that model variant.

## Window counts per split

| Split | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| train | 500 | 472 | 436 | 385 |
| val | 88 | 76 | 58 | 22 |
| test | 86 | 71 | 53 | 17 |

Splits with fewer than 30 test windows yield unstable metrics — treat those columns as supplementary, not headline.

## Test R² (raw OD units)

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Persistence | 0.9052 | -0.0371 | -9.7136 | -977.0226 |
| Linear extrap. | 0.8024 | -1.1011 | -16.3523 | -752.9852 |
| Liquid-ODE (Euler) | 0.9048 ± 0.0161 | 0.1108 ± 0.1535 | -9.5004 ± 0.1624 | -999.8978 ± 74.0053 |

*Key observation:* persistence R² collapses as horizon grows. Trained-model rows that hold a meaningful R² where persistence falls off are the paper's discriminating claim.

## Test MAE (raw OD units)

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Persistence | 0.0934 | 0.2718 | 0.5660 | 1.0035 |
| Linear extrap. | 0.1276 | 0.2989 | 0.4747 | 0.5494 |
| Liquid-ODE (Euler) | 0.0928 ± 0.0082 | 0.2491 ± 0.0250 | 0.5600 ± 0.0049 | 1.0115 ± 0.0403 |

## Validation R² (raw OD units)

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Persistence | 0.8824 | -0.4737 | -31.2204 | -782.4877 |
| Linear extrap. | 0.7335 | -2.6242 | -61.4906 | -743.3292 |
| Liquid-ODE (Euler) | 0.8918 ± 0.0171 | -0.1763 ± 0.2360 | -29.6445 ± 0.9006 | -837.4915 ± 75.8824 |

## Validation MAE (raw OD units)

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Persistence | 0.0824 | 0.2234 | 0.4395 | 0.7891 |
| Linear extrap. | 0.1210 | 0.3254 | 0.5953 | 0.6145 |
| Liquid-ODE (Euler) | 0.0785 ± 0.0070 | 0.1992 ± 0.0208 | 0.4310 ± 0.0075 | 0.8124 ± 0.0384 |

