# Empirical-Fisher effective dimension (Abbas et al. 2021)

Sample size n = 472. Trained-theta single-θ specialization.

| Model | D | mean d_norm | std | min | max | n_seeds |
|---|---|---|---|---|---|---|
| Classical Liquid-ODE (H=4) | 90 | 8.0290 | 1.2965 | 6.5461 | 9.8702 | 5 |
| QLNN (h=3) | 114 | 9.5144 | 4.7381 | 3.6760 | 14.7811 | 5 |

**Δd_norm = d(QLNN) − d(classical) = +1.4854**
**Pre-registered acceptance threshold (Claim 2): Δd_norm > 1.0 — MET.**

## Per-seed numbers

| Seed | classical d_norm | QLNN d_norm | Δ |
|---|---|---|---|
| 0 | 6.5461 | 6.0375 | -0.5086 |
| 1 | 9.8702 | 3.6760 | -6.1942 |
| 2 | 7.6995 | 14.7811 | +7.0816 |
| 3 | 7.2983 | 13.4992 | +6.2009 |
| 4 | 8.7310 | 9.5782 | +0.8472 |

## Monotonicity sanity check (d_norm vs n)

### classical_H4

| seed | n=100 | n=200 | n=350 | n=472 |
|---|---|---|---|---|
| 0 | 8.5726 | 7.1862 | 6.7123 | 6.5461 |
| 1 | 12.8854 | 10.7721 | 10.1339 | 9.8702 |
| 2 | 10.1209 | 8.4952 | 7.9173 | 7.6995 |
| 3 | 9.4932 | 7.9773 | 7.4736 | 7.2983 |
| 4 | 11.4524 | 9.5810 | 8.9600 | 8.7310 |

Monotonic increasing across n for every seed: **NO** (per-seed: {'0': False, '1': False, '2': False, '3': False, '4': False})

### qlnn_h3

| seed | n=100 | n=200 | n=350 | n=472 |
|---|---|---|---|---|
| 0 | 7.8788 | 6.6464 | 6.1857 | 6.0375 |
| 1 | 5.3579 | 4.2308 | 3.8291 | 3.6760 |
| 2 | 18.6589 | 15.9608 | 15.1223 | 14.7811 |
| 3 | 16.5160 | 14.3489 | 13.6298 | 13.4992 |
| 4 | 12.2034 | 10.3515 | 9.8155 | 9.5782 |

Monotonic increasing across n for every seed: **NO** (per-seed: {'0': False, '1': False, '2': False, '3': False, '4': False})

