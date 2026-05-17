# Classical baseline — 1h-ahead OD forecast

Dataset: qZETA bioreactor (778 rows). Splits: train=70%, val=15%, test=15% (chronological).
Window: 24 steps, horizon: 1 h. OD scaling: fixed MinMax [0.0, 3.8].
Metrics in raw OD units (MAE/RMSE) or unitless (R²). Mean ± std across seeds.

| Model | val MAE | val RMSE | val R² | test MAE | test RMSE | test R² | seeds |
|---|---|---|---|---|---|---|---|
| Persistence (OD(t+h)=OD(t)) | 0.0824 | 0.1044 | 0.8824 | 0.0934 | 0.1129 | 0.9052 | n/a |
| Linear extrapolation | 0.1210 | 0.1572 | 0.7335 | 0.1545 | 0.1930 | 0.7231 | n/a |
| Liquid-ODE (Euler) | 0.0688 ± 0.0199 | 0.0879 ± 0.0237 | 0.9118 ± 0.0442 | 0.0770 ± 0.0139 | 0.0927 ± 0.0184 | 0.9342 ± 0.0246 | 5 |
| Liquid-ODE (dopri5) | 0.0688 ± 0.0198 | 0.0878 ± 0.0236 | 0.9121 ± 0.0439 | 0.0771 ± 0.0137 | 0.0929 ± 0.0182 | 0.9339 ± 0.0245 | 5 |
| Liquid-ODE +physics(logistic-only) | 0.0557 ± 0.0101 | 0.0724 ± 0.0133 | 0.9419 ± 0.0210 | 0.0613 ± 0.0041 | 0.0734 ± 0.0048 | 0.9598 ± 0.0054 | 5 |
