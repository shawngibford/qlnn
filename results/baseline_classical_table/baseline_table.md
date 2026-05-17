# Classical baseline — 1h-ahead OD forecast

Dataset: qZETA bioreactor (778 rows). Splits: train=70%, val=15%, test=15% (chronological).
Window: 24 steps, horizon: 1 h. OD scaling: fixed MinMax [0.0, 3.8].
Metrics in raw OD units (MAE/RMSE) or unitless (R²). Mean ± std across seeds.

| Model | val MAE | val RMSE | val R² | test MAE | test RMSE | test R² | seeds |
|---|---|---|---|---|---|---|---|
| Persistence (OD(t+h)=OD(t)) | 0.0824 | 0.1044 | 0.8824 | 0.0934 | 0.1129 | 0.9052 | n/a |
| Linear extrapolation | 0.1210 | 0.1572 | 0.7335 | 0.1545 | 0.1930 | 0.7231 | n/a |
| Liquid-ODE (Euler) | 0.0688 ± 0.0178 | 0.0879 ± 0.0212 | 0.9118 ± 0.0395 | 0.0770 ± 0.0125 | 0.0927 ± 0.0164 | 0.9342 ± 0.0220 | 5 |
| Liquid-ODE (dopri5) | 0.0688 ± 0.0177 | 0.0878 ± 0.0211 | 0.9121 ± 0.0393 | 0.0771 ± 0.0123 | 0.0929 ± 0.0163 | 0.9339 ± 0.0219 | 5 |
| Liquid-ODE +physics | 0.0559 ± 0.0092 | 0.0728 ± 0.0123 | 0.9413 ± 0.0195 | 0.0615 ± 0.0035 | 0.0738 ± 0.0041 | 0.9594 ± 0.0046 | 5 |
