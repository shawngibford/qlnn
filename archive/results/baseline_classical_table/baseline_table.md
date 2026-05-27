# Classical baseline — 1h-ahead OD forecast

Dataset: qZETA bioreactor (778 rows). Splits: train=70%, val=15%, test=15% (chronological).
Window: 24 steps, horizon: 1 h. OD scaling: fixed MinMax [0.0, 3.8].
Metrics in raw OD units (MAE/RMSE) or unitless (R²). Mean ± std across seeds.

| Model | val MAE | val RMSE | val R² | test MAE | test RMSE | test R² | seeds |
|---|---|---|---|---|---|---|---|
| Persistence (OD(t+h)=OD(t)) | 0.0824 | 0.1044 | 0.8824 | 0.0934 | 0.1129 | 0.9052 | n/a |
| Linear extrapolation | 0.1210 | 0.1572 | 0.7335 | 0.1276 | 0.1631 | 0.8024 | n/a |
| Liquid-ODE (Euler, train-only OD) | 0.0785 ± 0.0070 | 0.0999 ± 0.0080 | 0.8918 ± 0.0171 | 0.0928 ± 0.0082 | 0.1128 ± 0.0097 | 0.9048 ± 0.0161 | 5 |
| Liquid-ODE (dopri5, train-only OD) | 0.0785 ± 0.0069 | 0.0999 ± 0.0079 | 0.8919 ± 0.0169 | 0.0929 ± 0.0081 | 0.1129 ± 0.0097 | 0.9047 ± 0.0160 | 5 |
| Liquid-ODE +physics (train-only OD) | 0.0753 ± 0.0061 | 0.0965 ± 0.0070 | 0.8992 ± 0.0147 | 0.0899 ± 0.0073 | 0.1094 ± 0.0089 | 0.9105 ± 0.0144 | 5 |
| Liquid-ODE (Euler, fixed [0,3.8] OD — leak sensitivity) | 0.0688 ± 0.0197 | 0.0879 ± 0.0234 | 0.9119 ± 0.0437 | 0.0741 ± 0.0150 | 0.0892 ± 0.0196 | 0.9386 ± 0.0256 | 5 |
