# Param-matched classical Liquid-ODE sweep (h=3)

Tier 2 #2.1 from the peer-review swarm: classical hidden_size ∈ {2,4,8,16,32} vs the QLNN (~100 params). All rows use the same h=3 protocol (`configs/param_sweep/*.yaml`), 5 seeds each.

| Model | hidden_size | params | test MAE_raw (mean ± std) | test R²_raw (mean ± std) | test ΔR²_raw (mean ± std) |
|---|---|---|---|---|---|
| classical | 2 | 42 | 0.2449 ± 0.0224 | 0.1564 ± 0.1545 | -2.3867 ± 0.6201 |
| classical | 4 | 90 | 0.2594 ± 0.0166 | 0.0534 ± 0.1211 | -2.8002 ± 0.4862 |
| classical | 8 | 210 | 0.2581 ± 0.0168 | 0.0579 ± 0.1207 | -2.7819 ± 0.4846 |
| classical | 16 | 546 | 0.2564 ± 0.0204 | 0.0777 ± 0.1501 | -2.7027 ± 0.6026 |
| classical | 32 | 1602 | 0.2491 ± 0.0250 | 0.1108 ± 0.1535 | -2.5698 ± 0.6163 |
| qlnn | — | -1 | 0.2655 ± 0.0000 | 0.0133 ± 0.0000 | -2.9611 ± 0.0000 |
