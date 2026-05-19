# Master comparison — all classical + quantum configurations

G1 accuracy bar = classical H=4 MAE **0.2594**; G2 σ gate = **0.00831** (½·σ_classical_H4 = ½·0.01662). Ranked by test MAE. σ-ratio = σ_classical_H4 / σ_row (Claim-1 needs ≥ 2×).

| Rank | Source | Config | Params | Test MAE | σ | σ-ratio | Test R² | G1 | G2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | classical_sweep | classical_best_param_sweep_cell_H2 | 42 | 0.2449 | 0.02236 | 0.74 | — | ✅ | ❌ |
| 2 | prior_search/depth | data_reuploading_depth_5 | — | 0.2466 | — | — | 0.120 | ✅ | ❌ |
| 3 | prior_search/promoted | strongly_entangling_top3_strongly_entangling_Q6_L3 | — | 0.2555 | 0.02529 | 0.66 | 0.077 | ✅ | ❌ |
| 4 | classical_sweep | classical_matched_param_H4 | 90 | 0.2594 | 0.01662 | 1.00 | — | ❌ | ❌ |
| 5 | prior_search/variational | hardware_efficient_variational_hardware_efficient_ry_rz | — | 0.2610 | — | — | 0.047 | ❌ | ❌ |
| 6 | prior_search/unknown | unknown_trial_0015 | — | 0.2612 | — | — | 0.046 | ❌ | ❌ |
| 7 | prior_search/unknown | unknown_trial_0005 | — | 0.2622 | — | — | 0.038 | ❌ | ❌ |
| 8 | prior_search/variational | strongly_entangling_variational_strongly_entangling | — | 0.2623 | — | — | 0.036 | ❌ | ❌ |
| 9 | prior_search/unknown | unknown_trial_0009 | — | 0.2623 | — | — | 0.037 | ❌ | ❌ |
| 10 | prior_search/unknown | unknown_trial_0016 | — | 0.2629 | — | — | 0.031 | ❌ | ❌ |
| 11 | prior_search/unknown | unknown_trial_0014 | — | 0.2629 | — | — | 0.033 | ❌ | ❌ |
| 12 | prior_search/promoted | data_reuploading_top1_data_reuploading_Q4_L5 | — | 0.2638 | 0.01250 | 1.33 | 0.023 | ❌ | ❌ |
| 13 | prior_search/unknown | unknown_trial_0000 | — | 0.2638 | — | — | 0.026 | ❌ | ❌ |
| 14 | prior_search/unknown | unknown_trial_0002 | — | 0.2639 | — | — | 0.024 | ❌ | ❌ |
| 15 | prior_search/unknown | unknown_trial_0003 | — | 0.2639 | — | — | 0.024 | ❌ | ❌ |
| 16 | prior_search/unknown | unknown_trial_0001 | — | 0.2641 | — | — | 0.023 | ❌ | ❌ |
| 17 | prior_search/unknown | unknown_trial_0011 | — | 0.2641 | — | — | 0.023 | ❌ | ❌ |
| 18 | prior_search/unknown | unknown_trial_0017 | — | 0.2645 | — | — | 0.021 | ❌ | ❌ |
| 19 | prior_search/depth | data_reuploading_depth_1 | — | 0.2648 | — | — | 0.017 | ❌ | ❌ |
| 20 | prior_search/unknown | unknown_trial_0013 | — | 0.2648 | — | — | 0.017 | ❌ | ❌ |
| 21 | prior_search/unknown | unknown_trial_0008 | — | 0.2652 | — | — | 0.014 | ❌ | ❌ |
| 22 | prior_search/reference | data_reuploading_reference | — | 0.2655 | 0.00437 | 3.80 | 0.013 | ❌ | ✅ |
| 23 | qlnn_reference | data_reuploading_4q3L_ring_rx | 114 | 0.2655 | 0.00437 | 3.81 | — | ❌ | ✅ |
| 24 | prior_search/promoted | hardware_efficient_top2_hardware_efficient_Q4_L3 | — | 0.2661 | 0.00846 | 1.96 | 0.008 | ❌ | ❌ |
| 25 | prior_search/entanglement | data_reuploading_entanglement_linear | — | 0.2663 | — | — | 0.009 | ❌ | ❌ |
| 26 | prior_search/unknown | unknown_trial_0007 | — | 0.2663 | — | — | 0.009 | ❌ | ❌ |
| 27 | prior_search/depth | data_reuploading_depth_2 | — | 0.2664 | — | — | 0.006 | ❌ | ❌ |
| 28 | prior_search/entanglement | data_reuploading_entanglement_all_to_all | — | 0.2681 | — | — | -0.005 | ❌ | ❌ |
| 29 | prior_search/unknown | unknown_trial_0006 | — | 0.2687 | — | — | -0.011 | ❌ | ❌ |
| 30 | prior_search/ansatz_family | brickwall_ansatz_brickwall | — | 0.2689 | — | — | -0.015 | ❌ | ❌ |
| 31 | prior_search/unknown | unknown_trial_0019 | — | 0.2689 | — | — | -0.015 | ❌ | ❌ |
| 32 | prior_search/unknown | unknown_trial_0020 | — | 0.2689 | — | — | -0.015 | ❌ | ❌ |
| 33 | prior_search/unknown | unknown_trial_0022 | — | 0.2689 | — | — | -0.015 | ❌ | ❌ |
| 34 | prior_search/encoding | data_reuploading_encoding_ry | — | 0.2699 | — | — | -0.020 | ❌ | ❌ |
| 35 | prior_search/unknown | unknown_trial_0010 | — | 0.2699 | — | — | -0.020 | ❌ | ❌ |
| 36 | prior_search/qubits | data_reuploading_qubits_2 | — | 0.2719 | — | — | -0.034 | ❌ | ❌ |
| 37 | prior_search/unknown | unknown_trial_0018 | — | 0.2720 | — | — | -0.039 | ❌ | ❌ |
| 38 | prior_search/unknown | unknown_trial_0004 | — | 0.2881 | — | — | -0.144 | ❌ | ❌ |
| 39 | prior_search/unknown | unknown_trial_0012 | — | 0.2881 | — | — | -0.144 | ❌ | ❌ |
| 40 | prior_search/qubits | data_reuploading_qubits_6 | — | 0.2918 | — | — | -0.159 | ❌ | ❌ |

**0 configuration(s) pass BOTH G1 (accuracy) and G2 (reproducibility)** — i.e. the Option-B 'best for all' target at the proxy level.
