# Circuit search — per-axis ablation (proxy budget: single seed, h=3)

Each row is a 1-seed run at the locked h=3 evaluation. The *reference* row is the historical 4q/3L data-reuploading/ring/RX circuit (matches `results/qlnn_hybrid_h3/seed_0/`).

| Axis | Level | Ansatz | Q | L | Test MAE | Test RMSE | Test R² | Test ΔOD R² |
|---|---|---|---|---|---|---|---|---|
| reference | data_reuploading_4q3L_ring_rx ★ | data_reuploading | 4 | 3 | 0.2655 | 0.2934 | 0.0133 | -2.9611 |
| ansatz_family | brickwall | brickwall | 4 | 3 | 0.2689 | 0.2977 | -0.0151 | -3.0750 |
| depth | L1 | data_reuploading | 4 | 1 | 0.2648 | 0.2929 | 0.0171 | -2.9459 |
| depth | L2 | data_reuploading | 4 | 2 | 0.2664 | 0.2945 | 0.0064 | -2.9889 |
| depth | L5 | data_reuploading | 4 | 5 | 0.2466 | 0.2771 | 0.1204 | -2.5310 |
| encoding | ry | data_reuploading | 4 | 3 | 0.2699 | 0.2983 | -0.0196 | -3.0931 |
| entanglement | all_to_all | data_reuploading | 4 | 3 | 0.2681 | 0.2962 | -0.0052 | -3.0353 |
| entanglement | linear | data_reuploading | 4 | 3 | 0.2663 | 0.2941 | 0.0088 | -2.9791 |
| promoted | top1_data_reuploading_Q4_L5 | data_reuploading | 4 | 5 | 0.2638 | 0.2919 | 0.0227 | -2.9233 |
| promoted | top2_hardware_efficient_Q4_L3 | hardware_efficient | 4 | 3 | 0.2661 | 0.2941 | 0.0082 | -2.9817 |
| promoted | top3_strongly_entangling_Q6_L3 | strongly_entangling | 6 | 3 | 0.2555 | 0.2829 | 0.0767 | -2.7066 |
| qubits | Q2 | data_reuploading | 2 | 3 | 0.2719 | 0.3005 | -0.0343 | -3.1520 |
| qubits | Q6 | data_reuploading | 6 | 3 | 0.2918 | 0.3180 | -0.1586 | -3.6513 |
| unknown | trial_0000 | unknown | -1 | -1 | 0.2638 | 0.2916 | 0.0257 | -2.9114 |
| unknown | trial_0001 | unknown | -1 | -1 | 0.2641 | 0.2920 | 0.0233 | -2.9209 |
| unknown | trial_0002 | unknown | -1 | -1 | 0.2639 | 0.2918 | 0.0244 | -2.9165 |
| unknown | trial_0003 | unknown | -1 | -1 | 0.2639 | 0.2918 | 0.0244 | -2.9166 |
| unknown | trial_0004 | unknown | -1 | -1 | 0.2881 | 0.3160 | -0.1439 | -3.5924 |
| unknown | trial_0005 | unknown | -1 | -1 | 0.2622 | 0.2898 | 0.0380 | -2.8619 |
| unknown | trial_0006 | unknown | -1 | -1 | 0.2687 | 0.2970 | -0.0108 | -3.0579 |
| unknown | trial_0007 | unknown | -1 | -1 | 0.2663 | 0.2941 | 0.0088 | -2.9791 |
| unknown | trial_0008 | unknown | -1 | -1 | 0.2652 | 0.2934 | 0.0136 | -2.9598 |
| unknown | trial_0009 | unknown | -1 | -1 | 0.2623 | 0.2899 | 0.0370 | -2.8660 |
| unknown | trial_0010 | unknown | -1 | -1 | 0.2699 | 0.2983 | -0.0196 | -3.0931 |
| unknown | trial_0011 | unknown | -1 | -1 | 0.2641 | 0.2920 | 0.0233 | -2.9209 |
| unknown | trial_0012 | unknown | -1 | -1 | 0.2881 | 0.3160 | -0.1439 | -3.5924 |
| unknown | trial_0013 | unknown | -1 | -1 | 0.2648 | 0.2929 | 0.0171 | -2.9459 |
| unknown | trial_0014 | unknown | -1 | -1 | 0.2629 | 0.2905 | 0.0334 | -2.8806 |
| unknown | trial_0015 | unknown | -1 | -1 | 0.2612 | 0.2886 | 0.0455 | -2.8318 |
| unknown | trial_0016 | unknown | -1 | -1 | 0.2629 | 0.2908 | 0.0309 | -2.8905 |
| unknown | trial_0017 | unknown | -1 | -1 | 0.2645 | 0.2922 | 0.0214 | -2.9284 |
| unknown | trial_0018 | unknown | -1 | -1 | 0.2720 | 0.3012 | -0.0392 | -3.1719 |
| unknown | trial_0019 | unknown | -1 | -1 | 0.2689 | 0.2977 | -0.0151 | -3.0750 |
| unknown | trial_0020 | unknown | -1 | -1 | 0.2689 | 0.2977 | -0.0151 | -3.0750 |
| unknown | trial_0022 | unknown | -1 | -1 | 0.2689 | 0.2977 | -0.0151 | -3.0750 |
| variational | rot_strongly_entangling | strongly_entangling | 4 | 3 | 0.2623 | 0.2901 | 0.0358 | -2.8707 |
| variational | ry_rz_hardware_efficient | hardware_efficient | 4 | 3 | 0.2610 | 0.2884 | 0.0468 | -2.8267 |
