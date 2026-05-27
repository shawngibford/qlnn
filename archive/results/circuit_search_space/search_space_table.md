# Quantum circuit search space (parametric)

Distinct QLNN circuit **topologies** in the unified matrix. A regime (R0–R3) is a *training* hyperparameter — it does NOT change the circuit diagram — so the 16 family×regime cells collapse to 4 baseline topologies; the rest are the axis-ablation + dedup'd Optuna + promoted topologies.

## Search axes

| Axis | Levels |
|---|---|
| ansatz family | brickwall, data_reuploading, hardware_efficient, strongly_entangling |
| num_qubits | 2, 4, 6 |
| num_layers | 1, 2, 3, 5 |
| encoding | rx, ry |
| entanglement | all_to_all, linear, ring, template (`template` = ansatz controls it internally) |

**28 distinct circuit topologies** (supplement gallery renders each via qml.draw_mpl).

## Topology list

| # | Family | Qubits | Layers | Encoding | Entanglement |
|---|---|---|---|---|---|
| 1 | brickwall | 2 | 1 | rx | template |
| 2 | brickwall | 4 | 3 | rx | template |
| 3 | brickwall | 4 | 5 | ry | template |
| 4 | brickwall | 6 | 1 | ry | template |
| 5 | brickwall | 6 | 5 | ry | template |
| 6 | data_reuploading | 2 | 3 | rx | ring |
| 7 | data_reuploading | 4 | 1 | rx | ring |
| 8 | data_reuploading | 4 | 2 | rx | linear |
| 9 | data_reuploading | 4 | 2 | rx | ring |
| 10 | data_reuploading | 4 | 3 | rx | all_to_all |
| 11 | data_reuploading | 4 | 3 | rx | linear |
| 12 | data_reuploading | 4 | 3 | rx | ring |
| 13 | data_reuploading | 4 | 3 | ry | ring |
| 14 | data_reuploading | 4 | 5 | rx | ring |
| 15 | data_reuploading | 4 | 5 | ry | linear |
| 16 | data_reuploading | 6 | 1 | rx | linear |
| 17 | data_reuploading | 6 | 3 | rx | ring |
| 18 | hardware_efficient | 2 | 3 | rx | linear |
| 19 | hardware_efficient | 4 | 2 | ry | linear |
| 20 | hardware_efficient | 4 | 3 | rx | linear |
| 21 | hardware_efficient | 4 | 3 | rx | ring |
| 22 | hardware_efficient | 4 | 5 | rx | linear |
| 23 | strongly_entangling | 2 | 2 | rx | template |
| 24 | strongly_entangling | 4 | 1 | rx | template |
| 25 | strongly_entangling | 4 | 3 | rx | template |
| 26 | strongly_entangling | 4 | 3 | ry | template |
| 27 | strongly_entangling | 6 | 2 | ry | template |
| 28 | strongly_entangling | 6 | 3 | rx | template |
