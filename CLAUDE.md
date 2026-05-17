# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research-code Python package implementing a quantum-liquid neural ODE pipeline for generating synthetic bioreactor time-series data. The dataset is small (778 samples of optical density and related features). `spec.md` is the full design/roadmap document. Stack is hybrid by design: PyTorch for the classical baseline (this repo), and a separate JAX + Equinox + Diffrax + PennyLane `qlnn_/` subpackage (forthcoming) for the quantum side. The three paper contributions are (1) synthetic data lift via QWGAN-GP, (2) expressivity via Fisher / effective dimension, (3) sample efficiency.

## Commands

### Setup (Python 3.11)
```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev]"
```

### Tests
```bash
.venv/bin/python -m pytest                # all tests (configured with -q via pyproject.toml)
.venv/bin/python -m pytest tests/test_liquid_cell.py  # single file
.venv/bin/python -m pytest tests/test_liquid_cell.py::test_liquid_cell_gradients_flow  # single test
.venv/bin/python -m pytest -k liquid_cell  # keyword filter
```

### Build wheel
```bash
.venv/bin/python -m pip wheel . -w dist
```

### No linter/formatter configured
Only `pytest` is in dev dependencies. No Black, ruff, mypy, or isort.

## Evaluation protocol (locked — every model in the paper uses this)

- Dataset: `data/raw/qZETA_data_copy.csv`, 778 rows, time-ordered.
- Splits: train 70%, val 15%, test 15% (chronological).
- Task: 1-hour-ahead OD forecast from a 24-step history window, stride 1.
- OD scaling: fixed MinMax bounds [0.0, 3.8].
- Metrics: MAE_raw, RMSE_raw, R²_raw, MSE_norm (+ DTW for trajectory-level
  evaluation once QWGAN lands).
- Baselines reported every run: persistence + linear extrapolation.
- Seeds: {0,1,2,3,4}, report mean ± std.
- Selection: best val MSE_norm checkpoint.

Any change to this protocol breaks comparability across milestones.

## Architecture

**Package**: `src/quantum_liquid_neuralode/` (src-layout, import as `quantum_liquid_neuralode`).

- **models/** — `LiquidCell` returns dh/dt (caller integrates). `LiquidODForecaster` is the full forecast model: encodes the first window step, evolves over the history with per-step inputs, then over the 1h horizon with the last input held constant. Residual delta around persistence: `OD(t+h) = OD(t) + tanh(delta_head(h)) * delta_scale`. `ode_method` selects `"euler"` / `"rk4"` (fixed-step, `forecast_steps` sub-steps) or `"dopri5"` (adaptive via `torchdiffeq.odeint`).
- **training/** — Physics losses (`logistic_growth_residual_loss`, `smoothness_loss`) and the reusable training loop `train_one` (early-stopping, eval cadence, optional physics regularizers). `TrainerConfig` / `PhysicsLossConfig` make every knob explicit. Multi-seed orchestration lives in `scripts/train_baseline.py`.
- **data_processing/** — `load_qzeta` (canonical loader, handles `TEMP EXT` rename and `DATE` parsing), `time_hours_from_date`, `split_indices`, `fit_minmax`/`apply_minmax` (with optional fixed bounds for OD), `make_horizon_windows` (drops windows whose realized horizon isn't within tolerance). `BioreactorDataPreprocessor` is legacy.
- **evaluation/** — `compute_metrics` (returns the `ForecastMetrics` bundle: MAE_raw, RMSE_raw, R²_raw, MSE_norm), `aggregate_seed_metrics`, `persistence_forecast`, `linear_extrapolation_forecast`.
- **utils/** — `select_device()`: MPS → CUDA → CPU. Note: `torchdiffeq.odeint` requires CPU (MPS lacks some ops); fixed-step Euler/RK4 work on MPS.

Tests in `tests/` mirror these modules.

### Intended pipeline (from spec.md)
```
Input Sequence → [Quantum Feature Encoder] → Latent → [Liquid Neural ODE] → [Decoder] → Generated Sequence
```
Phase 1 (classical Liquid-ODE baseline) is implemented and produces canonical results under `results/baseline_classical_*`. Phase 2 (quantum encoder via PennyLane in JAX `qlnn_/`), Phase 3 (full hybrid), Phase 4 (QWGAN-GP synthetic generator), and Phase 5 (Fisher / effective-dimension expressivity diagnostics) are planned.

### Scripts
- `scripts/train_baseline.py` — **canonical multi-seed trainer**. Reads YAML from `configs/`, writes `seeds_summary.json` (paper-table row).
- `scripts/summarize_baselines.py` — emits paper-ready markdown table across runs.
- `scripts/train_liquid_od_baseline.py` — legacy monolithic trainer (kept for HPO continuity).
- `scripts/hpo_liquid_od_forecast.py` — hyperparameter optimization (random search).
- `scripts/explore_qzeta_dataset.py` — dataset exploration.
- `scripts/visualize_quantum_circuit.py` — PennyLane quantum circuit visualization.

### Configs
- `configs/baseline.yaml` — headline (dopri5, no physics).
- `configs/baseline_euler_fast.yaml` — fast Euler variant for smoke tests / MPS.
- `configs/baseline_physics.yaml` — physics-informed ablation.

### Quantum subpackage: `src/qlnn_/` (step 2+)

JAX + Equinox + PennyLane. Lives alongside the PyTorch package, shares data via numpy arrays at the module boundary.

- **circuits/reuploading.py** — `DataReuploadingCircuit` (Pérez-Salinas 2020 universal pattern): interleaves angle-encoding, parameterized Rot, and ring-entangling layers. Returns a `(num_qubits,)` PauliZ-expectation vector. JAX-interfaced QNode, JIT- and grad-compatible.
- **encoders/quantum_feature_encoder.py** — `QuantumFeatureEncoder` Equinox module: `x ∈ ℝ^F → π·tanh(Wx+b) → PQC → ⟨Z⟩ ∈ [-1,1]^Q`. Trainable parameters are PyTree leaves; the PennyLane QNode is a static field. Use `jax.vmap` (or `encoder_apply_batched`) for batching. ~68 params at default (input_dim=7, num_qubits=4, num_layers=3).

End-to-end smoke test: `scripts/qlnn_smoke_encoder.py` consumes real qZETA windows through the PyTorch data pipeline and feeds them through the JAX encoder.

### Key dependencies
torch, torchdiffeq (ODE solvers), jax + jaxlib (quantum side), equinox, diffrax, optax, pennylane (quantum circuits), pandas, scikit-learn

### Data
Raw data lives in `data/` (gitignored). Dataset: `data/raw/qZETA_data_copy.csv` — 778 rows, features include PRE, TEMP_EXT, TEMP_CULTURE, PAR_LIGHT, PH, DO, OD, DRY, CELL.
