# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research code for a head-to-head comparison of classical Liquid Neural ODE
and Quantum Liquid Neural Network forecasters on a 778-sample
single-fermentation-run bioreactor OD dataset.

**Status: code + experiments complete; paper writing is the remaining work.**

The three pre-registered claims (`hypothesis.md` v2; original QWGAN-GP
claim was explicitly dropped after the Phase A/B/C peer-review-style
audit):

1. **Reproducibility** — QLNN test-MAE σ is ≥ 2× tighter than classical
   at matched params. ✅ MET (3.77× ratio, holds at every data fraction).
2. **Expressivity** — QLNN d_norm (Abbas et al. 2021 Eq. 4) exceeds
   classical by > 1.0 at matched params. ✅ MET (+1.49), caveated by
   QLNN's higher d_norm seed-variance (4.7 vs 1.3) — see
   `STEP5_MONOTONICITY_NOTE.md` for the corrected sanity-check criterion.
3. **Sample efficiency** — paired bootstrap shows QLNN wins at 10%
   (p=0.015) and 25% (p=0.002) of the training data, ties at 50%, loses
   at 100% (p=0.029). Stronger than the original pre-reg threshold
   required.

**Single source of truth for paper numbers: `PAPER_SUMMARY.md`.** It is
verified end-to-end by `scripts/verify_paper_integrity.py`.

Stack is hybrid by design: PyTorch + torchdiffeq for the classical
baseline (`src/quantum_liquid_neuralode/`) and JAX + Equinox + Diffrax +
PennyLane for the QLNN (`src/qlnn_/`). Both packages share the
data preprocessing, evaluation, bootstrap, and effective-dimension
modules so head-to-head numbers are bit-identical comparable.

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
  Plus h ∈ {3, 6, 12} ablation; h=3 is the discriminating regime.
- OD scaling: train-only MinMax (R3 leak fix); predictions clipped to
  [0, od_phys_max=3.8] in raw OD space at evaluation.
- Metrics: MAE_raw, RMSE_raw, R²_raw, MSE_norm, ΔOD_R²_raw.
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

### Pipeline status — feature-complete for the v2 paper

Step 1 (classical Liquid-ODE baseline), Step 2 (quantum feature encoder),
Step 3 (full hybrid forecaster), Step 5 (effective dimension), and Step 6
(sample efficiency) are all implemented, run, and committed. Step 4
(QWGAN-GP synthetic data lift) was explicitly **dropped** after the Phase
A/B/C peer-review-style audit because the single-run dataset cannot
support that claim without a held-out second fermentation run — see
`hypothesis.md` v2 "Deviations from v1" for the rationale.

Canonical results live under:
- `results/baseline_classical_{euler,dopri5,physics,euler_fixed_od}/`
- `results/horizon_sweep/euler_h{1,3,6,12}/`
- `results/param_sweep/euler_h3_hidden{2,4,8,16,32}/`
- `results/qlnn_hybrid_{h1,h3,h3_physics}/`
- `results/effective_dimension/`
- `results/sample_efficiency/`

`scripts/reproduce_paper.sh` regenerates everything from scratch;
`scripts/verify_paper_integrity.py` checks the regenerated numbers against
the values cited in `PAPER_SUMMARY.md`.

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
