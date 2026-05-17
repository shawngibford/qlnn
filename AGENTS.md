# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Quick orientation
- This is a small research-code Python package (src-layout). Import path: `quantum_liquid_neuralode`.
- The repository currently implements a few core, test-backed building blocks (liquid cell, losses, preprocessing). `spec.md` is the longer roadmap.

## Common commands

### Setup (Python 3.11)
```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip

# Install package + dev deps (pytest)
.venv/bin/python -m pip install -e ".[dev]"
```

### Build a wheel (optional)
```bash
mkdir -p dist
.venv/bin/python -m pip wheel . -w dist
```

### Run tests
Pytest is configured via `pyproject.toml` (tests live in `tests/`).

```bash
.venv/bin/python -m pytest
```

Run a single test file:
```bash
.venv/bin/python -m pytest tests/test_liquid_cell.py
```

Run a single test case:
```bash
.venv/bin/python -m pytest tests/test_liquid_cell.py::test_liquid_cell_gradients_flow
```

Filter by keyword:
```bash
.venv/bin/python -m pytest -k liquid_cell
```

### Linting / formatting
No linter/formatter/type-checker is currently configured in `pyproject.toml` or `requirements-dev.txt` (only `pytest`).

## High-level architecture

### Package layout
Core code lives under `src/quantum_liquid_neuralode/` and is organized by concern:
- `models/`: model components
- `training/`: loss functions (physics-informed + regularization)
- `data_processing/`: data preprocessing utilities for time series
- `utils/`: small utilities (e.g., device selection)

Tests in `tests/` mirror these modules (`test_liquid_cell.py`, `test_losses.py`, `test_preprocessor.py`).

### Key modules (what exists today)
- `src/quantum_liquid_neuralode/models/liquid_cell.py`
  - `LiquidCell` implements continuous-time dynamics by returning **dh/dt** given `(h, x)`.
  - Designed to be integrated by an external ODE solver (e.g., `torchdiffeq.odeint`); no solver wrapper exists in-repo yet.
  - The `forward(h, x, t=None)` signature is compatible with common ODE-solver calling conventions (time argument is accepted but unused).
  - `tau_unconstrained` is mapped to positive time constants via `softplus` + `tau_min`.
- `src/quantum_liquid_neuralode/training/losses.py`
  - `logistic_growth_residual_loss(od, time_points, mu, K)` computes a finite-difference residual vs logistic growth.
  - `smoothness_loss(sequence)` penalizes squared second differences along time.
- `src/quantum_liquid_neuralode/data_processing/preprocessor.py`
  - `BioreactorDataPreprocessor` is intentionally explicit: caller provides `feature_cols` and `target_col`.
  - `normalize_minmax()` applies per-column MinMax scaling to `[0, 1]` (useful for quantum-feature-encoding ranges).
  - `create_sequences(window_size, stride)` produces sliding windows for supervised sequence modeling.
- `src/quantum_liquid_neuralode/utils/mps.py`
  - `select_device(prefer_mps=True)` selects `mps` (Apple Silicon) → `cuda` → `cpu`.

### Roadmap / design notes
- `spec.md` describes the intended full pipeline (quantum feature encoder + liquid neural ODE + physics-informed training) and is the main design/roadmap document; the implemented code currently covers only the foundational components above.
