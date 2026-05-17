# Quantum-Liquid Neural Network — Bioreactor OD Modeling

Research code for a **quantum feature encoder + liquid neural ODE** time-series
generator on a small (778-sample) bioreactor dataset. Targets a publishable
three-pronged claim:

1. **Synthetic data lift** — QWGAN-GP-generated curves augment training and
   improve downstream OD-forecast accuracy.
2. **Expressivity** — Fisher / effective dimension shows the quantum model
   reaches representations a classical Liquid-ODE of matched parameter count
   can't.
3. **Sample efficiency** — QLNN converges with fewer epochs / less data.

The classical baseline (this milestone) is the head-to-head comparison floor
for all of the above.

## Stack

Hybrid by design:
- **PyTorch + torchdiffeq** — classical baseline (`quantum_liquid_neuralode/`).
  MPS-friendly. Adaptive `dopri5` or fixed-step `euler`/`rk4`.
- **JAX + Equinox + Diffrax + PennyLane** — quantum models (forthcoming
  `qlnn_/` subpackage). Real JIT through the quantum circuit, adjoint diff
  via Diffrax.

Both packages share the same data preprocessing path and evaluation protocol so
head-to-head numbers are comparable.

## Setup (Python 3.11)

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev]"

.venv/bin/python -m pytest
```

## Evaluation protocol (locked across every model)

- Dataset: `data/raw/qZETA_data_copy.csv`, 778 rows, time-ordered.
- Splits: train 70%, val 15%, test 15% (chronological — no shuffling across time).
- Task: 1-hour-ahead OD forecast from a 24-step history window.
- OD scaling: fixed MinMax bounds [0.0, 3.8].
- Metrics: MAE_raw, RMSE_raw, R²_raw, MSE_norm (+ DTW for trajectory-level
  evaluation once QWGAN lands).
- Baselines reported in every run: persistence and linear extrapolation.
- Seeds: {0, 1, 2, 3, 4}, mean ± std.
- Selection: best val MSE_norm checkpoint.

This is the contract; every model in the paper must run under it.

## Running the classical baseline

```bash
# Headline run: adaptive dopri5, 5 seeds, ~300 epochs (early-stopping)
.venv/bin/python scripts/train_baseline.py \
    --config configs/baseline.yaml \
    --output-dir results/baseline_classical_dopri5

# Fast variant (Euler, MPS-friendly):
.venv/bin/python scripts/train_baseline.py \
    --config configs/baseline_euler_fast.yaml \
    --output-dir results/baseline_classical_euler

# Physics-informed ablation:
.venv/bin/python scripts/train_baseline.py \
    --config configs/baseline_physics.yaml \
    --output-dir results/baseline_classical_physics
```

Each run writes:

```
<output_dir>/
    config.json            # frozen config that ran
    protocol.json          # locked data/split/window numbers
    baselines.json         # persistence + linear baseline metrics (deterministic)
    seed_{0..4}/
        metrics.json
        history.csv
        best_state.pt
    seeds_summary.json     # mean/std/min/max across seeds — the paper row
```

Build a paper-ready comparison table:

```bash
.venv/bin/python scripts/summarize_baselines.py \
    --runs results/baseline_classical_euler \
           results/baseline_classical_dopri5 \
           results/baseline_classical_physics \
    --labels "Liquid-ODE (Euler)" "Liquid-ODE (dopri5)" "+physics" \
    --output results/baseline_classical_table
```

## Package layout

```
src/quantum_liquid_neuralode/
    models/
        liquid_cell.py            # Continuous-time RNN cell — returns dh/dt
        forecaster.py             # LiquidODForecaster, swappable ODE solver
    training/
        losses.py                 # Physics losses (logistic, smoothness)
        trainer.py                # Reusable training loop with physics knobs
    data_processing/
        qzeta.py                  # Canonical qZETA loader
        windowing.py              # Splits, MinMax, sliding-window builder
        preprocessor.py           # General-purpose preprocessor (legacy entry)
    evaluation/
        metrics.py                # MAE / RMSE / R² / MSE_norm
        baselines.py              # Persistence + linear extrapolation
    utils/
        mps.py                    # Device selection
```

## Spec

See `spec.md` for the full implementation roadmap.

## License / status

Research code. Not for production.
