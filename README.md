# Quantum Liquid Neural Network — ODE/PDE solver & forecaster

[![integrity](https://img.shields.io/badge/verify__paper__integrity-passing-brightgreen)](scripts/verify_paper_integrity.py)
[![paper](https://img.shields.io/badge/paper-§1–§8%20drafted-blue)](paper/main.tex)
[![tests](https://img.shields.io/badge/pytest-passing-brightgreen)](tests/)
[![status](https://img.shields.io/badge/status-supplement%20%2B%20polish-orange)](HANDOFF.md)

Research code and paper draft for a controlled benchmark of **Quantum
Liquid Neural Networks (QLNNs) as both a physics-informed solver and a
data-driven forecaster for ODEs and PDEs**. Literature-grounded starting
circuits (Kyriienko DQC, Schuld data-reuploading, Lubasch multi-copy,
Berger TE-QPINN, QCPINN, RF-QRC), strong matched classical baselines
including a non-liquid Neural-ODE control, a pre-registered falsifiable
inductive-bias hypothesis, and a mechanistic trainability
characterization. Every number in the paper draft is gated by
`scripts/verify_paper_integrity.py` against committed JSONs (exit code
0 required for any commit to land).

---

## ✋ Start here (new contributors / colleagues)

In order:

1. **[`NEXT_AGENT_PICKUP.md`](NEXT_AGENT_PICKUP.md)** — one-line state
   + one-command sanity check.
2. **[`HANDOFF.md`](HANDOFF.md)** — operational pickup detail (what is
   done, what is in flight, what's next).
3. **[`paper/main.tex`](paper/main.tex)** — the 15-page paper draft
   (§1 Intro through §8 Conclusions); build with `bash paper/build.sh`.
4. **[`ODE_PDE_PRE_REG.md`](ODE_PDE_PRE_REG.md)** + **[`PRE_REG_AMENDMENT.md`](PRE_REG_AMENDMENT.md)**
   — pre-registration that locks the science; amendments are
   disclosed (no silent moves).

## Repository map (current, post-archive cleanup)

| Path | Purpose |
|---|---|
| `paper/main.tex` + `paper/supplement.tex` | The paper draft and supplement (LaTeX). |
| `paper/figures/` | Publication figures (PNG + PDF) regenerated from on-disk results. |
| `refs/CIRCUIT_SPECS.md` + `refs/_speccard_*.md` + `refs/_check_*.md` | PDF-grounded circuit specifications and dual-check files for every literature ansatz. The faithfulness gate that precedes any new ansatz implementation. |
| `src/qlnn_/circuits/` | Registered quantum ansatz families: `data_reuploading`, `hardware_efficient`, `strongly_entangling`, `brickwall`, `qcpinn`, `rf_qrc`, `te_qpinn`, plus the `pde_2d/` subpackage. |
| `src/qlnn_/cells/` | `liquid_quantum_cell` + `non_liquid_quantum_cell` (the mandatory non-liquid baseline that isolates the quantum-vs-liquid confound). |
| `src/quantum_liquid_neuralode/data_processing/` | ODE systems (`synthetic_ode.py`) and PDE systems (`pde_systems.py`). |
| `scripts/` | Trainers, summarizers, figure builders, integrity gates, and the SOTA-circuit search harnesses. |
| `configs/` | YAML configs for every training run. |
| `results/` | Frozen per-phase result tables and per-seed artifacts. The `results/*table*.md` files are the canonical headline numbers; never edited by hand. |
| **`archive/`** | **Legacy documents from the pre-pivot bioreactor-OD program.** See [`archive/README.md`](archive/README.md) for a legend. Preserved for reproducibility; not part of the current paper. |

## Methodology rigor

- **Pre-registration before analysis.** Hypothesis, tasks, metrics,
  baselines, and decision rules are locked in `ODE_PDE_PRE_REG.md`
  before any experiment that informs the headline. Subsequent
  methodological choices are disclosed in `PRE_REG_AMENDMENT.md`.
- **Integrity-gated numbers.** `scripts/verify_paper_integrity.py` exits
  0 on every committed headline; the paper LaTeX references results
  through this gate, so a mismatch fails the build.
- **5 seeds** per cell; mean ± ddof=1 std, 95% t-CI, and paired-bootstrap
  p-values reported for every comparison.
- **Equal HPO budget** between classical and quantum baselines
  (documented per-cell). The non-liquid Neural-ODE baseline is mandatory
  and isolates the quantum-vs-liquid confound that the project identity
  otherwise introduces.
- **Provenance** per run (git SHA + data SHA-256 + package versions +
  platform) in each `provenance.json`.
- `jax_enable_x64` intentionally off (Diffrax dtype-promotion
  constraint); reverse-mode `jax.jacrev` required through Diffrax
  `custom_vjp`.

## Stack

PyTorch + torchdiffeq for the classical Liquid-ODE baseline; JAX +
Equinox + Diffrax + PennyLane for the QLNN. Shared data, evaluation,
and bootstrap modules keep head-to-head numbers bit-identical
comparable. PennyLane device = `default.qubit` with the JAX interface
(effective `diff_method = backprop`) — the only configuration that
composes through the nested Diffrax + `jax.jacrev` autodiff path at
the qubit counts used here.

## Setup & reproduction

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev]"   # add ".[dev,search]" for Optuna

# One-command sanity check (run this first when joining):
PYTHONPATH=src .venv/bin/python -m pytest -q                          # full suite must pass
PYTHONPATH=src .venv/bin/python scripts/verify_paper_integrity.py    # must exit 0

# Rebuild figures from on-disk results (no training):
PYTHONPATH=src .venv/bin/python scripts/make_paper_figures.py
PYTHONPATH=src .venv/bin/python scripts/make_diagnostic_figures.py

# Build the paper:
bash paper/build.sh
bash paper/build_supplement.sh
```

To rebuild a phase from scratch see `scripts/reproduce_paper.sh` and
the per-phase notes in `results/*/README.md`.

## Honest limitations

The current paper draft is the result of a **deliberate pivot** away
from an earlier bioreactor-OD program that was honestly reassessed as
a rigorous null on an n=1 dataset (preserved in `archive/` with its
integrity check still green for continuity). The current program
addresses that limitation with a controlled hardness ladder, matched
baselines including a non-liquid control, and a falsifiable
inductive-bias hypothesis whose verdict — confirm, falsify, or
regime-dependent map — is publishable either way because the
pre-registered question and method are the contribution.

## Citation & contact

Research code (no production claims). Citation: see `paper/main.tex`
once the supplement is finalized. For author contact see the top-level
commit history.
