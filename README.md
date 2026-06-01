# Quantum Liquid Neural Network — ODE/PDE solver & forecaster

[![integrity](https://img.shields.io/badge/verify__paper__integrity-passing-brightgreen)](scripts/verify_paper_integrity.py)
[![paper](https://img.shields.io/badge/paper-main%2027pp%20%2B%20supp%208pp-blue)](paper/main.pdf)
[![tests](https://img.shields.io/badge/pytest-510%20passing-brightgreen)](tests/)
[![figures](https://img.shields.io/badge/figures-25%20(21%20main%20%2B%204%20supp)-blue)](paper/figures/)
[![bibliography](https://img.shields.io/badge/references-37%20verified-blue)](paper/references.bib)
[![status](https://img.shields.io/badge/status-post--audit%20re--runs%20pending%20(Anvil)-orange)](NEXT_STEPS.md)
[![amendments](https://img.shields.io/badge/pre--reg%20amendments-A1--A22-blue)](PRE_REG_AMENDMENT.md)

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

## ✋ Start here (new contributors / colleagues / advisors)

In order:

1. **[`ADVISOR_BRIEF.md`](ADVISOR_BRIEF.md)** — single-page,
   plain-language narrative of the work and the headline finding,
   plus the case for running the remaining compute on Purdue's Anvil
   HPC via an ACCESS allocation.
2. **[`NEXT_STEPS.md`](NEXT_STEPS.md)** — five-phase timeline from
   here to *PRX Quantum* submission, with explicit gates between
   phases. The single blocker is Phase A (ACCESS allocation —
   advisor letter required).
3. **[`paper/main.pdf`](paper/main.pdf)** — the rendered 27-page paper
   draft (§1 Intro through §8 Conclusions). 21 figures, 37 verified
   bibliography entries, 72 in-prose citations. Source:
   `paper/main.tex` + `paper/sections/0*.tex`; rebuild with
   `bash paper/build.sh`. Supplement
   (**[`paper/supplement.pdf`](paper/supplement.pdf)**, 8 pages, 4
   figures) via `bash paper/build_supplement.sh`. For a single-link
   print: **[`paper/main_with_supplement.pdf`](paper/main_with_supplement.pdf)**
   (35 pages combined).
4. **[`ODE_PDE_PRE_REG.md`](ODE_PDE_PRE_REG.md)** + **[`PRE_REG_AMENDMENT.md`](PRE_REG_AMENDMENT.md)**
   — pre-registration that locks the science; 22 amendments
   (A1–A22) are disclosed openly (no silent moves). A15–A19 landed
   2026-05-28 and close every reviewer-fairness concern from the
   internal audit; A20–A22 landed the same day from a five-reviewer
   adversarial peer-review pass (un-aliased te_qpinn_fnn readout,
   brickwall connectivity disclosure, latent docstring fix).
5. **[`CLAUDE.md`](CLAUDE.md)** — operational guidance for the
   coding agent (current state, what not to do, where the runners are).

## Repository map (current, post-archive cleanup)

| Path | Purpose |
|---|---|
| `paper/main.tex` + `paper/supplement.tex` | The paper draft and supplement (LaTeX). |
| `paper/figures/` | Publication figures (PNG + PDF) regenerated from on-disk results. |
| `refs/CIRCUIT_SPECS.md` + `refs/_speccard_*.md` + `refs/_check_*.md` | PDF-grounded circuit specifications and dual-check files for every literature ansatz. The faithfulness gate that precedes any new ansatz implementation. |
| `src/qlnn_/circuits/` | Registered quantum ansatz families: `data_reuploading`, `hardware_efficient`, `strongly_entangling`, `brickwall`, `qcpinn`, `rf_qrc`, `te_qpinn`, plus the `pde_2d/` subpackage. |
| `src/qlnn_/cells/` | `liquid_quantum_cell` + `non_liquid_quantum_cell` (the mandatory non-liquid baseline that isolates the quantum-vs-liquid confound). |
| `src/quantum_liquid_neuralode/data_processing/` | ODE systems (`synthetic_ode.py`), PDE systems (`pde_systems.py`), shared windowing utilities. OD-era loaders archived. |
| `scripts/` | Post-pivot runners (`run_p<n>_*.py`), figure builders (`make_p<n>_*.py`), the M3 staging wrapper (`run_p6_group.sh`), and the integrity gate. |
| `configs/` | YAML configs for every training run. |
| `results/` | Post-pivot per-phase result tables and per-seed artifacts. OD-era frozen results live in `archive/results/` and are still verified by the integrity gate. |
| **`archive/`** | **All pre-pivot bioreactor-OD artifacts** plus superseded pivot-era planning docs (`archive/superseded-2026-05-28/`). Preserved for reproducibility and integrity-gate continuity; not part of the active program. |

## Methodology rigor

- **Pre-registration before analysis.** Hypothesis, tasks, metrics,
  baselines, and decision rules are locked in `ODE_PDE_PRE_REG.md`
  before any experiment that informs the headline. Subsequent
  methodological choices are disclosed in `PRE_REG_AMENDMENT.md`.
- **Integrity-gated numbers.** `scripts/verify_paper_integrity.py` exits
  0 on every committed headline; the paper LaTeX references results
  through this gate, so a mismatch fails the build.
- **3 seeds** per cell (canonical solver matrix: 8 systems × 3 seeds
  = 24 PRIMARY cells; canonical forecaster matrix: 3 systems × 3
  seeds = 9 PRIMARY cells); mean ± ddof=1 std, 95% paired-bootstrap
  percentile CI, and Holm–Bonferroni multiple-comparison correction
  applied across the master verdict family.
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

# Rebuild any individual figure from on-disk results (no training).
# Each figure has its own standalone generator under scripts/:
PYTHONPATH=src .venv/bin/python scripts/make_tau_substrate_figure.py
PYTHONPATH=src .venv/bin/python scripts/make_compute_envelope_figure.py
# ...  (28 generators in scripts/make_*.py; see paper/figures/ for the
#       full set of 25 rendered figures.)

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

Research code (no production claims). The current rendered paper is
**[`paper/main.pdf`](paper/main.pdf)** (27pp, integrity-gated, on
master). Cite via the in-prose contributions enumeration in §1 until
the arXiv preprint / *PRX Quantum* submission tag lands. For author
contact see the top-level commit history.
