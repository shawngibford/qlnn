# Anvil Phase C — SLURM job-array infrastructure

Runs the ~222-cell audit re-run matrix (`NEXT_STEPS.md` Phase C /
`REMEDIATION_PLAN.md` Tier 3) on Purdue Anvil CPU in **~2–3 hr
wall-clock** (~115 core-hr billed), vs ~55 hr serial on a laptop.

## Workloads

| Script | Workload | Cells | Walltime/task | Est. core-hr |
|---|---|---:|---|---:|
| `01_kuramoto_kdv.sbatch` | M3 kuramoto + KdV @ 2000 steps | 30 | 3.0 h | ~50 |
| `02_a15_uniform_ode.sbatch` | A15 uniform-budget ODE re-runs (kuramoto excluded — covered by 01) | 48 | 1.5 h | ~16 |
| `03_a17_qcpinn_variants.sbatch` | A17 qcpinn PQC-ratio variants, ODE side | 45 | 1.5 h | ~30 |
| `04_a16_forecaster.sbatch` | A16/A18/A19 forecaster re-runs (7-family post-A18 roster) | 63 | 1.0 h | ~15 |
| `05_a19_baselines.sbatch` | A19 classical baselines @ 2000 steps (incl. classical_ltc) | 36 | 0.5 h | ~5 |
| **Total committed** | | **222** | | **~115** |
| `03b_a17_pde_optional.sbatch` | OPTIONAL PDE-side A17 (needs 2D variant ports wired first) | 36 | 3.0 h | ~40 |

Every workload writes to a NEW `results/p6_*` directory — the
committed result trees backing `verify_paper_integrity.py` are never
touched on Anvil.

## ⚡ The one-command path (recommended)

On an Anvil login node — this is ALL you have to type:

```bash
git clone https://github.com/shawngibford/qlnn.git
cd qlnn/slurm
./go.sh                          # account chm260071 pre-configured
```

Then log off. `go.sh` bootstraps the environment, runs the paper
integrity gate, and queues the ENTIRE pipeline with SLURM
dependencies: smoke (5 cells) → automated smoke-gate
verification → five production arrays (222 cells) → aggregation +
copy-back tarball. If any smoke cell fails, everything downstream
cancels automatically — no allocation is burned on a broken
environment. Total unattended runtime ≈ 4–5 hr.

```bash
# Check in whenever you like:
squeue --me
find $QLNN_ROOT/results/p6_* -name metrics.json | wc -l   # → 222 when done

# When done, send the tarball home:
scp $QLNN_ROOT/qlnn_phase_c_results_*.tar.gz you@home:
```

## Manual step-by-step path (fallback / debugging)

```bash
# 0. One-time, on an Anvil login node:
bash slurm/env_setup.sh          # clone → conda env "qlnn" → integrity gate

# 1. Smoke (5 representative cells):
cd $QLNN_ROOT/slurm
# account chm260071 + partition are embedded in every .sbatch header
sbatch 00_smoke.sbatch
# inspect logs/smoke_*.out; all five must end with "OK"
touch SMOKE_PASSED

# 2. Full committed scope (five arrays + dependent aggregation):
./submit_all.sh

# 3. Monitor:
squeue --me
find $QLNN_ROOT/results/p6_* -name metrics.json | wc -l   # → 222

# 4. After 99_aggregate completes: scp the tarball home, untar, and
#    run Phase D locally (verdict refresh + integrity + paper rebuild).
```

## Design notes

- **1 array task = 1 cell.** All cells are independent by
  `(family, system, seed)`; failures are isolated and requeue-safe
  (`scontrol requeue <jobid>_<idx>`), because every script either
  skips-if-`metrics.json`-exists (bash guard) or is natively
  resumable (job 01's runner).
- **2 cores + 8 GB per task.** The quantum circuits here are 3–8
  qubits — tiny state vectors; JAX/XLA is effectively single-threaded
  per cell. `config.env` pins `OMP_NUM_THREADS=2` and forces a single
  XLA host device so tasks don't oversubscribe shared nodes.
- **Environment = `module load anaconda` + `conda activate qlnn`**
  inside every task — the same pattern as the coauthor's working
  QPINN jobs. `env_setup.sh` creates the conda env (python 3.11 +
  `pip install -e ".[dev]"`) if it doesn't exist.
- **Account (`chm260071`) and partition (`shared`) are embedded in
  every `.sbatch` header** — plain `sbatch foo.sbatch` works with no
  flags, matching the coauthor's known-good QPINN script style.
  `go.sh`/`submit_all.sh` still pass `-A` at submit time (harmless
  duplicate; enables account override).
- **All five arrays run concurrently** (~222 tasks × 2 cores ≈ 444
  cores ≈ 3.5 nodes). The `%64` throttle per array is queue etiquette,
  not a requirement — raise it if the queue is empty.
- **Walltimes are ~2.5× the measured per-cell estimates** from the
  2026-05-28 laptop smokes (~0.9 hr/kuramoto, ~1.1 hr/KdV cell) to
  absorb JIT-compilation and node-speed variance.
- **`jax_enable_x64` stays OFF** (Diffrax dtype-promotion constraint,
  per CLAUDE.md); nothing in these scripts touches it, and the
  integrity gate run by `env_setup.sh` asserts it.

## What these scripts do NOT do

- Modify locked integrity numbers or committed `results/` trees
  (Phase D / M5 is local, human-reviewed).
- Run the Tier 4 experiments (BCa CI, forecaster n_broad expansion,
  τ-substrate theory) — advisor-gated, separate plan.
- Submit 03b automatically (2D variant ports aren't wired yet; the
  script fail-fasts with a clear message if invoked prematurely).
