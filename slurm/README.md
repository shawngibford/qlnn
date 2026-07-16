# Anvil Phase C — SLURM job-array infrastructure

Runs the ~222-cell audit re-run matrix (`NEXT_STEPS.md` Phase C /
`REMEDIATION_PLAN.md` Tier 3) on Purdue Anvil CPU. All five arrays run
concurrently; wall-clock is bounded by the slowest M3 cells.

**Anvil reality check (2026-07-14 runs):** laptop-derived per-cell
estimates do NOT transfer. XLA compilation of the big M3 autodiff
graphs (12-D kuramoto vmap, KdV jacrev³) took **1h25m per module** on
Anvil EPYC nodes under JAX's new CPU thunk runtime (observed job
19203640), and kuramoto cells OOM'd at 8 GB. Fixes now baked in:
`--xla_cpu_use_thunk_runtime=false` (restores fast legacy compile),
a persistent JAX compilation cache on scratch (same circuit family →
compile once, reuse across seeds/requeues), 32 GB + 8 cores + 12 h
for M3, and 32 GB + 4 cores + 8 h for A17 (its kuramoto rows).

## Workloads

| Script | Workload | Cells | Cap/task | Worst-case core-hr |
|---|---|---:|---|---:|
| `01_kuramoto_kdv.sbatch` | M3 kuramoto + KdV @ 2000 steps | 30 | 12 h × 8c × 32G | ~350 |
| `02_a15_uniform_ode.sbatch` | A15 uniform-budget ODE re-runs (kuramoto excluded — covered by 01) | 48 | 3 h × 2c | ~35 |
| `03_a17_qcpinn_variants.sbatch` | A17 qcpinn PQC-ratio variants, ODE side | 45 | 8 h × 4c × 32G | ~200 |
| `04_a16_forecaster.sbatch` | A16/A18/A19 forecaster re-runs (7-family post-A18 roster) | 63 | 2 h × 2c | ~30 |
| `05_a19_baselines.sbatch` | A19 classical baselines @ 2000 steps (incl. classical_ltc) | 36 | 1 h × 2c | ~10 |
| **Total committed** | | **222** | | **~625 worst-case** |
| `03b_a17_pde_optional.sbatch` | OPTIONAL PDE-side A17 (needs 2D variant ports wired first) | 36 | 3 h × 2c | ~40 |

Core-hr figures are walltime caps × cores (billing worst case). Actual
usage is typically 30–50% of cap: tasks exit the moment their cell
finishes, skip-if-done tasks exit in seconds, and the compilation
cache means only the FIRST cell of each circuit family pays the big
compile.

Every workload writes to a NEW `results/anvil/p6_*` directory — the
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
find $QLNN_ROOT/results/anvil/p6_* -name metrics.json | wc -l   # → 222 when done

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
find $QLNN_ROOT/results/anvil/p6_* -name metrics.json | wc -l   # → 222

# 4. After 99_aggregate completes: scp the tarball home, untar, and
#    run Phase D locally (verdict refresh + integrity + paper rebuild).
```

## Design notes

- **1 array task = 1 cell.** All cells are independent by
  `(family, system, seed)`; failures are isolated and requeue-safe
  (`scontrol requeue <jobid>_<idx>`), because every script either
  skips-if-`metrics.json`-exists (bash guard) or is natively
  resumable (job 01's runner).
- **2 cores per task.** Most cells request 8 GB/task; M3 and A17
  request 32 GB/task after OOM failures on Anvil. The quantum circuits
  here are 3–8 qubits — tiny state vectors; memory pressure is from
  JAX/XLA compilation and Diffrax traces, not state-vector size.
  `config.env` pins `OMP_NUM_THREADS=2` and forces a single XLA host
  device so tasks don't oversubscribe shared nodes.
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
- **Walltimes are set from Anvil-measured behavior, not laptop
  estimates.** The 2026-07-14 runs showed XLA compile alone can take
  1–1.5 h on the M3 graphs (see the reality-check note at the top);
  M3's 12 h cap absorbs two slow compiles + training + node variance.
  Small-system arrays keep tighter caps because their graphs compile
  in minutes.
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

## Troubleshooting (from the 2026-07-14 runs)

**Symptom: `CANCELLED ... DUE TO TIME LIMIT` on M3 cells, log shows
`[Compiling module jit_loss for CPU] Very slow compile` and an
operation taking >1 h.**
Cause: JAX's new CPU thunk runtime compile regression on the huge
M3 graphs. Fixed by `--xla_cpu_use_thunk_runtime=false` in
`config.env` (`QLNN_TASK_ENV`) + the 12 h walltime. If a cell STILL
times out at 12 h, check the log for repeated compile alarms — a
stale `.jax_cache` can be cleared with
`rm -rf $QLNN_ROOT/.jax_cache` (first run after clearing re-pays
one compile per family).

**Symptom: `oom_kill event ... Killed` a few minutes after START.**
Cause: XLA compile working set exceeds the task's memory. M3 and A17
now request `--mem=32G`. If an A15/A16/A19 cell ever OOMs, raise that
script's `--mem-per-cpu` to 8G (16 GB/task) — do not raise all of
them preemptively; billing scales with the request on shared nodes.

**Re-running only the failed cells** (all tasks are skip-if-done):
```bash
cd $QLNN_ROOT/slurm
# e.g. the three timed-out M3 cells — resubmit just those array indexes:
sbatch --array=17,23,29 01_kuramoto_kdv.sbatch     # use YOUR failed indexes
# find failed indexes from a past run:
sacct -j <jobid> --format=JobID%20,State,Elapsed | grep -E "TIMEOUT|OOM|FAILED"
```
Completed cells are never recomputed, so resubmitting the WHOLE array
(`sbatch 01_kuramoto_kdv.sbatch`) is also safe — finished tasks exit
in seconds.

**Benign warning you can ignore:**
`Explicitly requested dtype complex128 ... truncated to complex64` —
expected; `jax_enable_x64` is intentionally OFF for this project
(Diffrax dtype-promotion constraint, see CLAUDE.md).

## Provenance: the `results/anvil/` boundary

ALL output from these SLURM scripts lands under `results/anvil/` —
never directly in `results/`. This is deliberate: any cell under
`results/anvil/` is guaranteed to have been produced on Anvil by
these job arrays, and anything elsewhere in `results/` is a
laptop/local artifact. When the copy-back tarball is untarred at
home it recreates the same `results/anvil/...` tree, so provenance
survives the round-trip. Phase D aggregators should read Anvil cells
from `results/anvil/p6_*` explicitly.
