#!/usr/bin/env bash
# =====================================================================
# M3 — kuramoto + KdV completion. 30 cells, 1 cell per array task.
#   sbatch -A $QLNN_ACCOUNT -p $QLNN_PARTITION 01_kuramoto_kdv.sbatch
#
# Cell order is scripts/run_p7_8_h1_kuramoto_kdv.py::_build_cells()
# (deterministic): kuramoto QLNN 0-11, kuramoto cPINN 12-14,
# kdv QLNN 15-26, kdv cPINN 27-29.
# The runner is resumable (skips existing metrics.json) and
# error-isolated (writes error.json and exits nonzero on crash), so
# requeueing failed tasks is safe.
# Walltime: 12 h. Anvil-measured XLA compile is 1-1.5 h per module on
# these graphs (job 19203640) + training; laptop estimates do NOT apply.
# Output: results/anvil/p6_kuramoto_kdv/{system}_{ansatz}/seed_N/
# =====================================================================
#SBATCH --job-name=qlnn-m3
#SBATCH -A chm260071
#SBATCH -p shared
#SBATCH --array=0-29%64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/m3_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/config.env"
eval "${QLNN_TASK_ENV}"
# M3 graphs are huge (12-D kuramoto vmap / KdV jacrev³) — let XLA use
# all 8 requested cores for compile + train.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
module load anaconda
# conda activate references unset vars internally — suspend nounset
set +u; conda activate "${QLNN_CONDA_ENV}"; set -u
cd "${QLNN_ROOT}"
export PYTHONPATH=src

python -u scripts/run_p7_8_h1_kuramoto_kdv.py \
    --confirm --cell-index "${SLURM_ARRAY_TASK_ID}" \
    --out results/anvil/p6_kuramoto_kdv
