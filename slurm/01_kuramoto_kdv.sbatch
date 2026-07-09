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
# Walltime: KdV QLNN cells ~1.1 hr + JIT + margin.
# Output: results/p6_kuramoto_kdv/{system}_{ansatz}/seed_N/
# =====================================================================
#SBATCH --job-name=qlnn-m3
#SBATCH -A chm260071
#SBATCH -p shared
#SBATCH --array=0-29%64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/m3_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/config.env"
eval "${QLNN_TASK_ENV}"
module load anaconda
# conda activate references unset vars internally — suspend nounset
set +u; conda activate "${QLNN_CONDA_ENV}"; set -u
cd "${QLNN_ROOT}"
export PYTHONPATH=src

python -u scripts/run_p7_8_h1_kuramoto_kdv.py \
    --confirm --cell-index "${SLURM_ARRAY_TASK_ID}"
