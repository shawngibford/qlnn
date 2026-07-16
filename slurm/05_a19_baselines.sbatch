#!/usr/bin/env bash
# =====================================================================
# A19 — classical forecaster baselines at the 2000-step parity budget.
# 36 cells (4 baseline families × 3 ODE systems × 3 seeds), 1 cell
# per array task.
#   sbatch -A $QLNN_ACCOUNT -p $QLNN_PARTITION 05_a19_baselines.sbatch
#
# --no-verdict: each task trains ONE cell; the H1 verdict aggregation
# runs once in Phase D (locally, with human review) — never per-task.
# Roster includes classical_ltc (P7.10) so the refreshed verdict
# retains the LTC decomposition arm.
#
# Index decode: i = seed*12 + system*4 + family
#   family = i % 4, system = (i / 4) % 3, seed = i / 12
# Output: results/p6_a19_baselines/{system}_{family}/seed_N/
# =====================================================================
#SBATCH --job-name=qlnn-a19
#SBATCH -A chm260071
#SBATCH -p shared
#SBATCH --array=0-35%64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/a19_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/config.env"
eval "${QLNN_TASK_ENV}"
module load anaconda
# conda activate references unset vars internally — suspend nounset
set +u; conda activate "${QLNN_CONDA_ENV}"; set -u
cd "${QLNN_ROOT}"
export PYTHONPATH=src

FAMILIES=(plain_neuralode plain_mlp skyline classical_ltc)
SYSTEMS=(lotka_volterra van_der_pol lorenz)

i="${SLURM_ARRAY_TASK_ID}"
FAM="${FAMILIES[$(( i % 4 ))]}"
SYS="${SYSTEMS[$(( (i / 4) % 3 ))]}"
SEED=$(( i / 12 ))

OUT_CELL="results/p6_a19_baselines/${SYS}_${FAM}/seed_${SEED}"
if [ -f "${OUT_CELL}/metrics.json" ]; then
    echo "[a19 ${i}] SKIP — ${OUT_CELL}/metrics.json exists"
    exit 0
fi

echo "[a19 ${i}] ${FAM} × ${SYS} × seed ${SEED} @ 2000 train steps"
python -u scripts/run_p5_matched_baselines.py \
    --families "${FAM}" --systems "${SYS}" --seeds "${SEED}" \
    --train-steps 2000 --no-verdict --out results/p6_a19_baselines
