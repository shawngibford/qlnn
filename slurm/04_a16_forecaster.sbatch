#!/usr/bin/env bash
# =====================================================================
# A16/A19 — forecaster re-runs after the ansatz un-aliasing (A16),
# brickwall removal (A18), and 2000-step budget parity (A19).
# 63 cells (7 forecaster families × 3 ODE systems × 3 seeds),
# 1 cell per array task.
#   sbatch -A $QLNN_ACCOUNT -p $QLNN_PARTITION 04_a16_forecaster.sbatch
#
# Family roster is the post-A18 registry (brickwall removed;
# strongly_entangling now un-aliased per A16; 3 non_liquid variants
# from the P7.11 2x2 decomposition included so the refreshed verdict
# keeps the full fairness matrix).
#
# Index decode: i = seed*21 + system*7 + family
#   family = i % 7, system = (i / 7) % 3, seed = i / 21
# Output: results/anvil/p6_a16_forecaster/{system}_{family}/seed_N/
# =====================================================================
#SBATCH --job-name=qlnn-a16
#SBATCH -A chm260071
#SBATCH -p shared
#SBATCH --array=0-62%64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/a16_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/config.env"
eval "${QLNN_TASK_ENV}"
module load anaconda
# conda activate references unset vars internally — suspend nounset
set +u; conda activate "${QLNN_CONDA_ENV}"; set -u
cd "${QLNN_ROOT}"
export PYTHONPATH=src

FAMILIES=(data_reuploading hardware_efficient strongly_entangling rf_qrc
          non_liquid_data_reuploading non_liquid_hardware_efficient
          non_liquid_strongly_entangling)
SYSTEMS=(lotka_volterra van_der_pol lorenz)

i="${SLURM_ARRAY_TASK_ID}"
FAM="${FAMILIES[$(( i % 7 ))]}"
SYS="${SYSTEMS[$(( (i / 7) % 3 ))]}"
SEED=$(( i / 21 ))

OUT_CELL="results/anvil/p6_a16_forecaster/${SYS}_${FAM}/seed_${SEED}"
if [ -f "${OUT_CELL}/metrics.json" ]; then
    echo "[a16 ${i}] SKIP — ${OUT_CELL}/metrics.json exists"
    exit 0
fi

echo "[a16 ${i}] ${FAM} × ${SYS} × seed ${SEED} @ 2000 train steps"
python -u scripts/run_p4_forecaster_rollout.py \
    --families "${FAM}" --systems "${SYS}" --seeds "${SEED}" \
    --train-steps 2000 --out results/anvil/p6_a16_forecaster
