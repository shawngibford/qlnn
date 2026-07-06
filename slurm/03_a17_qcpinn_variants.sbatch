#!/usr/bin/env bash
# =====================================================================
# A17 — qcpinn quantum-attribution variants, ODE side. 45 cells
# (3 variants × 5 ODE systems × 3 seeds), 1 cell per array task.
#   sbatch -A $QLNN_ACCOUNT -p $QLNN_PARTITION 03_a17_qcpinn_variants.sbatch
#
# Variants sweep the PQC/(PQC+classical) parameter ratio per amendment
# A17: qcpinn_balanced (~24%), qcpinn_quantum (~45%), qcpinn_full_q
# (~87%); the committed qcpinn baseline sits at ~2%. kuramoto IS
# included here (job 01 covers only the four standard families).
#
# Index decode: i = seed*15 + system*3 + variant
#   variant = i % 3, system = (i / 3) % 5, seed = i / 15
# Output: results/p6_a17_qcpinn_variants/{variant}_{system}/seed_N/
# =====================================================================
#SBATCH --job-name=qlnn-a17
#SBATCH --array=0-44%64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:30:00
#SBATCH --output=logs/a17_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/config.env"
eval "${QLNN_TASK_ENV}"
cd "${QLNN_ROOT}"
export PYTHONPATH=src

VARIANTS=(qcpinn_balanced qcpinn_quantum qcpinn_full_q)
SYSTEMS=(lotka_volterra van_der_pol lorenz fitzhugh_nagumo kuramoto)

i="${SLURM_ARRAY_TASK_ID}"
VAR="${VARIANTS[$(( i % 3 ))]}"
SYS="${SYSTEMS[$(( (i / 3) % 5 ))]}"
SEED=$(( i / 15 ))

OUT_CELL="results/p6_a17_qcpinn_variants/${VAR}_${SYS}/seed_${SEED}"
if [ -f "${OUT_CELL}/metrics.json" ]; then
    echo "[a17 ${i}] SKIP — ${OUT_CELL}/metrics.json exists"
    exit 0
fi

echo "[a17 ${i}] ${VAR} × ${SYS} × seed ${SEED} @ 2000 steps"
"${QLNN_VENV}/bin/python" scripts/run_multi_state_demo.py \
    --families "${VAR}" --systems "${SYS}" --seeds "${SEED}" \
    --steps 2000 --out results/p6_a17_qcpinn_variants
