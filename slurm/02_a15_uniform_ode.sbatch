#!/usr/bin/env bash
# =====================================================================
# A15 — original ODE solver cells re-run at the uniform 2000-step
# budget. 48 cells (4 QLNN families × 4 original ODE systems × 3
# seeds), 1 cell per array task.
#   sbatch -A $QLNN_ACCOUNT -p $QLNN_PARTITION 02_a15_uniform_ode.sbatch
#
# kuramoto is EXCLUDED here — job 01 (M3) already trains kuramoto at
# the uniform 2000-step budget; duplicating it would waste ~11 core-hr.
#
# Index decode: i = seed*16 + system*4 + family
#   family = i % 4, system = (i / 4) % 4, seed = i / 16
# Output: results/p6_a15_uniform/{family}_{system}/seed_N/
# =====================================================================
#SBATCH --job-name=qlnn-a15
#SBATCH --array=0-47%64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:30:00
#SBATCH --output=logs/a15_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/config.env"
eval "${QLNN_TASK_ENV}"
cd "${QLNN_ROOT}"
export PYTHONPATH=src

FAMILIES=(chebyshev_dqc te_qpinn_fnn te_qpinn_qnn qcpinn)
SYSTEMS=(lotka_volterra van_der_pol lorenz fitzhugh_nagumo)

i="${SLURM_ARRAY_TASK_ID}"
FAM="${FAMILIES[$(( i % 4 ))]}"
SYS="${SYSTEMS[$(( (i / 4) % 4 ))]}"
SEED=$(( i / 16 ))

OUT_CELL="results/p6_a15_uniform/${FAM}_${SYS}/seed_${SEED}"
if [ -f "${OUT_CELL}/metrics.json" ]; then
    echo "[a15 ${i}] SKIP — ${OUT_CELL}/metrics.json exists"
    exit 0
fi

echo "[a15 ${i}] ${FAM} × ${SYS} × seed ${SEED} @ 2000 steps"
"${QLNN_VENV}/bin/python" scripts/run_multi_state_demo.py \
    --families "${FAM}" --systems "${SYS}" --seeds "${SEED}" \
    --steps 2000 --out results/p6_a15_uniform
