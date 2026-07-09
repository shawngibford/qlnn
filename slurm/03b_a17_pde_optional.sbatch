#!/usr/bin/env bash
# =====================================================================
# A17 (OPTIONAL) — qcpinn variants, PDE side. 36 cells (3 variants ×
# 4 PDEs × 3 seeds), ~40 core-hr. NOT submitted by submit_all.sh —
# submit manually only if the allocation has headroom:
#   sbatch -A $QLNN_ACCOUNT -p $QLNN_PARTITION 03b_a17_pde_optional.sbatch
#
# PREREQUISITE: the 2D ports of the qcpinn variants
# (qcpinn_balanced_2d etc.) must be registered in the P3.9 PDE-matrix
# QUANTUM_FAMILIES dict before this runs. As of the v1.2 tree they are
# NOT — this script fails fast with a clear message if so. Wiring the
# 2D variants is a ~half-day dev task documented in REMEDIATION_PLAN
# Tier 3 as the "optional PDE-side qcpinn attribution extension".
#
# Index decode: i = seed*12 + pde*3 + variant
#   variant = i % 3, pde = (i / 3) % 4, seed = i / 12
# =====================================================================
#SBATCH --job-name=qlnn-a17pde
#SBATCH -A chm260071
#SBATCH -p shared
#SBATCH --array=0-35%64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/a17pde_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/config.env"
eval "${QLNN_TASK_ENV}"
module load anaconda
# conda activate references unset vars internally — suspend nounset
set +u; conda activate "${QLNN_CONDA_ENV}"; set -u
cd "${QLNN_ROOT}"
export PYTHONPATH=src

VARIANTS=(qcpinn_balanced_2d qcpinn_quantum_2d qcpinn_full_q_2d)
PDES=(heat burgers_smooth allen_cahn burgers_shock)

i="${SLURM_ARRAY_TASK_ID}"
VAR="${VARIANTS[$(( i % 3 ))]}"
PDE="${PDES[$(( (i / 3) % 4 ))]}"
SEED=$(( i / 12 ))

# Fail fast if the 2D variant family isn't registered yet.
python -u - "$VAR" <<'EOF'
import sys
from qlnn_.training.p3_9_pde_matrix import QUANTUM_FAMILIES
fam = sys.argv[1]
if fam not in QUANTUM_FAMILIES:
    sys.exit(f"FATAL: {fam!r} not registered in p3_9_pde_matrix."
             f" Wire the 2D variant ports first (REMEDIATION_PLAN Tier 3"
             f" optional extension). Registered: {list(QUANTUM_FAMILIES)}")
EOF

OUT_CELL="results/p6_a17_pde_variants/${PDE}_${VAR}/seed_${SEED}"
if [ -f "${OUT_CELL}/metrics.json" ]; then
    echo "[a17pde ${i}] SKIP — exists"
    exit 0
fi

echo "[a17pde ${i}] ${VAR} × ${PDE} × seed ${SEED}"
python -u scripts/run_p3_9_pde_matrix.py \
    --families "${VAR}" --pdes "${PDE}" --seeds "${SEED}" \
    --out results/p6_a17_pde_variants
