#!/usr/bin/env bash
# =====================================================================
# One-time environment bootstrap on an Anvil LOGIN node.
#
# Usage:  bash slurm/env_setup.sh
#
# Idempotent: safe to re-run (git pull + pip install -e are no-ops
# when current). Ends by running the paper integrity gate so we prove
# the environment reproduces the committed numbers BEFORE any compute
# is spent (NEXT_STEPS.md Phase B gate).
# =====================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

REPO_URL="https://github.com/shawngibford/qlnn.git"

echo "[env_setup] target: ${QLNN_ROOT}"

# --- Python 3.11 module (Anvil provides python via modules) ----------
module purge
module load python/3.11 2>/dev/null || module load python 2>/dev/null || {
    echo "[env_setup] WARNING: no python module found; relying on system python3.11"
}
PYBIN="$(command -v python3.11 || command -v python3)"
"${PYBIN}" --version | grep -q "3\.1[123]" || {
    echo "[env_setup] FATAL: need Python >=3.11,<3.14 — got $(${PYBIN} --version)"
    exit 1
}

# --- Clone or update the repo ----------------------------------------
if [ -d "${QLNN_ROOT}/.git" ]; then
    git -C "${QLNN_ROOT}" pull --ff-only
else
    mkdir -p "$(dirname "${QLNN_ROOT}")"
    git clone "${REPO_URL}" "${QLNN_ROOT}"
fi
cd "${QLNN_ROOT}"

# --- Virtualenv --------------------------------------------------------
if [ ! -x "${QLNN_VENV}/bin/python" ]; then
    "${PYBIN}" -m venv "${QLNN_VENV}"
fi
"${QLNN_VENV}/bin/python" -m pip install -U pip
"${QLNN_VENV}/bin/python" -m pip install -e ".[dev]"

# --- Prove the env reproduces the committed numbers -------------------
echo "[env_setup] running paper integrity gate ..."
PYTHONPATH=src "${QLNN_VENV}/bin/python" scripts/verify_paper_integrity.py
echo "[env_setup] DONE. Next: sbatch slurm/00_smoke.sbatch"
