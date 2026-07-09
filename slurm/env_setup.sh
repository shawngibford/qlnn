#!/usr/bin/env bash
# =====================================================================
# One-time environment bootstrap on an Anvil LOGIN node.
#
# Usage:  bash slurm/env_setup.sh
#
# Uses Anvil's anaconda module + a conda env (the coauthor's working
# QPINN pattern), NOT a python-module venv. Idempotent: safe to
# re-run. Ends by running the paper integrity gate so we prove the
# environment reproduces the committed numbers BEFORE any compute is
# spent (NEXT_STEPS.md Phase B gate).
# =====================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

REPO_URL="https://github.com/shawngibford/qlnn.git"

echo "[env_setup] repo target : ${QLNN_ROOT}"
echo "[env_setup] conda env   : ${QLNN_CONDA_ENV}"

# --- Anaconda module (the Anvil-supported path) -----------------------
module load anaconda

# --- Clone or update the repo -----------------------------------------
if [ -d "${QLNN_ROOT}/.git" ]; then
    git -C "${QLNN_ROOT}" pull --ff-only || {
        echo "[env_setup] WARNING: git pull failed (local changes?); continuing with current checkout"
    }
else
    mkdir -p "$(dirname "${QLNN_ROOT}")"
    git clone "${REPO_URL}" "${QLNN_ROOT}"
fi
cd "${QLNN_ROOT}"

# --- Conda env: create if missing, then activate ----------------------
if ! conda env list | awk '{print $1}' | grep -qx "${QLNN_CONDA_ENV}"; then
    echo "[env_setup] creating conda env '${QLNN_CONDA_ENV}' (python 3.11) ..."
    conda create -y -n "${QLNN_CONDA_ENV}" python=3.11
fi
set +u; conda activate "${QLNN_CONDA_ENV}"; set -u

python --version | grep -q "3\.1[123]" || {
    echo "[env_setup] FATAL: env '${QLNN_CONDA_ENV}' is $(python --version), need >=3.11,<3.14"
    echo "            Recreate with: conda create -y -n ${QLNN_CONDA_ENV} python=3.11"
    exit 1
}

# --- Install the project into the conda env ---------------------------
python -m pip install -U pip
python -m pip install -e ".[dev]"

# --- Prove the env reproduces the committed numbers -------------------
echo "[env_setup] running paper integrity gate ..."
PYTHONPATH=src python scripts/verify_paper_integrity.py
echo "[env_setup] DONE. Next:  cd ${QLNN_ROOT}/slurm && ./go.sh"
