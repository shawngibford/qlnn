#!/usr/bin/env bash
# =====================================================================
# Smoke gate — automated verification of the 00_smoke.sbatch outputs.
# Submitted by go.sh with --dependency=afterok:<smoke job id>.
#
# Replaces the manual "inspect logs, then touch SMOKE_PASSED" step in
# the one-command pipeline. Checks, for each of the 5 smoke cells:
#   1. the log ends with "[smoke N] OK",
#   2. a metrics.json was written,
#   3. no error.json exists anywhere under the smoke output dirs.
# On success it touches SMOKE_PASSED, releasing the downstream arrays
# (which go.sh queued with afterok on THIS job). On failure it exits
# nonzero → SLURM cancels every downstream job automatically
# (--kill-on-invalid-dep=yes).
# =====================================================================
#SBATCH --job-name=qlnn-smoke-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/smoke_gate.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/config.env"
cd "${QLNN_ROOT}"

FAIL=0

echo "=== Smoke gate — $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# 1. Every smoke log ends OK.
for i in 0 1 2 3 4; do
    LOG="${SLURM_SUBMIT_DIR}/logs/smoke_${i}.out"
    if [ -f "${LOG}" ] && grep -q "\[smoke ${i}\] OK" "${LOG}"; then
        echo "  [${i}] log OK"
    else
        echo "  [${i}] FAIL — no OK marker in ${LOG}"
        FAIL=1
    fi
done

# 2. Each workload's smoke cell wrote a metrics.json.
declare -a SMOKE_OUTPUTS=(
  "results/p6_kuramoto_kdv/kuramoto_chebyshev_dqc/seed_0/metrics.json"
  "results/p6_a15_uniform/chebyshev_dqc_lotka_volterra/seed_0/metrics.json"
  "results/p6_a17_qcpinn_variants/qcpinn_balanced_lotka_volterra/seed_0/metrics.json"
  "results/p6_a16_forecaster/lotka_volterra_data_reuploading/seed_0/metrics.json"
  "results/p6_a19_baselines/lotka_volterra_plain_neuralode/seed_0/metrics.json"
)
for f in "${SMOKE_OUTPUTS[@]}"; do
    if [ -f "${f}" ]; then
        echo "  metrics OK   ${f}"
    else
        echo "  metrics MISS ${f}"
        FAIL=1
    fi
done

# 3. Zero error.json anywhere in the smoke output trees.
N_ERR=$(find results/p6_* -name error.json 2>/dev/null | wc -l)
echo "  error.json count: ${N_ERR}"
[ "${N_ERR}" -eq 0 ] || FAIL=1

if [ "${FAIL}" -ne 0 ]; then
    echo "SMOKE GATE FAILED — downstream arrays will be cancelled."
    exit 1
fi

touch "${SLURM_SUBMIT_DIR}/SMOKE_PASSED"
echo "SMOKE GATE PASSED — SMOKE_PASSED marker written; arrays released."
