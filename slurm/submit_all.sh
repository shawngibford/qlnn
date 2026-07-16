#!/usr/bin/env bash
# =====================================================================
# Submit the full Phase C committed scope (222 cells, ~115 core-hr,
# ~2-3 hr wall-clock) as five concurrent job arrays + a dependent
# aggregation job.
#
# Usage (from this directory, on an Anvil login node):
#   cd $QLNN_ROOT/slurm && ./submit_all.sh
#
# Refuses to run until:
#   - config.env has a real account string (not CHANGE_ME), and
#   - the SMOKE_PASSED marker exists (create it manually after
#     inspecting the 00_smoke.sbatch outputs).
#
# 03b (optional PDE-side A17, ~40 core-hr) is NOT submitted here.
# =====================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source config.env

[ "${QLNN_ACCOUNT}" != "CHANGE_ME" ] || {
    echo "FATAL: edit slurm/config.env — QLNN_ACCOUNT is still CHANGE_ME"; exit 1; }
[ -f SMOKE_PASSED ] || {
    echo "FATAL: SMOKE_PASSED marker missing."
    echo "  Run:  sbatch -A ${QLNN_ACCOUNT} -p ${QLNN_DEBUG_PARTITION} 00_smoke.sbatch"
    echo "  Inspect logs/smoke_*.out, then:  touch SMOKE_PASSED"
    exit 1; }

mkdir -p logs
SB="sbatch --parsable -A ${QLNN_ACCOUNT} -p ${QLNN_PARTITION}"

J1=$(${SB} 01_kuramoto_kdv.sbatch)
J2=$(${SB} 02_a15_uniform_ode.sbatch)
J3=$(${SB} 03_a17_qcpinn_variants.sbatch)
J4=$(${SB} 04_a16_forecaster.sbatch)
J5=$(${SB} 05_a19_baselines.sbatch)
echo "Submitted arrays: M3=${J1}  A15=${J2}  A17=${J3}  A16=${J4}  A19=${J5}"

J9=$(${SB} --dependency="afterok:${J1}:${J2}:${J3}:${J4}:${J5}" 99_aggregate.sbatch)
echo "Aggregation job ${J9} queued (runs after all five arrays succeed)."
echo ""
echo "Monitor:   squeue --me"
echo "Progress:  find ${QLNN_ROOT}/results/anvil/p6_* -name metrics.json | wc -l   # expect 222"
