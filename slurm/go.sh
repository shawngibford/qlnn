#!/usr/bin/env bash
# =====================================================================
# ONE-COMMAND Phase C pipeline for Anvil.
#
#   Usage (on an Anvil login node, from anywhere):
#       git clone https://github.com/shawngibford/qlnn.git
#       cd qlnn/slurm
#       ./go.sh                    # account chm260071 is pre-configured
#
#   (Optional: ./go.sh <other_account> to override.)
#
# That's it — log off. The script:
#   1. bootstraps the environment (anaconda module + conda env "qlnn"
#      + pip install + paper integrity gate),
#   2. queues the 5-cell smoke,
#   3. queues an automated smoke-verification gate (afterok:smoke),
#   4. queues all five production arrays gated on the smoke gate,
#   5. queues the final aggregation/tarball job gated on the arrays.
#
# Everything runs on the "shared" partition (same as the coauthor's
# working QPINN jobs). SLURM enforces the ordering. If ANY smoke cell
# fails, the gate job fails and every downstream job is cancelled
# automatically (--kill-on-invalid-dep=yes) — nothing burns allocation
# on a broken environment. Total unattended runtime: ~4-5 hr.
#
# Monitor any time with:   squeue --me
# Results land in:         $QLNN_ROOT/qlnn_phase_c_results_<date>.tar.gz
# =====================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# --- 0. Account handling ----------------------------------------------
if [ $# -ge 1 ]; then
    ACCOUNT="$1"
    # Persist into config.env so every later manual command works too.
    sed -i.bak "s/^export QLNN_ACCOUNT=.*/export QLNN_ACCOUNT=\"${ACCOUNT}\"/" config.env
    rm -f config.env.bak
fi
source config.env
[ "${QLNN_ACCOUNT}" != "CHANGE_ME" ] || {
    echo "FATAL: no account. Run:  ./go.sh <ACCESS_ACCOUNT>"; exit 1; }

echo "==> Account: ${QLNN_ACCOUNT}"

# --- 0b. Refuse to double-submit --------------------------------------
if squeue --me --noheader -o "%j" 2>/dev/null | grep -q "^qlnn-"; then
    echo "FATAL: qlnn-* jobs already in your queue. Inspect with"
    echo "       squeue --me    (cancel with scancel --me if intended)"
    exit 1
fi

# --- 1. Environment bootstrap (login node, ~5-10 min first time) ------
echo "==> [1/5] Environment bootstrap (conda env '${QLNN_CONDA_ENV}') + integrity gate ..."
bash env_setup.sh

mkdir -p logs
rm -f SMOKE_PASSED   # fresh gate every pipeline run

SB="sbatch --parsable -A ${QLNN_ACCOUNT}"

# --- 2. Smoke -----------------------------------------------------------
echo "==> [2/5] Queueing 5-cell smoke on '${QLNN_DEBUG_PARTITION}' ..."
J_SMOKE=$(${SB} -p "${QLNN_DEBUG_PARTITION}" 00_smoke.sbatch)
echo "    smoke job: ${J_SMOKE}"

# --- 3. Automated smoke gate -------------------------------------------
echo "==> [3/5] Queueing smoke gate (verifies outputs, writes SMOKE_PASSED) ..."
J_GATE=$(${SB} -p "${QLNN_PARTITION}" \
    --dependency="afterok:${J_SMOKE}" --kill-on-invalid-dep=yes \
    smoke_gate.sbatch)
echo "    gate job:  ${J_GATE}"

# --- 4. Production arrays, gated on the smoke gate ---------------------
echo "==> [4/5] Queueing five production arrays (222 cells) ..."
DEP="--dependency=afterok:${J_GATE} --kill-on-invalid-dep=yes"
J1=$(${SB} -p "${QLNN_PARTITION}" ${DEP} 01_kuramoto_kdv.sbatch)
J2=$(${SB} -p "${QLNN_PARTITION}" ${DEP} 02_a15_uniform_ode.sbatch)
J3=$(${SB} -p "${QLNN_PARTITION}" ${DEP} 03_a17_qcpinn_variants.sbatch)
J4=$(${SB} -p "${QLNN_PARTITION}" ${DEP} 04_a16_forecaster.sbatch)
J5=$(${SB} -p "${QLNN_PARTITION}" ${DEP} 05_a19_baselines.sbatch)
echo "    M3=${J1}  A15=${J2}  A17=${J3}  A16=${J4}  A19=${J5}"

# --- 5. Aggregation + tarball, gated on all arrays ---------------------
echo "==> [5/5] Queueing aggregation job ..."
J9=$(${SB} -p "${QLNN_PARTITION}" \
    --dependency="afterok:${J1}:${J2}:${J3}:${J4}:${J5}" \
    --kill-on-invalid-dep=yes \
    99_aggregate.sbatch)
echo "    aggregate: ${J9}"

cat <<EOF

========================================================================
 PIPELINE QUEUED — you can log off now.
========================================================================
 smoke(${J_SMOKE}) → gate(${J_GATE}) → arrays(${J1},${J2},${J3},${J4},${J5}) → aggregate(${J9})

 Watch progress:   squeue --me
 Smoke logs:       ${PWD}/logs/smoke_*.out
 Cell count:       find ${QLNN_ROOT}/results/anvil/p6_* -name metrics.json | wc -l   # → 222 when done
                   (smoke cells are index-0 array cells; skip-if-done
                    means they are never recomputed)
 Final artifact:   ${QLNN_ROOT}/qlnn_phase_c_results_<date>.tar.gz

 If anything fails, downstream jobs cancel automatically. Requeue a
 single failed array element with:  scontrol requeue <jobid>_<index>
 (all tasks are skip-if-done, so requeues never redo finished work).

 When the tarball exists, scp it back and hand it to Shawn for
 Phase D (verdict refresh + integrity + paper rebuild).
========================================================================
EOF
