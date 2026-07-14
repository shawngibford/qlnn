#!/usr/bin/env bash
# =====================================================================
# Smoke pass — ONE representative cell from each Phase C workload.
# Submit from the slurm/ directory:
#   cd $QLNN_ROOT/slurm
#   sbatch 00_smoke.sbatch     (account/partition embedded in headers)
#
# Gate per ACCESS_APPLICATION.md: "launch the SLURM array only after
# smoke passes." After all 5 tasks succeed, inspect outputs, then:
#   touch SMOKE_PASSED   (marker consumed by submit_all.sh)
# =====================================================================
#SBATCH --job-name=qlnn-smoke
#SBATCH -A chm260071
#SBATCH -p shared
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/smoke_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/config.env"
eval "${QLNN_TASK_ENV}"
module load anaconda
# conda activate references unset vars internally — suspend nounset
set +u; conda activate "${QLNN_CONDA_ENV}"; set -u
cd "${QLNN_ROOT}"
PY="python -u"
export PYTHONPATH=src

case "${SLURM_ARRAY_TASK_ID}" in
  0)  # M3: kuramoto chebyshev_dqc seed 0 (matches the 2026-05-28 laptop smoke)
      ${PY} scripts/run_p7_8_h1_kuramoto_kdv.py --confirm --cell-index 0
      ;;
  1)  # A15: one uniform-2000-step ODE cell
      ${PY} scripts/run_multi_state_demo.py \
          --families chebyshev_dqc --systems lotka_volterra --seeds 0 \
          --steps 2000 --out results/p6_a15_uniform
      ;;
  2)  # A17: one qcpinn-variant cell
      ${PY} scripts/run_multi_state_demo.py \
          --families qcpinn_balanced --systems lotka_volterra --seeds 0 \
          --steps 2000 --out results/p6_a17_qcpinn_variants
      ;;
  3)  # A16: one forecaster cell at the A19 2000-step budget
      ${PY} scripts/run_p4_forecaster_rollout.py \
          --families data_reuploading --systems lotka_volterra --seeds 0 \
          --train-steps 2000 --out results/p6_a16_forecaster
      ;;
  4)  # A19: one baseline cell at the 2000-step budget
      ${PY} scripts/run_p5_matched_baselines.py \
          --families plain_neuralode --systems lotka_volterra --seeds 0 \
          --train-steps 2000 --no-verdict --out results/p6_a19_baselines
      ;;
esac
echo "[smoke ${SLURM_ARRAY_TASK_ID}] OK"
