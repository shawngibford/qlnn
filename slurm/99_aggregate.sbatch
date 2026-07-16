#!/usr/bin/env bash
# =====================================================================
# Phase D prep — post-sweep audit + tarball. Submitted by
# submit_all.sh with --dependency=afterok:<job ids of 01..05>.
#
# Does NOT modify locked integrity numbers or run verdict refreshes —
# that is Phase D / M5, done locally with human review. This job only:
#   1. counts metrics.json vs expected per workload,
#   2. lists every error.json,
#   3. tars results/anvil/p6_* for scp copy-back.
# =====================================================================
#SBATCH --job-name=qlnn-aggregate
#SBATCH -A chm260071
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/aggregate.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/config.env"
cd "${QLNN_ROOT}"

declare -A EXPECTED=(
  [results/anvil/p6_kuramoto_kdv]=30
  [results/anvil/p6_a15_uniform]=48
  [results/anvil/p6_a17_qcpinn_variants]=45
  [results/anvil/p6_a16_forecaster]=63
  [results/anvil/p6_a19_baselines]=36
)

echo "===================================================================="
echo " Phase C completion audit — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "===================================================================="
TOTAL_GOT=0; TOTAL_WANT=0; FAIL=0
for dir in "${!EXPECTED[@]}"; do
    want="${EXPECTED[$dir]}"
    got=$(find "${dir}" -name metrics.json 2>/dev/null | wc -l)
    TOTAL_GOT=$(( TOTAL_GOT + got )); TOTAL_WANT=$(( TOTAL_WANT + want ))
    status="OK "
    [ "${got}" -lt "${want}" ] && { status="MISSING"; FAIL=1; }
    printf "  %-38s %3d / %3d  %s\n" "${dir}" "${got}" "${want}" "${status}"
done
echo "--------------------------------------------------------------------"
printf "  TOTAL %d / %d cells\n" "${TOTAL_GOT}" "${TOTAL_WANT}"

echo ""
echo "=== error.json inventory (must be empty after investigation) ==="
find results/anvil/p6_* -name error.json 2>/dev/null | tee logs/error_json_list.txt || true
N_ERR=$(wc -l < logs/error_json_list.txt || echo 0)
echo "  ${N_ERR} error.json file(s)"

echo ""
echo "=== Tarball for copy-back ==="
TARBALL="qlnn_phase_c_results_$(date -u +%Y%m%d).tar.gz"
tar czf "${TARBALL}" results/anvil/p6_*
echo "  wrote ${QLNN_ROOT}/${TARBALL} ($(du -h "${TARBALL}" | cut -f1))"

echo ""
echo "=== Phase D checklist (run LOCALLY after scp) ==="
cat <<'EOF'
  1. scp the tarball back and untar into the local repo.
  2. Re-run H1 aggregators over the refreshed matrix.
  3. Update paper tables + locked integrity values (M5, human review).
  4. PYTHONPATH=src .venv/bin/python scripts/verify_paper_integrity.py
  5. bash paper/build.sh && bash paper/build_supplement.sh
EOF

exit "${FAIL}"
