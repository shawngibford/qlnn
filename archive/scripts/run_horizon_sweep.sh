#!/usr/bin/env bash
# Canonical horizon-sweep for the QLNN paper.
#
# Runs the classical Liquid-ODE (Euler) baseline across forecast horizons
# {1, 3, 6, 12} hours, writing one multi-seed artifact directory per horizon.
# The R3 reviewer flagged that the 1h horizon is dominated by persistence
# (lag-6 autocorr ~= 0.99); this sweep produces the per-horizon table that
# makes any QLNN-vs-classical model claim discriminating.
#
# Outputs (one per horizon):
#   results/horizon_sweep/euler_h${H}/seeds_summary.json
#   results/horizon_sweep/euler_h${H}/baselines.json    (persistence + linear)
#   results/horizon_sweep/euler_h${H}/protocol.json     (horizon_hours stored here)
#
# After running this, generate the paper-style horizon table with:
#   python scripts/summarize_horizon_sweep.py \
#       --runs results/horizon_sweep/euler_h{1,3,6,12} \
#       --label "Liquid-ODE (Euler)" \
#       --output results/horizon_sweep
#
# h=12 has only 17 test windows — keep it but caveat the row as
# supplementary in the paper (see configs/horizon/baseline_euler_h12.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/horizon_sweep

for H in 1 3 6 12; do
  echo "=== Euler h${H} ==="
  .venv/bin/python scripts/train_baseline.py \
    --config configs/horizon/baseline_euler_h${H}.yaml \
    --output-dir results/horizon_sweep/euler_h${H} \
    --quiet
done

echo
echo "Sweep done. To produce the paper table:"
echo "  .venv/bin/python scripts/summarize_horizon_sweep.py \\"
echo "      --runs results/horizon_sweep/euler_h{1,3,6,12} \\"
echo "      --label 'Liquid-ODE (Euler)' \\"
echo "      --output results/horizon_sweep"
