#!/usr/bin/env bash
# Sample-efficiency sweep — implements Claim 3 from hypothesis.md v2.
#
# 8 multi-seed runs: classical H=4 and QLNN at h=3, at training-fraction
# {10, 25, 50, 100}%. The 100% rows would normally duplicate the existing
# param_sweep/euler_h3_hidden4/ and qlnn_hybrid_h3/ artifacts; we still
# re-run them here to be self-contained and to write predictions.npz with
# the (post-Phase-C) shape if the older runs are missing fields.
#
# Usage:
#   bash scripts/run_sample_efficiency.sh                # default: both stacks
#   bash scripts/run_sample_efficiency.sh classical      # only classical (~20 min)
#   bash scripts/run_sample_efficiency.sh qlnn           # only qlnn (~4 hours)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/sample_efficiency
which=${1:-both}

if [ "$which" = "both" ] || [ "$which" = "classical" ]; then
  for PCT in 10 25 50 100; do
    echo "=== classical H=4 pct=${PCT} ==="
    .venv/bin/python scripts/train_baseline.py \
      --config configs/sample_efficiency/classical_h4_h3_pct${PCT}.yaml \
      --output-dir results/sample_efficiency/classical_h4_h3_pct${PCT} \
      --quiet
  done
fi

if [ "$which" = "both" ] || [ "$which" = "qlnn" ]; then
  for PCT in 10 25 50 100; do
    echo "=== QLNN pct=${PCT} ==="
    .venv/bin/python scripts/train_qlnn.py \
      --config configs/sample_efficiency/qlnn_h3_pct${PCT}.yaml \
      --output-dir results/sample_efficiency/qlnn_h3_pct${PCT} \
      --quiet
  done
fi
