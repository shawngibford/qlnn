#!/usr/bin/env bash
# Param-matched classical Liquid-ODE sweep (Phase C, Tier 2 #2.1).
# Trains hidden_size ∈ {2, 4, 8, 16, 32} at h=3 horizon, 5 seeds each.
# Total ~25 runs; should finish in ~25 min on MPS / a similar CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/param_sweep

for H in 2 4 8 16 32; do
  echo "=== hidden_size=${H} ==="
  .venv/bin/python scripts/train_baseline.py \
    --config configs/param_sweep/baseline_euler_h3_hidden${H}.yaml \
    --output-dir results/param_sweep/euler_h3_hidden${H} \
    --quiet
done

echo "=== sweep complete ==="
echo "summarize with: .venv/bin/python scripts/summarize_param_sweep.py"
