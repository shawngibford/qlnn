#!/usr/bin/env bash
# Run the synthetic ODE benchmark suite: 5 systems × {classical, QLNN},
# locked h=3 protocol, 5 seeds each. Dispatches by config filename:
#   *__classical.yaml -> scripts/train_baseline.py
#   *__qlnn.yaml       -> scripts/train_qlnn.py
# Results land in results/synthetic_ode/<config-stem>/.
#
# GATED: do not launch while the Option-B sweep owns the machine — JAX is
# CPU-bound and the two would contend. Run after O-2 / tier-1 free it.
#
# Prereqs (zero-compute, safe anytime):
#   python scripts/generate_synthetic_ode_data.py
#   python scripts/generate_synthetic_ode_configs.py
#
# Usage:
#   bash scripts/run_synthetic_ode_suite.sh                # all 10
#   bash scripts/run_synthetic_ode_suite.sh --quiet
#   ONLY=lorenz bash scripts/run_synthetic_ode_suite.sh    # one system, both stacks

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
EXTRA_ARGS=("$@")
ONLY="${ONLY:-}"

# Worktree: force the edited src/ onto PYTHONPATH (editable install points
# at the main repo) so the synthetic_ode loader + ansatz registry resolve.
if [ -d "$REPO_ROOT/src/qlnn_/circuits" ]; then
    export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
fi

OUT_BASE="results/synthetic_ode"
mkdir -p "$OUT_BASE"

if [ ! -f data/synthetic/manifest.json ]; then
    echo "[synthetic_ode] missing data/synthetic/manifest.json — run:"
    echo "    $PYTHON scripts/generate_synthetic_ode_data.py"
    exit 1
fi

shopt -s nullglob
CONFIGS=(configs/synthetic_ode/*.yaml)
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "[synthetic_ode] no configs — run scripts/generate_synthetic_ode_configs.py"
    exit 1
fi

echo "[synthetic_ode] ${#CONFIGS[@]} configs; ONLY='${ONLY:-<all>}'"
for cfg in "${CONFIGS[@]}"; do
    stem="$(basename "$cfg" .yaml)"            # e.g. lorenz__qlnn
    system="${stem%%__*}"
    stack="${stem##*__}"
    if [ -n "$ONLY" ] && [ "$system" != "$ONLY" ]; then
        continue
    fi
    out_dir="$OUT_BASE/$stem"
    if [ "$stack" = "classical" ]; then
        trainer="scripts/train_baseline.py"
    else
        trainer="scripts/train_qlnn.py"
    fi
    echo "[synthetic_ode] ==> $stem  ($trainer)"
    "$PYTHON" "$trainer" \
        --config "$cfg" \
        --output-dir "$out_dir" \
        "${EXTRA_ARGS[@]}"
    echo
done
echo "[synthetic_ode] all done. Summarize next (T-suite figures TBD)."
