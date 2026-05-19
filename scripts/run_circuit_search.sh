#!/usr/bin/env bash
# Run the per-axis circuit-search ablation (single-seed proxy budget).
#
# Reads every YAML in configs/circuit_search/ and runs scripts/train_qlnn.py
# against it, writing results to results/circuit_search/<config-stem>/.
#
# The reference cell (configs/circuit_search/reference.yaml) intentionally
# matches the existing results/qlnn_hybrid_h3/ run — if you set
# REUSE_REFERENCE=1 (default) the runner copies the existing single-seed
# subset from qlnn_hybrid_h3 instead of retraining it. Saves ~25 min.
#
# Wall-clock estimate: 11 configs × ~5 min single-seed = ~55 min on M1.
#
# Usage:
#   bash scripts/run_circuit_search.sh
#   bash scripts/run_circuit_search.sh --quiet
#   REUSE_REFERENCE=0 bash scripts/run_circuit_search.sh    # force re-train ref

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
EXTRA_ARGS=("$@")
REUSE_REFERENCE="${REUSE_REFERENCE:-1}"
# Override these to run the promotion phase against the same harness:
#   CONFIGS_DIR=configs/circuit_search_promoted \
#   RESULTS_DIR=results/circuit_search_promoted \
#   REUSE_REFERENCE=0 bash scripts/run_circuit_search.sh --quiet
CONFIGS_DIR="${CONFIGS_DIR:-configs/circuit_search}"
RESULTS_DIR="${RESULTS_DIR:-results/circuit_search}"

# If invoked from a git worktree where the editable install points at the
# main repo, force PYTHONPATH to the worktree's src/ so the new ansatz
# registry / circuit modules are picked up. Harmless in the main repo.
if [ -d "$REPO_ROOT/src/qlnn_/circuits" ]; then
    export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
fi

OUT_BASE="$RESULTS_DIR"
mkdir -p "$OUT_BASE"

CONFIGS=()
while IFS= read -r f; do
    CONFIGS+=("$f")
done < <(ls -1 "$CONFIGS_DIR"/*.yaml | sort)

echo "[circuit_search] ${#CONFIGS[@]} configs to run"
echo "[circuit_search] REUSE_REFERENCE=$REUSE_REFERENCE"
echo

for cfg in "${CONFIGS[@]}"; do
    stem="$(basename "$cfg" .yaml)"
    out_dir="$OUT_BASE/$stem"

    # Reference cell shortcut: the existing 5-seed qlnn_hybrid_h3 run already
    # contains seed_0/ — reuse it directly.
    if [ "$stem" = "reference" ] && [ "$REUSE_REFERENCE" = "1" ] && [ -d "results/qlnn_hybrid_h3/seed_0" ]; then
        echo "[circuit_search] reference: REUSING results/qlnn_hybrid_h3/ (set REUSE_REFERENCE=0 to retrain)"
        mkdir -p "$out_dir"
        # Symlink avoids duplicating large eqx checkpoints; if the OS doesn't
        # support symlinks (rare on macOS/Linux dev), fall back to cp.
        if [ ! -e "$out_dir/seed_0" ]; then
            ln -s "$REPO_ROOT/results/qlnn_hybrid_h3/seed_0" "$out_dir/seed_0" \
                || cp -R "results/qlnn_hybrid_h3/seed_0" "$out_dir/seed_0"
        fi
        for f in seeds_summary.json config.json protocol.json provenance.json baselines.json; do
            [ -f "results/qlnn_hybrid_h3/$f" ] && cp -f "results/qlnn_hybrid_h3/$f" "$out_dir/$f" || true
        done
        continue
    fi

    echo "[circuit_search] ==> $stem"
    "$PYTHON" scripts/train_qlnn.py \
        --config "$cfg" \
        --output-dir "$out_dir" \
        "${EXTRA_ARGS[@]}"
    echo
done

echo
echo "[circuit_search] all done. summarize with:"
echo "    $PYTHON scripts/summarize_circuit_search.py"
