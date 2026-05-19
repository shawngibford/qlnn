#!/usr/bin/env bash
# Unified model×dataset matrix runner — 231 configs (21 models × 11
# datasets), 3-seed proxy, LOCKED protocol. Dispatches by stack:
#   *classical* -> scripts/train_baseline.py
#   *qlnn*      -> scripts/train_qlnn.py
# Results -> results/unified_matrix/<stem>/.
#
# GATED + dataset-grouped: 231 configs is multi-day. Run ONE dataset
# group at a time (≈21 configs ≈ proxy budget) so each is a gateable
# unit with a go/no-go between groups. NEVER launch while the Option-B
# sweep owns the machine (JAX is CPU-bound; they would contend).
#
# Prereqs (zero compute):
#   python scripts/generate_synthetic_ode_data.py
#   python scripts/generate_unified_matrix.py
#
# Usage:
#   ONLY=qzeta_od     bash scripts/run_unified_matrix.sh --quiet
#   ONLY=lorenz_m472  bash scripts/run_unified_matrix.sh --quiet
#   STACK=classical ONLY=lorenz_m472 bash scripts/run_unified_matrix.sh
#   # (omit ONLY to run the whole 231 — discouraged without grouping)

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
EXTRA_ARGS=("$@")
ONLY="${ONLY:-}"          # dataset key filter (recommended: one group)
STACK="${STACK:-}"        # optional: classical | qlnn

if [ -d "$REPO_ROOT/src/qlnn_/circuits" ]; then
    export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
fi

OUT_BASE="results/unified_matrix"
mkdir -p "$OUT_BASE"
shopt -s nullglob
CONFIGS=(configs/unified_matrix/*.yaml)
[ ${#CONFIGS[@]} -gt 0 ] || { echo "no configs — run generate_unified_matrix.py"; exit 1; }

count=0
for cfg in "${CONFIGS[@]}"; do
    stem="$(basename "$cfg" .yaml)"
    [ "$stem" = "matrix_manifest" ] && continue
    dataset="${stem%%__*}"
    model="${stem#*__}"
    if [ -n "$ONLY" ] && [ "$dataset" != "$ONLY" ]; then continue; fi
    case "$model" in
        classical_*) trainer="scripts/train_baseline.py"; this="classical";;
        qlnn_*)      trainer="scripts/train_qlnn.py";      this="qlnn";;
        *) echo "[unified] ?? unknown model $model"; continue;;
    esac
    if [ -n "$STACK" ] && [ "$STACK" != "$this" ]; then continue; fi
    echo "[unified] ==> $stem  ($trainer)"
    "$PYTHON" "$trainer" --config "$cfg" \
        --output-dir "$OUT_BASE/$stem" "${EXTRA_ARGS[@]}"
    echo
    count=$((count + 1))
done
echo "[unified] ran $count configs (ONLY='${ONLY:-<all>}' STACK='${STACK:-<all>}')"
echo "[unified] next: $PYTHON scripts/build_dataset_baseline_locks.py --dataset ${ONLY:-<dataset>}"
