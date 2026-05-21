#!/usr/bin/env bash
# Reproduce every paper-table number from scratch.
#
# Total wall-clock estimate: ~8 hours on a single MacBook Pro M1
# (~30 min classical, ~4 hr QLNN at h=1, ~4 hr QLNN at h=3, parallel possible).
# All sweeps are independent and can be launched concurrently.
#
# After completion, every numerical claim in PAPER_SUMMARY.md should be
# regenerable from the on-disk results/ tree, verified by:
#   .venv/bin/python scripts/verify_paper_integrity.py
#
# Usage:
#   bash scripts/reproduce_paper.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Sanity check the environment.
.venv/bin/python -m pytest -q --no-header > /dev/null
echo "[pytest] $(\.venv/bin/python -m pytest --collect-only -q 2>/dev/null | tail -1 | awk '{print $1, $2}') passing"

mkdir -p results

# -- Headline classical baselines (Phase A/B locked protocol; ~5 min each) --
echo ""
echo "[1/6] Classical baselines (Euler / dopri5 / +physics / fixed-OD-leak comparator)"
.venv/bin/python scripts/train_baseline.py --config configs/baseline_euler_fast.yaml \
    --output-dir results/baseline_classical_euler --quiet
.venv/bin/python scripts/train_baseline.py --config configs/baseline.yaml \
    --output-dir results/baseline_classical_dopri5 --device cpu --quiet
.venv/bin/python scripts/train_baseline.py --config configs/baseline_physics.yaml \
    --output-dir results/baseline_classical_physics --device cpu --quiet
.venv/bin/python scripts/train_baseline.py --config configs/baseline_euler_fixed_od.yaml \
    --output-dir results/baseline_classical_euler_fixed_od --quiet

# -- Horizon ablation (Phase B; ~10 min for h=1,3,6,12) --
echo ""
echo "[2/6] Horizon ablation (Euler at h=1, 3, 6, 12)"
bash scripts/run_horizon_sweep.sh

# -- Param-matched classical sweep (Phase C; ~25 min) --
echo ""
echo "[3/6] Param-matched classical sweep (H ∈ {2,4,8,16,32} at h=3)"
bash scripts/run_param_sweep.sh

# -- QLNN canonical runs (Phase C; ~4 hr) --
echo ""
echo "[4/6] QLNN n=5 runs at h=1 and h=3 (the slow ones — ~2 hr each)"
.venv/bin/python scripts/train_qlnn.py --config configs/qlnn_hybrid.yaml \
    --output-dir results/qlnn_hybrid_h1 --quiet
.venv/bin/python scripts/train_qlnn.py --config configs/horizon/qlnn_hybrid_h3.yaml \
    --output-dir results/qlnn_hybrid_h3 --quiet
.venv/bin/python scripts/train_qlnn.py --config configs/horizon/qlnn_hybrid_h3_physics.yaml \
    --output-dir results/qlnn_hybrid_h3_physics --quiet

# -- Step 5: effective dimension (Claim 2; ~5 min) --
echo ""
echo "[5/6] Effective dimension (Claim 2)"
.venv/bin/python scripts/run_effective_dimension.py

# -- Step 6: sample efficiency (Claim 3; ~4 hr QLNN side) --
echo ""
echo "[6/6] Sample-efficiency sweep (Claim 3)"
bash scripts/run_sample_efficiency.sh

# -- Aggregate paper tables --
echo ""
echo "[final] Aggregating paper tables..."
.venv/bin/python scripts/summarize_baselines.py \
    --runs results/baseline_classical_euler results/baseline_classical_dopri5 \
           results/baseline_classical_physics results/baseline_classical_euler_fixed_od \
    --labels "Liquid-ODE (Euler, train-only OD)" \
             "Liquid-ODE (dopri5, train-only OD)" \
             "Liquid-ODE +physics (train-only OD)" \
             "Liquid-ODE (Euler, fixed [0,3.8] OD - leak sensitivity)" \
    --output results/baseline_classical_table

.venv/bin/python scripts/summarize_horizon_sweep.py \
    --runs results/horizon_sweep/euler_h1 results/horizon_sweep/euler_h3 \
           results/horizon_sweep/euler_h6 results/horizon_sweep/euler_h12 \
    --label "Liquid-ODE (Euler)" \
    --output results/horizon_sweep_table

.venv/bin/python scripts/summarize_param_sweep.py --qlnn-run results/qlnn_hybrid_h3
.venv/bin/python scripts/summarize_sample_efficiency.py
.venv/bin/python scripts/make_paper_figures.py

echo ""
echo "======================================================================"
echo "Old OD program reproduced. Now the pivot program (P3.5 → P7.5)."
echo "======================================================================"
echo ""
echo "[P3.5/P3.6] Multi-state ODE solver matrix (4 quantum families × 3 ODE)"
echo "  → results/p3_6_multi_state/"
PYTHONPATH=src .venv/bin/python scripts/run_multi_state_demo.py
PYTHONPATH=src .venv/bin/python scripts/make_multi_state_figure.py

echo ""
echo "[P3.7/P3.8] PDE solver scaffolding + peer-review iteration"
echo "  → results/p3_7_pde_solver/, results/p3_8_review/"
PYTHONPATH=src .venv/bin/python scripts/run_pde_solver_demo.py
PYTHONPATH=src .venv/bin/python scripts/make_pde_solver_figure.py
PYTHONPATH=src .venv/bin/python scripts/run_p3_8_review_iteration.py
PYTHONPATH=src .venv/bin/python scripts/make_p3_8_review_figure.py

echo ""
echo "[P3.9] PDE multi-family port matrix (4 quantum × 3 PDEs)"
echo "  → results/p3_9_pde_matrix/"
PYTHONPATH=src .venv/bin/python scripts/run_p3_9_pde_matrix.py
PYTHONPATH=src .venv/bin/python scripts/make_p3_9_pde_matrix_figure.py

echo ""
echo "[P4] Forecaster autoregressive rollout (5 quantum families × 3 ODE)"
echo "  → results/p4_forecaster_rollout/"
PYTHONPATH=src .venv/bin/python scripts/run_p4_forecaster_rollout.py
PYTHONPATH=src .venv/bin/python scripts/make_p4_forecaster_rollout_figure.py

echo ""
echo "[P5] Mandatory baselines + FORECASTER-task H1 verdict"
echo "  → results/p5_matched_baselines/, results/p5_h1_verdict/"
PYTHONPATH=src .venv/bin/python scripts/run_p5_matched_baselines.py
PYTHONPATH=src .venv/bin/python scripts/make_p5_h1_verdict_figure.py

echo ""
echo "[P7] T3 mechanism diagnostics"
echo "  → results/p7_t3_mechanism/"
PYTHONPATH=src .venv/bin/python scripts/run_p7_t3_mechanism.py
PYTHONPATH=src .venv/bin/python scripts/make_p7_mechanism_figure.py

echo ""
echo "[P7.5] PRIMARY SOLVER-TASK H1 VERDICT + HPO sensitivity + H3 LOO"
echo "  → results/p7_5_solver_h1/, results/p7_5_hpo_sensitivity/,"
echo "    results/p7_5_h3_loo/"
PYTHONPATH=src .venv/bin/python scripts/run_p7_5_solver_h1.py
PYTHONPATH=src .venv/bin/python scripts/make_p7_5_solver_h1_figure.py
PYTHONPATH=src .venv/bin/python scripts/run_p7_5_hpo_sensitivity.py
PYTHONPATH=src .venv/bin/python scripts/run_p7_5_h3_loo.py

echo ""
echo "======================================================================"
echo "ALL DONE. Both the OD program AND the pivot program (P3.5 → P7.5)"
echo "are reproduced. Verify integrity end-to-end with:"
echo "  .venv/bin/python scripts/verify_paper_integrity.py"
echo "======================================================================"
