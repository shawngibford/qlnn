# OD-era Purge Plan

> Synthesized from a 4-agent audit swarm 2026-05-27, then ground-
> truthed against the actual file tree (Purge-A had wrong figure
> count; Purge-D had a wrong "qZETA is critical" claim — verified
> false: current paper cites zero OD-era data/results/figures).

## TL;DR

- **Active paper is already ODE/PDE-clean** — zero qZETA, bioreactor,
  fermentation, or OD-era results references in any `paper/main.tex`,
  `paper/sections/*.tex`, or `paper/supplement.tex`.
- **The contamination is in `paper/figures/`, `results/`, `scripts/`,
  and a couple of `src/` + root `*.md` files** — none of which the
  paper actually consumes.
- **The integrity gate is the only load-bearing dependency** on OD-era
  results (lines 42–95 of `verify_paper_integrity.py`). Moving those
  results requires a one-shot path patch.
- Disposition: **ARCHIVE everything OD-era under `archive/`** (preserving
  git-tracked history; never `rm` from disk except for clearly empty
  artifacts), and **PATCH the integrity gate paths**.

## Verified ground truth (this session)

```
paper/figures/  -- 37 PDF files
   8 cited in current paper (KEEP)
  ~7 post-pivot diagnostic outputs (KEEP)
  ~22 OD-era artifacts (ARCHIVE)

results/         -- 35+ subdirectories
  15 KEEP-ACTIVE (p3_6, p3_8_review, p3_9_pde_matrix,
                  p4_forecaster_rollout, p5_*, p7_*)
   5 KEEP-FROZEN-but-gated (param_sweep, qlnn_hybrid_h3,
                  effective_dimension, sample_efficiency,
                  horizon_sweep) -- ARCHIVE + patch gate
  11 ARCHIVE-OD safe (baselines, circuit_search variants,
                  qlnn_hybrid_h1/h3_physics, horizon_sweep_table)
   4 EMPTY/STALE (p6_smoke/cheap_smooth/kuramoto_kdv from
                  G7 wrapper tests; p7_5_hpo_sensitivity tiny)

scripts/         -- 66 files
  33 KEEP-ACTIVE/INFRA (run_p*, make_p*, verify_paper_integrity, etc.)
  32 ARCHIVE-OD (train_liquid_od_baseline, hpo_liquid_od_forecast,
                 explore_qzeta, circuit_search_*, option_b_*,
                 summarize_*, build_*, generate_unified_matrix, etc.)
   1 evaluate-case-by-case (run_p6_group.sh just landed; KEEP)

src/quantum_liquid_neuralode/data_processing/preprocessor.py
  -- BioreactorDataPreprocessor; ARCHIVE (no post-pivot importers)

data/raw/qZETA_data_copy.csv
  -- ARCHIVE (no current paper claim uses it; OD trainers being
     archived alongside)

Root *.md
  -- existing archive/ already has 14 OD-era docs (commit d554037).
     Check root for stragglers: hypothesis.md, STEP5_MONOTONICITY_NOTE.md,
     STEP6_PLAN.md, the REVIEW_*.md set, qlnn_*.md if present.
```

## Disposition table

### A. Figures — paper/figures/ (37 PDFs, 37 PNGs = 74 files)

**KEEP (in `paper/figures/`)** — 8 cited + ~2 post-pivot useful:
- fig_p3_6_multi_state, fig_p3_8_review_iteration, fig_p3_9_pde_matrix
- fig_p4_forecaster_rollout, fig_qq_forecaster_lorenz
- fig_p5_h1_verdict, fig_p7_6_qlnn_hpo, fig_p7_mechanism
- fig_p7_5_solver_h1 (post-pivot, uncited but useful — KEEP for M5
  paper update)
- fig_p3_7_pde_solver — own README: "NOT a paper claim" — DELETE
- fig_p3_solver_demo — superseded by multi_state — DELETE

**ARCHIVE to `archive/figures/`** — ~25 OD-era + post-pivot diagnostics
that pull from OD data:
- fig_ansatz_axis_effects, fig_circuit_pareto, fig_baseline_metrics
- fig_circuit_gallery_{brickwall, data_reuploading, hardware_efficient,
  strongly_entangling}
- fig_dataset_overview, fig_effective_dimension
- fig_horizon_ablation, fig_horizon_full_metrics
- fig_master_comparison, fig_param_sweep
- fig_promotion_validation, fig_quantum_circuit, fig_reproducibility
- fig_sample_efficiency, fig_sample_efficiency_full
- fig_learning_curves, fig_forecast_trajectory, fig_pred_vs_actual
- fig_residual_analysis, fig_qq_analysis (OD-era Q-Q), fig_paired_bootstrap
- fig_seed_strip, fig_all_circuit_diagrams (data-source TBD; if OD,
  archive)

### B. Results — results/ (35+ dirs)

**KEEP-ACTIVE in `results/`** — 15 post-pivot dirs (p3_6, p3_8_review,
p3_9_pde_matrix, p4_forecaster_rollout, p5_h1_verdict,
p5_matched_baselines, p7_5_*, p7_6_*, p7_8_*, p7_10_*, p7_11_*,
p7_t3_mechanism, p7_8_kdv_gate).

**ARCHIVE to `archive/results/`** — 16 OD-era dirs:
- The 5 gate-touched: param_sweep, qlnn_hybrid_h3, effective_dimension,
  sample_efficiency, horizon_sweep (REQUIRES integrity-gate path patch)
- The 11 not gate-touched: baseline_classical_{dopri5, euler,
  euler_fixed_od, physics, table}, circuit_search,
  circuit_search_optuna, circuit_search_promoted, circuit_search_space,
  qlnn_hybrid_h1, qlnn_hybrid_h3_physics, horizon_sweep_table

**DELETE** — 4 empty dirs:
- p6_smoke, p6_cheap_smooth, p6_kuramoto_kdv (artifacts of G7 wrapper
  test runs), p7_5_hpo_sensitivity (tiny exploratory; integrity gate
  doesn't read it)

### C. Scripts — scripts/ (66 files)

**KEEP-ACTIVE** in `scripts/` — the 33 post-pivot + infra scripts per
Purge-C's list.

**ARCHIVE to `archive/scripts/`** — the 32 OD-era scripts per Purge-C:
- train_liquid_od_baseline.py, hpo_liquid_od_forecast.py
- explore_qzeta_dataset.py, compare_scaling.py
- summarize_baselines.py, summarize_horizon_sweep.py,
  summarize_param_sweep.py, summarize_sample_efficiency.py
- visualize_quantum_circuit.py
- build_circuit_search_space.py, generate_circuit_search_configs.py,
  circuit_search_optuna.py, summarize_circuit_search.py,
  run_circuit_search.sh, check_circuit_regression.py,
  promote_top_circuits.py
- build_dataset_baseline_locks.py, build_master_comparison.py,
  generate_option_b_configs.py, summarize_option_b.py
- generate_unified_matrix.py, run_unified_matrix.sh
- qlnn_smoke_encoder.py
- Plus the 4 `*_sh` reproducer scripts that wrap OD-era runs

### D. Src + data + references

**ARCHIVE to `archive/src/`:**
- `src/quantum_liquid_neuralode/data_processing/preprocessor.py`
  (BioreactorDataPreprocessor)
- (Verify no other OD-only modules: check `models/`, `training/`,
  `diagnostics/` for OD-specific code)

**ARCHIVE to `archive/data/`:**
- `data/raw/qZETA_data_copy.csv` (no current paper uses it)
- Any other `data/` OD-era files (the agents didn't fully enumerate;
  inspect at execution time)

**Root *.md files** — already mostly archived per commit d554037:
- Check repo root for any stragglers: hypothesis.md, STEP*.md,
  REVIEW_*.md, PAPER_SUMMARY.md, PROJECT_DOSSIER.md (latter 2 already
  in archive/ per ls)

### E. Integrity gate patch

Required after moves: 6 path updates in `scripts/verify_paper_integrity.py`:
- L42: `results/param_sweep/...` → `archive/results/param_sweep/...`
- L43: `results/qlnn_hybrid_h3/...` → `archive/results/qlnn_hybrid_h3/...`
- L52: `results/effective_dimension/...` → `archive/results/effective_dimension/...`
- L73-74: `results/sample_efficiency/...` → `archive/results/sample_efficiency/...`
- L88-95: `results/horizon_sweep/...` → `archive/results/horizon_sweep/...`

Test after patch: gate exits 0 + reports all OD-era numbers verified
from new paths.

## Execution plan (atomic commits)

1. **Setup**: `mkdir -p archive/{figures,results,scripts,src,data}`
2. **Figures archive** (one commit) — `git mv` the 25 OD-era figure
   pairs to `archive/figures/`
3. **Figures delete** (one commit) — `git rm` the 2 redundant figures
4. **Results archive — safe set** (one commit) — `git mv` the 11
   ARCHIVE-OD result dirs to `archive/results/`
5. **Results archive — gate-touched set + gate patch** (single commit) —
   `git mv` the 5 KEEP-FROZEN dirs and patch the integrity gate paths
6. **Results delete — empty set** (one commit) — `git rm -r` the 4
   empty p6_* dirs
7. **Scripts archive** (one commit) — `git mv` the 32 OD-era scripts
8. **Src archive** (one commit) — `git mv` preprocessor.py + its test
9. **Data archive** (one commit) — `git mv` qZETA CSV + any OD data
10. **Doc archive** (one commit) — `git mv` any straggler OD docs
11. **V-gate**: 4-gate verification + integrity exit 0
12. **Push** (single push after all commits land green)

Risk-mitigation: between each commit, V-gate must stay green. If any
commit breaks the gate, stop and patch before continuing.

## Authorization

This is destructive (file moves) and touches the integrity gate. The
moves are reversible via git but the gate-patch is a one-shot
correctness change.

User decision needed: proceed?
