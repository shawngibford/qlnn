# NEXT AGENT — read this first

**Where:** worktree `/Users/shawngibford/dev/phd/qlnn/.claude/worktrees/upbeat-elbakyan-68b56a/`, branch `claude/upbeat-elbakyan-68b56a`, HEAD `019e771`.

**One-line state:** Paper draft complete (main 17pp, supplement 7pp, all integrity-gated). OD-era artifacts fully purged from the active surface. M0 prep done. M1+M2 smoke verified post-purge pipeline green. M3 (kuramoto + KdV solver compute, ~14-16 hr) is unblocked and waiting for `--confirm`.

**One-command sanity check:**
```bash
PY=/Users/shawngibford/dev/phd/qlnn/.venv/bin/python
cd /Users/shawngibford/dev/phd/qlnn/.claude/worktrees/upbeat-elbakyan-68b56a
PYTHONPATH=src $PY -m pytest -q --tb=short                # 508 tests green
PYTHONPATH=src $PY scripts/verify_paper_integrity.py     # exit-0, OD-frozen + post-pivot
bash paper/build.sh && bash paper/build_supplement.sh    # main 17pp / supp 7pp clean
```

All four must exit-0 / build clean. If any does not, STOP and inspect before proceeding.

## What's next: M3 — kuramoto + KdV solver sweep

```bash
PYTHONPATH=src $PY scripts/run_p7_8_h1_kuramoto_kdv.py --dry-run     # 30 cells, verify plan
PYTHONPATH=src $PY scripts/run_p7_8_h1_kuramoto_kdv.py --confirm     # ACTUAL — 14-16 hr
```

Cells: 12 kuramoto QLNN (4 families × 3 seeds) + 3 kuramoto classical-PINN + 12 KdV QLNN (4 families × 3 seeds) + 3 KdV classical-PINN = **30 cells**. Output to `results/p6_kuramoto_kdv/`.

**Hard rules during M3:**
- Never `rm` the `data` symlink while jobs run.
- Don't push to master without explicit user authorization.
- Don't `git add -A`. Use explicit paths.
- `verify_paper_integrity.py` MUST stay exit-0 across the M3 run.

**If M3 stalls or fails on a cell:** the runner writes per-cell results as it goes (look in `results/p6_kuramoto_kdv/<system>_<ansatz>/seed_N/metrics.json`). Partial completion is fine. Resume by re-invoking with `--max-cells N` to staging-test subsets.

## After M3: M5 — verdict refresh + paper update (~1 hr)

1. Re-run the H1 aggregator on the full 9-system manifest (script: `scripts/run_p7_8_h1_n24.py` — extend or wrap to include kuramoto + KdV cells).
2. Update `paper/sections/05_h1_verdict.tex` and `fig:h1-verdict` caption with the refreshed numbers.
3. Update Table 1 (master verdicts) with the new row.
4. Bump `scripts/verify_paper_integrity.py` to lock the refreshed numbers.
5. 4-gate verification (paper build, supplement build, pytest, integrity).
6. Commit + push.

## What NOT to do

- **Do NOT** revive any archived OD-era code. The 8-commit purge today moved `archive/results/{baseline_*, circuit_search, param_sweep, qlnn_hybrid_h3, sample_efficiency, horizon_sweep, effective_dimension, ...}`, `archive/figures/fig_{baseline_*, circuit_*, horizon_*, param_*, sample_efficiency, qq_analysis, paired_bootstrap, seed_strip, learning_curves, forecast_trajectory, pred_vs_actual, residual_analysis, all_circuit_diagrams, ...}`, `archive/scripts/{train_baseline, train_qlnn, *_search*, *_option_b*, summarize_*, build_*, reproduce_paper.sh, ...}`, `archive/src/quantum_liquid_neuralode/data_processing/{preprocessor, qzeta}`, `archive/tests/{test_preprocessor, test_unified_matrix, test_effective_dimension, test_summarize_baselines, test_forecaster, test_synthetic_ode, test_circuit_search_optuna}`. The integrity gate now reads OD-frozen numbers from `archive/results/` (lines 42-95 of `verify_paper_integrity.py`). Don't undo this.
- **Do NOT** modify `src/qlnn_/training/multi_state_solver.py`'s `VECTOR_ODES` dict beyond extending `kuramoto.t1` if the M3 sweep needs longer horizons. The other entries are integrity-gated indirectly via per-cell metrics.
- **Do NOT** touch `paper/PAPER_SUMMARY.md` — it's the integrity gate's source of truth.
- **Do NOT** push to `main`. Use the worktree branch only.

## Pre-reg amendments (14 total in `PRE_REG_AMENDMENT.md`)

- **A11** = PRIMARY SOLVER (n=24 FALSIFIED with sign-flip)
- **A12** = forecaster LTC decomposition (P7.10, 3 verdicts)
- **A13** = complete 2×2 + τ-cross-check disagreement (P7.11) — the centerpiece
- **A14** = post-hoc Q-Q residual diagnostics on Lorenz forecaster cell (P8-polish; this session)

## Two PRIMARY headline numbers to never get wrong

- **Solver n=24:** Δ_diff = **−0.084**, CI **[−0.278, +0.061]**, FALSIFIED, point-estimate-flipped
- **Forecaster combined:** Δ_diff = **−0.501**, CI **[−0.804, −0.244]**, FALSIFIED, CI excludes 0

`verify_paper_integrity.py` mechanically gates both.

## Today's commit ledger (origin/claude/upbeat-elbakyan-68b56a, since `7e1de2e`)

```
019e771 feat(M3-prep): register kuramoto in VECTOR_ODES
78113c7 docs(purge): PURGE_PLAN.md — OD-era cleanup plan
87f9caa chore(purge): archive OD trainers + src OD modules + OD-only tests
115c131 chore(purge): archive 31 OD-era scripts to archive/scripts/
39ad028 chore(purge): archive 5 gate-touched OD-era result dirs + patch gate
afe3624 chore(purge): archive 11 OD-era result dirs + 4 OD-era top-level files
d720b73 chore(purge): delete 2 redundant post-pivot figures
96a415e chore(purge): also archive missed fig_ansatz_axis_effects.png
77e0044 chore(purge): archive 26 OD-era figures to archive/figures/
be5a41b feat(M0-G8): kuramoto + KdV M3 runner scaffold
613953d feat(M0-G7): system-group go/no-go shell wrapper
9372b9c feat(M0-G6): forecaster underfit-guard (pre-reg amendment A6)
b25beb7 feat(M0-G4): per-system PDE dataset hash assertion
517bbce docs(P6): launch plan v0.2 — corrected architecture + M0-M5 staging
fe92460 docs(P8-polish): supplement §2.3 — multi-family solver matrix
99d7f3b docs(P8-polish): add headline H1 verdict figure to main §5
b919de2 docs(P8-polish): main-paper PRX-readiness pass
c808f26 docs(P8-polish): supplement caption + label hygiene pass
0db967a feat(P8-polish): Q-Q diagnostic extension to the post-pivot Lorenz cell
c9f69de docs(P8-polish): paper writeup of the Q-Q analysis — A14 + supplement §3.2
1ebee67 feat(figures): real Q-Q analysis — reusable helpers + fig_qq_analysis
d554037 chore(cleanup): archive 14 legacy docs + refresh README for canonical state
```

GitHub: <https://github.com/shawngibford/qlnn> — branch `claude/upbeat-elbakyan-68b56a`.
