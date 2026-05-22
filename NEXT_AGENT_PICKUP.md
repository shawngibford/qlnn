# NEXT AGENT — read this first

**Where:** worktree `/Users/shawngibford/dev/phd/qlnn/.claude/worktrees/upbeat-elbakyan-68b56a/`, branch `claude/upbeat-elbakyan-68b56a`, HEAD `ad65b33`.

**One-line state:** Paper body §1–§8 drafted (15 pages, builds clean, all numbers gated). Only the supplement remains.

**One-command sanity check:**
```bash
PY=/Users/shawngibford/dev/phd/qlnn/.venv/bin/python
cd /Users/shawngibford/dev/phd/qlnn/.claude/worktrees/upbeat-elbakyan-68b56a
PYTHONPATH=src $PY -m pytest -q --tb=short                # full suite green
PYTHONPATH=src $PY scripts/verify_paper_integrity.py     # exit-0
bash paper/build.sh                                       # 15-page PDF clean
```
All three must exit-0 / build clean. If any does not, STOP and inspect before proceeding.

## What to do

1. **Read** `HANDOFF.md` top-to-the-OLD-HEADER-line (it's the canonical pickup doc).
2. **Read** the headline finding in §4.3 (the τ-cross-check sign disagreement — this is the paper's mechanistic centerpiece, P7.11 sprint produced it).
3. **Verify both task sides are covered** (the user explicitly asked):
   - Solver task: 4 quantum × 4 ODE + 4 quantum 2D × 4 PDE + classical PINN, 120 cells total, fully in §3's all-to-all table. No liquid/non-liquid distinction needed (both sides are PINN-style — fairness is inherent). DONE.
   - Forecaster task: 5 quantum × 4 classical (incl. P7.10 classical_LTC and P7.11 non-liquid quantum). Complete 2×2 mechanism decomposition in §4.3. DONE.
4. **Draft the supplement** (`paper/supplement.tex`, NEW file):
   - S1 Circuit specs (verbatim from `refs/CIRCUIT_SPECS.md`)
   - S2 Per-cell error tables (24-cell solver + 9-cell forecaster, higher precision)
   - S3 Bootstrap diagnostics + guard reports
   - S4 Reproduction (`scripts/reproduce_paper.sh` + `verify_paper_integrity.py` walkthrough)
   - S5 Pre-registration verbatim (embed `ODE_PDE_PRE_REG.md`)
   - S6 Amendments verbatim (embed `PRE_REG_AMENDMENT.md` A1–A13)
5. **Final polish + tag** `v1.0-paper-draft` and push to GitHub.

## What NOT to do

- DO NOT modify `physics_residual_loss.py`, `pde_residual_loss.py`'s gate test, or the OD-program legacy gates in `verify_paper_integrity.py`. Those are immutable contracts.
- DO NOT regenerate any `results/` artifacts. They are checked-in and used by integrity gates with tight tolerances.
- DO NOT push to `main` on GitHub. Use the worktree branch.
- DO NOT touch P7.7 (QNG / causal training / L=5 reuploading) or Kuramoto or KdV. Those are explicitly deferred to the FOLLOW-UP paper.

## Pre-reg amendments to know (13 total in `PRE_REG_AMENDMENT.md`)

- **A11** = PRIMARY SOLVER (n=24 FALSIFIED with sign-flip)
- **A12** = forecaster LTC decomposition (P7.10, 3 verdicts)
- **A13** = complete 2×2 + τ-cross-check disagreement (P7.11) — the centerpiece

## Two PRIMARY headline numbers to never get wrong

- **Solver n=24:** Δ_diff = **−0.084**, CI **[−0.278, +0.061]**, FALSIFIED, point-estimate-flipped
- **Forecaster combined:** Δ_diff = **−0.501**, CI **[−0.804, −0.244]**, FALSIFIED, CI excludes 0

`verify_paper_integrity.py` mechanically gates both.

## Recent commits (last 8 — full ledger in HANDOFF.md)

```
ad65b33 feat(P7.11-wrap): integrity gates + A13 + §4 2×2 + §5 master table
b713507 feat(P7.11-sweep+decomp): non-liquid 36-cell sweep + 2×2 decomposition
db9ddd1 feat(P8-discuss-conclude): §7 Discussion + §8 Conclusions
a6ed83a feat(P8-mechanism): §6 Mechanism (T3 + KL-to-Haar LOO)
fc1e44c feat(P7.11-forecaster): NonLiquidVectorForecaster + dispatcher
76cd33e feat(P7.11-cell): NonLiquidQuantumCell + tests
7a35552 feat(P8-verdict): §5 H1 verdict aggregation
5d82e1c feat(P8-forecaster): §4 Forecaster + rf_qrc fix in LTC decomposition
```

GitHub: https://github.com/shawngibford/qlnn — branch `claude/upbeat-elbakyan-68b56a`.
