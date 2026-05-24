# Cross-Stack Integration Audit — Reviewer 4

**Verdict: comparable, with fairness caveats. No BLOCKERs.**

## Key positive finding (empirical)

`results/baseline_classical_euler/baselines.json` and `results/qlnn_hybrid/baselines.json` are **byte-identical**. This is the smoking-gun proof that the data path (load → split → MinMax scale → window) is genuinely shared across both stacks, not just structurally similar.

## Single-sourced shared modules verified

- `make_horizon_windows` — one implementation, both pipelines use it
- `compute_metrics`, `aggregate_seed_metrics`, `ForecastMetrics` — single canonical
- `persistence_forecast`, `linear_extrapolation_forecast` — single canonical
- `load_qzeta`, `time_hours_from_date` — single canonical
- `fit_minmax` / `apply_minmax` with identical `fixed_bounds={OD:(0,3.8)}` and `fit_end=split.train_end`

No shadow copies of any of these exist in `src/qlnn_/`.

## Schema equivalence verified

- Per-seed `metrics.json`: both serialize `{best_epoch, val.to_dict(), test.to_dict()}` with the same `ForecastMetrics` fields.
- `seeds_summary.json`: both use `aggregate_seed_metrics`, producing `{n_seeds, seeds, val:{...}, test:{...}}` where each field has `{mean, std, min, max, n_seeds}`.
- `summarize_baselines.py` consumes only fields both pipelines produce.

## HIGH issues (fairness, not validity)

| Field | baseline.yaml | qlnn_hybrid.yaml |
|---|---|---|
| epochs | 300 | 60 |
| batch_size | 64 | 32 |
| eval_every | 10 | 5 |
| patience | 10 | 6 |
| seeds | [0,1,2,3,4] (n=5) | [0,1,2] (n=3) |

These divergences are deliberate (quantum training is slow) and don't break the *evaluation* protocol, but they make the head-to-head "comparable" rather than "fair." Mitigation: align where practical OR disclose in methods.

## LOW

- `train_qlnn.py` doesn't call `np.random.seed(seed)` — currently no numpy RNG used in QLNN path, but a one-liner future-proofs it.
- Checkpoint formats differ (`best_state.pt` vs `best_model.eqx`) — fine for now, will need stack-aware loader if QWGAN-GP reuses checkpoints.
- `HistoryRow` columns differ (classical has `train_loss_total`, QLNN does not) — `history.csv` is not a shared schema across stacks; document this.

## Recommendations

1. Finish the multi-seed QLNN run and produce `results/qlnn_hybrid/seeds_summary.json` (already in flight).
2. Push QLNN to n=5 seeds before final paper table.
3. Either align training-knob divergences OR disclose them in methods.
4. Add `np.random.seed(seed)` defensively in `train_qlnn.py`.

## Verdict

**Comparable: yes. Fair head-to-head as currently configured: not quite — but trivially fixable.** Once the QLNN run actually completes, its numbers will be apples-to-apples with the classical baseline on the locked evaluation protocol.
