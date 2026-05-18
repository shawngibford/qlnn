# Step 6 — Sample-efficiency sweep (next session)

Implements Claim 3 from `hypothesis.md` v2: "QLNN reaches a target test MAE
(at h=3) with fewer training windows than the param-matched classical model."

## Pre-registered design (from hypothesis.md)

- **Truncation:** from the START of the training segment (chronological).
- **Fractions:** {10, 25, 50, 100}% of the 500 training windows = {50, 125, 250, 500}.
- **Seeds:** 5 per (model, fraction) cell.
- **Hyperparameters held constant** across fractions; only the training-window count changes.
- **Target metric:** test MAE at h=3.
- **Acceptance threshold:** QLNN reaches X at ≤ 50% data while classical needs > 50%, where X = param-matched classical test MAE at 100% data = 0.2594 (from `results/param_sweep/euler_h3_hidden4/`).

## Implementation plan

### Configs (8 new files)

```
configs/sample_efficiency/
    classical_h4_h3_pct10.yaml
    classical_h4_h3_pct25.yaml
    classical_h4_h3_pct50.yaml
    classical_h4_h3_pct100.yaml      (= results/param_sweep/euler_h3_hidden4 already exists; symlink or copy)
    qlnn_h3_pct10.yaml
    qlnn_h3_pct25.yaml
    qlnn_h3_pct50.yaml
    qlnn_h3_pct100.yaml              (= results/qlnn_hybrid_h3 already exists; symlink or copy)
```

Each new config inherits from `configs/param_sweep/baseline_euler_h3_hidden4.yaml`
(classical) or `configs/horizon/qlnn_hybrid_h3.yaml` (QLNN), with one new
field added to the `windows:` block:

```yaml
windows:
  ...
  train_fraction: 0.10    # NEW — truncate training windows to first 10%
```

### Code change

In `scripts/train_baseline.py` and `scripts/train_qlnn.py`, after the train
window builder produces `w_train`, if `cfg["windows"].get("train_fraction")` is
< 1.0, slice `w_train` to the first N windows (chronological truncation):

```python
train_fraction = float(cfg["windows"].get("train_fraction", 1.0))
if train_fraction < 1.0:
    n_keep = max(1, int(len(w_train) * train_fraction))
    # HorizonWindows is a frozen dataclass — rebuild via replace()
    w_train = HorizonWindows(
        x=w_train.x[:n_keep],
        t=w_train.t[:n_keep],
        y=w_train.y[:n_keep],
        od_last=w_train.od_last[:n_keep],
        od_prev=w_train.od_prev[:n_keep],
        dt_last=w_train.dt_last[:n_keep],
        end_idx=w_train.end_idx[:n_keep],
        target_idx=w_train.target_idx[:n_keep],
    )
```

(Make `HorizonWindows` allow mutation or add a `.slice(n)` helper — small
clean change.)

### Sweep runner

`scripts/run_sample_efficiency.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/sample_efficiency

for PCT in 10 25 50 100; do
  for MODEL in classical qlnn; do
    if [ "$MODEL" = "classical" ]; then
      .venv/bin/python scripts/train_baseline.py \
        --config configs/sample_efficiency/classical_h4_h3_pct${PCT}.yaml \
        --output-dir results/sample_efficiency/${MODEL}_h4_h3_pct${PCT} \
        --quiet
    else
      .venv/bin/python scripts/train_qlnn.py \
        --config configs/sample_efficiency/qlnn_h3_pct${PCT}.yaml \
        --output-dir results/sample_efficiency/${MODEL}_h3_pct${PCT} \
        --quiet
    fi
  done
done
```

Total: 8 multi-seed runs. Classical at 5 seeds × ~60s = 5 min × 4 fractions
≈ 20 min. QLNN at 5 seeds × ~12 min = 60 min × 4 fractions ≈ 4 hours.
**Total ~4.5 hours unattended.** Trivially parallelizable; can run while
the laptop is closed.

### Summarizer

`scripts/summarize_sample_efficiency.py`:

Produces a paper-style table:

```
| Model | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| Classical H=4 | MAE±CI | MAE±CI | MAE±CI | 0.2594 |
| QLNN          | MAE±CI | MAE±CI | MAE±CI | 0.2655 |
```

Plus a "sample-efficiency curve" plot (matplotlib) — log(n_train) vs test MAE
with shaded 95% CI bands.

### Bootstrap test for Claim 3

For each fraction at which QLNN is claimed to "match" classical-at-100%:
- Run `scripts/run_paired_comparison.py --reference-run results/sample_efficiency/qlnn_h3_pct<f> --candidate-run results/param_sweep/euler_h3_hidden4 --metric mae --split test`
- Headline: at which fraction does QLNN statistically tie (p > 0.05) or beat (p < 0.05) the 100%-data classical baseline?

## Tests to add

- `test_train_fraction_slices_correctly`: with `train_fraction=0.5`, the
  trainer sees exactly half the windows; full pipeline still works.
- `test_train_fraction_default_is_full`: omitting the field leaves the
  windows unchanged (back-compat).

## What to deliver in the Step 6 session

1. Codepath: HorizonWindows slicing + train_fraction config knob (~30 LOC).
2. 8 YAML configs.
3. Sweep runner + summarizer.
4. ~2 tests.
5. Run the full sweep (~4.5 hours background).
6. Paired-bootstrap analysis at each fraction.
7. Final paper-table row update in PAPER_SUMMARY.md.

## Risks / gotchas

- At 10% data (50 windows), the QLNN may not train at all (too few samples
  for a 100-param model with quantum circuit overhead). Expected — that's
  part of the result.
- At 10% data the classical H=4 also may collapse onto persistence.
  Compare BOTH against persistence in that regime.
- batch_size may need to scale with fraction: at 50 windows with batch=64,
  the model sees 1 batch per epoch. Either reduce batch_size at low fractions
  or just accept many seeds will early-stop at epoch 1.
- `n_iter` for the bootstrap may need to come down at small n_test if test
  windows shrink (they shouldn't — only train shrinks — but verify).

## When to run

Pick this up as the FIRST item in the next session. Step 5 (effective
dimension, in flight now) and the QLNN-+physics-at-h=3 run (also in flight)
should be committed before Step 6 starts.
