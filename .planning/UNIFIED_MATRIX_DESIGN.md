# Unified model × dataset matrix — design

## Why

The paper's deepest question: is the QLNN's accuracy↔variance behavior a
property of the **model** or the **dataset**? Answerable only if the
*same model suite* is evaluated *identically* on every dataset. The
bioreactor run alone can't separate the two.

## Model suite (21, dataset-agnostic — defined once in
`scripts/generate_unified_matrix.py`)

- **classical (5):** param-sweep capacity axis `hidden_size ∈
  {2,4,8,16,32}` (`train_baseline.py`). Classical's "regime" axis *is*
  capacity — `lr_schedule`/`init_circuit_std` are QLNN-trainer-only, so
  applying R1–R3 to classical would be ill-defined. Documented asymmetry.
- **qlnn (16):** 4 ansatz families {data_reuploading, hardware_efficient,
  strongly_entangling, brickwall} × 4 regimes {R0_control,
  R1_weight_decay, R2_physics_prior, R3_smooth_convergence}
  (`train_qlnn.py`). Generalizes Option-B's 3×4=12 to the full 4×4.

**Excluded (judgment calls — revisit if a reviewer wants them):** Optuna
trials / prior axis-ablation / promoted runs (search *scratch*, not
distinct models); bioreactor-only ablations (dopri5, +physics-on-classical,
fixed-OD-clip, horizon sweep — qZETA artifacts that don't transfer to
signed-state ODEs).

## Datasets (11)

`qzeta_od` + 5 ODE systems × {`m472`, `full`}.
- **m472** (~778 rows → ~472 train windows) = EXACT qZETA parity. The
  head-to-head: identical protocol AND identical data volume, so any
  cross-task difference is attributable to dynamics, not sample count.
- **full** (4000 rows → ~2774 windows) = data-scaling ablation.

## Protocol — identical everywhere

window=24, stride=1, h=3, 70/15/15 chronological, 3-seed **proxy**.
Promotion to the 5-seed locked protocol is a separate gated step (same
funnel as Option-B). qZETA keeps its physical OD clip [0,3.8]; ODE
systems set `od_phys_max=null` (signed states — the clip would be
actively wrong).

## Per-dataset comparable gates

`scripts/build_dataset_baseline_locks.py` derives, per dataset, from its
own trained classical H-sweep:
- G1 = classical H=4 test MAE on that dataset
- G2 = 0.5·σ(classical H=4) (the Claim-1 ≥2× rule)
→ `results/unified_matrix/baseline_lock__<dataset>.json`. So "passes
Option-B" means the same thing on Lorenz as on the bioreactor.

## Execution — gated, dataset-grouped (231 configs is multi-day)

`scripts/run_unified_matrix.sh` with `ONLY=<dataset>` runs one ~21-config
group (≈ one proxy budget) at a time. Sequencing:

  O-2 (running) → Option-B tier-1 → **then** unified matrix, one dataset
  group per gated go/no-go → T3.

Nothing contends with the critical path; each dataset group is a
reviewable checkpoint. Priority order: `qzeta_od` (re-confirms the
known baseline under the new harness) → the 5 `*_m472` (the head-to-head)
→ the 5 `*_full` (data-scaling ablation, lowest priority).

## Outcome value (both directions publishable)

- If a regime that fails Option-B on qZETA *passes* on, e.g., Van der Pol
  → "the tradeoff is dataset-dependent; QLNNs excel on stiff/oscillatory
  dynamics" — a strong mechanistic claim.
- If the ranking is stable across all 11 → "the accuracy↔variance
  behavior is an intrinsic model property, dataset-invariant" — equally
  strong, and rare to be able to assert with this much evidence.
