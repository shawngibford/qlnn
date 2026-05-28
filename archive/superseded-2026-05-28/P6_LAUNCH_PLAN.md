# P6 Launch Plan — Unified Matrix Sweep

> **Status: DRAFT v0.2, awaiting M0 execution.**
> Synthesized from a 4-agent pre-flight swarm 2026-05-27.
> **v0.2 correction**: v0.1 was based on Agent B's audit of
> `scripts/generate_unified_matrix.py`, which turns out to be
> stale OD-era infrastructure unused by the post-pivot program.
> The actual production architecture is **dedicated per-phase
> runners** (`run_p3_9_pde_matrix.py`, `run_p4_forecaster_rollout.py`,
> `run_p5_matched_baselines.py`, `run_p7_8_h1_n24.py`, etc.) that
> directly import ansatz modules and bypass the registry entirely.
> Gaps G1, G2, G3 are therefore non-issues. Real M0 is ~3-4 hours,
> not ~1 day. **User decision: M4 (matched HPO) is DEFERRED to
> follow-up paper.**

## TL;DR

- **HEAD:** `fe92460`+, on `claude/upbeat-elbakyan-68b56a`, fully
  pushed to its tracking branch.
- **PRIMARY verdicts already exist** (per `ODE_PDE_PRE_REG.md` +
  amendments A11, A12, A13):
  - Solver H1 at **n=24**: FALSIFIED, Δ_diff = −0.084,
    CI [−0.278, +0.061]
  - Forecaster H1 at **n=6** (2×2 decomposition): FALSIFIED,
    Δ_diff = −0.417, CI [−0.787, −0.046]
- **P6 is therefore not "the experiment that produces the verdict"** —
  the verdict exists. P6 is **the work that makes the verdict
  pre-reg-compliant at full manifest scope** (kuramoto + KdV cells,
  matched HPO per A3, forecaster underfit-guard per A6).
- **Honest scope question:** depending on what "P6 complete" means,
  the work is 12 hours of compute or 2 weeks of prep + compute. M0
  defines the prep; M1–M4 are the compute milestones.

---

## 1. Where we actually are (state inventory)

### What's already done

| Phase | Status | Evidence |
|---|---|---|
| P1 (pre-reg) | ✓ DONE | `ODE_PDE_PRE_REG.md` v1 locked 2026-05-19; 14 amendments documented |
| P2 (PDE generators) | ✓ DONE | `data_processing/pde_systems.py` + `data/pde/manifest.json`; SHA256 per-system lock active |
| P3 (solver path) | ✓ DONE | `chebyshev_dqc_2d`, `te_qpinn_*`, `qcpinn` running on heat/Burgers-smooth/Allen-Cahn |
| P3a (PDF faithfulness) | 5/6 DONE | RF-QRC, TE-QPINN ×2, QCPINN, Chebyshev-DQC spec-carded; **Lubasch deferred** |
| P4 (forecaster) | ✓ DONE | `p4_forecaster_rollout/` has 81 seed runs across ODE × ansatz |
| P5 (matched Neural-ODE baseline) | ✓ DONE | `p5_matched_baselines/` populated; mandatory non-liquid contrast in place |
| P6 (unified matrix) | **PARTIAL** | Generator exists but ODE-only forecaster-only; ~10% coverage |
| P7 (trainability + verdict) | ✓ DONE | n=24 PRIMARY solver verdict (A11); 2×2 forecaster decomposition (A12, A13); KL-to-Haar mechanism |
| P8 (dossier / paper) | ✓ DONE | main.pdf 17pp, supplement.pdf 7pp, integrity exit 0, fig:h1-verdict landed today |

### What P6 must close

From the pre-reg compliance audit (Agent D):

1. **System completeness** — pre-reg locks 5 ODE + 4 PDE = 9 systems.
   PRIMARY verdict ran 8 of 9. **Kuramoto** (12D, ~7 hr/cell estimated)
   and **KdV** (3rd-order derivative, ~8 hr/cell estimated) are
   deferred per A11.
2. **Matched HPO (A3)** — current results use fixed hyperparameters.
   A3 sensitivity scan (P7.5 commit 5) confirmed HPO-invariance on 3
   anchor cells, but pre-reg §6 requires *equal-budget HPO across all
   cells* for a fully-compliant verdict. A9 showed that swapping
   default-Adam for HPO-best can flip the verdict sign (CONFIRMED at
   n=9 default Adam → FALSIFIED at n=9 HPO-best).
3. **Forecaster underfit-guard (A6)** — `train_relative_l2` is logged
   on solver task but missing on forecaster. A6 mandates closing this
   gap for any forecaster verdict.

### What P6 does NOT need to add

- The PRIMARY verdict itself (already FALSIFIED both sides; further
  cells will widen CI precision, not flip the sign — CI [−0.278,
  +0.061] is comfortably negative).
- Lubasch-multicopy ansatz (deferred in the plan; user-acknowledged).
- New mechanism analysis (P7 trainability already maps it to
  KL-to-Haar expressivity distance).

---

## 2. Critical gaps in the matrix machinery (corrected v0.2)

### Stale-diagnosis (v0.1 retracted)

Three gaps in v0.1 of this document were based on auditing the wrong
script (the OD-era `generate_unified_matrix.py`, which is not the
production path). They are **not real gaps**:

- ~~**G1 PDE not in matrix generator**~~ — not a gap. PDEs are
  handled by per-phase runners (`run_p3_9_pde_matrix.py`,
  `run_p7_6_pde_solver_h1.py`, etc.) that read `data/pde/manifest.json`
  directly.
- ~~**G2 Unregistered ansätze**~~ — not a gap. The new families
  (chebyshev_dqc, te_qpinn_*, qcpinn, rf_qrc) are imported directly
  by their runners; they don't need the `AnsatzProtocol` registry.
- ~~**G3 Solver vs forecaster conflated**~~ — not a gap. Each runner
  is task-specific by file (`run_p7_8_h1_n24.py` is solver-only;
  `run_p4_forecaster_rollout.py` is forecaster-only).

### Real gaps that M0 must close

### Gap G4: Per-system data-hash gate not enforced

`data/pde/manifest.json` has SHA256 per system, but per-phase
training scripts don't verify the hash before consuming the field. A
silent drift between regenerated PDE fields and the manifest hash
would corrupt training data without detection. **Effort:** ~3 hours
(add `_assert_dataset_hash(name)` helper + call sites in PDE-consuming
runners).

### Gap G6: Forecaster underfit-guard (A6)

Per-cell `train_relative_l2` is logged on solver task but missing on
forecaster. A6 mandates an underfit-guard exclusion for the H1
aggregator on both tasks. **Effort:** ~3 hours (log
`train_relative_l2` in `run_p4_forecaster_rollout.py`'s training
loop; extend the aggregator in `run_p7_10_forecaster_decomposition.py`
to honor the exclusion).

### Gap G7: Operational group go/no-go wrapper

No code enforces "one go/no-go per system group." This is
operational (shell wrapper that waits for user confirmation between
groups). **Effort:** ~1 hour (`scripts/run_p6_group.sh`).

### Gap G8 (new): kuramoto + KdV runner scaffold

`run_p7_8_h1_n24.py` runs the n=24 solver verdict on 8 of 9 systems.
Adding kuramoto + KdV to the manifest requires either extending that
script or writing a parallel `run_p7_8_h1_kuramoto_kdv.py`. Per the
pre-reg, KdV has a mechanism gate that already PASSED
(`scripts/run_p7_8_kdv_gate.py`). **Effort:** ~half day (scaffolding
only; compute is M3, not M0).

### Gap G5 (DEFERRED per user decision)

~~Matched-HPO machinery (A3)~~ — deferred to follow-up paper. Will
be disclosed as an A3 deferral in the supplement; the existing P7.5
HPO-sensitivity spot-check (3 anchor cells, sign-invariant) is the
sensitivity evidence shipped with the current submission.

---

## 3. Coverage map (Agent C — sampled wall-clocks)

```
Per-seed wall-clock (Apple Silicon CPU, no GPU; JAX backend):
  Classical PINN solver:  ~1.5 sec
  ODE forecaster (mixed): ~21 sec
  ODE solver multi-state: ~23 sec
  PDE 2D quantum (Burgers): ~98 sec
  PDE classical PINN:     ~253 sec
  Mean: 59 sec (std 85)
```

Full pre-reg matrix:
- 9 systems × 10 ansätze × 2 tasks × 5 seeds = **900 seed runs nominal**
- Less Lubasch (deferred): 9 × 9 × 2 × 5 = **810 runs**
- Less already-done (~78 runs): **~732 remaining seed runs**

**Naive serial wall-clock:** ~12 hours. **With matched HPO at, say,
20 trials per cell:** multiply by 20 → ~10 days serial. With GPU
(per Agent C's note that GPU would roughly halve PDE cost): ~5 days
serial with HPO; ~6 hours without.

---

## 4. Milestone sequence (one user-authorized step at a time)

### M0 — Prep (no compute, ~3-4 hours of code)

**Required before any meaningful P6 cell runs.** Closes G4, G6, G7,
G8. G5 (matched HPO) is DEFERRED per user; G1, G2, G3 retracted as
stale-diagnosis.

- [ ] **G4**: add `_assert_dataset_hash(name)` helper; call from
      every runner that consumes `data/pde/*.npz` artifacts
- [ ] **G6**: log `train_relative_l2` in
      `run_p4_forecaster_rollout.py` training loop; extend
      `run_p7_10_forecaster_decomposition.py` aggregator to honor A1
      threshold (0.5) exclusion
- [ ] **G7**: `scripts/run_p6_group.sh` shell wrapper with explicit
      `read -p "Proceed with group N? (y/N): "` between system groups
- [ ] **G8**: scaffold `run_p7_8_h1_kuramoto_kdv.py` (or extend
      `run_p7_8_h1_n24.py` with `--systems kuramoto,kdv` flag);
      smoke-test that it constructs without raising
- [ ] All 4 gates green (pytest, integrity, paper build, supplement
      build)

**Verification:** unit test for `_assert_dataset_hash()` raises on
hash mismatch; forecaster-loop logs `train_relative_l2` to
`metrics.json`; group wrapper `--dry-run` prints the staged group
sequence without executing.

### M1 — Smoke (1 cell, ~10 min)

Single cell from the cheapest group (lotka_volterra,
data_reuploading, forecaster, seed 0) to prove the post-M0 pipeline
works end-to-end. **No verdict change; this is wiring validation.**

### M2 — Group 1: cheap-smooth (60-90 min)

`lotka_volterra` + `van_der_pol`, solver + forecaster, 4 ansatz
families × 5 seeds. Closes a complete sub-row of the matrix for the
smoothest systems. Worth doing because (a) cheap, (b) any failure
here points to wiring problems before we waste compute on PDEs.

### M3 — Deferred-system completion: kuramoto + KdV (~14-16 hours)

Adds the two systems A11 deferred. Per A11 this is the work that
takes the PRIMARY verdict from "n=24, 8 of 9 systems" to "n=N at full
9-system manifest." Sign is unlikely to flip (current CI is far from
zero), but precision improves and pre-reg compliance is complete.

### M4 — Matched-HPO refresh (DEFERRED per user 2026-05-27)

User-locked decision: matched HPO is deferred to a follow-up paper.
Rationale: PRIMARY verdict already FALSIFIED at n=24 with HPO-
sensitivity spot-check at 3 anchor cells (A3 was sign-invariant).
Chasing matched HPO at 810-cell scope is a 2-week diversion that
probably doesn't change the headline finding. M0 includes documenting
A3 as an explicit deferral in the supplement.

**Action for the current submission**: the supplement §5 amendments
table should add a row clarifying A3 is documented-deferred (not
silently violated). M5 will handle this.

### M5 — Verdict refresh + paper update

After M2+M3 land, re-run `scripts/run_p7_8_solver_h1.py` (or
equivalent) on the full 9-system manifest. Update the verdict number
in §5 / Table 1 / fig:h1-verdict caption. Bump integrity gate.
**Effort:** ~1 hour after M3.

---

## 5. Hard constraints (carried forward)

- **Never `rm` the `data` symlink while jobs run** (prior incident
  killed O-2; user-locked rule).
- **Explicit `git add <paths>` only**, never `-A`.
- **No `git push` without per-command authorization.**
- **No sweep > 30 min without user go-ahead** (`ODE_PDE_PRE_REG.md`
  §8; carried forward from HANDOFF).
- **`verify_paper_integrity.py` must stay exit 0** throughout. Any
  P6 result that lands in the paper must be wired into the integrity
  gate before commit.

---

## 6. Authorization status (updated 2026-05-27)

1. **M0 prep work** — ✓ AUTHORIZED, in-session, scoped to ~3-4 hours
   per v0.2 correction
2. **M1 smoke + M2 cheap-smooth** — pending after M0 V-gate
3. **M3 kuramoto + KdV** — pending; user decision on
   (both | one at a time | wait for fresh session) after M2
4. **M4 matched HPO** — ✓ DEFERRED to follow-up; M5 documents the
   deferral
5. **GPU availability** — open. Current estimates assume Apple
   Silicon CPU; GPU would roughly halve PDE costs

Next action: M0 swarm execution.

---

## 7. Risks specific to P6

1. **A9 HPO sign-flip risk** — A9 already showed that switching from
   default Adam to HPO-best changed the n=9 verdict from CONFIRMED to
   FALSIFIED. Current n=24 PRIMARY verdict uses fixed HP; an HPO sweep
   could change the magnitude (sign is unlikely to flip given CI is
   far from zero, but possible).
2. **Compute overrun** — naive estimate is 12 hours; with matched HPO
   it's 10 days. Estimate variance is wide because per-cell wall-clock
   varies 100× (1.5s for classical PINN, 253s for PDE PINN). Real
   number could be 2× the estimate.
3. **Per-system data hash drift** — if anyone regenerates a PDE field
   without updating the manifest, training silently consumes wrong
   data. M0 task G4 closes this.
4. **Operational: data symlink** — never `rm` it; if a long-running
   P6 group is going and the symlink is touched, all cells die.
5. **Scope drift into Lubasch** — the plan deferred Lubasch; M0
   should not silently include it.

---

## 8. What this document does NOT do

- Start any compute.
- Make the M4 (matched HPO) decision for you.
- Estimate cost on GPU (CPU only, per Agent C's measurements).
- Commit to a calendar.

It is a staging document. The next action is yours.
