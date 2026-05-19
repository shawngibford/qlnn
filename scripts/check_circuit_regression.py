"""Option-B regression gate for circuit-search candidates.

A candidate circuit is "best for all" (in the Option-B sense the user
locked) only if it passes EVERY gate in `results/baseline_lock.json`:

  G1 accuracy        candidate h=3 5-seed test MAE < classical matched-param
                     H=4 (0.2594) — beat the classical accuracy bar.
  G2 reproducibility candidate h=3 test MAE std <= 0.5 * sigma_classical_H4
                     — keep Claim-1's >=2x seed-tightness.
  G3 no-regression   candidate SE MAE at 10% data < classical 0.2788
                     — do NOT lose the sample-efficiency win at 10%.
  G4 no-regression   candidate SE MAE at 25% data < classical 0.2546
                     — do NOT lose the sample-efficiency win at 25%.
  G5 no-regression   candidate must not be WORSE than the current QLNN
                     reference at 50% and 100% fractions.

A candidate that passes G1+G2 but not G3/G4 is an accuracy win that
*regressed the paper's distinctive sample-efficiency story* — explicitly
disallowed by the user.

Usage:
    # Single h=3 run (G1+G2 only — partial check)
    python scripts/check_circuit_regression.py \\
        --h3-run results/circuit_search_promoted/top3_strongly_entangling_Q6_L3

    # Full Option-B check (needs the candidate run at all 4 SE fractions)
    python scripts/check_circuit_regression.py \\
        --h3-run results/circuit_search_promoted/topX \\
        --se-pct10 results/circuit_search_se/topX_pct10 \\
        --se-pct25 results/circuit_search_se/topX_pct25 \\
        --se-pct50 results/circuit_search_se/topX_pct50 \\
        --se-pct100 results/circuit_search_se/topX_pct100

Exit code 0 = all requested gates pass; 1 = at least one fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK = REPO_ROOT / "results" / "baseline_lock.json"


def _load(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


def _mae(run_dir: Path) -> dict:
    s = _load(run_dir / "seeds_summary.json")["test"]["mae_raw"]
    return {"mean": s["mean"], "std": s["std"],
            "ci95": s.get("ci95_half_width", 0.0),
            "n_seeds": _load(run_dir / "seeds_summary.json")["n_seeds"]}


def _line(ok: bool, gate: str, msg: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {gate}: {msg}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3-run", type=Path, required=True,
                    help="Candidate run dir with the locked h=3 5-seed protocol.")
    ap.add_argument("--se-pct10", type=Path)
    ap.add_argument("--se-pct25", type=Path)
    ap.add_argument("--se-pct50", type=Path)
    ap.add_argument("--se-pct100", type=Path)
    args = ap.parse_args()

    if not LOCK.exists():
        print(f"missing {LOCK} — regenerate the baseline lock first")
        return 1
    lock = _load(LOCK)

    h3 = _mae(args.h3_run if args.h3_run.is_absolute() else REPO_ROOT / args.h3_run)
    classical_h4 = lock["classical"]["matched_param_H4"]["test_mae"]["mean"]
    sigma_classical = lock["claim1_sigma_ratio"]["sigma_classical_H4"]
    sigma_gate = 0.5 * sigma_classical

    all_ok = True
    print(f"=== Option-B gate — candidate {args.h3_run} ===")
    print(f"(baseline lock @ git {lock['locked_at_git_sha'][:9]})\n")

    if h3["n_seeds"] < 5:
        print(f"  [WARN] candidate has only {h3['n_seeds']} seeds — "
              f"Option-B requires the full 5-seed protocol for a paper-grade verdict")

    # G1 accuracy
    all_ok &= _line(
        h3["mean"] < classical_h4, "G1 accuracy",
        f"candidate MAE={h3['mean']:.4f}  vs  classical_H4={classical_h4:.4f}  "
        f"(need <; margin={classical_h4 - h3['mean']:+.4f})")

    # G2 reproducibility
    all_ok &= _line(
        h3["std"] <= sigma_gate, "G2 reproducibility",
        f"candidate σ={h3['std']:.5f}  vs  0.5·σ_classical={sigma_gate:.5f}  "
        f"(need <=; ratio σ_cl/σ_cand="
        f"{(sigma_classical / h3['std']) if h3['std'] else float('inf'):.2f}x, "
        f"pre-reg needs >=2x)")

    # G3/G4/G5 — sample efficiency, only if the dirs were supplied.
    se_args = {10: args.se_pct10, 25: args.se_pct25,
               50: args.se_pct50, 100: args.se_pct100}
    if any(se_args.values()):
        for pct, gate_id in ((10, "G3"), (25, "G4")):
            d = se_args[pct]
            if d is None:
                print(f"  [SKIP] {gate_id} no-regression pct{pct}: "
                      f"--se-pct{pct} not supplied")
                continue
            cand = _mae(d if d.is_absolute() else REPO_ROOT / d)["mean"]
            cls = lock["sample_efficiency"][f"pct{pct}"]["classical"]["mean"]
            all_ok &= _line(
                cand < cls, f"{gate_id} no-regression pct{pct}",
                f"candidate SE MAE={cand:.4f} vs classical={cls:.4f} "
                f"(QLNN must keep this win; margin={cls - cand:+.4f})")
        for pct in (50, 100):
            d = se_args[pct]
            if d is None:
                print(f"  [SKIP] G5 no-regression-vs-reference pct{pct}: "
                      f"--se-pct{pct} not supplied")
                continue
            cand = _mae(d if d.is_absolute() else REPO_ROOT / d)["mean"]
            ref = lock["sample_efficiency"][f"pct{pct}"]["qlnn_reference"]["mean"]
            all_ok &= _line(
                cand <= ref + 1e-9, f"G5 no-regression-vs-reference pct{pct}",
                f"candidate SE MAE={cand:.4f} vs QLNN-reference={ref:.4f} "
                f"(must not be worse; margin={ref - cand:+.4f})")
    else:
        print("  [SKIP] G3/G4/G5 — no --se-pctXX dirs supplied "
              "(partial check: G1+G2 only)")

    print()
    if all_ok:
        print("OPTION-B GATE: ALL REQUESTED CHECKS PASS")
        return 0
    print("OPTION-B GATE: FAILED — candidate is not 'best for all'")
    return 1


if __name__ == "__main__":
    sys.exit(main())
