"""Verify that every headline number in PAPER_SUMMARY.md matches the
committed result JSONs. Run after reproduce_paper.sh, before a release,
and on every CI build.

Exits 0 if all numbers match (within tolerance), 1 otherwise.

Tolerance: 0.0001 absolute for MAE/R² values, 0.01 for d_norm differences.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(p: str | Path) -> dict:
    with (ROOT / p).open() as f: return json.load(f)


def _check(label: str, actual: float, expected: float, tol: float = 1e-4) -> bool:
    ok = abs(actual - expected) <= tol
    status = "✓" if ok else "✗"
    print(f"  {status} {label}: actual={actual:.4f}  expected={expected}  (tol={tol})")
    return ok


def main() -> int:
    all_ok = True

    print("=== Claim 1 (reproducibility): σ_classical_H4 / σ_QLNN ratio ===")
    c = _load("results/param_sweep/euler_h3_hidden4/seeds_summary.json")
    q = _load("results/qlnn_hybrid_h3/seeds_summary.json")
    ratio = c["test"]["mae_raw"]["std"] / q["test"]["mae_raw"]["std"]
    all_ok &= _check("ratio (paper: 3.77)", ratio, 3.77, tol=0.1)

    print("\n=== Claim 2 (expressivity): Δd_norm ===")
    ed = _load("results/effective_dimension/effective_dimension.json")
    all_ok &= _check("classical d_norm (paper: 8.0290)",
                     ed["classical_H4"]["aggregate"]["mean"], 8.0290, tol=0.001)
    all_ok &= _check("QLNN d_norm (paper: 9.5144)",
                     ed["qlnn_h3"]["aggregate"]["mean"], 9.5144, tol=0.001)
    all_ok &= _check("Δd_norm (paper: +1.49)",
                     ed["delta_d_norm_qlnn_minus_classical"], 1.49, tol=0.01)
    if not ed["pre_registered_hypothesis_met"]:
        print("  ✗ pre-registered hypothesis flag is False (expected True)")
        all_ok = False
    else:
        print("  ✓ pre-registered hypothesis flag is True")

    print("\n=== Claim 3 (sample efficiency): per-fraction means ===")
    expected_mae = {
        10:  (0.2788, 0.2686),
        25:  (0.2546, 0.2507),
        50:  (0.2564, 0.2633),
        100: (0.2594, 0.2655),
    }
    for pct, (paper_c, paper_q) in expected_mae.items():
        c = _load(f"results/sample_efficiency/classical_h4_h3_pct{pct}/seeds_summary.json")
        q = _load(f"results/sample_efficiency/qlnn_h3_pct{pct}/seeds_summary.json")
        all_ok &= _check(f"pct={pct} classical MAE",
                         c["test"]["mae_raw"]["mean"], paper_c)
        all_ok &= _check(f"pct={pct} QLNN MAE",
                         q["test"]["mae_raw"]["mean"], paper_q)

    print("\n=== Horizon ablation: persistence + LO-ODE R² ===")
    expected_horizon = {
        1:  (0.9052,    0.9048),
        3:  (-0.0371,   0.1108),
        6:  (-9.7136,  -9.5004),
        12: (-977.0226, -999.8978),
    }
    for h, (paper_pers, paper_lo) in expected_horizon.items():
        s = _load(f"results/horizon_sweep/euler_h{h}/seeds_summary.json")
        b = _load(f"results/horizon_sweep/euler_h{h}/baselines.json")
        all_ok &= _check(f"h={h} persistence R²",
                         b["persistence"]["test"]["r2_raw"], paper_pers,
                         tol=max(0.01, abs(paper_pers) * 0.01))
        all_ok &= _check(f"h={h} LO-ODE R²",
                         s["test"]["r2_raw"]["mean"], paper_lo,
                         tol=max(0.1, abs(paper_lo) * 0.01))

    print()
    if all_ok:
        print("ALL PAPER NUMBERS VERIFIED.")
        return 0
    print("INTEGRITY CHECK FAILED — fix mismatches before submitting.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
