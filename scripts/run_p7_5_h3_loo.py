"""P7.5 commit 6 — H3 leave-one-out cross-validation.

Closes audit concern A8 (H3 tentative trend at n=9): confirms the
KL_to_Haar ρ=+0.518 trend isn't driven by a single outlier cell.

For each of the 9 forecaster cells (P5 H1 verdict input):
  - Drop that cell
  - Re-compute Spearman ρ between T3 scalars and Δ across the
    remaining 8 cells
  - Report the LOO distribution (mean / std / min / max) per scalar

If KL_to_Haar's LOO ρ distribution stays consistently positive
(e.g. all 9 LOO ρ > 0), the trend is robust. If the distribution
straddles zero, the trend is fragile and was driven by 1 cell.

Output:
  results/p7_5_h3_loo/h3_loo_results.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "p7_5_h3_loo"
P5_IN = REPO_ROOT / "results" / "p5_h1_verdict"
P4_IN = REPO_ROOT / "results" / "p4_forecaster_rollout"
P7_IN = REPO_ROOT / "results" / "p7_t3_mechanism"

T3_SCALARS = (
    "expressibility_kl",
    "entangling_q",
    "gradient_variance",
    "fourier_bandwidth",
)

FAMILIES = [
    "data_reuploading", "hardware_efficient",
    "strongly_entangling", "brickwall",
]


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ via scipy.stats.spearmanr with tie handling."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x, y)
    return float(rho)


def _load_data() -> tuple[dict[str, dict], list[dict]]:
    t3 = json.loads((P7_IN / "t3_scalars.json").read_text())
    cells = json.loads((P5_IN / "per_cell_records.json").read_text())
    # Map each cell to its best-ansatz family.
    for c in cells:
        per_fam = {}
        for fam in FAMILIES:
            p = (P4_IN / f"{c['system']}_{fam}" /
                 f"seed_{c['seed']}" / "metrics.json")
            if p.exists():
                per_fam[fam] = float(
                    json.loads(p.read_text())["relative_l2"])
        c["best_family"] = (min(per_fam, key=per_fam.get)
                            if per_fam else None)
    return t3, cells


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t3, cells = _load_data()
    valid = [c for c in cells if c.get("best_family")]
    n = len(valid)
    print(f"P7.5 H3 leave-one-out cross-validation — n={n} cells\n")

    # Baseline full-sample Spearman per scalar.
    deltas_full = np.array([c["delta"] for c in valid])
    full_rho: dict[str, float] = {}
    for scalar in T3_SCALARS:
        vals_full = np.array(
            [t3[c["best_family"]][scalar] for c in valid],
            dtype=np.float64)
        full_rho[scalar] = _spearman_rho(vals_full, deltas_full)
        print(f"  Full-sample ρ({scalar}) = {full_rho[scalar]:+.4f}",
              flush=True)
    print()

    # LOO distribution per scalar.
    loo: dict[str, dict[str, Any]] = {}
    for scalar in T3_SCALARS:
        rhos = []
        for k_drop in range(n):
            remaining = [c for i, c in enumerate(valid) if i != k_drop]
            vals = np.array(
                [t3[c["best_family"]][scalar] for c in remaining],
                dtype=np.float64)
            d = np.array([c["delta"] for c in remaining])
            rhos.append(_spearman_rho(vals, d))
        rhos_arr = np.array(rhos)
        finite = rhos_arr[np.isfinite(rhos_arr)]
        if finite.size > 0:
            stable = (np.sign(finite[0]) == np.sign(finite)).all()
            all_positive = (finite > 0).all()
            all_negative = (finite < 0).all()
        else:
            stable = False
            all_positive = False
            all_negative = False
        loo[scalar] = {
            "full_rho": full_rho[scalar],
            "loo_rhos": rhos,
            "loo_mean": (float(np.nanmean(rhos_arr))
                          if finite.size > 0 else None),
            "loo_std": (float(np.nanstd(rhos_arr, ddof=1))
                         if finite.size > 1 else 0.0),
            "loo_min": (float(np.nanmin(rhos_arr))
                         if finite.size > 0 else None),
            "loo_max": (float(np.nanmax(rhos_arr))
                         if finite.size > 0 else None),
            "sign_stable": bool(stable),
            "all_positive": bool(all_positive),
            "all_negative": bool(all_negative),
        }
        if loo[scalar]["loo_mean"] is not None:
            print(f"  LOO({scalar}): "
                  f"mean={loo[scalar]['loo_mean']:+.4f}  "
                  f"std={loo[scalar]['loo_std']:.4f}  "
                  f"range [{loo[scalar]['loo_min']:+.4f}, "
                  f"{loo[scalar]['loo_max']:+.4f}]  "
                  f"sign_stable={loo[scalar]['sign_stable']}",
                  flush=True)
        else:
            print(f"  LOO({scalar}): NaN (no variance in scalar)",
                  flush=True)

    (OUT / "h3_loo_results.json").write_text(
        json.dumps(loo, indent=2) + "\n")

    print("\nWritten:", OUT / "h3_loo_results.json")

    # Headline assessment.
    print()
    kl = loo["expressibility_kl"]
    if kl["all_positive"]:
        verdict = (f"KL_to_Haar ρ is POSITIVE across all {n} LOO "
                   f"subsamples — the trend is ROBUST.")
    elif kl["sign_stable"]:
        verdict = (f"KL_to_Haar LOO ρ sign is stable across all {n} "
                   f"subsamples — trend direction is robust.")
    else:
        verdict = (f"KL_to_Haar LOO ρ FLIPS sign across subsamples — "
                   f"trend is FRAGILE (driven by 1-2 cells).")
    print(f"H3 robustness verdict: {verdict}")


if __name__ == "__main__":
    main()
