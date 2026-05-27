"""Run a paired-bootstrap head-to-head between two training runs.

Reads per-window predictions saved as ``seed_<k>/predictions.npz`` by
``train_baseline.py`` / ``train_qlnn.py`` and runs a per-seed paired
bootstrap on the chosen split (``val`` or ``test``). Results are
aggregated across seeds for a single combined report.

Usage:
    python scripts/run_paired_comparison.py \\
        --reference-run results/baseline_classical_euler \\
        --candidate-run results/qlnn_hybrid \\
        --split test --metric mae \\
        --output results/paired_qlnn_vs_classical.json

Special candidate ``persistence``: instead of loading a sibling run, the
script constructs the persistence prediction from ``od_last_norm`` (which
``predictions.npz`` already stores). This makes "is the trained model
beating persistence?" a one-line call without re-running anything.

Per-seed report contains the diff CI / p-value; the cross-seed report
contains the mean of diffs across seeds plus a combined p-value. We use
**Stouffer's method** (sum of one-sided Z scores, weighted equally) for
combination — chosen over a min-p strategy because it uses information
from every seed rather than the single most-extreme one. See
e.g. Whitlock 2005 ("Combining probability from independent tests: the
weighted Z-method is superior to Fisher's approach").
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from quantum_liquid_neuralode.evaluation import paired_bootstrap_diff


_SEED_DIR_RE = re.compile(r"^seed_(-?\d+)$")


def _seed_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    if not run_dir.exists():
        return out
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = _SEED_DIR_RE.match(child.name)
        if not m:
            continue
        if not (child / "predictions.npz").exists():
            continue
        out.append((int(m.group(1)), child))
    out.sort(key=lambda pair: pair[0])
    return out


def _load_preds(seed_dir: Path, split: str) -> dict[str, np.ndarray]:
    f = np.load(seed_dir / "predictions.npz")
    return {
        "y_true": np.asarray(f[f"{split}_y_true_norm"], dtype=np.float64),
        "y_pred": np.asarray(f[f"{split}_y_pred_norm"], dtype=np.float64),
        "od_last": np.asarray(f[f"{split}_od_last_norm"], dtype=np.float64),
    }


def _stouffer_combined_p(per_seed_p: list[float], per_seed_sign: list[int]) -> float:
    """Combine signed per-seed p-values into a single two-sided p-value.

    Convert each per-seed two-sided p to a one-sided Z (signed by the seed's
    diff direction), sum, then convert back to a two-sided p. ``per_seed_sign``
    is +1 if the seed's mean_diff > 0 (candidate worse if metric is
    lower-is-better) and -1 otherwise. Direction-consistent seeds reinforce;
    direction-flipping seeds cancel — exactly the behavior we want for a
    multi-seed claim.
    """
    if not per_seed_p:
        return float("nan")
    # Convert two-sided p to one-sided (signed) Z.
    # one_sided_p = p_two_sided / 2 in the direction indicated by sign.
    z_sum = 0.0
    k = 0
    for p, s in zip(per_seed_p, per_seed_sign):
        if p <= 0.0 or not np.isfinite(p):
            continue
        one_sided_p = max(min(p / 2.0, 1.0 - 1e-12), 1e-12)
        # If sign > 0, observation is "diff > 0" -> upper-tail one-sided p
        # corresponds to Z = ppf(1 - one_sided_p). If sign < 0, lower-tail:
        # Z = ppf(one_sided_p) (negative). Combined direction matters.
        if s >= 0:
            z = float(scipy_stats.norm.ppf(1.0 - one_sided_p))
        else:
            z = float(scipy_stats.norm.ppf(one_sided_p))
        z_sum += z
        k += 1
    if k == 0:
        return float("nan")
    z_combined = z_sum / np.sqrt(k)
    # Two-sided p from combined Z.
    p_two_sided = 2.0 * (1.0 - float(scipy_stats.norm.cdf(abs(z_combined))))
    return float(p_two_sided)


def _candidate_preds(
    seed_dir: Path | None, ref_payload: dict[str, np.ndarray], candidate_run: str
) -> np.ndarray:
    """Resolve candidate predictions for a seed.

    - ``persistence``: pred == od_last from the reference payload (zero-cost
      baseline; no candidate run needed).
    - otherwise: load ``seed_dir/predictions.npz`` and return the model's pred.
    """
    if candidate_run == "persistence":
        return ref_payload["od_last"]
    assert seed_dir is not None
    f = np.load(seed_dir / "predictions.npz")
    # We assume the candidate uses the same split key as the caller already
    # selected (this function is wrapped with the split string captured).
    raise NotImplementedError  # never reached — see main(); kept for clarity.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True,
                        help="Run dir whose predictions are model A (the reference).")
    parser.add_argument("--candidate-run", required=True,
                        help="Run dir for model B, or the literal 'persistence' "
                             "to use od_last from the reference run's predictions.npz.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--metric", choices=["mae", "rmse", "r2"], default="mae")
    parser.add_argument("--n-iter", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0,
                        help="Bootstrap RNG seed (deterministic across runs).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path for the JSON report. If omitted, "
                             "prints to stdout.")
    args = parser.parse_args()

    ref_seeds = _seed_dirs(args.reference_run)
    if not ref_seeds:
        raise SystemExit(f"no seed dirs with predictions.npz under {args.reference_run}")

    candidate_is_persistence = args.candidate_run == "persistence"
    if candidate_is_persistence:
        cand_run_path = None
        cand_label = "persistence"
        cand_seeds_by_id: dict[int, Path] = {}
    else:
        cand_run_path = Path(args.candidate_run)
        cand_label = str(cand_run_path)
        cand_seeds_list = _seed_dirs(cand_run_path)
        if not cand_seeds_list:
            raise SystemExit(f"no seed dirs with predictions.npz under {cand_run_path}")
        cand_seeds_by_id = {seed: d for seed, d in cand_seeds_list}

    per_seed_records: list[dict] = []
    per_seed_p: list[float] = []
    per_seed_sign: list[int] = []
    per_seed_diff: list[float] = []
    per_seed_ci_half: list[float] = []

    for seed, ref_dir in ref_seeds:
        ref_payload = _load_preds(ref_dir, args.split)
        if candidate_is_persistence:
            cand_pred = ref_payload["od_last"]
        else:
            if seed not in cand_seeds_by_id:
                raise SystemExit(
                    f"seed {seed} present in reference run but not in candidate run"
                )
            cand_payload = _load_preds(cand_seeds_by_id[seed], args.split)
            # Sanity-check the test ground truths agree (same dataset / split).
            if not np.allclose(ref_payload["y_true"], cand_payload["y_true"]):
                raise SystemExit(
                    f"seed {seed}: y_true mismatch between reference and candidate "
                    f"-- the runs aren't using the same split."
                )
            cand_pred = cand_payload["y_pred"]

        rep = paired_bootstrap_diff(
            pred_a=ref_payload["y_pred"],
            pred_b=cand_pred,
            y_true=ref_payload["y_true"],
            metric=args.metric,
            n_iter=args.n_iter,
            alpha=args.alpha,
            seed=args.seed,
        )
        rep["seed"] = seed
        per_seed_records.append(rep)
        per_seed_p.append(rep["p_value"])
        per_seed_sign.append(1 if rep["mean_diff"] > 0 else -1)
        per_seed_diff.append(rep["mean_diff"])
        per_seed_ci_half.append(rep["ci_half_width"])

    diffs = np.asarray(per_seed_diff, dtype=np.float64)
    halfs = np.asarray(per_seed_ci_half, dtype=np.float64)
    combined_p = _stouffer_combined_p(per_seed_p, per_seed_sign)

    report = {
        "reference_run": str(args.reference_run),
        "candidate_run": cand_label,
        "split": args.split,
        "metric": args.metric,
        "n_iter": int(args.n_iter),
        "alpha": float(args.alpha),
        "n_seeds": len(ref_seeds),
        "across_seeds": {
            # Mean / std / range of the per-seed bootstrap point estimates.
            "mean_diff": float(diffs.mean()),
            "std_diff": float(diffs.std(ddof=1)) if diffs.size > 1 else float("nan"),
            "min_diff": float(diffs.min()),
            "max_diff": float(diffs.max()),
            # Range of per-seed CI half-widths (a hint of bootstrap precision).
            "ci_half_width_mean": float(halfs.mean()),
            "ci_half_width_min": float(halfs.min()),
            "ci_half_width_max": float(halfs.max()),
            # Stouffer combined two-sided p-value across seeds. The per-seed
            # p-values are first signed by per-seed mean_diff direction, then
            # combined via summed z-scores (Whitlock 2005).
            "combined_p_value_stouffer": combined_p,
        },
        "per_seed": per_seed_records,
    }

    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}")

    # Human summary
    print(
        f"\npaired-bootstrap diff  ({args.metric}, {args.split})"
        f"\n  reference: {args.reference_run}"
        f"\n  candidate: {cand_label}"
        f"\n  n_seeds={len(ref_seeds)} n_windows={per_seed_records[0]['n_windows']} n_iter={args.n_iter}"
    )
    for rep in per_seed_records:
        print(
            f"  seed={rep['seed']:>2d} diff(a-b)={rep['mean_diff']:+.5f} "
            f"CI95=[{rep['ci_low']:+.5f}, {rep['ci_high']:+.5f}] p={rep['p_value']:.4f}"
        )
    print(
        f"  across-seeds mean_diff={diffs.mean():+.5f} "
        f"Stouffer combined p={combined_p:.4f}"
    )

    # Interpretation hint.
    # For lower-is-better metrics (mae/rmse): negative diff = A better; positive = B better.
    # For r2 (higher better): positive diff = A better.
    better_if_negative = args.metric in ("mae", "rmse")
    a_label = str(args.reference_run)
    if better_if_negative:
        if diffs.mean() < 0 and combined_p < 0.05:
            print(f"  -> A ({a_label}) statistically beats B at alpha={args.alpha}")
        elif diffs.mean() > 0 and combined_p < 0.05:
            print(f"  -> B ({cand_label}) statistically beats A at alpha={args.alpha}")
        else:
            print(f"  -> no statistical separation at alpha={args.alpha}")
    else:
        if diffs.mean() > 0 and combined_p < 0.05:
            print(f"  -> A ({a_label}) statistically beats B at alpha={args.alpha}")
        elif diffs.mean() < 0 and combined_p < 0.05:
            print(f"  -> B ({cand_label}) statistically beats A at alpha={args.alpha}")
        else:
            print(f"  -> no statistical separation at alpha={args.alpha}")


if __name__ == "__main__":
    main()
