"""Rank the Option-B Phase O-2 proxy results by the variance-aware
penalized objective and emit the top-K for tier-1 promotion.

penalized = mae_3seed_mean + 5.0 · relu(σ_3seed − G2_gate)

Lower is better. The penalty makes any G2 variance-gate violation
dominate, so a fast-but-unstable circuit can't win the proxy.

Emits:
    results/option_b/option_b_table.md   — circuit × regime grid + ranking
    results/option_b/option_b_table.json — machine-readable
    results/option_b/option_b_topk.json  — top-K for tier-1 promotion

Usage:
    python scripts/summarize_option_b.py [--top 3] [--penalty 5.0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "option_b"
CONFIGS = REPO_ROOT / "configs" / "option_b"
LOCK = REPO_ROOT / "results" / "baseline_lock.json"


def _load(p: Path):
    with p.open() as f:
        return json.load(f) if p.suffix == ".json" else yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--penalty", type=float, default=5.0)
    args = ap.parse_args()

    lock = _load(LOCK)
    gate = 0.5 * lock["claim1_sigma_ratio"]["sigma_classical_H4"]
    classical_h4 = lock["classical"]["matched_param_H4"]["test_mae"]["mean"]

    if not RESULTS.exists():
        print(f"no results in {RESULTS}/ — run the O-2 sweep first")
        return

    rows = []
    for run_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        summ = run_dir / "seeds_summary.json"
        if not summ.exists():
            continue
        s = _load(summ)["test"]["mae_raw"]
        mae, sigma = s["mean"], s["std"]
        cfg_path = CONFIGS / f"{run_dir.name}.yaml"
        meta = (_load(cfg_path).get("option_b", {}) if cfg_path.exists() else {})
        penalty = args.penalty * max(0.0, sigma - gate)
        rows.append({
            "run": run_dir.name,
            "circuit": meta.get("circuit", "?"),
            "regime": meta.get("regime", "?"),
            "n_seeds": _load(summ)["n_seeds"],
            "test_mae_mean": mae,
            "test_mae_sigma": sigma,
            "penalized": mae + penalty,
            "g1_accuracy_pass": mae < classical_h4,
            "g2_sigma_pass": sigma <= gate,
        })

    if not rows:
        print(f"no seeds_summary.json under {RESULTS}/*")
        return
    rows.sort(key=lambda r: r["penalized"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "option_b_table.json").write_text(json.dumps(rows, indent=2) + "\n")

    md = ["# Option-B Phase O-2 — penalized-objective ranking (3-seed proxy, h=3)\n"]
    md.append(f"Gate σ ≤ **{gate:.5f}** (0.5·σ_classical_H4); "
              f"G1 accuracy bar = classical_H4 MAE **{classical_h4:.4f}**. "
              f"penalized = MAE + {args.penalty:g}·relu(σ − gate); lower is better.\n")
    md.append("| Rank | Circuit | Regime | Test MAE | σ | penalized | G1 | G2 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        md.append(
            f"| {i} | {r['circuit']} | {r['regime']} | {r['test_mae_mean']:.4f} | "
            f"{r['test_mae_sigma']:.5f} | {r['penalized']:.4f} | "
            f"{'✅' if r['g1_accuracy_pass'] else '❌'} | "
            f"{'✅' if r['g2_sigma_pass'] else '❌'} |"
        )
    both = [r for r in rows if r["g1_accuracy_pass"] and r["g2_sigma_pass"]]
    md.append("")
    if both:
        md.append(f"**{len(both)} config(s) pass BOTH G1+G2 at the 3-seed "
                  f"proxy** — promote to tier-1 (5-seed) for the paper-grade "
                  f"verdict:")
        for r in both:
            md.append(f"- `{r['run']}` (MAE {r['test_mae_mean']:.4f}, "
                      f"σ {r['test_mae_sigma']:.5f})")
    else:
        md.append("**No config passes both G1+G2 at the 3-seed proxy.** "
                  "The top-K by penalized score are still promoted (proxy σ "
                  "is noisy at 3 seeds; tier-1 re-checks at 5).")
    (RESULTS / "option_b_table.md").write_text("\n".join(md) + "\n")

    topk = rows[: args.top]
    (RESULTS / "option_b_topk.json").write_text(json.dumps(topk, indent=2) + "\n")

    print(f"wrote {RESULTS}/option_b_table.{{md,json}} + option_b_topk.json")
    print(f"\nTop {args.top} by penalized objective:")
    for i, r in enumerate(topk, 1):
        print(f"  {i}. {r['circuit']:9s} {r['regime']:22s} "
              f"MAE={r['test_mae_mean']:.4f} σ={r['test_mae_sigma']:.5f} "
              f"pen={r['penalized']:.4f}  "
              f"G1={'P' if r['g1_accuracy_pass'] else 'F'} "
              f"G2={'P' if r['g2_sigma_pass'] else 'F'}")


if __name__ == "__main__":
    main()
