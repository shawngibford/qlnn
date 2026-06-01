"""Render a side-by-side comparison of the solver-task and
forecaster-task PRIMARY H1 verdicts.

Why it exists: §5 reports two PRIMARY verdicts — solver (n=24,
Δ_diff=-0.084, CI crosses zero) and forecaster (n=9, Δ_diff=-0.501,
CI excludes zero negatively). The numbers tell different parts of the
same null story — same hypothesis, same falsification direction,
different statistical strength. The reader needs to SEE the contrast.

Reads:
  results/p7_8_solver_h1_n24/h1_analysis_combined_n24.json
  results/p7_11_decomposition/h1_combined.json

Emits paper/figures/fig_cross_task_verdict.{png,pdf}.

Standalone — does NOT enter the integrity contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper" / "figures"

ROWS = [
    ("Solver task",
     "results/p7_8_solver_h1_n24/h1_analysis_combined_n24.json",
     "#0072B2", 24),
    ("Forecaster task",
     "results/p7_11_decomposition/h1_combined.json",
     "#D55E00", 9),
]


def main() -> None:
    rows = []
    for label, path, color, n in ROWS:
        with (REPO_ROOT / path).open() as f:
            b = json.load(f)["bootstrap"]
        rows.append(dict(
            label=label, color=color, n=n,
            d_smooth=float(b["delta_smooth_mean"]),
            d_broad=float(b["delta_broad_mean"]),
            d_diff=float(b["delta_diff_mean"]),
            ci_lo=float(b["ci_low"]),
            ci_hi=float(b["ci_high"]),
        ))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0),
                              constrained_layout=True,
                              gridspec_kw=dict(width_ratios=(1.0, 1.0)))

    # Left panel: side-by-side Δ_smooth and Δ_broad per task.
    ax = axes[0]
    bar_w = 0.34
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        ax.bar(i - bar_w / 2, r["d_smooth"], bar_w,
                color=r["color"], alpha=0.55, edgecolor="black",
                lw=0.6, label="smooth/periodic" if i == 0 else "")
        ax.bar(i + bar_w / 2, r["d_broad"], bar_w,
                color=r["color"], alpha=1.0, edgecolor="black",
                lw=0.6, label="broadband/multiscale" if i == 0 else "")
        # Numeric labels on bar tops
        ax.text(i - bar_w / 2, r["d_smooth"] + 0.015,
                f"{r['d_smooth']:+.3f}", ha="center", va="bottom",
                fontsize=8.5, color=r["color"], fontweight="bold")
        ax.text(i + bar_w / 2, r["d_broad"] + 0.015,
                f"{r['d_broad']:+.3f}", ha="center", va="bottom",
                fontsize=8.5, color=r["color"], fontweight="bold")
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['label']}\n(n={r['n']})" for r in rows],
                        fontsize=10)
    ax.set_ylabel(r"$\Delta_{\mathrm{regime}}$  "
                  r"(per-regime mean per-cell gap)")
    ax.set_title("Per-regime $\\Delta$ contributions",
                  fontsize=10.5)
    ax.legend(loc="upper center", fontsize=8.5,
              bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    # Right panel: Δ_diff with CI per task — the headline statistic.
    ax = axes[1]
    for i, r in enumerate(rows):
        err = np.array([[r["d_diff"] - r["ci_lo"]],
                         [r["ci_hi"] - r["d_diff"]]])
        ax.errorbar(r["d_diff"], i, xerr=err, fmt="o",
                     color=r["color"], markersize=12, capsize=6,
                     capthick=1.8, lw=2.2,
                     markeredgecolor="black", markeredgewidth=0.7,
                     zorder=3)
        ax.annotate(f"{r['d_diff']:+.3f}\n[{r['ci_lo']:+.3f},  "
                     f"{r['ci_hi']:+.3f}]",
                     xy=(r["d_diff"], i),
                     xytext=(0, -22), textcoords="offset points",
                     ha="center", va="top",
                     fontsize=9.5, color=r["color"], fontweight="bold")
        ci_status = ("CI excludes 0" if r["ci_hi"] < 0 or r["ci_lo"] > 0
                      else "CI crosses 0")
        ax.text(0.98, i - 0.32, ci_status,
                 transform=ax.get_yaxis_transform(),
                 ha="right", va="top", fontsize=8,
                 fontstyle="italic", color="#555555")
    ax.axvline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{r['label']}\n(n={r['n']})" for r in rows],
                        fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta_{\mathrm{diff}} = \Delta_{\mathrm{smooth}} "
                   r"- \Delta_{\mathrm{broad}}$  "
                   r"(95% paired-bootstrap CI)")
    ax.set_title(r"Headline test statistic  $\Delta_{\mathrm{diff}}$",
                  fontsize=10.5)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    # Pad y for label space.
    ax.set_ylim(len(rows) - 0.5 + 0.55, -0.5 - 0.30)

    fig.suptitle(
        "Cross-task verdict: solver (n=24) and forecaster (n=9) both "
        "FALSIFIED in the same direction\nwith different CI strengths "
        "— consistent null story, not coincidental sign",
        y=1.08, fontsize=11)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_cross_task_verdict.png")
    fig.savefig(OUT_DIR / "fig_cross_task_verdict.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_cross_task_verdict.pdf'}")


if __name__ == "__main__":
    main()
