"""Render the sample-size sensitivity figure.

Visualizes the headline n-sensitivity result: the solver-task Δ_diff
estimate changes SIGN between the original n=18 (RAW pre-guards) and
the broadband-expanded n=24 PRIMARY analysis. This is the most-asked
reviewer concern about robustness of the verdict.

Reads:
  results/p7_5_solver_h1/h1_analysis_solver_task_raw.json    (n=18 raw)
  results/p7_8_solver_h1_n24/h1_analysis_combined_n24.json   (n=24)
  results/p7_11_decomposition/h1_combined.json               (forecaster)

Emits paper/figures/fig_sample_size_sensitivity.{png,pdf}.

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
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper" / "figures"


def _read_bootstrap(rel: str) -> dict:
    with (REPO_ROOT / rel).open() as f:
        return json.load(f)["bootstrap"]


def main() -> None:
    n18 = _read_bootstrap(
        "results/p7_5_solver_h1/h1_analysis_solver_task_raw.json")
    n24 = _read_bootstrap(
        "results/p7_8_solver_h1_n24/h1_analysis_combined_n24.json")
    fc  = _read_bootstrap(
        "results/p7_11_decomposition/h1_combined.json")

    rows = [
        ("Solver  n=18 (pre-amendment)", n18, "#0072B2"),
        ("Solver  n=24 (PRIMARY, post-A12)", n24, "#D55E00"),
        ("Forecaster  n=9 (PRIMARY)", fc, "#009E73"),
    ]

    fig, ax = plt.subplots(figsize=(9.0, 4.0),
                            constrained_layout=True)
    y = np.arange(len(rows))
    for i, (label, b, color) in enumerate(rows):
        mean = float(b["delta_diff_mean"])
        lo   = float(b["ci_low"])
        hi   = float(b["ci_high"])
        err = np.array([[mean - lo], [hi - mean]])
        ax.errorbar(mean, i, xerr=err, fmt="o", color=color,
                    markersize=11, capsize=5, capthick=1.6, lw=2.0,
                    markeredgecolor="black", markeredgewidth=0.7,
                    zorder=3)
        # Numeric value label BELOW the point (consistent with the
        # τ-substrate figure; never collides with the title or arms).
        ax.annotate(f"{mean:+.3f}", xy=(mean, i),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", va="top",
                    fontsize=9.5, fontweight="bold", color=color)
        # n_smooth / n_broad shown to the RIGHT of the point label in
        # the same row, via axis-relative offset so it never clips.
        n_smooth = b.get("n_smooth")
        n_broad  = b.get("n_broad")
        if n_smooth is not None and n_broad is not None:
            ax.annotate(f"$n_{{\\mathrm{{s}}}}={n_smooth}$, "
                         f"$n_{{\\mathrm{{b}}}}={n_broad}$",
                         xy=(mean, i),
                         xytext=(0, 16), textcoords="offset points",
                         ha="center", va="bottom",
                         fontsize=8.5, color="#555555")

    ax.axvline(0.0, color="black", lw=0.8, ls="--", alpha=0.6, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta_{\mathrm{diff}}$  (95% paired-bootstrap CI)")
    ax.set_title("Sample-size + binning sensitivity: solver "
                 r"$\Delta_{\mathrm{diff}}$ flipped sign "
                 "between $n{=}18$ (raw) and $n{=}24$ (broadband expanded)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.30, lw=0.5)
    # Pad x-axis so the annotations have breathing room.
    x_lo = min(float(b["ci_low"])  for _, b, _ in rows)
    x_hi = max(float(b["ci_high"]) for _, b, _ in rows)
    pad = 0.10 * (x_hi - x_lo)
    ax.set_xlim(x_lo - pad, x_hi + pad)
    # Pad y-axis so labels above row 0 and below row N-1 don't clip.
    ax.set_ylim(len(rows) - 0.5 + 0.25, -0.5 - 0.25)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_sample_size_sensitivity.png")
    fig.savefig(OUT_DIR / "fig_sample_size_sensitivity.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_sample_size_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
