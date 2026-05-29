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

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(rows))
    for i, (label, b, color) in enumerate(rows):
        mean = float(b["delta_diff_mean"])
        lo   = float(b["ci_low"])
        hi   = float(b["ci_high"])
        err = np.array([[mean - lo], [hi - mean]])
        ax.errorbar(mean, i, xerr=err, fmt="o", color=color,
                    markersize=10, capsize=4, capthick=1.5, lw=1.8,
                    markeredgecolor="black", markeredgewidth=0.6)
        ax.text(mean, i + 0.18, f"{mean:+.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color)
        n_smooth = b.get("n_smooth")
        n_broad  = b.get("n_broad")
        if n_smooth is not None and n_broad is not None:
            ax.text(hi + 0.04, i, f"n_s={n_smooth}, n_b={n_broad}",
                    fontsize=7.5, va="center", color="#444444")

    ax.axvline(0.0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta_{\mathrm{diff}}$  (95% paired-bootstrap CI)")
    ax.set_title("Sample-size + binning sensitivity: solver Δ flipped sign\n"
                 "between n=18 (raw) and n=24 (broadband bin expanded)",
                 fontsize=10.5)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_sample_size_sensitivity.png")
    fig.savefig(OUT_DIR / "fig_sample_size_sensitivity.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_sample_size_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
