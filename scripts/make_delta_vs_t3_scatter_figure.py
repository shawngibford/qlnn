"""Render the per-cell Δ vs T3-scalar 4-panel scatter.

For each of the 4 T3 scalars (Meyer-Wallach Q, KL-to-Haar, gradient
variance, Fourier K_max), plot per-cell Δ_combined against the
T3 value of the cell's best-quantum family. Color-coded by regime
(smooth_periodic vs broadband_multiscale).

This is the visual companion to Table~\\ref{tab:h3}: instead of
reporting Spearman ρ only, show the actual scatter so the reader
sees the spread and the (sometimes weak) trend.

Reads:
  results/p7_11_decomposition/per_cell_records.json
  results/p7_t3_mechanism/t3_scalars.json

Emits paper/figures/fig_delta_vs_t3_scatter.{png,pdf}.

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
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper" / "figures"

SCALARS = [
    ("expressibility_kl",  "KL-to-Haar  (lower = more expressive)",
     "ρ_full = +0.518   sign-stable"),
    ("entangling_q",       "Meyer-Wallach  Q",
     "ρ_full = +0.179   sign-unstable"),
    ("gradient_variance",  "Mean-square gradient variance",
     "ρ_full = −0.179   sign-unstable"),
    ("fourier_bandwidth",  "Fourier  K_max",
     "constant across families at L=1"),
]
REGIME_COLOR = {
    "smooth_periodic":     "#0072B2",
    "broadband_multiscale":"#D55E00",
}
REGIME_LABEL = {
    "smooth_periodic":     "smooth/periodic",
    "broadband_multiscale":"broadband/multiscale",
}


def main() -> None:
    with (REPO_ROOT / "results" / "p7_11_decomposition" /
          "per_cell_records.json").open() as f:
        records = json.load(f)
    with (REPO_ROOT / "results" / "p7_t3_mechanism" /
          "t3_scalars.json").open() as f:
        scalars = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0),
                              constrained_layout=True)
    seen_regimes: set[str] = set()
    for ax, (scalar_key, label, rho_str) in zip(axes.ravel(), SCALARS):
        for r in records:
            family = r["liquid_qlnn_best_family"]
            if family not in scalars:
                continue
            x = float(scalars[family][scalar_key])
            y = float(r["delta_combined"])
            regime = r["regime"]
            ax.scatter(x, y, s=90, color=REGIME_COLOR[regime],
                        edgecolor="black", lw=0.6, zorder=3,
                        alpha=0.85,
                        label=(REGIME_LABEL[regime]
                                if regime not in seen_regimes else None))
            seen_regimes.add(regime)
            # System abbreviation as marker annotation
            sys_abbrev = "".join(p[0].upper() for p in r["system"].split("_"))[:3]
            ax.annotate(f"  {sys_abbrev}{r['seed']}",
                         (x, y), fontsize=7, color="#444444",
                         ha="left", va="center")
        ax.axhline(0.0, color="black", lw=0.7, ls="--", alpha=0.5,
                    zorder=1)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\Delta_{\mathrm{combined}}$ per cell")
        ax.set_title(rho_str, fontsize=9.5, color="#555555")
        ax.grid(alpha=0.25, lw=0.5)

    # Shared legend in top band.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=len(handles), frameon=False, fontsize=10)
    fig.suptitle(
        r"Per-cell  $\Delta_{\mathrm{combined}}$  vs.  T3 ansatz scalars  "
        r"(n=9 forecaster cells; marker = best-family scalar)",
        y=1.07, fontsize=11)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_delta_vs_t3_scatter.png")
    fig.savefig(OUT_DIR / "fig_delta_vs_t3_scatter.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_delta_vs_t3_scatter.pdf'}")


if __name__ == "__main__":
    main()
