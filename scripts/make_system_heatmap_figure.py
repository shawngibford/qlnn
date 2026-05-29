"""Render the per-system mean-relL² heatmap.

System × {QLNN-best-family, classical PINN} grid of mean relL² across
seeds for the PRIMARY solver-task verdict (n=24, post-A12). Shows the
per-system structure that is hidden by the smooth/broad bin
aggregation in fig_p5_h1_verdict.

Reads:
  results/p7_8_solver_h1_n24/per_cell_records.json
    — flat list of 24 records (8 systems × 3 seeds) with fields
      {system, seed, qlnn_best_family, qlnn_relL2,
       classical_pinn_relL2, delta, regime}.

Emits paper/figures/fig_system_heatmap.{png,pdf}.

Standalone — does NOT enter the integrity contract.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = REPO_ROOT / "results" / "p7_8_solver_h1_n24" / "per_cell_records.json"
OUT_DIR = REPO_ROOT / "paper" / "figures"


def main() -> None:
    with IN_PATH.open() as f:
        records = json.load(f)

    # Aggregate per (system, side)
    by_sys: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"qlnn": [], "classical": []})
    regime_by_sys: dict[str, str] = {}
    best_family_by_sys: dict[str, list[str]] = defaultdict(list)
    for r in records:
        sys = r["system"]
        by_sys[sys]["qlnn"].append(float(r["qlnn_relL2"]))
        by_sys[sys]["classical"].append(float(r["classical_pinn_relL2"]))
        regime_by_sys[sys] = r["regime"]
        best_family_by_sys[sys].append(r["qlnn_best_family"])

    # Sort systems by regime so SMOOTH band sits above BROAD.
    systems = sorted(
        by_sys.keys(),
        key=lambda s: (regime_by_sys[s] != "smooth_periodic", s))
    sides = ["qlnn", "classical"]
    n_s = len(systems)

    # Mean and std per cell
    mat_mean = np.zeros((n_s, 2))
    mat_std = np.zeros((n_s, 2))
    for i, sys in enumerate(systems):
        for j, side in enumerate(sides):
            arr = np.array(by_sys[sys][side])
            mat_mean[i, j] = float(arr.mean())
            mat_std[i, j]  = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(np.log10(np.clip(mat_mean, 1e-5, None)),
                   aspect="auto", cmap="viridis_r")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["QLNN best", "classical PINN"], fontsize=9.5)
    yticklabels = []
    for s in systems:
        most_common = max(set(best_family_by_sys[s]),
                          key=best_family_by_sys[s].count)
        regime_short = "S" if regime_by_sys[s] == "smooth_periodic" else "B"
        yticklabels.append(f"{s}  [{regime_short}]  ({most_common})")
    ax.set_yticks(np.arange(n_s))
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_title("Per-system mean relative L²  (n=24, PRIMARY)",
                 fontsize=10.5)
    for i in range(n_s):
        for j in range(2):
            txt_color = "white" if mat_mean[i, j] < 0.04 else "black"
            ax.text(j, i,
                    f"{mat_mean[i, j]:.3f}\n±{mat_std[i, j]:.3f}",
                    ha="center", va="center",
                    fontsize=8, color=txt_color)

    # Regime band separator
    smooth_systems = [s for s in systems
                      if regime_by_sys[s] == "smooth_periodic"]
    if smooth_systems and len(smooth_systems) < n_s:
        ax.axhline(len(smooth_systems) - 0.5, color="red", lw=1.5, alpha=0.7)
        ax.text(-0.55, len(smooth_systems) - 0.5, "—— regime band ——",
                color="red", fontsize=7.5, va="center")

    cbar = plt.colorbar(im, ax=ax, label="log₁₀(mean relL²)",
                        shrink=0.7, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_system_heatmap.png")
    fig.savefig(OUT_DIR / "fig_system_heatmap.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_system_heatmap.pdf'}")


if __name__ == "__main__":
    main()
