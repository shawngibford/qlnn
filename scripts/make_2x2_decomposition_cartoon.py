"""Render a schematic of the 2×2 mechanism-decomposition design.

A12/A13 introduce a 2×2 factor design for the forecaster H1:
  rows    = cell substrate         (classical MLP  vs  quantum cell)
  columns = baseline reference     (LTC liquid    vs  nonliquid)

Each cell of the 2×2 yields a Δ_diff number. The four numbers
algebraically partition Δ_combined into a sum + an interaction term;
the interaction term is precisely the τ-substrate effect surfaced by
fig_tau_substrate.

This is a pure schematic — no run data. Useful for §6 mechanism prose.

Emits paper/figures/fig_2x2_decomposition.{png,pdf}.

Standalone — does NOT enter the integrity contract.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper" / "figures"


def _cell(ax, x, y, w, h, title, sub, value, color, value_color="black"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        fc=color, ec="black", lw=1.0, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.22, sub, ha="center", va="top",
            fontsize=8.5, style="italic", color="#333333")
    ax.text(x + w / 2, y + 0.10, value, ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=value_color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Column / row headers
    ax.text(0.30, 0.97, "baseline reference  →",
            ha="center", va="top", fontsize=10, color="#555555")
    ax.text(0.25, 0.88, "LTC-liquid", ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(0.55, 0.88, "nonliquid", ha="center", va="center",
            fontsize=11, fontweight="bold")

    ax.text(0.06, 0.55, "cell substrate ↓", ha="center", va="center",
            fontsize=10, color="#555555", rotation=90)
    ax.text(0.13, 0.68, "classical\nMLP", ha="center", va="center",
            fontsize=10.5, fontweight="bold")
    ax.text(0.13, 0.34, "quantum\ncell", ha="center", va="center",
            fontsize=10.5, fontweight="bold")

    # The four boxes
    box_w, box_h = 0.27, 0.30
    # row 1: classical substrate
    _cell(ax, 0.20, 0.50, box_w, box_h,
          "liquid_via_classical",
          "τ-isolating (classical)",
          r"$\Delta = {+}0.115$", "#BBDEFB")
    _cell(ax, 0.50, 0.50, box_w, box_h,
          "quantum_via_nonliquid",
          "control",
          r"$\Delta = {-}0.167$", "#E0E0E0")
    # row 2: quantum substrate
    _cell(ax, 0.20, 0.14, box_w, box_h,
          "liquid_via_quantum",
          "τ-isolating (quantum)",
          r"$\Delta = {-}0.334$", "#FFCCBC", value_color="#B71C1C")
    _cell(ax, 0.50, 0.14, box_w, box_h,
          "quantum_via_ltc",
          "control",
          r"$\Delta = {-}0.616$", "#E0E0E0")

    # Sign-flip annotation linking the two τ-isolating cells
    ax.annotate("", xy=(0.27, 0.45), xytext=(0.27, 0.49),
                arrowprops=dict(arrowstyle="<->", color="#B71C1C", lw=1.8))
    ax.text(0.35, 0.47, "sign flip\non τ machinery",
            fontsize=9, color="#B71C1C", fontweight="bold", va="center")

    # Combined headline
    ax.text(0.50, 0.05,
            r"$\Delta_{\mathrm{combined}} = {-}0.501$  "
            r"(95% CI excludes 0 negatively)",
            ha="center", va="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow",
                      ec="goldenrod", lw=0.8))

    ax.set_title(
        "2×2 mechanism decomposition: substrate × baseline\n"
        "(forecaster task, n=9 per cell)", fontsize=11)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_2x2_decomposition.png")
    fig.savefig(OUT_DIR / "fig_2x2_decomposition.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_2x2_decomposition.pdf'}")


if __name__ == "__main__":
    main()
