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
import matplotlib.pyplot as plt

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.10,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper" / "figures"

# Per-cell content: (path, role, Δ value, face color, value color).
CELLS = {
    # (row_idx, col_idx) → cell content
    (0, 0): dict(path="liquid_via_classical",
                 role=r"$\tau$-isolating  (classical substrate)",
                 value=r"$\Delta = +0.115$",
                 fc="#BBDEFB", vc="#0D47A1"),
    (0, 1): dict(path="quantum_via_nonliquid",
                 role="control",
                 value=r"$\Delta = -0.167$",
                 fc="#E0E0E0", vc="#424242"),
    (1, 0): dict(path="liquid_via_quantum",
                 role=r"$\tau$-isolating  (quantum substrate)",
                 value=r"$\Delta = -0.334$",
                 fc="#FFCCBC", vc="#B71C1C"),
    (1, 1): dict(path="quantum_via_ltc",
                 role="control",
                 value=r"$\Delta = -0.616$",
                 fc="#E0E0E0", vc="#424242"),
}

ROW_LABELS = ["classical MLP\nsubstrate", "quantum cell\nsubstrate"]
COL_LABELS = ["LTC-liquid\nreference", "nonliquid\nreference"]


def main() -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(8.8, 5.6),
        gridspec_kw=dict(wspace=0.18, hspace=0.30,
                         left=0.18, right=0.96,
                         top=0.82, bottom=0.12))

    for (r, c), info in CELLS.items():
        ax = axes[r, c]
        # Color the axes patch directly — no manual FancyBboxPatch overlap.
        ax.set_facecolor(info["fc"])
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # In-cell content via transAxes — never collides with anything.
        ax.text(0.5, 0.80, info["path"], ha="center", va="center",
                transform=ax.transAxes, fontsize=10.5,
                fontweight="bold", family="monospace")
        ax.text(0.5, 0.62, info["role"], ha="center", va="center",
                transform=ax.transAxes, fontsize=9, style="italic",
                color="#333333")
        ax.text(0.5, 0.30, info["value"], ha="center", va="center",
                transform=ax.transAxes, fontsize=18,
                fontweight="bold", color=info["vc"])

    # Column headers via fig.text in the slim band above the 2 axes.
    for c, lbl in enumerate(COL_LABELS):
        ax = axes[0, c]
        bbox = ax.get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.04, lbl,
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Row labels via fig.text on the LEFT band.
    for r, lbl in enumerate(ROW_LABELS):
        ax = axes[r, 0]
        bbox = ax.get_position()
        fig.text(bbox.x0 - 0.02, (bbox.y0 + bbox.y1) / 2, lbl,
                 ha="right", va="center", fontsize=11, fontweight="bold")

    # Column / row axis-class labels above the headers.
    fig.text(0.55, 0.96, "baseline reference",
             ha="center", va="top", fontsize=10, color="#555555")
    fig.text(0.03, 0.47, "cell substrate",
             ha="center", va="center", fontsize=10, color="#555555",
             rotation=90)

    # Suptitle.
    fig.suptitle(
        "2×2 mechanism decomposition: substrate × baseline  "
        "(forecaster task, n=9 per cell)",
        y=0.995, fontsize=12, fontweight="bold")

    # Footer with the combined headline — below all axes, no overlap.
    fig.text(0.55, 0.025,
             r"$\Delta_{\mathrm{combined}} = -0.501$  "
             r"(95% CI excludes 0 negatively;   "
             r"$|\Delta_{\tau,\mathrm{classical}} - "
             r"\Delta_{\tau,\mathrm{quantum}}| = 0.449$)",
             ha="center", va="center", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.35",
                       fc="lightyellow", ec="goldenrod", lw=0.8))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_2x2_decomposition.png")
    fig.savefig(OUT_DIR / "fig_2x2_decomposition.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_2x2_decomposition.pdf'}")


if __name__ == "__main__":
    main()
