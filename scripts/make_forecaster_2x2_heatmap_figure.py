"""Render the forecaster 2×2 decomposition per-cell heatmap.

Visualizes the 4 decomposition paths (rows) × 9 forecaster cells
(columns) as a heatmap of per-cell Δ values. Each row is one of the
A12/A13 decomposition components (liquid_via_classical /
liquid_via_quantum / quantum_via_ltc / quantum_via_nonliquid); each
column is one (system, seed) cell. Symmetric divergent colormap so
positive (favoring quantum/LTC) is blue and negative (favoring
classical/nonliquid) is red.

Reads:
  results/p7_11_decomposition/per_cell_records.json
    — list of 9 records with all 4 delta_* fields.

Emits paper/figures/fig_forecaster_2x2_heatmap.{png,pdf}.

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
    "xtick.labelsize": 8.5, "ytick.labelsize": 9,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper" / "figures"

PATHS = [
    ("delta_liquid_via_classical",   r"$\tau$-isolating  (classical substrate)"),
    ("delta_liquid_via_quantum",     r"$\tau$-isolating  (quantum substrate)"),
    ("delta_quantum_via_nonliquid",  "control  (quantum vs nonliquid)"),
    ("delta_quantum_via_ltc",        "control  (quantum vs LTC)"),
]


def main() -> None:
    with (REPO_ROOT / "results" / "p7_11_decomposition" /
          "per_cell_records.json").open() as f:
        records = json.load(f)

    # Sort cells: smooth first, then broad; within regime by system+seed.
    records.sort(key=lambda r: (r["regime"] != "smooth_periodic",
                                  r["system"], int(r["seed"])))
    cell_labels = [f"{r['system'][:10]}\ns{r['seed']}" for r in records]

    mat = np.zeros((len(PATHS), len(records)))
    for i, (key, _) in enumerate(PATHS):
        for j, r in enumerate(records):
            mat[i, j] = float(r[key])

    fig, ax = plt.subplots(figsize=(10.5, 4.6),
                            constrained_layout=True)

    vmax = float(np.abs(mat).max())
    im = ax.imshow(mat, aspect="auto", cmap="RdBu",
                    vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(records)))
    ax.set_xticklabels(cell_labels, fontsize=8.5, rotation=0)
    ax.set_yticks(np.arange(len(PATHS)))
    ax.set_yticklabels([p[1] for p in PATHS], fontsize=9.5)

    # Annotate each cell with its numeric value
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            rel = (val + vmax) / max(2 * vmax, 1e-9)
            txt_color = "white" if abs(rel - 0.5) > 0.32 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold")

    # Regime band marker
    smooth_count = sum(1 for r in records
                        if r["regime"] == "smooth_periodic")
    if 0 < smooth_count < len(records):
        ax.axvline(smooth_count - 0.5, color="black", lw=1.5,
                    alpha=0.7, zorder=3)
        ax.text(smooth_count / 2 - 0.5, -0.85, "smooth/periodic",
                 ha="center", va="bottom", fontsize=9, color="#0D47A1",
                 fontweight="bold")
        ax.text(smooth_count + (len(records) - smooth_count) / 2 - 0.5,
                 -0.85, "broadband/multiscale",
                 ha="center", va="bottom", fontsize=9, color="#B71C1C",
                 fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02,
                         label=r"per-cell $\Delta$  "
                         r"(blue = favors quantum/LTC; "
                         r"red = favors classical/nonliquid)")

    ax.set_title(
        "Forecaster 2$\\times$2 decomposition: 4 paths $\\times$ 9 cells  "
        "(per-cell substrate $\\times$ baseline gap)",
        fontsize=11)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_forecaster_2x2_heatmap.png")
    fig.savefig(OUT_DIR / "fig_forecaster_2x2_heatmap.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_forecaster_2x2_heatmap.pdf'}")


if __name__ == "__main__":
    main()
