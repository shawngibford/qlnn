"""Render the end-to-end quantum-PINN pipeline schematic.

A high-level block diagram that walks a newcomer through the data
flow from problem specification → trial solution → quantum circuit
evaluation → physics-residual loss → optimizer step. This is the
``how does the whole thing fit together?'' explainer figure that a
QML-newcomer reviewer would want in §2.

Three stacked rows:
  Row 1 — Problem specification    (PDE/ODE, IC, regime tag)
  Row 2 — Trial solution + circuit (Lagaris hard-IC ansatz, QNode)
  Row 3 — Loss + optimizer         (physics-residual, Adam, integrity)

Each row has 3-4 boxes with arrows. No data — pure schematic.

Emits paper/figures/fig_pipeline_diagram.{png,pdf}.

Standalone — does NOT enter the integrity contract.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 9.5,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper" / "figures"

COLORS = dict(
    input    = "#FFE0B2",   # problem spec (warm)
    func     = "#BBDEFB",   # classical function blocks (cool blue)
    quantum  = "#A5D6A7",   # quantum blocks (green)
    loss     = "#FFCCBC",   # loss & verdict (red-orange)
    output   = "#E1BEE7",   # final output / artifact
    arrow    = "#444444",
    band     = "#F5F5F5",
)


def _box(ax, x, y, w, h, label, fc, fontsize=9.5):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.04",
        fc=fc, ec="black", lw=0.7)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize)


def _arrow(ax, x1, y1, x2, y2, *, lw=1.0, ls="-"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=lw, ls=ls,
                         color=COLORS["arrow"]))


def _row_band(ax, y0, y1, label, color=None):
    fc = color or COLORS["band"]
    rect = mpatches.Rectangle((0, y0), 1.0, y1 - y0,
                               fc=fc, ec="none", alpha=0.4, zorder=0)
    ax.add_patch(rect)
    ax.text(0.005, (y0 + y1) / 2, label, ha="left", va="center",
            rotation=90, fontsize=8.5, color="#444444",
            fontweight="bold", alpha=0.9)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # === Row 1: problem specification (top) ============================
    _row_band(ax, 0.78, 0.99, "Problem specification", "#FFF3E0")
    _box(ax, 0.05, 0.83, 0.22, 0.13,
         "Pre-registered\nhardness ladder\n(8 systems × 3 seeds)",
         COLORS["input"])
    _box(ax, 0.33, 0.83, 0.22, 0.13,
         "ODE / PDE\n+ analytic IC $u_0$\n+ regime tag {S, B}",
         COLORS["input"])
    _box(ax, 0.61, 0.83, 0.22, 0.13,
         "Reference solver\n(Diffrax RK4 / FFT)\n→ $u_{\\mathrm{ref}}(t, x)$",
         COLORS["input"])
    _arrow(ax, 0.27, 0.895, 0.33, 0.895)
    _arrow(ax, 0.55, 0.895, 0.61, 0.895)
    # Down to row 2
    _arrow(ax, 0.16, 0.83, 0.16, 0.72, lw=1.5)
    _arrow(ax, 0.44, 0.83, 0.44, 0.72, lw=1.5)

    # === Row 2: trial solution + quantum circuit ======================
    _row_band(ax, 0.42, 0.78, "Trial solution + quantum circuit",
              "#E8F5E9")
    _box(ax, 0.05, 0.60, 0.22, 0.12,
         r"Lagaris hard-IC" "\n"
         r"$u_\theta(t,x) = u_0(x)$"
         "\n"
         r"$+\,(t{-}t_0)[s\,\mathrm{circ}_\theta(t,x){+}b]$",
         COLORS["func"], fontsize=8.5)
    _box(ax, 0.33, 0.60, 0.22, 0.12,
         "Quantum encoder\n(Chebyshev T / FNN / QNN)",
         COLORS["quantum"])
    _box(ax, 0.61, 0.60, 0.22, 0.12,
         "Variational core\n+ entangling layer\n(ring CNOT / linear)",
         COLORS["quantum"])
    _arrow(ax, 0.27, 0.66, 0.33, 0.66)
    _arrow(ax, 0.55, 0.66, 0.61, 0.66)
    _box(ax, 0.85, 0.60, 0.12, 0.12,
         r"$\Sigma\,\langle Z_i\rangle$",
         COLORS["quantum"])
    _arrow(ax, 0.83, 0.66, 0.85, 0.66)

    # PennyLane / JAX stack annotation under row 2
    ax.text(0.50, 0.46,
             "PennyLane  default.qubit  +  JAX  "
             r"$\Rightarrow$ "
             "jacrev through nested Diffrax + QNode",
             ha="center", va="center", fontsize=10,
             style="italic", color="#1B5E20",
             family="monospace")

    # === Row 3: loss + optimizer + verdict ============================
    _row_band(ax, 0.04, 0.42, "Loss · optimizer · verdict", "#FFEBEE")
    _box(ax, 0.05, 0.25, 0.22, 0.13,
         "Physics-residual loss\n"
         r"$\mathcal{L} = \mathrm{MSE}\,[\,$residual$\,(u_\theta)\,]$",
         COLORS["loss"])
    _box(ax, 0.33, 0.25, 0.22, 0.13,
         "Adam optimizer\n($lr=10^{-2}$, up to 2000 steps)",
         COLORS["func"])
    _box(ax, 0.61, 0.25, 0.22, 0.13,
         "Trained " r"$u_\theta^*$"
         "\n→ evaluate on dense grid\n→ relative-$L^2$ vs $u_{\\mathrm{ref}}$",
         COLORS["loss"])
    _box(ax, 0.85, 0.25, 0.12, 0.13,
         "Paired bootstrap\n→ $\\Delta_{\\mathrm{diff}}$",
         COLORS["output"])
    _arrow(ax, 0.27, 0.315, 0.33, 0.315)
    _arrow(ax, 0.55, 0.315, 0.61, 0.315)
    _arrow(ax, 0.83, 0.315, 0.85, 0.315)
    # Vertical: row 2 readout → row 3 loss
    _arrow(ax, 0.91, 0.60, 0.91, 0.40, lw=1.5, ls="--")
    ax.text(0.91, 0.50, r" $u_\theta$", fontsize=9, color="#444444",
             va="center")
    # Adam back-propagation loop arrow
    _arrow(ax, 0.44, 0.25, 0.44, 0.18, lw=1.3)
    ax.annotate(
        "",
        xy=(0.69, 0.605), xytext=(0.44, 0.18),
        arrowprops=dict(arrowstyle="->", lw=1.0, ls=":",
                         color="#B71C1C",
                         connectionstyle="arc3,rad=-0.45"))
    ax.text(0.21, 0.12, r"back-prop loop  ($\nabla_\theta$ via jacrev)",
             fontsize=9, color="#B71C1C",
             ha="left", va="center", fontstyle="italic")

    # === Title ========================================================
    ax.text(0.50, 0.995,
             "End-to-end quantum-PINN benchmark pipeline",
             ha="center", va="top", fontsize=13, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_DIR / "fig_pipeline_diagram.png")
    fig.savefig(OUT_DIR / "fig_pipeline_diagram.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_pipeline_diagram.pdf'}")


if __name__ == "__main__":
    main()
