"""Render a schematic of the 4 quantum ansatz families used as the
solver-side surface for H1.

Each panel is a block-diagram cartoon of the family's structural
signature — input encoding, variational core, readout — emphasizing
WHERE each family differs from the others. This is a readability aid
for §3 (solver methodology), NOT a faithful gate-level rendering.

Families (matches src/qlnn_/training/solver_demo.py FAMILIES):
  - chebyshev_dqc      — Chebyshev-T angle encoding + entangling layers
  - te_qpinn_fnn       — classical FNN encoder + variational + Σ Z readout
  - te_qpinn_qnn       — quantum encoder + variational + Σ Z readout
  - qcpinn             — classical pre-net + quantum block + classical post-net
                         (A17 adds 3 step-wise variants along Q/(Q+C) ratio)

Emits paper/figures/fig_circuit_zoo.{png,pdf}.

Standalone — does NOT enter the integrity contract.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10.5,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper" / "figures"

# Wong-style palette for block classes.
COLORS = {
    "input":      "#FFE0B2",   # warm orange (classical input)
    "cl_pre":     "#BBDEFB",   # cool blue (classical pre-net)
    "q_enc":      "#C8E6C9",   # quantum encoder (green)
    "q_var":      "#A5D6A7",   # quantum variational (darker green)
    "q_ent":      "#81C784",   # quantum entanglement layer
    "cl_post":    "#BBDEFB",   # cool blue (classical post-net)
    "readout":    "#FFCCBC",   # readout (red-orange)
}


def _box(ax, x, y, w, h, label, fc, fontsize=8.5):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.03",
        fc=fc, ec="black", lw=0.7)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize)


def _arrow(ax, x1, y, x2):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"))


def _draw_chebyshev(ax):
    ax.set_title("chebyshev_dqc", fontsize=10.5, fontweight="bold")
    _box(ax, 0.02, 0.40, 0.13, 0.25, "t  ∈ ℝ", COLORS["input"])
    _arrow(ax, 0.15, 0.525, 0.20)
    _box(ax, 0.20, 0.40, 0.20, 0.25,
         "Chebyshev-T\nangle encoding", COLORS["q_enc"])
    _arrow(ax, 0.40, 0.525, 0.45)
    _box(ax, 0.45, 0.40, 0.18, 0.25,
         "variational\nR(α,β,γ)", COLORS["q_var"])
    _arrow(ax, 0.63, 0.525, 0.66)
    _box(ax, 0.66, 0.40, 0.16, 0.25, "ring CNOT", COLORS["q_ent"])
    _arrow(ax, 0.82, 0.525, 0.85)
    _box(ax, 0.85, 0.40, 0.13, 0.25, "Σ ⟨Z⟩", COLORS["readout"])
    ax.text(0.50, 0.20, r"× L layers", ha="center", fontsize=8,
            style="italic", color="#555555")
    ax.text(0.50, 0.06, "Schuld/Pérez-Salinas re-uploading; Chebyshev T_k(t)",
            ha="center", fontsize=7.5, color="#444444")


def _draw_te_qpinn_fnn(ax):
    ax.set_title("te_qpinn_fnn", fontsize=10.5, fontweight="bold")
    _box(ax, 0.02, 0.40, 0.13, 0.25, "t  ∈ ℝ", COLORS["input"])
    _arrow(ax, 0.15, 0.525, 0.20)
    _box(ax, 0.20, 0.40, 0.20, 0.25,
         "FNN encoder\nMLP(t)→θ", COLORS["cl_pre"])
    _arrow(ax, 0.40, 0.525, 0.45)
    _box(ax, 0.45, 0.40, 0.18, 0.25,
         "variational\nR(θ)", COLORS["q_var"])
    _arrow(ax, 0.63, 0.525, 0.66)
    _box(ax, 0.66, 0.40, 0.16, 0.25, "linear CNOT", COLORS["q_ent"])
    _arrow(ax, 0.82, 0.525, 0.85)
    _box(ax, 0.85, 0.40, 0.13, 0.25, "Σ ⟨Z⟩", COLORS["readout"])
    ax.text(0.50, 0.20, r"× L layers", ha="center", fontsize=8,
            style="italic", color="#555555")
    ax.text(0.50, 0.06, "Berger 2025; CLASSICAL encoder, QUANTUM core",
            ha="center", fontsize=7.5, color="#444444")


def _draw_te_qpinn_qnn(ax):
    ax.set_title("te_qpinn_qnn", fontsize=10.5, fontweight="bold")
    _box(ax, 0.02, 0.40, 0.13, 0.25, "t  ∈ ℝ", COLORS["input"])
    _arrow(ax, 0.15, 0.525, 0.20)
    _box(ax, 0.20, 0.40, 0.20, 0.25,
         "QNN encoder\nReUp(t)", COLORS["q_enc"])
    _arrow(ax, 0.40, 0.525, 0.45)
    _box(ax, 0.45, 0.40, 0.18, 0.25,
         "variational\nR(α)", COLORS["q_var"])
    _arrow(ax, 0.63, 0.525, 0.66)
    _box(ax, 0.66, 0.40, 0.16, 0.25, "linear CNOT", COLORS["q_ent"])
    _arrow(ax, 0.82, 0.525, 0.85)
    _box(ax, 0.85, 0.40, 0.13, 0.25, "Σ ⟨Z⟩", COLORS["readout"])
    ax.text(0.50, 0.20, r"× L layers", ha="center", fontsize=8,
            style="italic", color="#555555")
    ax.text(0.50, 0.06,
            "Berger 2025; QUANTUM encoder + core (paired with te_qpinn_fnn)",
            ha="center", fontsize=7.5, color="#444444")


def _draw_qcpinn(ax):
    ax.set_title("qcpinn  (+ A17 variants)", fontsize=10.5, fontweight="bold")
    _box(ax, 0.02, 0.40, 0.13, 0.25, "t  ∈ ℝ", COLORS["input"])
    _arrow(ax, 0.15, 0.525, 0.20)
    _box(ax, 0.20, 0.20, 0.18, 0.30,
         "classical\npre-net", COLORS["cl_pre"])
    _arrow(ax, 0.38, 0.35, 0.41)
    _box(ax, 0.41, 0.10, 0.22, 0.55,
         "quantum\nblock\n(IQP-style)", COLORS["q_var"])
    _arrow(ax, 0.63, 0.35, 0.66)
    _box(ax, 0.66, 0.20, 0.18, 0.30,
         "classical\npost-net", COLORS["cl_post"])
    _arrow(ax, 0.84, 0.35, 0.87)
    _box(ax, 0.87, 0.20, 0.11, 0.30, "u(t)", COLORS["readout"])
    ax.text(0.50, 0.94,
            "A17 sweep: Q/(Q+C) ≈ 2% → 24% → 59% → 96%",
            ha="center", fontsize=7.5, color="#B71C1C",
            fontweight="bold")
    ax.text(0.50, 0.04,
            "Sedykh 2024; hybrid quantum-classical (Q-only baseline)",
            ha="center", fontsize=7.5, color="#444444")


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.6))
    for ax in axes.ravel():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        ax.axis("off")
    _draw_chebyshev   (axes[0, 0])
    _draw_te_qpinn_fnn(axes[0, 1])
    _draw_te_qpinn_qnn(axes[1, 0])
    _draw_qcpinn      (axes[1, 1])

    fig.suptitle(
        "Quantum ansatz families on the solver-task surface "
        "(L = num_layers, n = num_qubits = 3)",
        y=1.005, fontsize=11)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_circuit_zoo.png")
    fig.savefig(OUT_DIR / "fig_circuit_zoo.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_circuit_zoo.pdf'}")


if __name__ == "__main__":
    main()
