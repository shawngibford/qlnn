"""Original explainer diagrams for the advisor deck.

Diagram A — What is a Liquid Neural Network (Hasani 2021 LTC):
  a leaky-integrator neuron whose time-constant tau is modulated by
  the input; contrast against a fixed-tau RNN cell.

Diagram B — Where the quantum goes (our LiquidQuantumCell):
  the PQC replaces the classical nonlinearity f(.); its PauliZ
  readout q(x) enters the ODE BOTH as an extra leak conductance and
  as the drive.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

NAVY   = "#1E2761"
ICE    = "#CADCFC"
VIOLET = "#7C3AED"   # quantum accent
GREY   = "#6B7280"
GREEN  = "#0F9D58"

def card(ax, x, y, w, h, fc, ec, lw=1.6, r=0.03):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, zorder=2))

def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=2.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=16, color=color, lw=lw, zorder=3))

# ============ Diagram A: classical LNN =====================================
fig, ax = plt.subplots(figsize=(11, 5.6), dpi=200)
ax.set_xlim(0, 11); ax.set_ylim(0, 5.6); ax.axis("off")

# Left column: ordinary RNN neuron (fixed tau)
card(ax, 0.35, 3.1, 4.4, 2.1, "#F4F6FB", GREY)
ax.text(2.55, 4.95, "Ordinary continuous-time neuron", ha="center",
        fontsize=12.5, color=GREY, weight="bold")
ax.text(2.55, 4.35, r"$\dfrac{dh}{dt} \;=\; -\dfrac{h}{\tau} \;+\; f(x,h)$",
        ha="center", fontsize=15, color="#333")
ax.text(2.55, 3.55, "fixed time-constant  →  one temporal scale,\nset at design time",
        ha="center", fontsize=10.5, color=GREY)

# Right column: LTC neuron
card(ax, 6.2, 3.1, 4.4, 2.1, "#EEF3FF", NAVY)
ax.text(8.4, 4.95, "Liquid time-constant neuron (Hasani 2021)",
        ha="center", fontsize=12.5, color=NAVY, weight="bold")
ax.text(8.4, 4.32, r"$\dfrac{dh}{dt} = -\left[\dfrac{1}{\tau} + f(x,h)\right]\, h \;+\; A\, f(x,h)$",
        ha="center", fontsize=14.5, color=NAVY)
ax.text(8.4, 3.52, "input modulates the effective time-constant →\nthe neuron speeds up / slows down with the signal",
        ha="center", fontsize=10.5, color=NAVY)

arrow(ax, 4.95, 4.15, 6.0, 4.15, color=NAVY)
ax.text(5.48, 4.42, "make τ\n“liquid”", ha="center", fontsize=10, color=NAVY, style="italic")

# Bottom: the mechanism cartoon — input signal modulating a leaky reservoir
axb_y = 0.35
card(ax, 0.35, axb_y, 10.25, 2.3, "white", ICE, lw=1.4)
ax.text(0.7, 2.28, "Intuition", fontsize=11.5, color=NAVY, weight="bold")

# input signal
t = np.linspace(0, 2.4, 300)
sig = 0.36*np.sin(5*t) * np.exp(-0.4*t) + 0.5
ax.plot(0.85 + t, axb_y + 0.55 + sig, color=GREY, lw=2)
ax.text(2.05, axb_y + 1.75, "input  x(t)", fontsize=10, color=GREY, ha="center")

arrow(ax, 3.6, axb_y + 1.1, 4.45, axb_y + 1.1, color=GREY)

# neuron as a "leaky bucket" with adjustable valve
circ = Circle((5.35, axb_y + 1.12), 0.62, fc="#EEF3FF", ec=NAVY, lw=2, zorder=2)
ax.add_patch(circ)
ax.text(5.35, axb_y + 1.26, "h(t)", ha="center", fontsize=12, color=NAVY, weight="bold")
ax.text(5.35, axb_y + 0.86, r"leak $\propto \frac{1}{\tau(x)}$", ha="center",
        fontsize=9, color=NAVY)
ax.text(5.35, axb_y - 0.02, "τ adapts to the input:  fast when the signal is busy, slow when calm",
        ha="center", fontsize=9.5, color=NAVY)

arrow(ax, 6.15, axb_y + 1.1, 7.0, axb_y + 1.1, color=NAVY)

# output trajectory
sig2 = 0.42*np.tanh(np.sin(3.1*t)) + 0.5
ax.plot(7.2 + t, axb_y + 0.55 + sig2, color=NAVY, lw=2.4)
ax.text(8.4, axb_y + 1.75, "adaptive-memory state  h(t)", fontsize=10, color=NAVY, ha="center")

plt.tight_layout()
plt.savefig("diagram_lnn.png", bbox_inches="tight", facecolor="white")
plt.close()

# ============ Diagram B: where the quantum goes ============================
fig, ax = plt.subplots(figsize=(11, 5.8), dpi=200)
ax.set_xlim(0, 11); ax.set_ylim(0, 5.8); ax.axis("off")

ax.text(5.5, 5.55, "Our cell:  swap the classical nonlinearity  f(x)  for a quantum circuit",
        ha="center", fontsize=13.5, color=NAVY, weight="bold")

# input
card(ax, 0.3, 3.3, 1.7, 1.0, "#F4F6FB", GREY)
ax.text(1.15, 3.8, "input\nx(t)", ha="center", va="center", fontsize=11.5, color="#333")

# PQC box (violet — the quantum part)
card(ax, 2.6, 2.55, 3.55, 2.5, "#F3EBFF", VIOLET, lw=2.2)
ax.text(4.38, 4.72, "Parameterized quantum circuit", ha="center",
        fontsize=10.8, color=VIOLET, weight="bold")
# mini circuit sketch: 3 qubit wires with boxes
for i, qy in enumerate([4.25, 3.8, 3.35]):
    ax.plot([2.85, 5.85], [qy, qy], color=VIOLET, lw=1.4)
    for bx in [3.3, 4.2, 5.1]:
        card(ax, bx, qy - 0.14, 0.34, 0.28, "white", VIOLET, lw=1.2, r=0.01)
ax.text(4.38, 2.85, r"angle-encode $x$  →  entangling layers  →  $\langle Z_j\rangle$",
        ha="center", fontsize=9, color=VIOLET)

arrow(ax, 2.05, 3.8, 2.55, 3.8, color=GREY)

# q(x) output
arrow(ax, 6.25, 3.8, 6.9, 3.8, color=VIOLET)
ax.text(6.57, 4.12, r"$q(x)$", ha="center", fontsize=12, color=VIOLET)
ax.text(6.57, 3.45, r"$\in[-1,1]^Q$", ha="center", fontsize=9, color=VIOLET)

# ODE box
card(ax, 6.95, 2.75, 3.75, 2.15, "#EEF3FF", NAVY, lw=2.0)
ax.text(8.82, 4.55, "Liquid ODE  (Diffrax-integrated)", ha="center",
        fontsize=11.5, color=NAVY, weight="bold")
ax.text(8.82, 3.85,
        r"$\dfrac{dh}{dt} = -\left[\dfrac{1}{\tau} + q(x)\right]\odot h \;+\; A \odot q(x)$",
        ha="center", fontsize=13, color=NAVY)
ax.text(8.82, 3.05, r"learnable per-qubit $\tau$ (softplus > $\tau_{min}$)"
        "\nand amplitude $A$", ha="center", fontsize=9.5, color=NAVY)

# Two roles annotation
arrow(ax, 7.9, 2.7, 7.3, 1.95, color=VIOLET, lw=1.8)
arrow(ax, 9.7, 2.7, 10.15, 1.95, color=VIOLET, lw=1.8)
card(ax, 5.9, 1.25, 2.7, 0.72, "white", VIOLET, lw=1.4)
ax.text(7.25, 1.61, "role 1 — leak conductance:\nquantum output modulates τ", ha="center",
        fontsize=9.5, color=VIOLET)
card(ax, 8.85, 1.25, 2.0, 0.72, "white", VIOLET, lw=1.4)
ax.text(9.85, 1.61, "role 2 — drive:\ninjects the signal", ha="center",
        fontsize=9.5, color=VIOLET)

# ablation strip
card(ax, 0.3, 0.15, 10.4, 0.72, "#F7F7F9", GREY, lw=1.2)
ax.text(5.55, 0.51,
        "Fairness 2×2:   remove τ-leak → non-liquid quantum cell    |    swap PQC for MLP → classical LTC    |    remove both → plain Neural-ODE",
        ha="center", fontsize=9.8, color="#333")

# label on classical baseline of h dynamics
ax.text(1.15, 2.55, "history window\n(T, d) trajectory", ha="center", fontsize=9, color=GREY)

plt.tight_layout()
plt.savefig("diagram_quantum_cell.png", bbox_inches="tight", facecolor="white")
plt.close()
print("diagrams written")
