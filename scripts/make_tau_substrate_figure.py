"""Render the τ-substrate disagreement figure.

The single most novel mechanism finding of the paper: the LTC
time-constant machinery shows SIGN-OPPOSITE Δ_diff on classical-MLP vs
quantum-cell hidden state. The two decomposition paths that ISOLATE the
τ machinery disagree in sign; the two that DON'T (control paths)
roughly agree.

Reads:
  results/p7_11_decomposition/h1_liquid_via_classical.json
  results/p7_11_decomposition/h1_liquid_via_quantum.json
  results/p7_11_decomposition/h1_quantum_via_ltc.json
  results/p7_11_decomposition/h1_quantum_via_nonliquid.json

Emits paper/figures/fig_tau_substrate.{png,pdf}.

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
IN_DIR = REPO_ROOT / "results" / "p7_11_decomposition"
OUT_DIR = REPO_ROOT / "paper" / "figures"

# Four decomposition paths, ordered: τ-ISOLATING first, CONTROLS second.
PATHS = [
    ("liquid_via_classical", "τ-isolating\n(classical substrate)",
     "+", "#0072B2"),
    ("liquid_via_quantum",   "τ-isolating\n(quantum substrate)",
     "−", "#D55E00"),
    ("quantum_via_ltc",      "Control\n(quantum vs LTC)",
     "·", "#666666"),
    ("quantum_via_nonliquid","Control\n(quantum vs nonliquid)",
     "·", "#999999"),
]


def _load_delta(name: str) -> tuple[float, float, float]:
    path = IN_DIR / f"h1_{name}.json"
    with path.open() as f:
        d = json.load(f)
    b = d["bootstrap"]
    return (float(b["delta_diff_mean"]),
            float(b["ci_low"]), float(b["ci_high"]))


def main() -> None:
    deltas = [(label, role, color, *_load_delta(name))
              for name, label, role, color in PATHS]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(deltas))
    for i, (label, role, color, mean, lo, hi) in enumerate(deltas):
        err = np.array([[mean - lo], [hi - mean]])
        ax.errorbar(mean, i, xerr=err, fmt="o", color=color,
                    markersize=9, capsize=4, capthick=1.5, lw=1.5,
                    markeredgecolor="black", markeredgewidth=0.6)
        ax.text(mean, i + 0.28, role, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=color)

    ax.axvline(0.0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in deltas], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta_{\mathrm{diff}} = \Delta_{\mathrm{smooth}} -"
                  r"\Delta_{\mathrm{broad}}$  (95% paired-bootstrap CI)")
    ax.set_title("Substrate-dependent τ signature: liquid-cell τ machinery\n"
                 "flips sign between classical and quantum hidden state",
                 fontsize=10.5)
    ax.set_xlim(-1.3, 0.6)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    # Annotate the headline disagreement
    a_mean, a_lo, a_hi = _load_delta("liquid_via_classical")
    b_mean, b_lo, b_hi = _load_delta("liquid_via_quantum")
    ax.annotate(
        f"sign flip: {a_mean:+.3f} vs {b_mean:+.3f}\n(|gap| = {abs(a_mean - b_mean):.3f})",
        xy=(0.02, 0.5), xycoords="figure fraction",
        fontsize=8.5, ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                  ec="goldenrod", lw=0.8))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tau_substrate.png")
    fig.savefig(OUT_DIR / "fig_tau_substrate.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_tau_substrate.pdf'}")


if __name__ == "__main__":
    main()
