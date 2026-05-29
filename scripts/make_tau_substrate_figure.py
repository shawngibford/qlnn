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
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.10,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = REPO_ROOT / "results" / "p7_11_decomposition"
OUT_DIR = REPO_ROOT / "paper" / "figures"

# Four decomposition paths. τ-isolating paths first, controls below.
PATHS = [
    ("liquid_via_classical",  "τ-isolating  (classical substrate)",
     "#0072B2", "tau"),
    ("liquid_via_quantum",    "τ-isolating  (quantum substrate)",
     "#D55E00", "tau"),
    ("quantum_via_nonliquid", "control  (quantum vs nonliquid)",
     "#888888", "ctrl"),
    ("quantum_via_ltc",       "control  (quantum vs LTC)",
     "#BBBBBB", "ctrl"),
]


def _load(name: str) -> tuple[float, float, float]:
    with (IN_DIR / f"h1_{name}.json").open() as f:
        b = json.load(f)["bootstrap"]
    return (float(b["delta_diff_mean"]),
            float(b["ci_low"]), float(b["ci_high"]))


def main() -> None:
    deltas = [(label, color, kind, *_load(name))
              for name, label, color, kind in PATHS]

    fig, ax = plt.subplots(figsize=(9.6, 5.2),
                            constrained_layout=True)

    y = np.arange(len(deltas))
    for i, (label, color, kind, mean, lo, hi) in enumerate(deltas):
        err = np.array([[mean - lo], [hi - mean]])
        ax.errorbar(mean, i, xerr=err, fmt="o", color=color,
                    markersize=10, capsize=5, capthick=1.6, lw=2.0,
                    markeredgecolor="black", markeredgewidth=0.7,
                    zorder=3)
        # Numeric value label offset slightly BELOW the point — the
        # title sits above row 0 so we never want labels going up.
        ax.annotate(f"{mean:+.3f}", xy=(mean, i),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", va="top",
                    fontsize=9, fontweight="bold", color=color)

    ax.axvline(0.0, color="black", lw=0.8, ls="--", alpha=0.6,
                zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in deltas], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta_{\mathrm{diff}} = \Delta_{\mathrm{smooth}} -"
                  r"\Delta_{\mathrm{broad}}$  "
                  r"(95% paired-bootstrap CI)")
    a_mean, *_ = _load("liquid_via_classical")
    b_mean, *_ = _load("liquid_via_quantum")
    ax.set_title(
        "Substrate-dependent $\\tau$ signature: the two $\\tau$-isolating "
        "decomposition paths\n"
        f"disagree in sign  ({a_mean:+.3f}  vs  {b_mean:+.3f};   "
        f"$|\\Delta_{{\\mathrm{{classical}}}} - "
        f"\\Delta_{{\\mathrm{{quantum}}}}|$ = {abs(a_mean - b_mean):.3f})",
        fontsize=11)
    ax.grid(axis="x", alpha=0.30, lw=0.5)

    # Reasonable x-range with padding so error bars don't kiss the spine.
    all_lo  = min(lo for *_, lo, _ in deltas)
    all_hi  = max(hi for *_, _, hi in deltas)
    pad = 0.10 * (all_hi - all_lo)
    ax.set_xlim(all_lo - pad, all_hi + pad)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_tau_substrate.png")
    fig.savefig(OUT_DIR / "fig_tau_substrate.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_tau_substrate.pdf'}")


if __name__ == "__main__":
    main()
