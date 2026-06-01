"""Render a loss-history convergence grid for the 4 ODE systems.

For each of the 4 P3.6 systems (lotka_volterra, van_der_pol,
fitzhugh_nagumo, lorenz), show training-loss-vs-step on a log-y axis
for the 4 quantum families (chebyshev_dqc, te_qpinn_fnn, te_qpinn_qnn,
qcpinn), with all 3 seeds shown as semitransparent overlays + the
seed-mean as a solid line. This is the "did training work?" diagnostic
every reviewer expects in a benchmark paper.

Reads:
  results/p3_6_multi_state/{family}_{system}/seed_N/curves.npz
    — key 'loss_history' shape (steps,).

Emits paper/figures/fig_loss_history_grid.{png,pdf}.

Standalone — does NOT enter the integrity contract.
"""
from __future__ import annotations

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
IN_DIR = REPO_ROOT / "results" / "p3_6_multi_state"
OUT_DIR = REPO_ROOT / "paper" / "figures"

SYSTEMS = [
    ("lotka_volterra", "Lotka-Volterra  [S]"),
    ("van_der_pol",    "Van der Pol  [S]"),
    ("fitzhugh_nagumo","FitzHugh-Nagumo  [B]"),
    ("lorenz",         "Lorenz  [B]"),
]
FAMILIES = [
    ("chebyshev_dqc", "#0072B2"),
    ("te_qpinn_fnn",  "#D55E00"),
    ("te_qpinn_qnn",  "#CC79A7"),
    ("qcpinn",        "#009E73"),
]
SEEDS = (0, 1, 2)


def _load_loss(family: str, system: str, seed: int) -> np.ndarray | None:
    p = IN_DIR / f"{family}_{system}" / f"seed_{seed}" / "curves.npz"
    if not p.exists():
        return None
    return np.load(p)["loss_history"]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4),
                              constrained_layout=True)
    handles_by_family: dict[str, plt.Line2D] = {}
    for ax, (system, label) in zip(axes.ravel(), SYSTEMS):
        for family, color in FAMILIES:
            seed_curves = []
            for seed in SEEDS:
                lh = _load_loss(family, system, seed)
                if lh is None:
                    continue
                seed_curves.append(lh)
                ax.plot(lh, color=color, lw=0.6, alpha=0.25, zorder=2)
            if seed_curves:
                # Align to common length and plot seed-mean
                n_min = min(len(c) for c in seed_curves)
                stacked = np.stack([c[:n_min] for c in seed_curves])
                mean = stacked.mean(axis=0)
                line, = ax.plot(mean, color=color, lw=1.7, zorder=3,
                                 label=family)
                handles_by_family[family] = line
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("physics-residual loss")
        ax.set_title(label, fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.5, which="both")

    # Single shared legend in a top band — no per-axes duplication.
    handles = [handles_by_family[f] for f, _ in FAMILIES
               if f in handles_by_family]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="lower center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(handles), frameon=False, fontsize=10)
    fig.suptitle(
        "Training-loss histories: 4 quantum families × 4 systems × 3 seeds  "
        "(faint = per-seed, bold = seed-mean)",
        y=1.075, fontsize=11)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_loss_history_grid.png")
    fig.savefig(OUT_DIR / "fig_loss_history_grid.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_loss_history_grid.pdf'}")


if __name__ == "__main__":
    main()
