"""Render a per-system trajectory comparison grid.

For each of the 4 P3.6 systems (lotka_volterra, van_der_pol,
fitzhugh_nagumo, lorenz), show the QLNN prediction vs reference
numerical integrator at seed 0 for the best-performing family per
system. First-component trajectory only (2-state systems plot u[0];
Lorenz plots x).

Reads:
  results/p3_6_multi_state/{family}_{system}/seed_0/curves.npz
  results/p3_6_multi_state/{family}_{system}/seeds_summary.json

Best family per system selected by lowest mean relative L² across
seeds (from seeds_summary.json).

Emits paper/figures/fig_trajectory_grid.{png,pdf}.

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
IN_DIR = REPO_ROOT / "results" / "p3_6_multi_state"
OUT_DIR = REPO_ROOT / "paper" / "figures"

SYSTEMS = [
    ("lotka_volterra", "Lotka-Volterra  [S, predator-prey]", "smooth"),
    ("van_der_pol",    "Van der Pol  [S, limit cycle]",       "smooth"),
    ("fitzhugh_nagumo","FitzHugh-Nagumo  [B, fast-slow]",     "broad"),
    ("lorenz",         "Lorenz  [B, chaotic]",                "broad"),
]
FAMILIES = ("chebyshev_dqc", "te_qpinn_fnn", "te_qpinn_qnn", "qcpinn")


def _summary_relL2(family: str, system: str) -> float:
    p = IN_DIR / f"{family}_{system}" / "seeds_summary.json"
    if not p.exists():
        return float("inf")
    with p.open() as f:
        d = json.load(f)
    # Schema: {"metrics": {"relative_l2": {"mean": ...}}}
    try:
        return float(d["metrics"]["relative_l2"]["mean"])
    except Exception:
        return float("inf")


def _best_family(system: str) -> str:
    return min(FAMILIES, key=lambda fam: _summary_relL2(fam, system))


def _load_curves(family: str, system: str, seed: int = 0):
    p = IN_DIR / f"{family}_{system}" / f"seed_{seed}" / "curves.npz"
    z = np.load(p)
    return z["t_eval"], z["u_pred"], z["u_ref"]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 5.6), sharex=False)
    for ax, (system, label, _regime) in zip(axes.ravel(), SYSTEMS):
        family = _best_family(system)
        t, u_pred, u_ref = _load_curves(family, system, seed=0)
        relL2 = _summary_relL2(family, system)
        # Plot first state component (or x for Lorenz).
        ax.plot(t, u_ref[:, 0], color="black", lw=1.6, label="reference")
        ax.plot(t, u_pred[:, 0], color="#0072B2", lw=1.4, ls="--",
                label=f"{family}")
        ax.set_title(f"{label}\nbest family: {family}, "
                     f"⟨relL²⟩={relL2:.3g}",
                     fontsize=10)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$u_0(t)$")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle(
        "Per-system QLNN-vs-reference trajectories  (seed 0, "
        "first state component)", y=1.01, fontsize=11)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_trajectory_grid.png")
    fig.savefig(OUT_DIR / "fig_trajectory_grid.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_trajectory_grid.pdf'}")


if __name__ == "__main__":
    main()
