"""Render the symmetric-HPO sensitivity heatmap (A9 / P7.6).

For each of the 3 anchor cells the symmetric HPO sweep ran a 3 LR × 2
training-step grid, with the CLASSICAL side tuned at the same grid as
the QUANTUM side (per A9). For each cell of that grid we have a per-
configuration Δ = cPINN_relL2 - QLNN_best_relL2. Sign-stability of Δ
across the grid is the headline robustness test.

Reads:
  results/p7_5_hpo_sensitivity/{anchor}/cell_results.json

Emits paper/figures/fig_hpo_heatmap.{png,pdf}.

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
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = REPO_ROOT / "results" / "p7_5_hpo_sensitivity"
OUT_DIR = REPO_ROOT / "paper" / "figures"

ANCHORS = [
    ("lotka_volterra_seed_2", "Lotka-Volterra  s2  [S]"),
    ("van_der_pol_seed_1",    "Van der Pol  s1  [S]"),
    ("lorenz_seed_2",          "Lorenz  s2  [B]"),
]


def _grid_from_runs(runs: list[dict]) -> tuple[np.ndarray, list[float], list[int]]:
    """Build a (n_lr × n_steps) Δ matrix from the per-config runs."""
    lrs    = sorted({float(r["lr"]) for r in runs})
    steps  = sorted({int(r["train_steps"]) for r in runs})
    mat = np.full((len(lrs), len(steps)), np.nan)
    for r in runs:
        i = lrs.index(float(r["lr"]))
        j = steps.index(int(r["train_steps"]))
        mat[i, j] = float(r["delta"])
    return mat, lrs, steps


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8),
                              constrained_layout=True,
                              gridspec_kw=dict(wspace=0.30))

    # Common color limits across panels for direct visual comparison.
    all_vals = []
    grids = []
    for anchor, _ in ANCHORS:
        with (IN_DIR / anchor / "cell_results.json").open() as f:
            cell = json.load(f)
        mat, lrs, steps = _grid_from_runs(cell["runs"])
        grids.append((mat, lrs, steps, cell))
        all_vals.append(mat.ravel())
    all_vals_arr = np.concatenate(all_vals)
    vmax = float(np.nanmax(np.abs(all_vals_arr)))
    vmin = -vmax

    for ax, (label_path, label), (mat, lrs, steps, cell) in zip(
            axes, ANCHORS, grids):
        im = ax.imshow(mat, aspect="auto", origin="lower",
                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(steps)))
        ax.set_xticklabels([str(s) for s in steps])
        ax.set_yticks(np.arange(len(lrs)))
        ax.set_yticklabels([f"{lr:g}" for lr in lrs])
        ax.set_xlabel("training steps")
        if label_path.startswith("lotka"):
            ax.set_ylabel("learning rate  (Adam)")
        # Annotate each cell with its Δ value
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                # Color text by brightness — light on dark, dark on light
                rel = (val - vmin) / max(vmax - vmin, 1e-9)
                txt_color = "white" if abs(rel - 0.5) > 0.30 else "black"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=8.5, color=txt_color, fontweight="bold")
        stab = cell["sign_stability"]
        stab_color = ("#0a8a3a" if stab == "all_positive"
                      else "#c0392b" if stab == "mixed"
                      else "#555555")
        ax.set_title(f"{label}\nsign-stability: "
                      f"$\\bf{{{stab.replace('_', ' ')}}}$",
                      fontsize=9.5, color=stab_color)

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02,
                         label=r"$\Delta$ = relL²(cPINN) − relL²(QLNN best)")

    fig.suptitle(
        "Amendment A9 — symmetric HPO sensitivity at 3 anchor cells  "
        "(3 LR × 2 training-step budgets per cell, both sides tuned)",
        y=1.06, fontsize=11)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_hpo_heatmap.png")
    fig.savefig(OUT_DIR / "fig_hpo_heatmap.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_hpo_heatmap.pdf'}")


if __name__ == "__main__":
    main()
