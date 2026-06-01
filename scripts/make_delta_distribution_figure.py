"""Render the per-system Δ-distribution box plot for the n=24 PRIMARY
solver-task cells.

The headline solver-task verdict aggregates 24 (8 systems × 3 seeds)
per-cell Δ values into a single Δ_diff with CI. This figure shows
the underlying distribution per system: median, IQR, whiskers, and
individual seed points. The reader can see at a glance whether the
verdict is driven by a few extreme cells or by a broad pattern.

Reads:
  results/p7_8_solver_h1_n24/per_cell_records.json
    — 24 records with {system, seed, qlnn_relL2, classical_pinn_relL2,
      delta, regime}.

Emits paper/figures/fig_delta_distribution.{png,pdf}.

Standalone — does NOT enter the integrity contract.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = (REPO_ROOT / "results" / "p7_8_solver_h1_n24" /
            "per_cell_records.json")
OUT_DIR = REPO_ROOT / "paper" / "figures"


def main() -> None:
    with IN_PATH.open() as f:
        records = json.load(f)

    by_sys: dict[str, list[float]] = defaultdict(list)
    regime_by_sys: dict[str, str] = {}
    for r in records:
        by_sys[r["system"]].append(float(r["delta"]))
        regime_by_sys[r["system"]] = r["regime"]

    # Sort: smooth above broad band; alphabetical within regime.
    systems = sorted(by_sys.keys(),
                      key=lambda s: (regime_by_sys[s] != "smooth_periodic",
                                       s))

    fig, ax = plt.subplots(figsize=(10.5, 4.8),
                            constrained_layout=True)

    box_data = [by_sys[s] for s in systems]
    bp = ax.boxplot(box_data, positions=np.arange(len(systems)),
                     widths=0.55, showmeans=True, meanline=True,
                     patch_artist=True,
                     medianprops=dict(color="black", lw=1.5),
                     meanprops=dict(color="#B71C1C", lw=1.2, ls="--"),
                     whiskerprops=dict(color="#444444", lw=1.0),
                     capprops=dict(color="#444444", lw=1.0))
    for box, sys in zip(bp["boxes"], systems):
        color = ("#BBDEFB" if regime_by_sys[sys] == "smooth_periodic"
                 else "#FFCCBC")
        box.set(facecolor=color, edgecolor="#444444", lw=0.8)

    # Overlay individual seed points
    for i, sys in enumerate(systems):
        ys = by_sys[sys]
        xs = np.full(len(ys), i) + np.random.default_rng(42).normal(
            0, 0.06, size=len(ys))
        color = ("#0D47A1" if regime_by_sys[sys] == "smooth_periodic"
                  else "#B71C1C")
        ax.scatter(xs, ys, s=45, color=color, edgecolor="black",
                    lw=0.5, alpha=0.85, zorder=4)

    ax.axhline(0.0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_xticks(np.arange(len(systems)))
    ax.set_xticklabels(systems, rotation=22, ha="right", fontsize=9.5)
    ax.set_ylabel(r"per-cell  $\Delta = "
                   r"\mathrm{relL}^2(\mathrm{cPINN}) - "
                   r"\mathrm{relL}^2(\mathrm{QLNN\;best})$")
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    # Regime separator vertical line
    smooth_idx = [i for i, s in enumerate(systems)
                   if regime_by_sys[s] == "smooth_periodic"]
    broad_idx = [i for i, s in enumerate(systems)
                  if regime_by_sys[s] != "smooth_periodic"]
    if smooth_idx and broad_idx:
        ax.axvline(max(smooth_idx) + 0.5, color="black", lw=1.2,
                    alpha=0.5)
    # Color-code the x-tick labels by regime instead of relying on
    # a top-margin band (which collides with the suptitle).
    for tick, sys in zip(ax.get_xticklabels(), systems):
        tick.set_color("#0D47A1"
                        if regime_by_sys[sys] == "smooth_periodic"
                        else "#B71C1C")

    ax.set_title(
        r"Per-system $\Delta$ distribution underlying the $n{=}24$ "
        "PRIMARY solver verdict\n"
        "(box = IQR; solid line = median; dashed line = mean; "
        "scatter = individual seeds;  "
        "blue x-tick = smooth/periodic, red x-tick = broadband)",
        fontsize=10.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_delta_distribution.png")
    fig.savefig(OUT_DIR / "fig_delta_distribution.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_delta_distribution.pdf'}")


if __name__ == "__main__":
    main()
