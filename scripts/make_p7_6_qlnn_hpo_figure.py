"""P7.6 QLNN HPO sensitivity figure.

Renders a 2-panel publication figure summarizing the symmetric
QLNN HPO sweep:

  Panel A (left, large): per-(family, anchor cell) QLNN relL² range
    across the 6 HPO combinations (3 LRs × 2 train_steps). One bar
    per family × cell, error-bar showing the [min, max] across HPO,
    marker on the HPO-best. Reveals which cells are HPO-sensitive
    vs HPO-invariant.

  Panel B (right): final per-cell Δ = NeuralODE − QLNN at HPO-best
    for both sides, partitioned by regime, with the n=9 H1 verdict
    Δ_smooth − Δ_broad ± 95% CI annotation.

Reads:
  results/p7_6_qlnn_hpo/summary.json
  results/p7_6_qlnn_hpo/h1_verdict_full_hpo_best.json
  results/p7_6_qlnn_hpo/per_cell_records_full_hpo_best.json

Writes:
  paper/figures/fig_p7_6_qlnn_hpo.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results" / "p7_6_qlnn_hpo"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Wong palette
FAMILY_COLORS = {
    "chebyshev_dqc":  "#0072B2",  # blue
    "te_qpinn_fnn":   "#009E73",  # green
    "te_qpinn_qnn":   "#D55E00",  # vermilion
    "qcpinn":         "#CC79A7",  # pink
}
C_SMOOTH = "#56B4E9"   # sky-blue
C_BROAD  = "#E69F00"   # orange


def _load_json(p: Path) -> dict | list:
    return json.loads(p.read_text())


def main() -> None:
    summary = _load_json(IN / "summary.json")
    verdict = _load_json(IN / "h1_verdict_full_hpo_best.json")
    cells = _load_json(IN / "per_cell_records_full_hpo_best.json")

    families = summary["families"]
    anchor_cells = summary["anchor_cells"]  # e.g. lotka_volterra_seed2
    stab = summary["family_sign_stability"]

    fig = plt.figure(figsize=(13.5, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.32)

    # ---- Panel A — HPO range per (family, anchor cell) -------------------
    axA = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(anchor_cells))
    width = 0.18
    for i, fam in enumerate(families):
        rng_lo, rng_hi, fam_best = [], [], []
        for ac in anchor_cells:
            d = stab[fam][ac]["delta_range"]
            rng_lo.append(d[0])
            rng_hi.append(d[1])
            fam_best.append(max(d))
        rng_lo = np.array(rng_lo); rng_hi = np.array(rng_hi)
        fam_best = np.array(fam_best)
        xx = x_pos + (i - 1.5) * width
        # Bar = HPO range; marker = HPO-best Δ for this family @ this cell.
        axA.bar(xx, rng_hi - rng_lo, bottom=rng_lo, width=width,
                color=FAMILY_COLORS[fam], alpha=0.55,
                edgecolor=FAMILY_COLORS[fam], label=fam)
        axA.scatter(xx, fam_best, color=FAMILY_COLORS[fam],
                    edgecolor="black", linewidth=0.6, zorder=4,
                    s=28, marker="o")

    axA.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    axA.set_xticks(x_pos)
    axA.set_xticklabels([ac.replace("_seed", " s") for ac in anchor_cells])
    axA.set_ylabel("Δ = relL²(c-PINN) − relL²(QLNN) across HPO grid", fontsize=10)
    axA.set_title("(A) Symmetric QLNN HPO: per-family Δ range across "
                  "3 LRs × 2 train_steps", fontsize=10.5)
    axA.legend(loc="lower right", ncol=2, frameon=True, fontsize=8)
    axA.grid(True, alpha=0.25, axis="y")

    # ---- Panel B — Per-cell Δ at full HPO-best + verdict -----------------
    axB = fig.add_subplot(gs[0, 1])
    sys_label = [f"{c['system'].replace('lotka_volterra', 'LV')[:8]} s{c['seed']}"
                 for c in cells]
    deltas = [c["delta"] for c in cells]
    regimes = [c["regime"] for c in cells]
    is_hpo_anchor = ["HPO-best" in c["qlnn_source"] for c in cells]
    colors = [C_SMOOTH if r == "smooth_periodic" else C_BROAD for r in regimes]
    edges = ["black" if h else "0.5" for h in is_hpo_anchor]
    bar_lw = [1.6 if h else 0.6 for h in is_hpo_anchor]

    yy = np.arange(len(cells))
    for j in range(len(cells)):
        axB.barh(yy[j], deltas[j], color=colors[j], alpha=0.85,
                 edgecolor=edges[j], linewidth=bar_lw[j])

    axB.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    axB.set_yticks(yy)
    axB.set_yticklabels(sys_label, fontsize=8.5)
    axB.set_xlabel("Δ = relL²(c-PINN) − relL²(QLNN)", fontsize=10)
    axB.invert_yaxis()

    b = verdict["bootstrap"]
    title = (
        f"(B) HPO-best per-cell Δ — H1: {verdict['outcome']}\n"
        f"Δ_smooth = {b['delta_smooth_mean']:+.3f}, "
        f"Δ_broad = {b['delta_broad_mean']:+.3f}, "
        f"Δ_diff = {b['delta_diff_mean']:+.3f}\n"
        f"95% CI [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] "
        f"(n={b['n_smooth']}+{b['n_broad']})"
    )
    axB.set_title(title, fontsize=9.5)

    # Legend for panel B
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_SMOOTH, ec="black",
                      label="smooth_periodic"),
        plt.Rectangle((0, 0), 1, 1, fc=C_BROAD, ec="black",
                      label="broadband_multiscale"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=1.6,
                      label="HPO-best anchor seed"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="0.5", lw=0.6,
                      label="default-Adam other seed"),
    ]
    axB.legend(handles=handles, loc="lower right", fontsize=7.5,
               frameon=True)
    axB.grid(True, alpha=0.25, axis="x")

    fig.suptitle(
        "P7.6 — Symmetric QLNN HPO sensitivity on ODE solver task   "
        "(72 retrains: 4 families × 3 anchor cells × 3 LRs × 2 train_steps)",
        fontsize=11.5, y=1.02,
    )

    out_png = OUT / "fig_p7_6_qlnn_hpo.png"
    out_pdf = OUT / "fig_p7_6_qlnn_hpo.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
