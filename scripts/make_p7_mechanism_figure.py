"""Render the P7 H3 mechanism figure — cross-tab + correlation analysis.

Reads:
  results/p7_t3_mechanism/t3_scalars.json
    — 4 T3 scalars per forecaster family at the P4 config.
  results/p4_forecaster_rollout/{system}_{family}/seed_N/metrics.json
    — per-cell QLNN relative-L² (to identify the BEST-ansatz family
       per (system, seed)).
  results/p5_h1_verdict/per_cell_records.json
    — per-cell Δ = NeuralODE − QLNN_best and regime tag.
  results/p7_t3_mechanism/gradient_scaling.json
    — barren-plateau decay curves per family.

Emits paper/figures/fig_p7_mechanism.{png,pdf} as a 4-panel figure:

  Top-left:    Per-cell Δ scatter, color = best-ansatz family.
               Visualizes which family wins per cell.
  Top-right:   T3 scalars per family (4-bar grid for each family).
               LV-favored cells, Lorenz-favored cells highlighted.
  Bottom-left: Barren-plateau scaling — Var(grad) vs n_qubits per
               family. log-scale y.
  Bottom-right: H3 correlation table — Δ vs each T3 scalar across
               cells. Spearman ρ + p-value annotated.

Standalone — does NOT plug into make_paper_figures.py or the
paper-integrity contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
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
P4_IN = REPO_ROOT / "results" / "p4_forecaster_rollout"
P5_IN = REPO_ROOT / "results" / "p5_h1_verdict"
P7_IN = REPO_ROOT / "results" / "p7_t3_mechanism"
OUT = REPO_ROOT / "paper" / "figures"

FAMILIES = [
    "data_reuploading", "hardware_efficient",
    "strongly_entangling", "brickwall",
]

FAMILY_COLOR = {
    "data_reuploading":    "#0072B2",
    "hardware_efficient":  "#D55E00",
    "strongly_entangling": "#009E73",
    "brickwall":           "#CC79A7",
}


def _spearman_rho_pvalue(x, y):
    """Spearman rank correlation + p-value using scipy.stats.spearmanr.

    scipy handles tied ranks correctly via average-rank assignment
    (essential here — many T3 scalars are constant across cells
    because the same family gets selected as best multiple times).

    Returns (rho, p_value); if either input has zero variance after
    tie-handling, returns (nan, nan) — the correct null signal.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 3:
        return 0.0, 1.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        # No variance in one variable → Spearman is undefined.
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except ImportError:
        # Fallback: use the naive ranks (will be wrong for ties but
        # we already filtered out the constant-variance case).
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        rho = float(np.corrcoef(rx, ry)[0, 1])
        return rho, float("nan")


def _load_data():
    t3 = json.loads((P7_IN / "t3_scalars.json").read_text())
    bp = json.loads((P7_IN / "gradient_scaling.json").read_text())
    cells = json.loads((P5_IN / "per_cell_records.json").read_text())

    # Map each cell to its BEST-ansatz family (from P4 results).
    for c in cells:
        per_fam = {}
        for fam in FAMILIES:
            p = (P4_IN / f"{c['system']}_{fam}" /
                 f"seed_{c['seed']}" / "metrics.json")
            if p.exists():
                per_fam[fam] = float(
                    json.loads(p.read_text())["relative_l2"])
        if per_fam:
            c["best_family"] = min(per_fam, key=per_fam.get)
        else:
            c["best_family"] = None
    return t3, bp, cells


# ---------------------------------------------------------------------------
# Top-left: per-cell Δ scatter, colored by best-ansatz family
# ---------------------------------------------------------------------------


def _topleft_per_cell_delta(ax, cells) -> None:
    smooth = [c for c in cells if c["regime"] == "smooth_periodic"]
    broad = [c for c in cells if c["regime"] == "broadband_multiscale"]

    # x-axis: ordered first by regime, then by best_family alphabetically.
    def order(c):
        return (0 if c["regime"] == "smooth_periodic" else 1,
                c["system"], c["seed"])
    ordered = sorted(cells, key=order)

    xs = list(range(len(ordered)))
    ys = [c["delta"] for c in ordered]
    colors = [FAMILY_COLOR.get(c["best_family"], "gray")
              for c in ordered]
    edgecolors = ["black"] * len(ordered)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    sc = ax.scatter(xs, ys, c=colors, edgecolors=edgecolors,
                    s=130, linewidth=0.7, zorder=3)

    # Family legend.
    legend_handles = [
        plt.scatter([], [], c=FAMILY_COLOR[fam], edgecolors="black",
                    s=80, label=fam)
        for fam in FAMILIES
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=7, ncol=2)

    # Regime separator and annotations.
    n_smooth = sum(1 for c in ordered if c["regime"] == "smooth_periodic")
    if 0 < n_smooth < len(ordered):
        ax.axvline(n_smooth - 0.5, color="gray", linestyle="--",
                   linewidth=1.0, alpha=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{c['system'][:3]}_{c['seed']}" for c in ordered],
        rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Δ = NeuralODE − QLNN_best\n(positive ⇒ QLNN better)")
    ax.set_title("Per-cell Δ, colored by best-ansatz family",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Top-right: T3 scalars per family (grouped bars)
# ---------------------------------------------------------------------------


def _topright_t3_scalars(ax, t3) -> None:
    # 4 scalars × 4 families. Normalize each scalar to [0, 1] across
    # families so they sit on the same axis.
    metric_keys = [
        ("expressibility_kl", "KL to Haar\n(lower=more exp.)"),
        ("entangling_q",      "Entangling Q\n(higher=more ent.)"),
        ("gradient_variance", "Var(grad)\n(higher=more trainable)"),
        ("fourier_bandwidth", "Fourier K_max\n(higher=more freq.)"),
    ]

    n_metrics = len(metric_keys)
    n_families = len(FAMILIES)
    bar_w = 0.18
    xpos = np.arange(n_metrics)

    for i, family in enumerate(FAMILIES):
        vals = []
        for k, _ in metric_keys:
            v = t3[family][k]
            vals.append(float(v))
        # Normalize each metric across families to [0, 1] for plotting.
        all_per_metric = [
            [t3[f][k] for f in FAMILIES] for k, _ in metric_keys
        ]
        vmin = [min(per) for per in all_per_metric]
        vmax = [max(per) for per in all_per_metric]
        normalized = []
        for j, v in enumerate(vals):
            denom = max(vmax[j] - vmin[j], 1e-9)
            normalized.append((v - vmin[j]) / denom)
        offsets = (i - (n_families - 1) / 2) * bar_w
        x = xpos + offsets
        ax.bar(x, normalized, bar_w, label=family,
               color=FAMILY_COLOR[family],
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(xpos)
    ax.set_xticklabels([label for _, label in metric_keys],
                       rotation=0, fontsize=7)
    ax.set_ylabel("Min-max normalized\nacross families")
    ax.set_title("T3 scalars per family at P4 config (n=3, L=1)",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Bottom-left: barren-plateau scaling curves
# ---------------------------------------------------------------------------


def _bottomleft_bp_scaling(ax, bp) -> None:
    for family in FAMILIES:
        if family not in bp:
            continue
        d = bp[family]
        ns = sorted(int(k) for k in d.keys())
        vals = [d[str(n)] for n in ns]
        ax.plot(ns, vals, marker="o", color=FAMILY_COLOR[family],
                label=family, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("number of qubits")
    ax.set_ylabel("Var(∂⟨Z₀⟩/∂θ) (log)")
    ax.set_title("Barren-plateau scaling (McClean 2018) per family",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# Bottom-right: Δ vs T3-scalar correlation (annotation panel)
# ---------------------------------------------------------------------------


def _bottomright_correlations(ax, t3, cells) -> None:
    """Compute Spearman ρ between per-cell Δ and the best-ansatz's
    T3 scalar value per cell.

    For each cell, take its best-family's T3 scalar; correlate
    across all 9 cells against the per-cell Δ.
    """
    metric_keys = [
        ("expressibility_kl", "KL to Haar"),
        ("entangling_q", "Entangling Q"),
        ("gradient_variance", "Var(grad)"),
        ("fourier_bandwidth", "Fourier K_max"),
    ]

    rows = []
    valid_cells = [c for c in cells if c.get("best_family")]
    deltas = np.array([c["delta"] for c in valid_cells])
    for k, label in metric_keys:
        t3_vals = np.array([t3[c["best_family"]][k]
                             for c in valid_cells], dtype=np.float64)
        rho, p = _spearman_rho_pvalue(t3_vals, deltas)
        rows.append((label, t3_vals, rho, p))

    # Render as a 4-row table.
    ax.axis("off")
    ax.set_title("Cross-tabulation: per-cell Δ vs T3 scalar of best-ansatz",
                 fontsize=10)

    header = ["T3 scalar", "best-ansatz values across 9 cells",
              "Spearman ρ", "p-value"]
    col_widths = [0.18, 0.46, 0.18, 0.18]

    # Header row
    y = 0.88
    x0 = 0.02
    for i, h in enumerate(header):
        ax.text(x0 + sum(col_widths[:i]), y, h, fontsize=8,
                weight="bold", transform=ax.transAxes)

    for j, (label, t3_vals, rho, p) in enumerate(rows):
        y = 0.78 - j * 0.16
        ax.text(x0, y, label, fontsize=8, transform=ax.transAxes)
        vals_str = ", ".join(f"{v:.2g}" for v in t3_vals[:5])
        vals_str += " …"
        ax.text(x0 + col_widths[0], y, vals_str, fontsize=7,
                transform=ax.transAxes)
        # Render NaN as "undefined" (the correct interpretation
        # when one variable is constant across cells).
        if np.isnan(rho):
            ax.text(x0 + col_widths[0] + col_widths[1], y,
                    "undefined",
                    fontsize=8, color="#888888", style="italic",
                    transform=ax.transAxes)
            ax.text(x0 + col_widths[0] + col_widths[1] + col_widths[2],
                    y, "(no variance)",
                    fontsize=8, color="#888888", style="italic",
                    transform=ax.transAxes)
        else:
            color = ("#009E73" if p < 0.05
                     else ("#CC79A7" if p < 0.2 else "#888888"))
            ax.text(x0 + col_widths[0] + col_widths[1], y,
                    f"{rho:+.3f}", fontsize=9, color=color,
                    weight="bold", transform=ax.transAxes)
            ax.text(x0 + col_widths[0] + col_widths[1] + col_widths[2],
                    y, f"{p:.3f}", fontsize=9, color=color,
                    transform=ax.transAxes)

    ax.text(0.5, 0.04,
            "(green = p<0.05; pink = p<0.20; gray = no signal; n=9 cells)",
            fontsize=7, color="gray", ha="center",
            style="italic", transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t3, bp, cells = _load_data()

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.28,
                           height_ratios=[1.0, 1.0])

    ax_tl = fig.add_subplot(gs[0, 0])
    _topleft_per_cell_delta(ax_tl, cells)

    ax_tr = fig.add_subplot(gs[0, 1])
    _topright_t3_scalars(ax_tr, t3)

    ax_bl = fig.add_subplot(gs[1, 0])
    _bottomleft_bp_scaling(ax_bl, bp)

    ax_br = fig.add_subplot(gs[1, 1])
    _bottomright_correlations(ax_br, t3, cells)

    fig.suptitle(
        "P7 — H3 mechanism: T3 circuit properties × per-cell Δ\n"
        "(post-H1-FALSIFIED: which property predicts the INVERTED "
        "regime advantage?)",
        y=0.995, fontsize=12, weight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fp = OUT / f"fig_p7_mechanism.{ext}"
        fig.savefig(fp)
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
