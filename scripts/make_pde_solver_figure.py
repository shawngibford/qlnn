"""Render the P3.7 PDE solver comparison figure.

Reads `results/p3_7_pde_solver/{pde}/seed_N/field.npz` +
`seeds_summary.json` produced by `scripts/run_pde_solver_demo.py`
and emits `paper/figures/fig_p3_7_pde_solver.{png,pdf}` as a 4-row
figure:

    Row 0 (heat):           t=0  |  t=T/2  |  t=T   snapshots
    Row 1 (burgers_smooth): t=0  |  t=T/2  |  t=T   snapshots
    Row 2 (allen_cahn):     t=0  |  t=T/2  |  t=T   snapshots
    Row 3:  rel-L2 bar chart (3 PDEs × mean ± 95% t-CI, n=3 seeds)

Each snapshot panel overlays the predicted u(t, ·) (color) against
the reference field (grey). The bar chart uses log-scale (the H1
prediction is that broadband PDEs like Allen-Cahn fail by orders of
magnitude on physics-residual-only training, just like Lorenz did
in P3.6).

Standalone — does NOT plug into `make_paper_figures.py` or the
paper-integrity contract (P3.7 is exploratory).
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
IN = REPO_ROOT / "results" / "p3_7_pde_solver"
OUT = REPO_ROOT / "paper" / "figures"

PDES = ["heat", "burgers_smooth", "allen_cahn"]
PDE_COLOR = {
    "heat":           "#0072B2",     # cool blue
    "burgers_smooth": "#D55E00",     # vermilion
    "allen_cahn":     "#009E73",     # cool green
}
PDE_PRETTY = {
    "heat":           "Heat (analytic ref, SMOOTH)",
    "burgers_smooth": "Burgers smooth (npz ref, SMOOTH/PERIODIC)",
    "allen_cahn":     "Allen–Cahn (npz ref, BROADBAND/MULTISCALE)",
}


def _load_field(pde: str, seed: int = 0) -> dict:
    p = IN / pde / f"seed_{seed}" / "field.npz"
    d = np.load(p)
    return {"t": d["t_eval"], "x": d["x_eval"],
            "u_pred": d["u_pred"], "u_ref": d["u_ref"]}


def _load_summary(pde: str) -> dict:
    p = IN / pde / "seeds_summary.json"
    return json.loads(p.read_text())


def _snapshot_panel(ax, pde: str, t_index: int, t_label: str) -> None:
    """Overlay u_pred(t, ·) (color) on u_ref (grey) at a fixed time."""
    field = _load_field(pde, seed=0)
    x = field["x"]
    ax.plot(x, field["u_ref"][t_index, :], color="#555555", lw=1.0,
            alpha=0.8, label="reference")
    ax.plot(x, field["u_pred"][t_index, :], color=PDE_COLOR[pde],
            lw=1.4, alpha=0.95, label="predicted (seed 0)")
    ax.set_title(f"{pde}  |  {t_label}", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, framealpha=0.9)


def _bar_panel(ax) -> None:
    """relative-L2 across PDEs (mean ± 95% t-CI, n=3 seeds)."""
    means, errs, colors = [], [], []
    for pde in PDES:
        s = _load_summary(pde)["metrics"]["relative_l2"]
        means.append(max(s["mean"], 1e-4))
        errs.append(s["ci95_half_width"])
        colors.append(PDE_COLOR[pde])
    x = np.arange(len(PDES))
    ax.bar(x, means, yerr=errs, capsize=4, color=colors,
            edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    labels = [PDE_PRETTY[p] for p in PDES]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("relative-L2 vs reference  (mean ± 95% t-CI, n=3)")
    ax.set_yscale("log")
    ax.set_title("Cross-PDE solver accuracy (chebyshev_dqc_2d, 8 qubits, 5 layers)",
                  fontsize=10)
    ax.grid(True, alpha=0.3, which="both", axis="y")
    # Reference line at relative-L2 = 1.0 (the trivial-zero-prediction
    # accuracy floor — anything above this is WORSE than predicting 0).
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8,
                alpha=0.7, label="rel-L2 = 1 (predict-zero floor)")
    ax.legend(loc="best", fontsize=7, framealpha=0.9)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((IN / "config.json").read_text())

    fig = plt.figure(figsize=(13.5, 12.5))
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           height_ratios=[1.0, 1.0, 1.0, 0.95],
                           hspace=0.42, wspace=0.28)

    # Rows 0-2: per-PDE snapshot triplets (t=0, t=T/2, t=T)
    for row, pde in enumerate(PDES):
        field = _load_field(pde, seed=0)
        n_t = field["u_pred"].shape[0]
        # 0-indexed interior eval grid; t=0 is the first interior step
        # (t≈t0+δ), t=T/2 ≈ index n_t//2, t=T ≈ last index.
        for col, (idx, label) in enumerate([
                (0, f"t ≈ {float(field['t'][0]):.2f}"),
                (n_t // 2, f"t ≈ {float(field['t'][n_t // 2]):.2f}"),
                (n_t - 1, f"t ≈ {float(field['t'][-1]):.2f}")]):
            ax = fig.add_subplot(gs[row, col])
            _snapshot_panel(ax, pde, idx, label)

    # Row 3: bar chart spanning all 3 columns
    ax_bar = fig.add_subplot(gs[3, :])
    _bar_panel(ax_bar)

    fig.suptitle(
        f"P3.7 — Chebyshev-DQC 2D solver on 3 PDEs "
        f"(seeds {cfg['seeds']}; nested jacrev∘jacrev autodiff "
        "validated through PennyLane QNode)",
        fontsize=12, y=0.995)
    fig.text(
        0.5, 0.002,
        "Each row = one PDE; columns = predicted u(t, x) snapshots at "
        "t≈0 / T/2 / T overlaid on reference. "
        "Same model (8 qubits, 5 HEA layers); per-PDE step budget varied. "
        "Demo artifacts; NOT in verify_paper_integrity contract.",
        ha="center", fontsize=7.5, style="italic", color="#555555")

    for ext in ("png", "pdf"):
        path = OUT / f"fig_p3_7_pde_solver.{ext}"
        fig.savefig(path)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
