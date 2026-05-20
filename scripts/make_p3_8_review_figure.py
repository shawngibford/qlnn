"""Render the P3.8 peer-review iteration figure.

Reads `results/p3_8_review/{pde_or_system}_{model}/` (per-seed
field.npz + per-(thing, model) seeds_summary.json) produced by
`scripts/run_p3_8_review_iteration.py` and emits
`paper/figures/fig_p3_8_review_iteration.{png,pdf}` as a 4-row
diagnostic figure designed to address the peer-review audit:

  Row 0: Cross-PDE quantum-vs-classical-PINN relative-L2 bar chart.
         3 PDEs × 2 models grouped by PDE, log-scale, predict-zero
         floor marked. This is the headline of the audit: classical
         PINN at matched capacity vs the chebyshev_dqc_2d quantum
         solver. If classical wins broadly, the "quantum advantage"
         framing on the SOLVER side is invalidated.

  Row 1: Per-PDE loss-trajectory comparison (quantum vs classical
         at seed 0). 3 panels (heat / burgers / allen_cahn) showing
         loss-vs-step. Diagnoses whether AC's high final_loss in
         P3.7 was under-convergence or genuine saturation.

  Row 2: BC-violation bar chart per (PDE, model). Quantifies the
         audit's prediction that Lagaris IC does NOT implicitly
         enforce periodicity. The smoke already showed ~40% BC
         violation for the quantum heat solver.

  Row 3: Lorenz extended (T=5, ~5.5 Lyapunov times) per-family
         relL2 bars with the predict-mean baseline overlay (the
         honest chaotic baseline) and the predict-zero=1.0 line.
         Replaces P3.6's misleading "all families fail" framing.

Standalone — does NOT plug into make_paper_figures.py or the paper
integrity contract.
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
IN = REPO_ROOT / "results" / "p3_8_review"
OUT = REPO_ROOT / "paper" / "figures"

PDES = ["heat", "burgers_smooth", "allen_cahn"]
MODELS = ["chebyshev_dqc_2d", "classical_pinn"]
LORENZ_FAMILIES = [
    "chebyshev_dqc", "te_qpinn_fnn", "te_qpinn_qnn", "qcpinn",
]

MODEL_COLOR = {
    "chebyshev_dqc_2d": "#0072B2",   # quantum (cool blue)
    "classical_pinn":   "#F0E442",   # classical (Wong yellow)
}
FAMILY_COLOR = {
    "chebyshev_dqc": "#0072B2",
    "te_qpinn_fnn":  "#D55E00",
    "te_qpinn_qnn":  "#CC79A7",
    "qcpinn":        "#009E73",
}

PDE_PRETTY = {
    "heat":           "Heat (smooth, analytic ref)",
    "burgers_smooth": "Burgers smooth",
    "allen_cahn":     "Allen-Cahn (audit re-run: 64×32 colloc, 1800 steps)",
}


def _try_load_summary(pde: str, model: str) -> dict | None:
    p = IN / f"{pde}_{model}" / "seeds_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _try_load_lorenz_summary(family: str) -> dict | None:
    p = IN / f"lorenz_{family}" / "seeds_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _try_load_field(pde: str, model: str, seed: int = 0) -> dict | None:
    p = IN / f"{pde}_{model}" / f"seed_{seed}" / "field.npz"
    if not p.exists():
        return None
    return dict(np.load(p))


# ---------------------------------------------------------------------------
# Row 0: cross-PDE quantum-vs-classical relL2 bar chart
# ---------------------------------------------------------------------------


def _panel_pde_bars(ax) -> None:
    width = 0.36
    x = np.arange(len(PDES))
    drew_any = False
    for i, model in enumerate(MODELS):
        means, errs = [], []
        for pde in PDES:
            s = _try_load_summary(pde, model)
            if s is None:
                means.append(np.nan); errs.append(0.0); continue
            means.append(max(s["metrics"]["relative_l2"]["mean"], 1e-4))
            errs.append(s["metrics"]["relative_l2"]["ci95_half_width"])
            drew_any = True
        offset = (i - 0.5) * width
        valid = ~np.isnan(means)
        if np.any(valid):
            xs = x[valid] + offset
            ms = np.array(means)[valid]
            es = np.array(errs)[valid]
            ax.bar(xs, ms, width, yerr=es, capsize=3,
                    label=model, color=MODEL_COLOR[model],
                    edgecolor="black", linewidth=0.5,
                    alpha=0.85 if model == "chebyshev_dqc_2d" else 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([PDE_PRETTY[p] for p in PDES], fontsize=8)
    ax.set_ylabel("relative-L2 vs reference  (mean ± 95% t-CI, n=3)")
    ax.set_title("PDE solver: quantum vs classical PINN at matched capacity",
                  fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8,
                alpha=0.7, label="rel-L2 = 1 (predict-zero floor)")
    if drew_any:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "P3.8 sweep results not yet available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#888888")


# ---------------------------------------------------------------------------
# Row 1: per-PDE loss trajectory (quantum vs classical, seed 0)
# ---------------------------------------------------------------------------


def _panel_loss_trajectory(ax, pde: str) -> None:
    ax.set_title(f"loss trajectory — {pde}", fontsize=10)
    drew = False
    for model in MODELS:
        f = _try_load_field(pde, model, seed=0)
        if f is None:
            continue
        loss = np.asarray(f["loss_history"])
        steps = np.arange(len(loss))
        ax.semilogy(steps, loss, color=MODEL_COLOR[model], lw=1.2,
                     label=model)
        drew = True
    ax.set_xlabel("step")
    ax.set_ylabel("residual loss")
    ax.grid(True, alpha=0.3, which="both")
    if drew:
        ax.legend(loc="best", fontsize=7, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#aaaaaa")


# ---------------------------------------------------------------------------
# Row 2: BC violation bars
# ---------------------------------------------------------------------------


def _panel_bc_violation(ax) -> None:
    """max_t |u(t, x_lo+ε) − u(t, x_hi−ε)| / max|u| per (PDE, model)."""
    width = 0.36
    x = np.arange(len(PDES))
    drew = False
    for i, model in enumerate(MODELS):
        vals, errs = [], []
        for pde in PDES:
            s = _try_load_summary(pde, model)
            if s is None or "bc_violation" not in s["metrics"]:
                vals.append(np.nan); errs.append(0.0); continue
            vals.append(s["metrics"]["bc_violation"]["mean"])
            errs.append(s["metrics"]["bc_violation"]["ci95_half_width"])
            drew = True
        offset = (i - 0.5) * width
        valid = ~np.isnan(vals)
        if np.any(valid):
            ax.bar(x[valid] + offset, np.array(vals)[valid], width,
                    yerr=np.array(errs)[valid], capsize=3,
                    label=model, color=MODEL_COLOR[model],
                    edgecolor="black", linewidth=0.5,
                    alpha=0.85 if model == "chebyshev_dqc_2d" else 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([PDE_PRETTY[p] for p in PDES], fontsize=8)
    ax.set_ylabel("BC violation  (max_t |Δu| / max|u|)")
    ax.set_title("Periodic-BC implicit-enforcement check "
                  "(audit prediction: Lagaris IC does NOT enforce periodicity)",
                  fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0.05, color="#888888", linestyle="--", linewidth=0.8,
                alpha=0.7, label="5% threshold")
    if drew:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)


# ---------------------------------------------------------------------------
# Row 3: Lorenz extended bars with predict-mean baseline
# ---------------------------------------------------------------------------


def _panel_lorenz_extended(ax) -> None:
    width = 0.6
    x = np.arange(len(LORENZ_FAMILIES))
    means, errs, baselines = [], [], []
    drew = False
    for fam in LORENZ_FAMILIES:
        s = _try_load_lorenz_summary(fam)
        if s is None:
            means.append(np.nan); errs.append(0.0); baselines.append(np.nan)
            continue
        means.append(max(s["metrics"]["relative_l2"]["mean"], 1e-4))
        errs.append(s["metrics"]["relative_l2"]["ci95_half_width"])
        baselines.append(
            s["metrics"]["relative_l2_predict_mean_baseline"]["mean"])
        drew = True
    valid = ~np.isnan(means)
    if np.any(valid):
        ax.bar(x[valid], np.array(means)[valid], width,
                yerr=np.array(errs)[valid], capsize=3,
                color=[FAMILY_COLOR[LORENZ_FAMILIES[i]]
                       for i in range(len(LORENZ_FAMILIES)) if valid[i]],
                edgecolor="black", linewidth=0.5, alpha=0.85,
                label="rel-L2 (predicted)")
        # Predict-mean baseline overlay (one horizontal segment per family)
        for i in range(len(LORENZ_FAMILIES)):
            if valid[i] and not np.isnan(baselines[i]):
                ax.hlines(baselines[i], x[i] - width/2, x[i] + width/2,
                          color="#000000", linewidth=1.5,
                          label="predict-mean baseline" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(LORENZ_FAMILIES, fontsize=8, rotation=20)
    ax.set_ylabel("relative-L2  (n=3 seeds, T=5 ≈ 5.5 Lyapunov times)")
    ax.set_title("Lorenz extended (P3.8): proper chaotic baselines",
                  fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8,
                alpha=0.7, label="rel-L2 = 1 (predict-zero floor)")
    if drew:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "P3.8 Lorenz extended results not yet available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#888888")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = IN / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    fig = plt.figure(figsize=(13.5, 14.0))
    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        height_ratios=[1.0, 0.9, 0.9, 1.0],
        hspace=0.55, wspace=0.35)

    # Row 0: cross-PDE bar (full width)
    ax0 = fig.add_subplot(gs[0, :])
    _panel_pde_bars(ax0)

    # Row 1: per-PDE loss trajectories (3 panels)
    for col, pde in enumerate(PDES):
        ax = fig.add_subplot(gs[1, col])
        _panel_loss_trajectory(ax, pde)

    # Row 2: BC violation bar (full width)
    ax2 = fig.add_subplot(gs[2, :])
    _panel_bc_violation(ax2)

    # Row 3: Lorenz extended bars (full width)
    ax3 = fig.add_subplot(gs[3, :])
    _panel_lorenz_extended(ax3)

    fig.suptitle(
        "P3.8 — Peer-review iteration: classical PINN baseline + "
        "audit-corrected re-runs",
        fontsize=12, y=0.998)
    fig.text(
        0.5, -0.005,
        "Row 0: quantum vs classical PINN at matched capacity on 3 PDEs. "
        "Row 1: loss trajectories (seed 0). "
        "Row 2: BC-violation (audit predicted Lagaris IC does NOT enforce periodicity). "
        "Row 3: Lorenz at T=5 (~5.5 LTE; P3.6 used T=2) with the more honest predict-mean baseline. "
        "Demo artifacts; H1 verdict still requires P5's Neural-ODE.",
        ha="center", fontsize=7.5, style="italic", color="#555555")

    for ext in ("png", "pdf"):
        path = OUT / f"fig_p3_8_review_iteration.{ext}"
        fig.savefig(path)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
