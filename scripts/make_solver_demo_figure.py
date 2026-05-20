"""Render the P3.5 solver-comparison figure.

Reads `results/p3_solver_demo/{family}_{ode}/{seed_N/curves.npz,
seeds_summary.json}` produced by `scripts/run_solver_demo.py` and
emits `paper/figures/fig_p3_solver_demo.{png,pdf}` as a 2×3 panel
grid:

    +-----------+-----------+-----------+
    | chebyshev | te_fnn    | te_qnn    |    (curve overlays,
    +-----------+-----------+-----------+     seed 0,
    | qcpinn    | MAE bars  | residual  |     both ODEs vs exact)
    +-----------+-----------+-----------+

The MAE bar chart shows mean ± 95% t-CI across seeds {0,1,2} for all
4 families × 2 ODEs (8 bars grouped by family). The residual panel
overlays |u_pred − exact| over t for all 4 families on the logistic
ODE (the discriminating one — the chebyshev tower saturates near the
sigmoid plateau).

Standalone — does NOT plug into `scripts/make_paper_figures.py` or
the paper-integrity contract; the demo is exploratory, not a paper
claim (per the plan).
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
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
IN = REPO_ROOT / "results" / "p3_solver_demo"
OUT = REPO_ROOT / "paper" / "figures"

FAMILIES = ["chebyshev_dqc", "te_qpinn_fnn", "te_qpinn_qnn", "qcpinn"]
ODES = ["expdecay", "logistic"]

# Wong palette assignment, matching solver_demo.FAMILY_COLORS.
FAMILY_COLOR = {
    "chebyshev_dqc": "#0072B2",     # cool blue
    "te_qpinn_fnn":  "#D55E00",     # vermilion
    "te_qpinn_qnn":  "#CC79A7",     # pink
    "qcpinn":        "#009E73",     # cool green
}
ODE_STYLE = {"expdecay": "-", "logistic": "--"}


def _load_curves(family: str, ode: str, seed: int = 0) -> dict:
    p = IN / f"{family}_{ode}" / f"seed_{seed}" / "curves.npz"
    d = np.load(p)
    return {"t": d["t_eval"], "u_pred": d["u_pred"], "exact": d["exact"]}


def _load_summary(family: str, ode: str) -> dict:
    p = IN / f"{family}_{ode}" / "seeds_summary.json"
    return json.loads(p.read_text())


def _load_config() -> dict:
    return json.loads((IN / "config.json").read_text())


def _panel_family_curves(ax, family: str) -> None:
    """4-panel overlay: predicted (seed 0) vs exact for both ODEs."""
    ax.set_title(family, fontsize=11)
    color = FAMILY_COLOR[family]
    for ode in ODES:
        c = _load_curves(family, ode, seed=0)
        s = _load_summary(family, ode)
        mae = s["metrics"]["mae"]["mean"]
        # exact = thin grey ground truth
        ax.plot(c["t"], c["exact"], color="#555555", lw=1.0,
                ls=ODE_STYLE[ode], alpha=0.7,
                label=f"exact ({ode})" if family == FAMILIES[0] else None)
        ax.plot(c["t"], c["u_pred"], color=color, lw=1.6,
                ls=ODE_STYLE[ode], alpha=0.9,
                label=f"{ode} (MAE={mae:.3f})")
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9, fontsize=7)


def _panel_mae_bars(ax) -> None:
    """Mean ± 95% t-CI MAE bars, 4 families × 2 ODEs grouped by family."""
    width = 0.36
    x = np.arange(len(FAMILIES))
    for i, ode in enumerate(ODES):
        means, errs = [], []
        for fam in FAMILIES:
            s = _load_summary(fam, ode)["metrics"]["mae"]
            means.append(s["mean"])
            errs.append(s["ci95_half_width"])
        offset = (i - 0.5) * width
        # Slightly desaturate by ODE: solid for expdecay, hatched for logistic
        bars = ax.bar(x + offset, means, width, yerr=errs, capsize=3,
                      label=ode,
                      edgecolor="black", linewidth=0.5,
                      color=[FAMILY_COLOR[f] for f in FAMILIES],
                      alpha=(0.95 if ode == "expdecay" else 0.55))
        if ode == "logistic":
            for b in bars:
                b.set_hatch("//")
    ax.set_xticks(x)
    # Disclose param counts in the tick label (R1 mitigation per the plan)
    labels = []
    for fam in FAMILIES:
        s = _load_summary(fam, ODES[0])
        p = s["pqc_params"]
        c = s["classical_params"]
        if c > 0:
            labels.append(f"{fam}\n({p} pqc + {c} cls)")
        else:
            labels.append(f"{fam}\n({p} pqc)")
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("interior MAE vs exact  (mean ± 95% t-CI, n=3)")
    ax.set_title("Cross-family accuracy on both ODEs")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)


def _panel_residual_logistic(ax) -> None:
    """|u_pred − exact|(t) for the logistic ODE — the discriminating
    test. The chebyshev tower's saturation near sigmoid plateaus
    should be visible as an elevated residual at large t."""
    ax.set_title("Residual |û − u| on logistic (seed 0)")
    for fam in FAMILIES:
        c = _load_curves(fam, "logistic", seed=0)
        ax.semilogy(c["t"], np.abs(c["u_pred"] - c["exact"]),
                    color=FAMILY_COLOR[fam], lw=1.4, label=fam)
    ax.set_xlabel("t")
    ax.set_ylabel("|u_pred − u_exact|")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=7, framealpha=0.9)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    # Row 1: 3 curve overlays
    _panel_family_curves(axes[0, 0], "chebyshev_dqc")
    _panel_family_curves(axes[0, 1], "te_qpinn_fnn")
    _panel_family_curves(axes[0, 2], "te_qpinn_qnn")
    # Row 2: 4th family overlay + MAE bars + residual
    _panel_family_curves(axes[1, 0], "qcpinn")
    _panel_mae_bars(axes[1, 1])
    _panel_residual_logistic(axes[1, 2])

    fig.suptitle(
        "P3.5 — 4 SOTA solver families on two analytic ODEs "
        f"(seeds {cfg['seeds']}; per-family natural defaults — disclosed)",
        fontsize=12, y=1.00)
    fig.text(
        0.5, -0.02,
        "Demo artifacts; numbers are seed-dependent CPU JAX, NOT pinned "
        "by scripts/verify_paper_integrity.py. Param matching is "
        "intentionally OFF (P5 territory); per-family counts shown.",
        ha="center", fontsize=7, style="italic", color="#555555")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = OUT / f"fig_p3_solver_demo.{ext}"
        fig.savefig(path)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
