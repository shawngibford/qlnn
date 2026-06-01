"""Render the training-budget sensitivity figure.

Two complementary panels addressing the question "why 2000 steps and
not 500 / 1000 / 4000?":

  Left  — HPO direct probe: for each anchor cell × LR, plot the
          1500-step relL² vs the 3000-step relL² on the classical
          PINN side. Diminishing-returns curve makes the
          ``additional steps beyond 1500 buy little'' argument
          visible.

  Right — Loss-history-derived budget sweep: for each (family, system)
          pair on the P3.6 ODE matrix, plot training-loss at
          checkpoint budgets {500, 1000, 1500, 2000} divided by the
          asymptotic seed-mean loss. Values ≤ 1.0 indicate the
          configuration has substantially converged by step N.

Reads:
  results/p7_5_hpo_sensitivity/{anchor}/cell_results.json
  results/p3_6_multi_state/{family}_{system}/seed_N/curves.npz

Emits paper/figures/fig_training_budget_sensitivity.{png,pdf}.

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
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper" / "figures"

HPO_ANCHORS = [
    ("lotka_volterra_seed_2", "Lotka-Volterra s2 [S]",  "#0072B2", "o"),
    ("van_der_pol_seed_1",    "Van der Pol s1 [S]",     "#D55E00", "s"),
    ("lorenz_seed_2",          "Lorenz s2 [B]",           "#009E73", "^"),
]
CHECKPOINTS = (500, 1000, 1500, 2000)
P36_SYSTEMS = ("lotka_volterra", "van_der_pol",
               "fitzhugh_nagumo", "lorenz")
P36_FAMILIES = ("chebyshev_dqc", "te_qpinn_fnn",
                "te_qpinn_qnn", "qcpinn")
SEEDS = (0, 1, 2)


def _load_p36_loss(family: str, system: str, seed: int) -> np.ndarray | None:
    p = (REPO_ROOT / "results" / "p3_6_multi_state" /
         f"{family}_{system}" / f"seed_{seed}" / "curves.npz")
    if not p.exists():
        return None
    return np.load(p)["loss_history"]


def main() -> None:
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(13.0, 5.2),
        constrained_layout=True,
        gridspec_kw=dict(width_ratios=(1.0, 1.05)))

    # ===== Left: HPO 1500 vs 3000 cPINN relL² ===========================
    for anchor, label, color, marker in HPO_ANCHORS:
        with (REPO_ROOT / "results" / "p7_5_hpo_sensitivity" /
              anchor / "cell_results.json").open() as f:
            cell = json.load(f)
        for lr in sorted({float(r["lr"]) for r in cell["runs"]}):
            run_1500 = next(
                (r for r in cell["runs"]
                 if r["train_steps"] == 1500 and float(r["lr"]) == lr),
                None)
            run_3000 = next(
                (r for r in cell["runs"]
                 if r["train_steps"] == 3000 and float(r["lr"]) == lr),
                None)
            if run_1500 is None or run_3000 is None:
                continue
            x = float(run_1500["cpinn_relL2"])
            y = float(run_3000["cpinn_relL2"])
            ax_l.plot([x, x], [x, y], color=color, lw=0.8, alpha=0.4,
                       zorder=2)
            ax_l.scatter(x, y, s=70, color=color, marker=marker,
                          edgecolor="black", lw=0.5, zorder=3,
                          label=(label if lr == 0.001 else None))
    # y = x diagonal: "more steps did nothing"
    lo, hi = ax_l.get_xlim()[0], ax_l.get_xlim()[1]
    lo = min(lo, ax_l.get_ylim()[0])
    hi = max(hi, ax_l.get_ylim()[1])
    ax_l.plot([lo, hi], [lo, hi], color="black", lw=0.7, ls="--",
               alpha=0.5, label="$y = x$  (no improvement)")
    ax_l.set_xlabel("classical PINN  relL²  at 1500 steps")
    ax_l.set_ylabel("classical PINN  relL²  at 3000 steps")
    ax_l.set_title(
        "Cross-step diminishing returns on the cPINN baseline\n"
        "(per anchor cell × 3 LRs; points below diagonal = more "
        "steps helped)",
        fontsize=10)
    ax_l.legend(loc="upper left", fontsize=8, frameon=True)
    ax_l.grid(alpha=0.25, lw=0.5)
    ax_l.set_xlim(lo, hi)
    ax_l.set_ylim(lo, hi)

    # ===== Right: loss-at-step-N normalized to asymptotic ==============
    # For each (family, system), compute mean-across-seeds loss at each
    # checkpoint, normalize by the asymptotic seed-mean loss.
    family_means = {fam: [] for fam in P36_FAMILIES}
    for system in P36_SYSTEMS:
        for family in P36_FAMILIES:
            seed_curves = [_load_p36_loss(family, system, s)
                            for s in SEEDS]
            seed_curves = [c for c in seed_curves if c is not None]
            if not seed_curves:
                continue
            n_min = min(len(c) for c in seed_curves)
            stacked = np.stack([c[:n_min] for c in seed_curves])
            seed_mean = stacked.mean(axis=0)
            asymptote = float(seed_mean[-50:].mean())
            for cp in CHECKPOINTS:
                if cp >= len(seed_mean):
                    val = float(seed_mean[-1])
                else:
                    val = float(seed_mean[cp])
                rel = val / max(asymptote, 1e-12)
                family_means[family].append((cp, rel))
    # Average across the 4 systems
    fam_colors = {
        "chebyshev_dqc": "#0072B2",
        "te_qpinn_fnn":  "#D55E00",
        "te_qpinn_qnn":  "#CC79A7",
        "qcpinn":        "#009E73",
    }
    for family in P36_FAMILIES:
        pts = family_means[family]
        if not pts:
            continue
        by_cp: dict[int, list[float]] = {}
        for cp, rel in pts:
            by_cp.setdefault(cp, []).append(rel)
        cps = sorted(by_cp)
        means = [np.mean(by_cp[cp]) for cp in cps]
        ax_r.plot(cps, means, "o-", color=fam_colors[family],
                   lw=1.8, markersize=9, markeredgecolor="black",
                   markeredgewidth=0.5, label=family)
        for cp, m in zip(cps, means):
            ax_r.annotate(f"{m:.1f}", (cp, m),
                           xytext=(0, 9), textcoords="offset points",
                           ha="center", va="bottom", fontsize=8,
                           color=fam_colors[family])
    ax_r.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6,
                  label="asymptotic loss (1.0)")
    ax_r.set_xlabel("training step budget")
    ax_r.set_ylabel(r"loss($N$) / asymptotic loss   "
                     r"(seed × system mean)")
    ax_r.set_yscale("log")
    ax_r.set_xticks(list(CHECKPOINTS))
    ax_r.set_title(
        "Convergence ratio across the 4 ODE solver families\n"
        "(values approaching 1.0 = budget is enough)",
        fontsize=10)
    ax_r.legend(loc="upper right", fontsize=8.5)
    ax_r.grid(alpha=0.25, lw=0.5, which="both")

    fig.suptitle(
        "Training-budget sensitivity:  is 2000 steps the right choice?",
        y=1.05, fontsize=11.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_training_budget_sensitivity.png")
    fig.savefig(OUT_DIR / "fig_training_budget_sensitivity.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_training_budget_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
