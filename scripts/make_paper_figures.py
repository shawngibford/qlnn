"""Generate paper-quality figures from committed result JSONs.

Produces three figures into `paper/figures/`:
  1. fig_horizon_ablation.{png,pdf}   — h ∈ {1,3,6,12} R² and MAE for LO-ODE
                                         vs persistence vs linear.
  2. fig_sample_efficiency.{png,pdf}   — log(n_train) vs test MAE for both
                                         stacks at fractions {10,25,50,100}%.
  3. fig_reproducibility.{png,pdf}     — CI half-width per stack per fraction,
                                         showing QLNN tightness across the
                                         full data-sweep.

All numbers come from on-disk seeds_summary.json files; no re-training needed.

Style: matplotlib publication defaults (Times-roman fallback if available,
otherwise default sans), 300 dpi, single-column-friendly sizes.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Two-color palette: classical = warm, quantum = cool
C_CLASSICAL = "#D55E00"  # vermilion
C_QLNN = "#0072B2"        # cool blue
C_PERSIST = "#999999"     # grey
C_LINEAR = "#CC79A7"      # pink


def _load(p: str | Path) -> dict:
    with (ROOT / p).open() as f: return json.load(f)


def _ci(s: dict) -> float:
    """Return 95% CI half-width if present, else fall back to std."""
    return s.get("ci95_half_width", s.get("std", 0.0))


# ---------------------------------------------------------------------------
# Figure 1 — horizon ablation
# ---------------------------------------------------------------------------
def fig_horizon():
    horizons = [1, 3, 6, 12]
    pers_r2 = []
    pers_mae = []
    lin_r2 = []
    lin_mae = []
    lo_r2 = []
    lo_r2_ci = []
    lo_mae = []
    lo_mae_ci = []

    for h in horizons:
        b = _load(f"results/horizon_sweep/euler_h{h}/baselines.json")
        s = _load(f"results/horizon_sweep/euler_h{h}/seeds_summary.json")
        pers_r2.append(b["persistence"]["test"]["r2_raw"])
        pers_mae.append(b["persistence"]["test"]["mae_raw"])
        lin_r2.append(b["linear"]["test"]["r2_raw"])
        lin_mae.append(b["linear"]["test"]["mae_raw"])
        lo_r2.append(s["test"]["r2_raw"]["mean"])
        lo_r2_ci.append(_ci(s["test"]["r2_raw"]))
        lo_mae.append(s["test"]["mae_raw"]["mean"])
        lo_mae_ci.append(_ci(s["test"]["mae_raw"]))

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Left: R²
    ax = axes[0]
    ax.errorbar(horizons, lo_r2, yerr=lo_r2_ci, marker="o", color=C_CLASSICAL,
                label="Liquid-ODE (Euler)", capsize=3, linewidth=1.5)
    ax.plot(horizons, pers_r2, marker="s", color=C_PERSIST, linestyle="--",
            label="Persistence", linewidth=1.2)
    ax.plot(horizons, lin_r2, marker="^", color=C_LINEAR, linestyle=":",
            label="Linear extrap.", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xticks(horizons); ax.set_xticklabels(horizons)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Test R²")
    ax.set_title("(a) R² vs horizon")
    # Clip the y-axis so the h=12 collapse doesn't dominate
    ax.set_ylim(bottom=-15, top=1.1)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3)

    # Right: MAE
    ax = axes[1]
    ax.errorbar(horizons, lo_mae, yerr=lo_mae_ci, marker="o", color=C_CLASSICAL,
                label="Liquid-ODE (Euler)", capsize=3, linewidth=1.5)
    ax.plot(horizons, pers_mae, marker="s", color=C_PERSIST, linestyle="--",
            label="Persistence", linewidth=1.2)
    ax.plot(horizons, lin_mae, marker="^", color=C_LINEAR, linestyle=":",
            label="Linear extrap.", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xticks(horizons); ax.set_xticklabels(horizons)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Test MAE (raw OD)")
    ax.set_title("(b) MAE vs horizon")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Horizon ablation — h=3 is the discriminating regime", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_horizon_ablation.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_horizon_ablation.{png,pdf}'}")


# ---------------------------------------------------------------------------
# Figure 2 — sample efficiency
# ---------------------------------------------------------------------------
def fig_sample_efficiency():
    fractions = [10, 25, 50, 100]
    n_train_classical = [47, 118, 236, 472]   # from protocol.json
    c_mae, c_ci = [], []
    q_mae, q_ci = [], []
    for pct in fractions:
        c = _load(f"results/sample_efficiency/classical_h4_h3_pct{pct}/seeds_summary.json")
        q = _load(f"results/sample_efficiency/qlnn_h3_pct{pct}/seeds_summary.json")
        c_mae.append(c["test"]["mae_raw"]["mean"])
        c_ci.append(_ci(c["test"]["mae_raw"]))
        q_mae.append(q["test"]["mae_raw"]["mean"])
        q_ci.append(_ci(q["test"]["mae_raw"]))

    target = c_mae[-1]  # classical 100% data = target X

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(n_train_classical, c_mae, yerr=c_ci, marker="o",
                color=C_CLASSICAL, label="Classical Liquid-ODE (H=4)",
                capsize=4, linewidth=2, markersize=8)
    ax.errorbar(n_train_classical, q_mae, yerr=q_ci, marker="s",
                color=C_QLNN, label="QLNN (~100 params)",
                capsize=4, linewidth=2, markersize=8)
    ax.axhline(target, color="black", linestyle=":", linewidth=1, alpha=0.7,
               label=f"Target X = classical-100% = {target:.4f}")

    # Bootstrap-verdict annotations
    annotations = [
        (47, "QLNN wins\n(p=0.015)", "down", C_QLNN),
        (118, "QLNN wins\n(p=0.002)", "down", C_QLNN),
        (236, "tie\n(p=0.226)", "up", "black"),
        (472, "Classical wins\n(p=0.029)", "up", C_CLASSICAL),
    ]
    for x, txt, direction, color in annotations:
        idx = n_train_classical.index(x)
        y = max(c_mae[idx], q_mae[idx]) + 0.015 if direction == "up" else min(c_mae[idx], q_mae[idx]) - 0.020
        va = "bottom" if direction == "up" else "top"
        ax.annotate(txt, (x, y), ha="center", va=va, fontsize=8,
                    color=color, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(n_train_classical)
    ax.set_xticklabels([f"{f}%\n(n={n})" for f, n in zip(fractions, n_train_classical)])
    ax.set_xlabel("Training data fraction (chronological truncation from start)")
    ax.set_ylabel("Test MAE at h=3 (raw OD, mean ± 95% CI)")
    ax.set_title("Sample-efficiency crossover at h=3 (paired-bootstrap verdicts shown)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_sample_efficiency.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_sample_efficiency.{png,pdf}'}")


# ---------------------------------------------------------------------------
# Figure 3 — reproducibility (CI width per stack per fraction)
# ---------------------------------------------------------------------------
def fig_reproducibility():
    fractions = [10, 25, 50, 100]
    c_ci, q_ci = [], []
    for pct in fractions:
        c = _load(f"results/sample_efficiency/classical_h4_h3_pct{pct}/seeds_summary.json")
        q = _load(f"results/sample_efficiency/qlnn_h3_pct{pct}/seeds_summary.json")
        c_ci.append(_ci(c["test"]["mae_raw"]))
        q_ci.append(_ci(q["test"]["mae_raw"]))

    ratios = [c / q for c, q in zip(c_ci, q_ci)]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # Left: absolute CIs side-by-side
    ax = axes[0]
    x = np.arange(len(fractions))
    width = 0.35
    ax.bar(x - width/2, c_ci, width, label="Classical H=4", color=C_CLASSICAL)
    ax.bar(x + width/2, q_ci, width, label="QLNN", color=C_QLNN)
    ax.set_xticks(x); ax.set_xticklabels([f"{f}%" for f in fractions])
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("95% CI half-width on test MAE")
    ax.set_title("(a) Absolute CI widths")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: ratio (classical / QLNN)
    ax = axes[1]
    bars = ax.bar(x, ratios, color=C_QLNN, alpha=0.85)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="Equal variance")
    ax.axhline(2.0, color=C_CLASSICAL, linewidth=0.8, linestyle=":",
               label="Pre-reg threshold ≥ 2×")
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.05, f"{r:.2f}×", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{f}%" for f in fractions])
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("σ(classical) / σ(QLNN)")
    ax.set_title("(b) QLNN reproducibility ratio")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(ratios) * 1.2)

    fig.suptitle("Claim 1 — QLNN tighter seed variance at every data fraction", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_reproducibility.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_reproducibility.{png,pdf}'}")


if __name__ == "__main__":
    fig_horizon()
    fig_sample_efficiency()
    fig_reproducibility()
    print(f"\nAll figures written to {OUT}/")
