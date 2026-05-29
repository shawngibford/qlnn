"""Render the A17 qcpinn quantum-parameter step-wise sweep figure.

A17 introduces 3 additional qcpinn variants along the Q/(Q+C)
parameter ratio: balanced (≈24%), quantum (≈45%), full_q (≈87%).
With the original qcpinn (≈2%) this gives a 4-point sweep that
addresses Reviewer 2's "is the quantum substrate actually contributing
anything in qcpinn?" concern.

Reads SMOKE numbers from results/smoke_post_audit/smoke_post_audit_runtimes.json
as a placeholder; full PRODUCTION numbers will land after Phase C
(Anvil) re-runs and this script will refresh against the same file
locations. The figure clearly labels itself as "smoke-budget, 50 steps"
so it cannot be misread as a verdict.

Emits paper/figures/fig_qcpinn_variants.{png,pdf}.

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
OUT_DIR = REPO_ROOT / "paper" / "figures"

# Wall-clock + final-loss after a SMOKE-budget run on kuramoto seed 0.
SMOKE_PATH = REPO_ROOT / "results" / "smoke_post_audit" / \
             "smoke_post_audit_runtimes.json"


def main() -> None:
    with SMOKE_PATH.open() as f:
        smoke = json.load(f)
    cfg = smoke["config"]
    rows = {r["family"]: r for r in smoke["records"]}

    variants = ["qcpinn_balanced", "qcpinn_quantum", "qcpinn_full_q"]
    ratios = []
    losses = []
    wall_h = []
    pqcs = []
    for fam in variants:
        r = rows[fam]
        pqc = float(r["pqc_params"])
        cl  = float(r["classical_params"])
        ratios.append(100.0 * pqc / (pqc + cl))
        losses.append(float(r["loss_step_final"]))
        wall_h.append(float(r["est_full_cell_hours"]))
        pqcs.append(int(pqc))

    cpinn_loss = float(rows["classical_pinn"]["loss_step_final"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8),
                                    gridspec_kw={"width_ratios": (1.1, 1.0)})

    # Left: loss vs Q-ratio
    ax1.plot(ratios, losses, "o-", color="#0072B2", lw=1.8,
             markersize=10, markeredgecolor="black", markeredgewidth=0.6)
    for r, l, pqc in zip(ratios, losses, pqcs):
        ax1.annotate(f"PQC={pqc}", (r, l), textcoords="offset points",
                     xytext=(8, 8), fontsize=8, color="#444444")
    ax1.axhline(cpinn_loss, color="#D55E00", lw=1.2, ls="--",
                label=f"classical PINN  ({cpinn_loss:.4f})")
    ax1.set_xlabel("Q-parameter ratio  PQC / (PQC + classical)  [%]")
    ax1.set_ylabel(f"loss after {cfg['smoke_steps']} steps  (smoke)")
    ax1.set_yscale("log")
    ax1.set_title("A17 step-wise qcpinn ratio sweep "
                  f"({cfg['system']}, seed {cfg['seed']})", fontsize=10.5)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.25, lw=0.5)

    # Right: wall-clock vs Q-ratio
    bars = ax2.bar(np.arange(len(variants)), wall_h,
                   color=["#9FC5E8", "#6FA8DC", "#3D85C6"],
                   edgecolor="black", lw=0.6)
    for x, v, fam in zip(np.arange(len(variants)), wall_h, variants):
        ax2.text(x, v + 0.02, f"{v:.2f}h", ha="center", va="bottom",
                 fontsize=8.5)
    ax2.set_xticks(np.arange(len(variants)))
    ax2.set_xticklabels([v.replace("qcpinn_", "") for v in variants],
                        rotation=0, fontsize=9)
    ax2.set_ylabel("est. wall-clock per cell  (hr, CPU)")
    ax2.set_title("Cost of adding quantum capacity", fontsize=10.5)
    ax2.grid(axis="y", alpha=0.25, lw=0.5)

    fig.suptitle(
        "A17: quantum-parameter step-wise sweep — does adding PQC help?\n"
        f"(SMOKE-budget {cfg['smoke_steps']} steps; production = "
        f"{cfg['prod_steps']} steps via Phase C on Anvil)",
        y=1.04, fontsize=10.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_qcpinn_variants.png")
    fig.savefig(OUT_DIR / "fig_qcpinn_variants.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_qcpinn_variants.pdf'}")


if __name__ == "__main__":
    main()
