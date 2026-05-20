"""Render the P3.6 multi-state ODE solver comparison figure.

Reads `results/p3_6_multi_state/{family}_{system}/` (per-seed
`curves.npz` + `seeds_summary.json`) produced by
`scripts/run_multi_state_demo.py` and emits
`paper/figures/fig_p3_6_multi_state.{png,pdf}` as a 2×2 panel grid:

    +------------------+------------------+
    | Lotka-Volterra   | Van der Pol      |   (curve overlays, seed 0,
    +------------------+------------------+    first component of each
    | Lorenz           | rel-L2 bar chart |    family vs reference)
    +------------------+------------------+

The bar-chart shows mean ± 95% t-CI relative-L2 across seeds {0,1,2}
for all 4 families × 3 systems, grouped by family. The figure caption
discloses per-family param counts (cumulative across components for
multi-state cells).

Standalone — does NOT plug into `make_paper_figures.py` or the
paper-integrity contract (demo is exploratory, not a paper claim).
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
IN = REPO_ROOT / "results" / "p3_6_multi_state"
OUT = REPO_ROOT / "paper" / "figures"

FAMILIES = ["chebyshev_dqc", "te_qpinn_fnn", "te_qpinn_qnn", "qcpinn"]
SYSTEMS = ["lotka_volterra", "van_der_pol", "lorenz"]

# Wong palette — same as P3.5 for visual continuity.
FAMILY_COLOR = {
    "chebyshev_dqc": "#0072B2",
    "te_qpinn_fnn":  "#D55E00",
    "te_qpinn_qnn":  "#CC79A7",
    "qcpinn":        "#009E73",
}

SYSTEM_PRETTY = {
    "lotka_volterra": "Lotka–Volterra (d=2, SMOOTH/PERIODIC)",
    "van_der_pol":    "Van der Pol (d=2, stiff periodic)",
    "lorenz":         "Lorenz (d=3, BROADBAND/CHAOTIC)",
}


def _load_curves(family: str, system: str, seed: int = 0) -> dict:
    p = IN / f"{family}_{system}" / f"seed_{seed}" / "curves.npz"
    d = np.load(p)
    return {"t": d["t_eval"], "u_pred": d["u_pred"], "u_ref": d["u_ref"]}


def _load_summary(family: str, system: str) -> dict:
    p = IN / f"{family}_{system}" / "seeds_summary.json"
    return json.loads(p.read_text())


def _panel_system_overlay(ax, system: str) -> None:
    """First-component curve overlay for one system."""
    ax.set_title(SYSTEM_PRETTY[system], fontsize=10)
    ref_drawn = False
    for fam in FAMILIES:
        c = _load_curves(fam, system, seed=0)
        s = _load_summary(fam, system)
        rl2 = s["metrics"]["relative_l2"]["mean"]
        if not ref_drawn:
            ax.plot(c["t"], c["u_ref"][:, 0],
                    color="#555555", lw=1.2, alpha=0.7, label="reference")
            ref_drawn = True
        ax.plot(c["t"], c["u_pred"][:, 0],
                color=FAMILY_COLOR[fam], lw=1.5, alpha=0.9,
                label=f"{fam}  rel-L2={rl2:.3f}")
    ax.set_xlabel("t")
    ax.set_ylabel("u_1(t)")    # always plot first component
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9, fontsize=7)


def _panel_rel_l2_bars(ax) -> None:
    """Mean ± 95% t-CI relative-L2 bars, 4 families × 3 systems."""
    width = 0.22
    x = np.arange(len(FAMILIES))
    hatches = {"lotka_volterra": "", "van_der_pol": "//", "lorenz": "xx"}
    for i, system in enumerate(SYSTEMS):
        means, errs = [], []
        for fam in FAMILIES:
            s = _load_summary(fam, system)["metrics"]["relative_l2"]
            means.append(max(s["mean"], 1e-4))   # log-safe floor
            errs.append(s["ci95_half_width"])
        offset = (i - 1.0) * width
        bars = ax.bar(x + offset, means, width, yerr=errs, capsize=3,
                       label=system, edgecolor="black", linewidth=0.5,
                       color=[FAMILY_COLOR[f] for f in FAMILIES],
                       alpha=0.85 if system == "lotka_volterra" else 0.55)
        if hatches[system]:
            for b in bars:
                b.set_hatch(hatches[system])
    ax.set_xticks(x)
    labels = []
    for fam in FAMILIES:
        # Use the LV summary for label counts (same per-component family).
        s = _load_summary(fam, "lotka_volterra")
        p = s["pqc_params"]
        c = s["classical_params"]
        per = s.get("config_str", "")
        # Show per-component count if available (multi-state scales linearly)
        per_comp = int(p / max(s["dim"], 1)) if s["dim"] else p
        if c > 0:
            labels.append(f"{fam}\n({per_comp} pqc + {c // s['dim']} cls each)")
        else:
            labels.append(f"{fam}\n({per_comp} pqc each)")
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("relative-L2 vs reference  (mean ± 95% t-CI, n=3)")
    ax.set_title("Cross-family accuracy on 3 vector ODEs")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((IN / "config.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    _panel_system_overlay(axes[0, 0], "lotka_volterra")
    _panel_system_overlay(axes[0, 1], "van_der_pol")
    _panel_system_overlay(axes[1, 0], "lorenz")
    _panel_rel_l2_bars(axes[1, 1])

    fig.suptitle(
        "P3.6 — 4 SOTA solver families on 3 vector-state ODEs "
        f"(seeds {cfg['seeds']}; per-component scalar circuits; "
        "first-component overlay)", fontsize=11, y=1.00)
    fig.text(
        0.5, -0.02,
        "Per-component dispatch: each family's scalar circuit is instantiated d "
        "times (d = state dim) with independent weights. NO quantum entanglement "
        "across components. Reference = canonical numpy RK4 (synthetic_ode.py). "
        "Demo artifacts; not pinned by verify_paper_integrity.py.",
        ha="center", fontsize=7, style="italic", color="#555555")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = OUT / f"fig_p3_6_multi_state.{ext}"
        fig.savefig(path)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
