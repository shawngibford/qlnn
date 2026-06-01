"""Render the per-family compute envelope figure.

Two panels:
  Left  — per-family wall-clock per cell (smoke-extrapolated, hours)
          stratified by task surface: ODE solver (kuramoto smoke,
          1D scalar) and PDE solver (KdV smoke, 2D scalar). Classical
          PINN shown as a horizontal reference.
  Right — Phase C compute envelope: stacked bar per re-run scope
          (A15 / A16 / A17 / A19 / M3) with the per-cell est × per-
          scope cell count = scope total. Phase C grand total
          annotated.

Reads:
  results/smoke_kuramoto/smoke_kuramoto_runtimes.json
  results/smoke_kdv/smoke_kdv_runtimes.json
  results/smoke_post_audit/smoke_post_audit_runtimes.json

Emits paper/figures/fig_compute_envelope.{png,pdf}.

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


def _load_hrs(path: str) -> dict[str, float]:
    with (REPO_ROOT / path).open() as f:
        d = json.load(f)
    return {r["family"]: float(r["est_full_cell_hours"])
            for r in d["records"]}


# Phase C compute envelope composition (from NEXT_STEPS.md + CLAUDE.md
# audit-driven re-run scope notes). Each entry is (label, cell count,
# per-cell hours derived from smoke).
def _phase_c_scopes(ode_hrs: dict[str, float],
                     pde_hrs: dict[str, float],
                     post_audit_hrs: dict[str, float]) -> list[dict]:
    # Smoke-mean wall-clock per ODE / PDE cell, used as the per-scope
    # mean estimate (the per-cell number bakes in the family-mix of
    # the scope).
    ode_mean = float(np.mean(list(ode_hrs.values())))
    pde_mean = float(np.mean(list(pde_hrs.values())))
    qcpinn_var_mean = float(np.mean([
        post_audit_hrs.get("qcpinn_balanced", 0.0),
        post_audit_hrs.get("qcpinn_quantum", 0.0),
        post_audit_hrs.get("qcpinn_full_q", 0.0),
    ]))
    return [
        dict(scope="A15 solver re-runs\n(7 QLNN + cPINN × 9 sys × 3 seeds)",
             cells=216, hrs_per=ode_mean, color="#0072B2"),
        dict(scope="A16 forecaster re-runs\n(strongly_ent un-aliased)",
             cells=18,  hrs_per=ode_mean, color="#56B4E9"),
        dict(scope="A17 qcpinn variants\n(3 variants × 8 sys × 3 seeds)",
             cells=72,  hrs_per=qcpinn_var_mean, color="#009E73"),
        dict(scope="A19 forecaster parity\n(2000-step budget)",
             cells=36,  hrs_per=ode_mean, color="#F0E442"),
        dict(scope="M3 kuramoto + KdV\n(5 fam × 2 sys × 3 seeds)",
             cells=30,  hrs_per=(ode_mean + pde_mean) / 2.0,
             color="#D55E00"),
    ]


def main() -> None:
    ode = _load_hrs("results/smoke_kuramoto/smoke_kuramoto_runtimes.json")
    pde = _load_hrs("results/smoke_kdv/smoke_kdv_runtimes.json")
    audit = _load_hrs(
        "results/smoke_post_audit/smoke_post_audit_runtimes.json")
    cpinn = audit.get("classical_pinn", 0.001)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11.5, 4.6),
        constrained_layout=True,
        gridspec_kw=dict(width_ratios=(1.0, 1.0)))

    # ===== Left panel: per-family per-cell wall-clock ===================
    families = sorted(set(ode) | set(pde) | set(audit) - {"classical_pinn"})
    families_ode = [f for f in families if f in ode]
    families_pde = [f for f in families if f in pde]
    families_qcpinn_var = [f for f in families
                            if f.startswith("qcpinn_") and f in audit]

    # Three category groups, plotted as grouped bars.
    groups = [
        ("ODE solver  (kuramoto smoke)",
         families_ode, ode, "#0072B2"),
        ("PDE solver  (KdV smoke)",
         families_pde, pde, "#D55E00"),
        ("ODE qcpinn-variants  (A17)",
         families_qcpinn_var, audit, "#009E73"),
    ]
    n_per_group = max(len(g[1]) for g in groups)
    bar_w = 0.8 / len(groups)
    x = np.arange(n_per_group)
    for gi, (label, fams, hrs_map, color) in enumerate(groups):
        vals = [hrs_map[f] for f in fams]
        labs = [f.replace("qcpinn_", "qc_").replace("_2d", "")
                for f in fams]
        ax_l.bar(x[:len(vals)] + (gi - 1) * bar_w, vals,
                  bar_w, color=color, edgecolor="black", lw=0.6,
                  label=label, alpha=0.85)
        for xi, (xv, val) in enumerate(zip(x[:len(vals)], vals)):
            ax_l.text(xv + (gi - 1) * bar_w, val + 0.04,
                       f"{val:.2f}h",
                       ha="center", va="bottom", fontsize=7.5,
                       color=color, fontweight="bold")
    ax_l.axhline(cpinn, color="black", lw=1.2, ls="--",
                  label=f"classical PINN  ({cpinn:.3f}h)",
                  alpha=0.7)
    # Use the longest set as the canonical x-label order
    longest = max(groups, key=lambda g: len(g[1]))
    ax_l.set_xticks(x[:len(longest[1])])
    ax_l.set_xticklabels(
        [f.replace("qcpinn_", "qc_").replace("_2d", "")
         for f in longest[1]],
        fontsize=8.5, rotation=18, ha="right")
    ax_l.set_ylabel("wall-clock per cell  (hours, CPU)")
    ax_l.set_title("Per-family per-cell wall-clock\n"
                    "(smoke-extrapolated to 2000 steps)",
                    fontsize=10.5)
    ax_l.legend(loc="upper left", fontsize=8, frameon=True)
    ax_l.grid(axis="y", alpha=0.25, lw=0.5)

    # ===== Right panel: Phase C envelope, scope-stacked =================
    scopes = _phase_c_scopes(ode, pde, audit)
    scope_labels = [s["scope"] for s in scopes]
    scope_hrs = [s["cells"] * s["hrs_per"] for s in scopes]
    scope_colors = [s["color"] for s in scopes]
    bars = ax_r.barh(np.arange(len(scopes)), scope_hrs,
                       color=scope_colors, edgecolor="black", lw=0.6)
    for i, (s, hrs) in enumerate(zip(scopes, scope_hrs)):
        ax_r.text(hrs + 1.2, i,
                   f"{s['cells']} cells × {s['hrs_per']:.2f}h ≈ "
                   f"{hrs:.0f} CPU-hr",
                   va="center", fontsize=8.5, color="#222222")
    total_hrs = float(sum(scope_hrs))
    ax_r.set_yticks(np.arange(len(scopes)))
    ax_r.set_yticklabels(scope_labels, fontsize=9)
    ax_r.invert_yaxis()
    ax_r.set_xlabel("CPU-hours  (serial-equivalent)")
    ax_r.set_title(
        f"Phase C re-run compute envelope\n"
        f"≈ {total_hrs:.0f} CPU-hr serial; embarrassingly parallel "
        "on Anvil GPU",
        fontsize=10.5)
    ax_r.grid(axis="x", alpha=0.25, lw=0.5)
    ax_r.set_xlim(0, max(scope_hrs) * 1.65)

    fig.suptitle(
        "Compute envelope:  per-family per-cell  +  Phase C re-run scopes",
        y=1.05, fontsize=11.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_compute_envelope.png")
    fig.savefig(OUT_DIR / "fig_compute_envelope.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_compute_envelope.pdf'}")


if __name__ == "__main__":
    main()
