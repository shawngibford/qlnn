"""Render the P4 forecaster autoregressive rollout figure.

Reads `results/p4_forecaster_rollout/{system}_{family}/` (per-seed
field.npz + per-(system, family) seeds_summary.json) produced by
`scripts/run_p4_forecaster_rollout.py` and emits
`paper/figures/fig_p4_forecaster_rollout.{png,pdf}` as a 4-row
diagnostic figure:

  Row 0: cross-system relative-L2 bar chart (5 families per system,
         log scale, persistence floor marked). The headline.

  Row 1: per-system valid-prediction-time (VPT) bars. For Lorenz
         (chaotic), VPT is reported in Lyapunov times; for LV / VdP
         in physical time. 3 panels, one per system.

  Row 2: per-system rollout trajectory comparison (best-seed of each
         family vs reference). 3 panels showing 1st state component
         vs time, with persistence floor as a faint gray line.

  Row 3: per-system spectral-error bars (Fourier-bias mechanism
         probe per pre-reg §5).

Handles partial-sweep data gracefully (missing cells get hatched
placeholder bars).

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
IN = REPO_ROOT / "results" / "p4_forecaster_rollout"
OUT = REPO_ROOT / "paper" / "figures"

SYSTEMS = ["lotka_volterra", "van_der_pol", "lorenz"]
FAMILIES = [
    "data_reuploading", "hardware_efficient", "strongly_entangling",
    "brickwall", "rf_qrc",
]

# Wong palette.
FAMILY_COLOR = {
    "data_reuploading":    "#0072B2",
    "hardware_efficient":  "#D55E00",
    "strongly_entangling": "#009E73",
    "brickwall":           "#CC79A7",
    "rf_qrc":              "#F0E442",
}

SYSTEM_PRETTY = {
    "lotka_volterra": "Lotka-Volterra (2D smooth/periodic)",
    "van_der_pol":    "Van der Pol μ=5 (2D stiff/relax)",
    "lorenz":         "Lorenz '63 (3D chaotic, λ≈0.906)",
}


def _try_load_summary(system: str, family: str) -> dict | None:
    p = IN / f"{system}_{family}" / "seeds_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _try_load_seed_field(system: str, family: str, seed: int) -> dict | None:
    p = IN / f"{system}_{family}" / f"seed_{seed}" / "field.npz"
    if not p.exists():
        return None
    return dict(np.load(p))


def _try_load_seed_metrics(system: str, family: str, seed: int) -> dict | None:
    p = IN / f"{system}_{family}" / f"seed_{seed}" / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Row 0: cross-system relative-L2 bars
# ---------------------------------------------------------------------------


def _row0_rel_l2_bars(ax) -> None:
    n_fams = len(FAMILIES)
    n_sys = len(SYSTEMS)
    bar_w = 0.14
    xpos = np.arange(n_sys)

    # Persistence floor bars (a flat gray reference per system).
    pers_means = []
    for system in SYSTEMS:
        # Pull the median persistence from any family's summary (it's
        # the same trajectory's floor for any forecaster).
        for family in FAMILIES:
            s = _try_load_summary(system, family)
            if s is not None and "metrics" in s:
                pers = s["metrics"].get(
                    "persistence_floor_relative_l2", {}).get("mean")
                if pers is not None:
                    pers_means.append(pers)
                    break
        else:
            pers_means.append(np.nan)

    # Forecaster bars.
    for i, family in enumerate(FAMILIES):
        means, errs, missing = [], [], []
        for system in SYSTEMS:
            s = _try_load_summary(system, family)
            if s is None or "metrics" not in s:
                means.append(np.nan)
                errs.append(0.0)
                missing.append(True)
            else:
                means.append(s["metrics"]["relative_l2"]["mean"])
                errs.append(s["metrics"]["relative_l2"]["ci95_half_width"])
                missing.append(False)
        offsets = (i - (n_fams - 1) / 2) * bar_w
        x = xpos + offsets
        means_arr = np.asarray(means)
        plot_means = np.where(np.isnan(means_arr), 1e-6, means_arr)
        bars = ax.bar(x, plot_means, bar_w,
                      yerr=errs, capsize=2,
                      color=FAMILY_COLOR[family], label=family,
                      edgecolor="black", linewidth=0.4)
        for j, miss in enumerate(missing):
            if miss:
                bars[j].set_hatch("////")
                bars[j].set_alpha(0.4)

    # Persistence floor horizontal lines per system.
    for j, (system, pers) in enumerate(zip(SYSTEMS, pers_means)):
        if not np.isnan(pers):
            ax.hlines(pers, xpos[j] - 0.5, xpos[j] + 0.5,
                      colors="gray", linestyles="--", linewidth=1.0,
                      label="persistence floor" if j == 0 else None)

    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([SYSTEM_PRETTY[s] for s in SYSTEMS],
                       rotation=12, ha="right")
    ax.set_ylabel("relative-L² (log)")
    ax.set_title("P4 forecaster rollout: 5 quantum families × 3 ODE "
                 "systems (3 seeds each)", fontsize=11)
    ax.set_ylim(1e-3, 10)
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.grid(True, axis="y", which="minor", alpha=0.1)


# ---------------------------------------------------------------------------
# Row 1: VPT bars
# ---------------------------------------------------------------------------


def _row1_vpt(axes) -> None:
    for ax, system in zip(axes, SYSTEMS):
        n_fams = len(FAMILIES)
        bar_w = 0.6
        xpos = np.arange(n_fams)
        means, errs, missing, lyap_mode = [], [], [], False
        for family in FAMILIES:
            s = _try_load_summary(system, family)
            if s is None or "metrics" not in s:
                means.append(np.nan)
                errs.append(0.0)
                missing.append(True)
                continue
            # Lyapunov-time VPT for Lorenz; physical time elsewhere.
            if "vpt_lyapunov" in s["metrics"]:
                means.append(s["metrics"]["vpt_lyapunov"]["mean"])
                errs.append(
                    s["metrics"]["vpt_lyapunov"]["ci95_half_width"])
                lyap_mode = True
            else:
                means.append(s["metrics"]["vpt_time"]["mean"])
                errs.append(s["metrics"]["vpt_time"]["ci95_half_width"])
            missing.append(False)

        colors = [FAMILY_COLOR[f] for f in FAMILIES]
        bars = ax.bar(xpos, np.where(np.isnan(means), 0.01, means),
                      bar_w, yerr=errs, capsize=2, color=colors,
                      edgecolor="black", linewidth=0.4)
        for j, miss in enumerate(missing):
            if miss:
                bars[j].set_hatch("////")
                bars[j].set_alpha(0.3)
        ax.set_xticks(xpos)
        ax.set_xticklabels(FAMILIES, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("VPT" + (" (Lyapunov times)" if lyap_mode
                                else " (physical time)"))
        ax.set_title(SYSTEM_PRETTY[system], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Row 2: rollout trajectory comparison (best seed per family)
# ---------------------------------------------------------------------------


def _row2_trajectories(axes) -> None:
    for ax, system in zip(axes, SYSTEMS):
        any_loaded = False
        # Reference trajectory: any seed's u_ref is the same per system.
        ref_field = None
        for family in FAMILIES:
            f = _try_load_seed_field(system, family, 0)
            if f is not None:
                ref_field = np.asarray(f["u_ref"])
                break

        if ref_field is not None:
            ax.plot(ref_field[:, 0], color="black", linewidth=1.5,
                    label="reference", linestyle="-", alpha=0.8)
            any_loaded = True

        for family in FAMILIES:
            # Pick the seed with the lowest relative-L2 — the "best"
            # representative of this family.
            best_seed = None
            best_relL2 = np.inf
            for seed in (0, 1, 2):
                m = _try_load_seed_metrics(system, family, seed)
                if m is None:
                    continue
                if m["relative_l2"] < best_relL2:
                    best_relL2 = m["relative_l2"]
                    best_seed = seed
            if best_seed is None:
                continue
            f = _try_load_seed_field(system, family, best_seed)
            if f is None:
                continue
            pred = np.asarray(f["u_pred"])
            ax.plot(pred[:, 0], color=FAMILY_COLOR[family],
                    label=f"{family} (s{best_seed})",
                    linewidth=0.9, alpha=0.85)
            any_loaded = True

        ax.set_xlabel("rollout step")
        ax.set_ylabel("state[0]")
        ax.set_title(SYSTEM_PRETTY[system], fontsize=9)
        if any_loaded:
            ax.legend(fontsize=6, loc="upper right", ncol=2)
        else:
            ax.text(0.5, 0.5, "trajectories\nnot yet available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="grey", style="italic")
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Row 3: spectral error bars
# ---------------------------------------------------------------------------


def _row3_spectral_error(ax) -> None:
    n_fams = len(FAMILIES)
    n_sys = len(SYSTEMS)
    bar_w = 0.14
    xpos = np.arange(n_sys)
    for i, family in enumerate(FAMILIES):
        means, errs, missing = [], [], []
        for system in SYSTEMS:
            s = _try_load_summary(system, family)
            if s is None or "metrics" not in s:
                means.append(np.nan)
                errs.append(0.0)
                missing.append(True)
            else:
                means.append(s["metrics"]["spectral_error"]["mean"])
                errs.append(
                    s["metrics"]["spectral_error"]["ci95_half_width"])
                missing.append(False)
        offsets = (i - (n_fams - 1) / 2) * bar_w
        x = xpos + offsets
        means_arr = np.asarray(means)
        plot_means = np.where(np.isnan(means_arr), 1e-3, means_arr)
        bars = ax.bar(x, plot_means, bar_w, yerr=errs, capsize=2,
                      color=FAMILY_COLOR[family], label=family,
                      edgecolor="black", linewidth=0.4)
        for j, miss in enumerate(missing):
            if miss:
                bars[j].set_hatch("////")
                bars[j].set_alpha(0.4)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([SYSTEM_PRETTY[s] for s in SYSTEMS],
                       rotation=12, ha="right")
    ax.set_ylabel("spectral error (log)")
    ax.set_title("Spectral error: Fourier-bias mechanism probe (H1/H3)",
                 fontsize=10)
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    ax.grid(True, axis="y", which="major", alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    fig = plt.figure(figsize=(13.5, 14.0))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.30,
                           height_ratios=[1.1, 0.95, 1.05, 0.9])

    ax0 = fig.add_subplot(gs[0, :])
    _row0_rel_l2_bars(ax0)

    axes1 = [fig.add_subplot(gs[1, c]) for c in range(3)]
    _row1_vpt(axes1)

    axes2 = [fig.add_subplot(gs[2, c]) for c in range(3)]
    _row2_trajectories(axes2)

    ax3 = fig.add_subplot(gs[3, :])
    _row3_spectral_error(ax3)

    fig.suptitle(
        "P4 — forecaster autoregressive rollout (5 families × 3 ODE "
        "systems × 3 seeds = 45 cells)\n"
        "NOT yet H1 evidence (the pre-reg defines H1 as the QLNN−"
        "NeuralODE gap; awaits P5).",
        y=0.995, fontsize=11)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fp = OUT / f"fig_p4_forecaster_rollout.{ext}"
        fig.savefig(fp)
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
