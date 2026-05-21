"""Render the P3.9 PDE multi-family matrix figure.

Reads from TWO directories:
  - `results/p3_8_review/{pde}_chebyshev_dqc_2d/` + `..._classical_pinn/`
    (P3.8's chebyshev_dqc_2d + classical_pinn baselines at the
    audit-corrected configs)
  - `results/p3_9_pde_matrix/{pde}_{family}/` for the 3 new families
    (qcpinn_2d, te_qpinn_fnn_2d, te_qpinn_qnn_2d)

Emits `paper/figures/fig_p3_9_pde_matrix.{png,pdf}` as a 4-row figure:

  Row 0: Cross-PDE bar chart, ALL 4 quantum families + classical PINN.
         3 PDEs × 5 bars per PDE, log-scale relL2, predict-zero floor
         marked. The audit-corrected PDE coverage that closes the
         P3.8 single-family gap.

  Row 1: Per-PDE loss trajectories (4 quantum families overlaid +
         classical PINN reference line). Diagnoses whether the new
         ports are under-convergent vs saturating on each PDE.

  Row 2: BC violation per (PDE, family). The 1.0+ BC violations we
         saw in qcpinn_2d's heat smoke run get tracked across the
         matrix; quantifies the Lagaris-IC ≠ periodic-BC gap.

  Row 3: Per-family param accounting bar chart. PQC params (quantum
         scalars) vs classical params (pre/post-NN scalars for qcpinn
         + FNN scalars for te_qpinn_fnn). Disclosure of the
         "classical-heavy capacity confound" the CIRCUIT_SPECS
         amendment flags.

Handles partial sweep data transparently — missing entries become
hatched placeholder bars with "not yet available" annotations.

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
IN_P38 = REPO_ROOT / "results" / "p3_8_review"
IN_P39 = REPO_ROOT / "results" / "p3_9_pde_matrix"
OUT = REPO_ROOT / "paper" / "figures"

PDES = ["heat", "burgers_smooth", "allen_cahn"]
QUANTUM_FAMILIES = [
    "chebyshev_dqc_2d", "qcpinn_2d", "te_qpinn_fnn_2d", "te_qpinn_qnn_2d",
]
ALL_MODELS = QUANTUM_FAMILIES + ["classical_pinn"]

# Wong palette, colorblind-safe.
MODEL_COLOR = {
    "chebyshev_dqc_2d": "#0072B2",   # cool blue
    "qcpinn_2d":        "#009E73",   # green
    "te_qpinn_fnn_2d":  "#D55E00",   # vermilion
    "te_qpinn_qnn_2d":  "#CC79A7",   # pink
    "classical_pinn":   "#F0E442",   # yellow
}

MODEL_PRETTY = {
    "chebyshev_dqc_2d": "chebyshev_dqc_2d",
    "qcpinn_2d":        "qcpinn_2d",
    "te_qpinn_fnn_2d":  "te_qpinn_fnn_2d",
    "te_qpinn_qnn_2d":  "te_qpinn_qnn_2d",
    "classical_pinn":   "classical_pinn (MLP)",
}

PDE_PRETTY = {
    "heat":           "Heat (analytic ref, 1200 steps)",
    "burgers_smooth": "Burgers smooth (npz ref, 1500 steps)",
    "allen_cahn":     "Allen-Cahn (npz ref, 64×32 colloc, 1800 steps)",
}


def _summary_path(pde: str, model: str) -> Path | None:
    """Route to whichever results dir holds this (pde, model)."""
    if model in ("chebyshev_dqc_2d", "classical_pinn"):
        cand = IN_P38 / f"{pde}_{model}" / "seeds_summary.json"
    else:
        cand = IN_P39 / f"{pde}_{model}" / "seeds_summary.json"
    return cand if cand.exists() else None


def _try_load_summary(pde: str, model: str) -> dict | None:
    p = _summary_path(pde, model)
    if p is None:
        return None
    return json.loads(p.read_text())


def _try_load_seed_metrics(pde: str, model: str, seed: int) -> dict | None:
    if model in ("chebyshev_dqc_2d", "classical_pinn"):
        cand = IN_P38 / f"{pde}_{model}" / f"seed_{seed}" / "metrics.json"
    else:
        cand = IN_P39 / f"{pde}_{model}" / f"seed_{seed}" / "metrics.json"
    return json.loads(cand.read_text()) if cand.exists() else None


def _try_load_seed_field(pde: str, model: str, seed: int) -> dict | None:
    if model in ("chebyshev_dqc_2d", "classical_pinn"):
        cand = IN_P38 / f"{pde}_{model}" / f"seed_{seed}" / "field.npz"
    else:
        cand = IN_P39 / f"{pde}_{model}" / f"seed_{seed}" / "field.npz"
    if not cand.exists():
        return None
    return dict(np.load(cand))


def _row0_rel_l2_bars(ax) -> None:
    """3 PDEs × 5 models per PDE, log-scale relL2 with 95% CI bars."""
    n_models = len(ALL_MODELS)
    n_pdes = len(PDES)
    bar_w = 0.15
    xpos = np.arange(n_pdes)

    for i, model in enumerate(ALL_MODELS):
        means, errs, missing = [], [], []
        for pde in PDES:
            s = _try_load_summary(pde, model)
            if s is None or "metrics" not in s:
                means.append(np.nan)
                errs.append(0.0)
                missing.append(True)
            else:
                means.append(s["metrics"]["relative_l2"]["mean"])
                errs.append(s["metrics"]["relative_l2"]["ci95_half_width"])
                missing.append(False)
        offsets = (i - (n_models - 1) / 2) * bar_w
        x = xpos + offsets
        means_arr = np.asarray(means)
        errs_arr = np.asarray(errs)
        # Replace NaN with a small visible placeholder height for missing.
        plot_means = np.where(np.isnan(means_arr), 1e-6, means_arr)
        bars = ax.bar(x, plot_means, bar_w,
                      yerr=errs_arr, capsize=2,
                      color=MODEL_COLOR[model],
                      label=MODEL_PRETTY[model],
                      edgecolor="black", linewidth=0.4)
        for j, miss in enumerate(missing):
            if miss:
                bars[j].set_hatch("////")
                bars[j].set_alpha(0.4)

    ax.set_yscale("log")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8,
               label="predict-zero floor")
    ax.set_xticks(xpos)
    ax.set_xticklabels([PDE_PRETTY[p] for p in PDES], rotation=15, ha="right")
    ax.set_ylabel("relative-L² (log)")
    ax.set_title("P3.9 PDE matrix: 4 quantum families + classical PINN "
                  "(audit-corrected configs)", fontsize=11)
    ax.set_ylim(1e-4, 5)
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.grid(True, axis="y", which="minor", alpha=0.1)


def _row1_loss_trajectories(axes) -> None:
    """Per-PDE loss-vs-step for all 5 models at seed 0."""
    for ax, pde in zip(axes, PDES):
        any_loaded = False
        for model in ALL_MODELS:
            field = _try_load_seed_field(pde, model, 0)
            if field is None:
                continue
            loss = np.asarray(field["loss_history"], dtype=np.float64)
            steps = np.arange(1, len(loss) + 1)
            ax.plot(steps, np.clip(loss, 1e-10, None),
                    color=MODEL_COLOR[model],
                    label=MODEL_PRETTY[model], linewidth=1.0, alpha=0.85)
            any_loaded = True
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.set_title(PDE_PRETTY[pde], fontsize=9)
        if any_loaded:
            ax.legend(fontsize=6, loc="upper right")
        else:
            ax.text(0.5, 0.5, "loss data\nnot yet available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="grey", style="italic")
        ax.grid(True, alpha=0.3)


def _row2_bc_violation(ax) -> None:
    """BC violation per (PDE, model). Bar grouping: PDE on x-axis,
    models within each group."""
    n_models = len(ALL_MODELS)
    n_pdes = len(PDES)
    bar_w = 0.15
    xpos = np.arange(n_pdes)

    for i, model in enumerate(ALL_MODELS):
        vals = []
        errs = []
        missing = []
        for pde in PDES:
            s = _try_load_summary(pde, model)
            if (s is None or "metrics" not in s
                    or "bc_violation" not in s["metrics"]):
                vals.append(np.nan)
                errs.append(0.0)
                missing.append(True)
            else:
                vals.append(s["metrics"]["bc_violation"]["mean"])
                errs.append(s["metrics"]["bc_violation"]["ci95_half_width"])
                missing.append(False)
        offsets = (i - (n_models - 1) / 2) * bar_w
        x = xpos + offsets
        vals_arr = np.asarray(vals)
        plot_vals = np.where(np.isnan(vals_arr), 0.001, vals_arr)
        bars = ax.bar(x, plot_vals, bar_w,
                      yerr=errs, capsize=2,
                      color=MODEL_COLOR[model],
                      label=MODEL_PRETTY[model],
                      edgecolor="black", linewidth=0.4)
        for j, miss in enumerate(missing):
            if miss:
                bars[j].set_hatch("////")
                bars[j].set_alpha(0.4)

    ax.axhline(0.05, color="grey", linestyle=":", linewidth=0.8,
               label="5% threshold (declared design)")
    ax.set_xticks(xpos)
    ax.set_xticklabels([PDE_PRETTY[p] for p in PDES], rotation=15, ha="right")
    ax.set_ylabel("BC violation (rel. to max|u|)")
    ax.set_title("Periodic-BC violation (Lagaris IC does NOT enforce "
                  "periodicity)", fontsize=10)
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)


def _row3_param_accounting(ax) -> None:
    """Per-family PQC params + classical params (stacked bars)."""
    # Pull one seed_0 (heat) per model for the param counts — they're
    # identical across PDEs by construction.
    pqcs, classs, labels, colors = [], [], [], []
    for model in QUANTUM_FAMILIES + ["classical_pinn"]:
        m = _try_load_seed_metrics("heat", model, 0)
        if m is None:
            pqcs.append(0)
            classs.append(0)
            labels.append(MODEL_PRETTY[model] + "\n(not yet)")
            colors.append(MODEL_COLOR[model])
        else:
            pqcs.append(int(m["pqc_params"]))
            classs.append(int(m["classical_params"]))
            labels.append(MODEL_PRETTY[model])
            colors.append(MODEL_COLOR[model])

    xpos = np.arange(len(labels))
    bw = 0.6
    ax.bar(xpos, pqcs, bw, label="PQC params", color="#0072B2",
           edgecolor="black", linewidth=0.4)
    ax.bar(xpos, classs, bw, bottom=pqcs, label="classical params",
           color="#F0E442", edgecolor="black", linewidth=0.4)

    for i, (pq, cl) in enumerate(zip(pqcs, classs)):
        total = pq + cl
        if total > 0:
            ax.text(i, total + max(total * 0.04, 8),
                    f"{total}\n(PQC={pq})",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("trainable scalar count")
    ax.set_title("Per-family capacity (PQC scalars vs classical pre/post-NN "
                  "scalars)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def main() -> None:
    fig = plt.figure(figsize=(13.5, 14.0))
    gs = gridspec.GridSpec(4, 3, figure=fig,
                            hspace=0.55, wspace=0.30,
                            height_ratios=[1.1, 1.0, 0.9, 0.9])

    # Row 0: spans the full width.
    ax0 = fig.add_subplot(gs[0, :])
    _row0_rel_l2_bars(ax0)

    # Row 1: 3 panels.
    axes1 = [fig.add_subplot(gs[1, c]) for c in range(3)]
    _row1_loss_trajectories(axes1)

    # Row 2: spans the full width.
    ax2 = fig.add_subplot(gs[2, :])
    _row2_bc_violation(ax2)

    # Row 3: spans the full width.
    ax3 = fig.add_subplot(gs[3, :])
    _row3_param_accounting(ax3)

    fig.suptitle(
        "P3.9 — PDE multi-family matrix "
        "(4 quantum PINN-style ports + classical PINN baseline)\n"
        "rf_qrc OUT of scope (frozen-reservoir architecture; deferred "
        "to P4 as forecaster). NOT yet H1 evidence (the pre-reg defines "
        "H1 as the QLNN−NeuralODE gap; awaits P5).",
        y=0.995, fontsize=11)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fp = OUT / f"fig_p3_9_pde_matrix.{ext}"
        fig.savefig(fp)
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
