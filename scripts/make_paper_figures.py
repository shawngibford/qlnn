"""Generate paper-quality figures from committed result JSONs.

Produces a publication-grade figure set into `paper/figures/`:

  Core claims (legacy, kept for backward compat):
    1. fig_horizon_ablation          — R²+MAE vs horizon
    2. fig_sample_efficiency         — Test MAE vs n_train, both stacks
    3. fig_reproducibility           — CI-width per stack per fraction

  Architecture / data context:
    4. fig_quantum_circuit           — qml.draw_mpl of the 4q/3L
                                       data-reuploading PQC used in the QLNN
    5. fig_dataset_overview          — OD + 6 covariates with chronological
                                       train/val/test split shading

  Per-metric expansions (the "as many metrics as we use" set):
    6. fig_baseline_metrics          — 4-panel bar chart MAE/RMSE/R²/MSE_norm
                                       across all four baseline variants plus
                                       persistence and linear extrapolation
    7. fig_param_sweep               — Params vs test MAE and R² across
                                       hidden_size ∈ {2,4,8,16,32}; QLNN
                                       overlaid (efficiency-Pareto view)
    8. fig_horizon_full_metrics      — h ∈ {1,3,6,12} for all four core
                                       metrics, not just MAE+R²
    9. fig_sample_efficiency_full    — Sample-efficiency curves for
                                       MAE / RMSE / R² / ΔR² (both stacks)
   10. fig_effective_dimension       — d_norm vs n per seed for both
                                       stacks plus aggregate bar with the
                                       +1.0 pre-registered Claim-2 threshold

All numbers come from on-disk JSON in `results/`. The only figures that touch
the raw CSV are `fig_dataset_overview` and `fig_quantum_circuit` (the latter
needs no data — only the locked architecture).

Style: matplotlib publication defaults, 300 dpi, two-color palette
(classical = vermilion, quantum = cool blue) — chosen to remain readable in
grayscale print.
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
    n_train_classical: list[int] = []
    c_mae, c_ci = [], []
    q_mae, q_ci = [], []
    for pct in fractions:
        c = _load(f"results/sample_efficiency/classical_h4_h3_pct{pct}/seeds_summary.json")
        q = _load(f"results/sample_efficiency/qlnn_h3_pct{pct}/seeds_summary.json")
        # Read n_train_windows from protocol.json (not hardcoded — H-02 fix).
        c_proto = _load(f"results/sample_efficiency/classical_h4_h3_pct{pct}/protocol.json")
        q_proto = _load(f"results/sample_efficiency/qlnn_h3_pct{pct}/protocol.json")
        n_c = int(c_proto["n_train_windows"])
        n_q = int(q_proto["n_train_windows"])
        if n_c != n_q:
            print(f"WARN: classical/QLNN n_train_windows differ at pct={pct}: {n_c} vs {n_q}")
        n_train_classical.append(n_c)
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


# ---------------------------------------------------------------------------
# Figure 4 — quantum circuit diagram (the actual PQC the QLNN uses)
# ---------------------------------------------------------------------------
def fig_quantum_circuit():
    """Render the locked 4-qubit, 3-layer data-reuploading PQC from the
    QLNN hybrid run, using PennyLane's matplotlib drawer.

    Config is read from `results/qlnn_hybrid_h3/config.json` so this figure
    always reflects the architecture that produced the paper numbers.
    """
    import pennylane as qml

    cfg = _load("results/qlnn_hybrid_h3/config.json")
    num_qubits = int(cfg["model"]["num_qubits"])
    num_layers = int(cfg["model"]["num_layers"])

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        for layer in range(num_layers):
            for i in range(num_qubits):
                qml.RX(inputs[i], wires=i)
            for i in range(num_qubits):
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1],
                        weights[layer, i, 2], wires=i)
            if num_qubits >= 2:
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if num_qubits > 2:
                    qml.CNOT(wires=[num_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    rng = np.random.default_rng(0)
    inputs = rng.uniform(-np.pi, np.pi, size=(num_qubits,))
    weights = rng.normal(0, 0.1, size=(num_layers, num_qubits, 3))

    fig, _ax = qml.draw_mpl(circuit, style="pennylane")(inputs, weights)
    fig.suptitle(
        f"QLNN feature encoder — data-reuploading PQC\n"
        f"({num_qubits} qubits, {num_layers} layers, ring entanglement; "
        f"{num_layers * num_qubits * 3} trainable rotations + linear "
        f"angle-encoder)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_quantum_circuit.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_quantum_circuit.{png,pdf}'}")


# ---------------------------------------------------------------------------
# Figure 5 — dataset overview
# ---------------------------------------------------------------------------
def fig_dataset_overview():
    """OD target trace + 6 covariates, with chronological train/val/test
    split shading. Provides reader-context for the locked evaluation
    protocol described in §3.

    Skipped with a warning if the raw CSV isn't on disk (e.g. running in a
    git worktree where `data/` is gitignored and unmounted).
    """
    csv_path = ROOT / "data" / "raw" / "qZETA_data_copy.csv"
    if not csv_path.exists():
        print(f"SKIP fig_dataset_overview: {csv_path} not present "
              f"(symlink data/ to the main repo to regenerate)")
        return

    from quantum_liquid_neuralode.data_processing.qzeta import (
        load_qzeta, time_hours_from_date,
    )
    from quantum_liquid_neuralode.data_processing.windowing import split_indices

    df = load_qzeta(csv_path)
    t = time_hours_from_date(df)
    n = len(df)
    s = split_indices(n, train_ratio=0.7, val_ratio=0.15)
    train_idx = np.arange(0, s.train_end)
    val_idx = np.arange(s.train_end, s.val_end)
    test_idx = np.arange(s.val_end, n)

    cov_cols = ["PRE", "TEMP_EXT", "TEMP_CULTURE", "PAR_LIGHT", "PH", "DO"]

    fig = plt.figure(figsize=(9.5, 7.5))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.25)

    # Top row: OD target across full series, spans both columns
    ax_od = fig.add_subplot(gs[0, :])
    ax_od.plot(t, df["OD"].to_numpy(), color="black", linewidth=1.0)
    ax_od.axvspan(t[train_idx[0]], t[train_idx[-1]], color="#0072B2",
                  alpha=0.10, label=f"train ({len(train_idx)})")
    ax_od.axvspan(t[val_idx[0]], t[val_idx[-1]], color="#E69F00",
                  alpha=0.18, label=f"val ({len(val_idx)})")
    ax_od.axvspan(t[test_idx[0]], t[test_idx[-1]], color="#D55E00",
                  alpha=0.18, label=f"test ({len(test_idx)})")
    ax_od.set_ylabel("OD (raw)")
    ax_od.set_xlabel("Time (hours)")
    ax_od.set_title(f"(a) OD target across single fermentation run "
                    f"(n={n} samples)")
    ax_od.legend(loc="upper left", fontsize=8, ncol=3, frameon=True)
    ax_od.grid(True, alpha=0.3)

    # Bottom: 6 covariates in a 3x2 grid
    for k, col in enumerate(cov_cols):
        r, c = 1 + k // 2, k % 2
        ax = fig.add_subplot(gs[r, c])
        ax.plot(t, df[col].to_numpy(), color="#444444", linewidth=0.8)
        ax.set_title(f"({chr(ord('b') + k)}) {col}", fontsize=9)
        ax.set_xlabel("Time (hours)" if r == 3 else "")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "qZETA dataset overview — 778-row single-run bioreactor "
        "(chronological 70/15/15 split)",
        y=0.995, fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_dataset_overview.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_dataset_overview.{png,pdf}'}")


# ---------------------------------------------------------------------------
# Figure 6 — baseline metrics (4 panels, all baselines + persistence/linear)
# ---------------------------------------------------------------------------
def _baseline_metric_rows():
    """Pull (label, mean, std) per metric from each canonical baseline run.

    Returns dict keyed by metric name → list of (label, mean, std). For the
    two zero-parameter baselines (persistence, linear), std is set to 0.
    """
    runs = [
        ("Liquid-ODE\n(Euler)",      "results/baseline_classical_euler"),
        ("Liquid-ODE\n(dopri5)",     "results/baseline_classical_dopri5"),
        ("Liquid-ODE\n+physics",     "results/baseline_classical_physics"),
        ("Liquid-ODE\nfixed [0,3.8]",
                                     "results/baseline_classical_euler_fixed_od"),
    ]
    # Persistence + linear come from any baselines.json (h=1 ⇒ baseline runs).
    base = _load("results/baseline_classical_euler/baselines.json")

    metrics = {
        "MAE (raw OD)":        ("mae_raw",  False),
        "RMSE (raw OD)":       ("rmse_raw", False),
        "R²":                  ("r2_raw",   False),
        "MSE (normalized)":    ("mse_norm", False),
    }
    out = {pretty: [] for pretty in metrics}

    # Persistence
    for pretty, (key, _) in metrics.items():
        out[pretty].append(("Persistence",
                            base["persistence"]["test"][key], 0.0))
        out[pretty].append(("Linear extrap.",
                            base["linear"]["test"][key], 0.0))
    for label, runpath in runs:
        s = _load(f"{runpath}/seeds_summary.json")
        for pretty, (key, _) in metrics.items():
            mu = s["test"][key]["mean"]
            sd = s["test"][key].get("std", 0.0)
            out[pretty].append((label, mu, sd))
    return out


def fig_baseline_metrics():
    metric_rows = _baseline_metric_rows()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    axes = axes.flatten()
    bar_colors = (
        [C_PERSIST, C_LINEAR] + [C_CLASSICAL] * 4
    )

    for ax, (pretty, rows) in zip(axes, metric_rows.items()):
        labels = [r[0] for r in rows]
        means = [r[1] for r in rows]
        stds = [r[2] for r in rows]
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color=bar_colors,
               edgecolor="black", linewidth=0.5, capsize=3, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(pretty)
        ax.set_title(pretty)
        ax.grid(True, axis="y", alpha=0.3)
        for xi, (mu, sd) in enumerate(zip(means, stds)):
            ax.text(xi, mu + (sd if sd else 0) + 0.01 * (abs(mu) + 1e-3),
                    f"{mu:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        "Baseline comparison on h=1 — all canonical metrics, 5 seeds (mean ± std)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_baseline_metrics.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_baseline_metrics.{png,pdf}'}")


# ---------------------------------------------------------------------------
# Figure 7 — parameter sweep / efficiency Pareto
# ---------------------------------------------------------------------------
def fig_param_sweep():
    # Locked from results/param_sweep/param_sweep_table.md (Tier 2 #2.1).
    HIDDEN_TO_PARAMS = {2: 42, 4: 90, 8: 210, 16: 546, 32: 1602}
    hidden_sizes = [2, 4, 8, 16, 32]
    params = [HIDDEN_TO_PARAMS[h] for h in hidden_sizes]
    mae, mae_ci = [], []
    r2, r2_ci = [], []
    dr2, dr2_ci = [], []

    for h in hidden_sizes:
        s = _load(f"results/param_sweep/euler_h3_hidden{h}/seeds_summary.json")
        mae.append(s["test"]["mae_raw"]["mean"]); mae_ci.append(_ci(s["test"]["mae_raw"]))
        r2.append(s["test"]["r2_raw"]["mean"]);   r2_ci.append(_ci(s["test"]["r2_raw"]))
        dr2.append(s["test"]["delta_r2_raw"]["mean"]); dr2_ci.append(_ci(s["test"]["delta_r2_raw"]))

    # QLNN headline (h=3). D=114 trainable params (verified from the
    # rebuilt skeleton in run_effective_dimension and matches the "D" field
    # in results/effective_dimension/effective_dimension.json).
    q = _load("results/qlnn_hybrid_h3/seeds_summary.json")
    q_params = 114
    q_mae = q["test"]["mae_raw"]["mean"];  q_mae_ci = _ci(q["test"]["mae_raw"])
    q_r2 = q["test"]["r2_raw"]["mean"];    q_r2_ci = _ci(q["test"]["r2_raw"])
    q_dr2 = q["test"]["delta_r2_raw"]["mean"]; q_dr2_ci = _ci(q["test"]["delta_r2_raw"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    panels = [
        ("(a) Test MAE vs params",  "Test MAE (raw OD)",     mae, mae_ci, q_mae, q_mae_ci),
        ("(b) Test R² vs params",   "Test R²",               r2,  r2_ci,  q_r2,  q_r2_ci),
        ("(c) ΔOD R² vs params",    "Test ΔOD R²",           dr2, dr2_ci, q_dr2, q_dr2_ci),
    ]
    for ax, (title, ylab, vals, cis, qv, qci) in zip(axes, panels):
        ax.errorbar(params, vals, yerr=cis, marker="o", color=C_CLASSICAL,
                    capsize=4, linewidth=1.8, markersize=7,
                    label="Classical Liquid-ODE (h=3)")
        ax.errorbar([q_params], [qv], yerr=[qci], marker="s",
                    color=C_QLNN, capsize=4, linewidth=1.8, markersize=10,
                    label=f"QLNN (~{q_params} params)")
        for p, v in zip(params, vals):
            ax.annotate(f"H={hidden_sizes[params.index(p)]}", (p, v),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=C_CLASSICAL)
        ax.set_xscale("log")
        ax.set_xlabel("Trainable params")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle("Param-matched sweep at h=3 (5 seeds, mean ± 95% CI)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_param_sweep.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_param_sweep.{png,pdf}'}")


# ---------------------------------------------------------------------------
# Figure 8 — horizon sweep, all four metrics
# ---------------------------------------------------------------------------
def fig_horizon_full_metrics():
    horizons = [1, 3, 6, 12]
    metrics = [
        ("mae_raw",  "Test MAE (raw OD)",  False),
        ("rmse_raw", "Test RMSE (raw OD)", False),
        ("r2_raw",   "Test R²",            True),   # may need ylim clipping
        ("mse_norm", "Test MSE (norm)",    False),
    ]

    # Collect per metric
    table = {m[0]: {"lo": [], "lo_ci": [], "pers": [], "lin": []} for m in metrics}
    for h in horizons:
        b = _load(f"results/horizon_sweep/euler_h{h}/baselines.json")
        s = _load(f"results/horizon_sweep/euler_h{h}/seeds_summary.json")
        for key, _, _ in metrics:
            table[key]["lo"].append(s["test"][key]["mean"])
            table[key]["lo_ci"].append(_ci(s["test"][key]))
            table[key]["pers"].append(b["persistence"]["test"][key])
            table[key]["lin"].append(b["linear"]["test"][key])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for ax, (key, ylab, clip_r2) in zip(axes.flatten(), metrics):
        d = table[key]
        ax.errorbar(horizons, d["lo"], yerr=d["lo_ci"], marker="o",
                    color=C_CLASSICAL, capsize=3, linewidth=1.5,
                    label="Liquid-ODE (Euler)")
        ax.plot(horizons, d["pers"], marker="s", color=C_PERSIST,
                linestyle="--", linewidth=1.2, label="Persistence")
        ax.plot(horizons, d["lin"], marker="^", color=C_LINEAR,
                linestyle=":", linewidth=1.2, label="Linear extrap.")
        if clip_r2:
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            ax.set_ylim(bottom=-15, top=1.1)
        ax.set_xscale("log")
        ax.set_xticks(horizons); ax.set_xticklabels(horizons)
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle(
        "Horizon ablation — all canonical metrics, classical Liquid-ODE (5 seeds)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_horizon_full_metrics.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_horizon_full_metrics.{png,pdf}'}")


# ---------------------------------------------------------------------------
# Figure 9 — sample-efficiency full metric panel
# ---------------------------------------------------------------------------
def fig_sample_efficiency_full():
    fractions = [10, 25, 50, 100]
    metrics = [
        ("mae_raw",       "Test MAE (raw OD)"),
        ("rmse_raw",      "Test RMSE (raw OD)"),
        ("r2_raw",        "Test R²"),
        ("delta_r2_raw",  "Test ΔOD R²"),
    ]
    n_train = []
    rows = {k: {"c": [], "c_ci": [], "q": [], "q_ci": []} for k, _ in metrics}
    for pct in fractions:
        c = _load(f"results/sample_efficiency/classical_h4_h3_pct{pct}/seeds_summary.json")
        q = _load(f"results/sample_efficiency/qlnn_h3_pct{pct}/seeds_summary.json")
        cp = _load(f"results/sample_efficiency/classical_h4_h3_pct{pct}/protocol.json")
        n_train.append(int(cp["n_train_windows"]))
        for key, _ in metrics:
            rows[key]["c"].append(c["test"][key]["mean"])
            rows[key]["c_ci"].append(_ci(c["test"][key]))
            rows[key]["q"].append(q["test"][key]["mean"])
            rows[key]["q_ci"].append(_ci(q["test"][key]))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for ax, (key, ylab) in zip(axes.flatten(), metrics):
        d = rows[key]
        ax.errorbar(n_train, d["c"], yerr=d["c_ci"], marker="o",
                    color=C_CLASSICAL, capsize=4, linewidth=1.8,
                    markersize=7, label="Classical H=4")
        ax.errorbar(n_train, d["q"], yerr=d["q_ci"], marker="s",
                    color=C_QLNN, capsize=4, linewidth=1.8,
                    markersize=7, label="QLNN")
        ax.set_xscale("log")
        ax.set_xticks(n_train)
        ax.set_xticklabels([f"{f}%\n(n={n})" for f, n in zip(fractions, n_train)],
                           fontsize=8)
        ax.set_xlabel("Training data fraction")
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle(
        "Sample efficiency at h=3 — all canonical metrics, mean ± 95% CI",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_sample_efficiency_full.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_sample_efficiency_full.{png,pdf}'}")


# ---------------------------------------------------------------------------
# Figure 10 — effective dimension (Claim 2)
# ---------------------------------------------------------------------------
def fig_effective_dimension():
    """Per-seed d_norm curves vs n for both stacks, plus an aggregate bar
    that visualizes the Claim-2 threshold (Δd_norm > 1.0).
    """
    ed = _load("results/effective_dimension/effective_dimension.json")
    ns = ed["protocol"]["curve_ns"]
    seeds = ed["protocol"]["seeds"]

    # Build (seed → list-of-d_norm) by reading the monotonicity CSV
    csv_path = ROOT / "results" / "effective_dimension" / "monotonicity_check.csv"
    import csv as _csv
    per_seed = {"classical_H4": {s: [] for s in seeds},
                "qlnn_h3":      {s: [] for s in seeds}}
    with csv_path.open() as f:
        r = _csv.DictReader(f)
        rows = list(r)
    # Group by (model, seed) preserving n-order
    for row in rows:
        per_seed[row["model"]][int(row["seed"])].append(
            (int(row["n"]), float(row["d_norm"]))
        )
    for m in per_seed:
        for s in per_seed[m]:
            per_seed[m][s].sort(key=lambda t: t[0])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # (a) Classical per-seed curves
    ax = axes[0]
    for s in seeds:
        xs, ys = zip(*per_seed["classical_H4"][s])
        ax.plot(xs, ys, marker="o", linewidth=1, alpha=0.7,
                color=C_CLASSICAL, label=f"seed {s}" if s == seeds[0] else None)
    # Mean curve overlay
    mean_curve = []
    for i, n in enumerate(ns):
        mean_curve.append(np.mean([per_seed["classical_H4"][s][i][1] for s in seeds]))
    ax.plot(ns, mean_curve, color="black", linewidth=2, marker="D",
            markersize=6, label="mean (5 seeds)")
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel("n (windows used for empirical Fisher)")
    ax.set_ylabel("d_norm (Abbas et al. Eq. 4)")
    ax.set_title("(a) Classical H=4 — per-seed d_norm curves")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8, frameon=True)

    # (b) QLNN per-seed curves
    ax = axes[1]
    for s in seeds:
        xs, ys = zip(*per_seed["qlnn_h3"][s])
        ax.plot(xs, ys, marker="s", linewidth=1, alpha=0.7,
                color=C_QLNN, label=f"seed {s}" if s == seeds[0] else None)
    mean_curve = []
    for i, n in enumerate(ns):
        mean_curve.append(np.mean([per_seed["qlnn_h3"][s][i][1] for s in seeds]))
    ax.plot(ns, mean_curve, color="black", linewidth=2, marker="D",
            markersize=6, label="mean (5 seeds)")
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel("n (windows used for empirical Fisher)")
    ax.set_ylabel("d_norm (Abbas et al. Eq. 4)")
    ax.set_title("(b) QLNN — per-seed d_norm curves")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8, frameon=True)

    # (c) Aggregate comparison bar with threshold
    ax = axes[2]
    c_agg = ed["classical_H4"]["aggregate"]
    q_agg = ed["qlnn_h3"]["aggregate"]
    means = [c_agg["mean"], q_agg["mean"]]
    stds = [c_agg["std"], q_agg["std"]]
    x = np.arange(2)
    bars = ax.bar(x, means, yerr=stds, color=[C_CLASSICAL, C_QLNN],
                  edgecolor="black", linewidth=0.5, capsize=5, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["Classical\nH=4", "QLNN"])
    ax.set_ylabel("d_norm at n=472 (mean ± std)")
    ax.set_title("(c) Aggregate d_norm — Claim 2 verdict")
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.2, f"{m:.2f}\n±{s:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    # Show Δ and the pre-reg threshold visually
    delta = ed["delta_d_norm_qlnn_minus_classical"]
    thr = ed["threshold"]
    ax.annotate(
        f"Δd_norm = +{delta:.2f}  (pre-reg threshold > +{thr:.1f})",
        xy=(0.5, 0.02), xycoords="axes fraction",
        ha="center", va="bottom", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Claim 2 — Effective dimension (Abbas et al. 2021), 5-seed estimate",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_effective_dimension.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_effective_dimension.{png,pdf}'}")


# ---------------------------------------------------------------------------
# Figure 11 — circuit-search per-axis effect plot (Phase 2 ablation)
# ---------------------------------------------------------------------------
def fig_ansatz_axis_effects():
    """Small-multiples bar chart: for each search axis, show test MAE & R²
    by level. Skipped with a clear message if `results/circuit_search/` is
    empty (the figure has nothing to plot until Phase 2 of the search runs).
    """
    table = ROOT / "results" / "circuit_search" / "circuit_search_table.json"
    if not table.exists():
        print(f"SKIP fig_ansatz_axis_effects: {table} not present "
              f"(run scripts/run_circuit_search.sh + scripts/summarize_circuit_search.py)")
        return

    rows = _load("results/circuit_search/circuit_search_table.json")
    if not rows:
        print("SKIP fig_ansatz_axis_effects: empty table")
        return

    # Find the reference row's metrics (drawn as a horizontal dashed line on
    # every panel for easy visual comparison).
    ref = next((r for r in rows if r["is_reference"]), None)
    ref_mae = ref["test_mae_raw"]["mean"] if ref else None
    ref_r2 = ref["test_r2_raw"]["mean"] if ref else None

    # Group by axis (excluding the reference itself — it gets its own line).
    axes_present: dict[str, list[dict]] = {}
    for r in rows:
        if r["is_reference"]:
            continue
        axes_present.setdefault(r["axis"], []).append(r)

    if not axes_present:
        print("SKIP fig_ansatz_axis_effects: only reference row present")
        return

    n_axes = len(axes_present)
    fig, axes_mpl = plt.subplots(2, n_axes, figsize=(3.0 * n_axes + 1.5, 6.0),
                                 squeeze=False)

    for col, (axis_name, axis_rows) in enumerate(sorted(axes_present.items())):
        labels = [r["level"] for r in axis_rows]
        x = np.arange(len(labels))

        # Top row: MAE
        ax = axes_mpl[0, col]
        ax.bar(x, [r["test_mae_raw"]["mean"] for r in axis_rows],
               color=C_QLNN, edgecolor="black", linewidth=0.4, alpha=0.85)
        if ref_mae is not None:
            ax.axhline(ref_mae, color="black", linestyle="--", linewidth=1,
                       alpha=0.7, label="reference")
        ax.set_title(f"Axis: {axis_name}", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        if col == 0:
            ax.set_ylabel("Test MAE (raw OD)")
            ax.legend(loc="best", fontsize=7, frameon=True)
        ax.grid(True, axis="y", alpha=0.3)

        # Bottom row: R²
        ax = axes_mpl[1, col]
        ax.bar(x, [r["test_r2_raw"]["mean"] for r in axis_rows],
               color=C_QLNN, edgecolor="black", linewidth=0.4, alpha=0.85)
        if ref_r2 is not None:
            ax.axhline(ref_r2, color="black", linestyle="--", linewidth=1,
                       alpha=0.7, label="reference")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        if col == 0:
            ax.set_ylabel("Test R²")
            ax.legend(loc="best", fontsize=7, frameon=True)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Circuit search — per-axis effect on QLNN forecast quality "
        "(single-seed proxy budget, h=3)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_ansatz_axis_effects.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_ansatz_axis_effects.{png,pdf}'}")


# ---------------------------------------------------------------------------
# Figure 12 — circuit-search Pareto (params vs test MAE, by ansatz family)
# ---------------------------------------------------------------------------
def fig_circuit_pareto():
    """Scatter of trainable params vs test MAE, one marker per circuit,
    colored by ansatz family. Overlays the classical Pareto from
    `fig_param_sweep` for direct comparison.

    Skipped if circuit_search table is missing.
    """
    table = ROOT / "results" / "circuit_search" / "circuit_search_table.json"
    if not table.exists():
        print(f"SKIP fig_circuit_pareto: {table} not present")
        return
    rows = _load("results/circuit_search/circuit_search_table.json")
    if not rows:
        print("SKIP fig_circuit_pareto: empty table")
        return

    # Estimate per-circuit params: matches the locked rule of thumb
    #   D = (encoder linear) + (PQC weights) + (cell tau+A) + (head+initial)
    # Easier: read it from each run's seeds_summary if it stored num_params,
    # else compute from ansatz shape contracts.
    def _approx_params(r):
        Q = r["num_qubits"]
        L = r["num_layers"]
        # 7 input features + biases on the encoder linear.
        enc_linear = 7 * Q + Q
        # ansatz weights (matches AnsatzProtocol.weight_shape):
        if r["ansatz_name"] in ("data_reuploading", "strongly_entangling"):
            pqc = L * Q * 3
        elif r["ansatz_name"] in ("hardware_efficient", "brickwall"):
            pqc = L * Q * 2
        else:  # unknown — best-effort
            pqc = L * Q * 3
        # cell tau_unconstrained + A
        cell = 2 * Q
        # initial_h_W + initial_h_b + delta_head_W + delta_head_b + delta_scale
        head = 7 * Q + Q + Q + 1 + 1
        return enc_linear + pqc + cell + head

    fams = sorted(set(r["ansatz_name"] for r in rows))
    palette = {
        "data_reuploading":      C_QLNN,
        "hardware_efficient":    "#E69F00",
        "strongly_entangling":   "#009E73",
        "brickwall":             "#CC79A7",
    }

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # Classical Pareto for context (read from param_sweep).
    cls_hidden = [2, 4, 8, 16, 32]
    cls_params = [42, 90, 210, 546, 1602]
    cls_mae = []
    for h in cls_hidden:
        s = _load(f"results/param_sweep/euler_h3_hidden{h}/seeds_summary.json")
        cls_mae.append(s["test"]["mae_raw"]["mean"])
    ax.plot(cls_params, cls_mae, marker="o", linestyle="--", color=C_CLASSICAL,
            linewidth=1.2, alpha=0.7, label="Classical Liquid-ODE (h=3)")

    # Circuit search points.
    for fam in fams:
        rs = [r for r in rows if r["ansatz_name"] == fam]
        xs = [_approx_params(r) for r in rs]
        ys = [r["test_mae_raw"]["mean"] for r in rs]
        ax.scatter(xs, ys, color=palette.get(fam, "grey"), label=f"QLNN — {fam}",
                   s=70, edgecolor="black", linewidth=0.5, alpha=0.9, zorder=3)

    # Annotate the reference cell.
    ref = next((r for r in rows if r["is_reference"]), None)
    if ref:
        ax.annotate("reference (4q/3L)",
                    (_approx_params(ref), ref["test_mae_raw"]["mean"]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Test MAE at h=3 (raw OD)")
    ax.set_title("Circuit-search Pareto — QLNN ansatz family vs classical baseline")
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_circuit_pareto.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_circuit_pareto.{png,pdf}'}")


# ---------------------------------------------------------------------------
# Figure 13 — proxy-vs-promoted comparison (Phase 4 validation story)
# ---------------------------------------------------------------------------
def fig_promotion_validation():
    """Per-circuit head-to-head: proxy budget (1-seed) vs the full 5-seed
    locked protocol. Communicates the proxy/full divergence — and shows
    the corrected ranking after promotion.

    Skipped if either the proxy table or the promoted runs are missing.
    """
    table = ROOT / "results" / "circuit_search" / "circuit_search_table.json"
    promoted_dir = ROOT / "results" / "circuit_search_promoted"
    if not (table.exists() and promoted_dir.exists()):
        print(f"SKIP fig_promotion_validation: need both "
              f"results/circuit_search/circuit_search_table.json AND "
              f"results/circuit_search_promoted/ to draw this figure")
        return

    rows = _load("results/circuit_search/circuit_search_table.json")
    promoted_rows = [r for r in rows if r["axis"] == "promoted"]
    if not promoted_rows:
        print("SKIP fig_promotion_validation: no `promoted` rows in table")
        return

    # For each promoted row, look up its source proxy run (the YAML's
    # `source_run` field) and read that proxy's test MAE from the table.
    proxy_lookup = {r["run"]: r for r in rows}

    promoted_meta = []
    for pr in promoted_rows:
        cfg_path = ROOT / "configs" / "circuit_search_promoted" / f"{pr['run']}.yaml"
        import yaml as _yaml
        proxy_run = None
        if cfg_path.exists():
            y = _yaml.safe_load(cfg_path.read_text())
            proxy_run = (y.get("circuit_search") or {}).get("source_run")
        proxy = proxy_lookup.get(proxy_run)
        if proxy is None:
            continue
        promoted_meta.append({
            "label": pr["run"].replace("top", "T").replace("_", "\n").replace("\nQ", " Q"),
            "ansatz": pr["ansatz_name"],
            "q": pr["num_qubits"], "l": pr["num_layers"],
            "proxy_mae": proxy["test_mae_raw"]["mean"],
            "promoted_mae": pr["test_mae_raw"]["mean"],
            "promoted_ci": pr["test_mae_raw"].get("ci95_half_width")
                            or pr["test_mae_raw"].get("std", 0.0),
            "proxy_run": proxy_run,
        })

    if not promoted_meta:
        print("SKIP fig_promotion_validation: no proxy↔promoted match found")
        return

    # Reference (5-seed) from the existing headline run.
    ref = _load("results/qlnn_hybrid_h3/seeds_summary.json")
    ref_mae = ref["test"]["mae_raw"]["mean"]
    ref_ci = ref["test"]["mae_raw"].get("ci95_half_width",
                                        ref["test"]["mae_raw"].get("std", 0.0))

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    n = len(promoted_meta)
    x = np.arange(n)
    width = 0.38

    proxy_vals = [m["proxy_mae"] for m in promoted_meta]
    promoted_vals = [m["promoted_mae"] for m in promoted_meta]
    promoted_cis = [m["promoted_ci"] for m in promoted_meta]
    labels = [m["label"] for m in promoted_meta]

    ax.bar(x - width/2, proxy_vals, width, color="#999999",
           edgecolor="black", linewidth=0.4, label="Proxy (1 seed)")
    ax.bar(x + width/2, promoted_vals, width, color=C_QLNN, yerr=promoted_cis,
           edgecolor="black", linewidth=0.4, capsize=4,
           label="Promoted (5 seeds, mean ± 95% CI)")

    # Annotate the bars.
    for xi, (p, q, ci) in enumerate(zip(proxy_vals, promoted_vals, promoted_cis)):
        ax.text(xi - width/2, p + 0.002, f"{p:.4f}", ha="center", va="bottom",
                fontsize=8, color="#555")
        ax.text(xi + width/2, q + ci + 0.002, f"{q:.4f}", ha="center", va="bottom",
                fontsize=8, color=C_QLNN, fontweight="bold")

    # Reference horizontal band (5-seed, with CI shading).
    ax.axhline(ref_mae, color="black", linestyle="--", linewidth=1, alpha=0.8,
               label=f"Reference 5-seed (data_reuploading 4q/3L): {ref_mae:.4f} ± {ref_ci:.4f}")
    ax.axhspan(ref_mae - ref_ci, ref_mae + ref_ci,
               color="black", alpha=0.06, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test MAE at h=3 (raw OD)")
    ax.set_title(
        "Phase 4 promotion — proxy ranking vs locked 5-seed protocol\n"
        "(proxy ranks 1→2→3 do NOT match 5-seed ranks; "
        "strongly_entangling 6q/3L is the true winner)",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_promotion_validation.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_promotion_validation.{png,pdf}'}")


if __name__ == "__main__":
    # Legacy 3 (kept exactly as before)
    fig_horizon()
    fig_sample_efficiency()
    fig_reproducibility()
    # New publication-grade additions
    fig_quantum_circuit()
    fig_dataset_overview()
    fig_baseline_metrics()
    fig_param_sweep()
    fig_horizon_full_metrics()
    fig_sample_efficiency_full()
    fig_effective_dimension()
    # Circuit-search figures (gracefully skipped until search results land)
    fig_ansatz_axis_effects()
    fig_circuit_pareto()
    fig_promotion_validation()
    print(f"\nAll figures written to {OUT}/")
