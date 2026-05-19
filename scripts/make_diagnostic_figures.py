"""Reviewer-diagnostic figure set (separate from the headline-claim figures
in scripts/make_paper_figures.py).

These fill the training-dynamics / error-analysis / statistical-rigor /
quantum-trainability gaps that ML (NeurIPS/ICML) and QC (Quantum/PRX-Q)
reviewers flag first. Three tiers:

  T1 (zero compute, on-disk data):
    fig_learning_curves        train/val loss vs epoch, 5-seed band
    fig_forecast_trajectory    actual vs predicted OD across test window
    fig_pred_vs_actual         calibration scatter + R²
    fig_residual_analysis      residual-vs-time, hist+QQ, ACF
    fig_paired_bootstrap       QLNN−classical paired-diff bootstrap vs null
    fig_seed_strip             every seed's test MAE/R² + mean±CI
    fig_all_circuit_diagrams   qml.draw_mpl for all 4 ansätze (2×2)

  T2 (after the Option-B O-2 sweep lands):
    fig_accuracy_variance_frontier   (MAE, σ) space + G1/G2 feasible box
    fig_regularization_arrows        R0→{R1,R2,R3} vectors per circuit
    fig_circuit_regime_heatmap       3×4 grid by penalized objective

  T3 (dedicated ~2-4 h compute, gated):
    fig_barren_plateau / fig_expressibility / fig_entangling_capability
    / fig_fisher_eigenspectrum  (see scripts/analyze_quantum_trainability.py)

Every function gracefully skips (prints SKIP …) if its inputs are absent,
mirroring scripts/make_paper_figures.py. Predictions on disk are in
normalized OD space; raw OD = norm·(od_data_max−od_data_min)+od_data_min
using the per-run protocol.json bounds.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

C_CLASSICAL = "#D55E00"
C_QLNN = "#0072B2"
C_NULL = "#999999"

# Canonical 5-seed reference runs (the Claim-1 head-to-head pair).
CLASSICAL_RUN = "results/param_sweep/euler_h3_hidden4"
QLNN_RUN = "results/qlnn_hybrid_h3"
SEEDS = (0, 1, 2, 3, 4)


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def _load_json(rel: str) -> dict:
    with (ROOT / rel).open() as f:
        return json.load(f)


def _seed_dirs(run: str) -> list[int]:
    base = ROOT / run
    if not base.exists():
        return []
    out = []
    for s in SEEDS:
        if (base / f"seed_{s}" / "predictions.npz").exists():
            out.append(s)
    return out


def _history(run: str, seed: int) -> dict[str, np.ndarray] | None:
    p = ROOT / run / f"seed_{seed}" / "history.csv"
    if not p.exists():
        return None
    cols: dict[str, list[float]] = {}
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.asarray(v) for k, v in cols.items()}


def _preds(run: str, seed: int) -> dict[str, np.ndarray] | None:
    p = ROOT / run / f"seed_{seed}" / "predictions.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


def _raw_bounds(run: str) -> tuple[float, float]:
    proto = _load_json(f"{run}/protocol.json")
    return float(proto["od_data_min"]), float(proto["od_data_max"])


def _to_raw(norm: np.ndarray, run: str) -> np.ndarray:
    lo, hi = _raw_bounds(run)
    return norm * (hi - lo) + lo


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / (name + '.{png,pdf}')}")


# ---------------------------------------------------------------------------
# T1.1 — learning curves with seed-variance band
# ---------------------------------------------------------------------------
def fig_learning_curves():
    if not (_exists(CLASSICAL_RUN) and _exists(QLNN_RUN)):
        print("SKIP fig_learning_curves: reference runs missing")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (run, name, col) in zip(
        axes,
        [(CLASSICAL_RUN, "Classical Liquid-ODE (H=4)", C_CLASSICAL),
         (QLNN_RUN, "QLNN (data_reuploading 4q/3L)", C_QLNN)],
    ):
        seeds = _seed_dirs(run)
        val_curves = []
        for s in seeds:
            h = _history(run, s)
            if h is None:
                continue
            ax.plot(h["epoch"], h["train_mse_norm"], color=col, alpha=0.18,
                    linewidth=0.8)
            ax.plot(h["epoch"], h["val_mse_norm"], color=col, alpha=0.30,
                    linewidth=0.8, linestyle="--")
            val_curves.append((h["epoch"], h["val_mse_norm"]))
        # Mean ± band over the common epoch prefix (early-stop ⇒ ragged).
        if val_curves:
            min_len = min(len(v) for _, v in val_curves)
            ep = val_curves[0][0][:min_len]
            stack = np.stack([v[:min_len] for _, v in val_curves])
            mu, sd = stack.mean(0), stack.std(0)
            ax.plot(ep, mu, color=col, linewidth=2.2, label="val mean")
            ax.fill_between(ep, mu - sd, mu + sd, color=col, alpha=0.20,
                            label="val ±1σ (seeds)")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        # Proxy legend entries for the thin lines.
        ax.plot([], [], color=col, alpha=0.4, linewidth=0.8, label="train (per seed)")
        ax.plot([], [], color=col, alpha=0.4, linewidth=0.8, linestyle="--",
                label="val (per seed)")
        ax.legend(loc="upper right", frameon=True)
    axes[0].set_ylabel("MSE (normalized, log)")
    fig.suptitle("Training dynamics — the dynamic view of Claim 1 "
                 "(QLNN val band is far tighter across seeds)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_learning_curves")


# ---------------------------------------------------------------------------
# T1.2 — forecast trajectory overlay
# ---------------------------------------------------------------------------
def fig_forecast_trajectory():
    if not (_exists(CLASSICAL_RUN) and _exists(QLNN_RUN)):
        print("SKIP fig_forecast_trajectory: reference runs missing")
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, (run, name, col) in zip(
        axes,
        [(CLASSICAL_RUN, "Classical Liquid-ODE (H=4)", C_CLASSICAL),
         (QLNN_RUN, "QLNN (data_reuploading 4q/3L)", C_QLNN)],
    ):
        seeds = _seed_dirs(run)
        preds = [_preds(run, s) for s in seeds]
        preds = [p for p in preds if p is not None]
        if not preds:
            continue
        y_true = _to_raw(preds[0]["test_y_true_norm"], run)
        idx = np.arange(len(y_true))
        stack = np.stack([_to_raw(p["test_y_pred_norm"], run) for p in preds])
        mu, sd = stack.mean(0), stack.std(0)
        # Persistence reference (od_last) — the residual baseline.
        od_last = _to_raw(preds[0]["test_od_last_norm"], run)
        ax.plot(idx, y_true, color="black", linewidth=1.8, label="actual OD")
        ax.plot(idx, od_last, color=C_NULL, linewidth=1.0, linestyle=":",
                label="persistence (OD$_{t}$)")
        ax.plot(idx, mu, color=col, linewidth=1.8, label=f"{name} mean")
        ax.fill_between(idx, mu - sd, mu + sd, color=col, alpha=0.25,
                        label="±1σ (5 seeds)")
        ax.set_ylabel("OD (raw)")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", frameon=True, ncol=2)
    axes[1].set_xlabel("test window index (chronological, stride 1)")
    fig.suptitle("Forecast trajectory — actual vs predicted OD on the "
                 "held-out test window (h=3)", y=1.01)
    fig.tight_layout()
    _save(fig, "fig_forecast_trajectory")


# ---------------------------------------------------------------------------
# T1.3 — predicted vs actual calibration
# ---------------------------------------------------------------------------
def fig_pred_vs_actual():
    if not (_exists(CLASSICAL_RUN) and _exists(QLNN_RUN)):
        print("SKIP fig_pred_vs_actual: reference runs missing")
        return

    def _r2(yt, yp):
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, (run, name, col) in zip(
        axes,
        [(CLASSICAL_RUN, "Classical Liquid-ODE (H=4)", C_CLASSICAL),
         (QLNN_RUN, "QLNN (data_reuploading 4q/3L)", C_QLNN)],
    ):
        seeds = _seed_dirs(run)
        preds = [p for p in (_preds(run, s) for s in seeds) if p is not None]
        if not preds:
            continue
        yt = _to_raw(preds[0]["test_y_true_norm"], run)
        yp = np.stack([_to_raw(p["test_y_pred_norm"], run) for p in preds]).mean(0)
        ax.scatter(yt, yp, s=28, color=col, edgecolor="black", linewidth=0.4,
                   alpha=0.8)
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lim, lim, color="black", linestyle="--", linewidth=1,
                label="y = ŷ")
        ax.set_title(f"{name}\nR² = {_r2(yt, yp):.3f} (seed-mean prediction)")
        ax.set_xlabel("actual OD (raw)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", frameon=True)
    axes[0].set_ylabel("predicted OD (raw)")
    fig.suptitle("Predicted-vs-actual calibration (h=3 test window)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_pred_vs_actual")


# ---------------------------------------------------------------------------
# T1.4 — residual analysis
# ---------------------------------------------------------------------------
def fig_residual_analysis():
    if not _exists(QLNN_RUN):
        print("SKIP fig_residual_analysis: QLNN run missing")
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for r, (run, name, col) in enumerate(
        [(CLASSICAL_RUN, "Classical H=4", C_CLASSICAL),
         (QLNN_RUN, "QLNN 4q/3L", C_QLNN)],
    ):
        seeds = _seed_dirs(run)
        preds = [p for p in (_preds(run, s) for s in seeds) if p is not None]
        if not preds:
            continue
        yt = _to_raw(preds[0]["test_y_true_norm"], run)
        yp = np.stack([_to_raw(p["test_y_pred_norm"], run) for p in preds]).mean(0)
        res = yp - yt
        idx = np.arange(len(res))

        # (a) residual vs time
        ax = axes[r, 0]
        ax.axhline(0, color="black", linewidth=0.6)
        ax.plot(idx, res, color=col, linewidth=1.0, marker="o", markersize=2.5)
        ax.set_title(f"({chr(97 + r * 3)}) {name} — residual vs time")
        ax.set_xlabel("test window index"); ax.set_ylabel("residual (raw OD)")
        ax.grid(True, alpha=0.3)

        # (b) histogram + normal QQ
        ax = axes[r, 1]
        ax.hist(res, bins=20, color=col, alpha=0.55, edgecolor="black",
                linewidth=0.4, density=True)
        xs = np.linspace(res.min(), res.max(), 100)
        mu, sd = res.mean(), res.std()
        if sd > 0:
            ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) /
                    (sd * np.sqrt(2 * np.pi)), color="black", linewidth=1.2,
                    label=f"N({mu:.3f},{sd:.3f})")
        ax.set_title(f"({chr(98 + r * 3)}) {name} — residual histogram")
        ax.set_xlabel("residual (raw OD)"); ax.set_ylabel("density")
        ax.legend(frameon=True); ax.grid(True, alpha=0.3)

        # (c) autocorrelation
        ax = axes[r, 2]
        rc = res - res.mean()
        denom = np.sum(rc ** 2)
        max_lag = min(20, len(rc) - 1)
        acf = [1.0 if denom == 0 else
               np.sum(rc[:len(rc) - k] * rc[k:]) / denom
               for k in range(max_lag + 1)]
        ax.bar(range(max_lag + 1), acf, color=col, alpha=0.7)
        ci = 1.96 / np.sqrt(len(res))
        ax.axhline(ci, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(-ci, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"({chr(99 + r * 3)}) {name} — residual ACF")
        ax.set_xlabel("lag"); ax.set_ylabel("autocorr")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Residual analysis — structure in the residuals reveals "
                 "missed dynamics (h=3, seed-mean)", y=1.01)
    fig.tight_layout()
    _save(fig, "fig_residual_analysis")


# ---------------------------------------------------------------------------
# T1.5 — paired-bootstrap distributions (sample-efficiency regime)
# ---------------------------------------------------------------------------
def fig_paired_bootstrap():
    fractions = [10, 25, 50, 100]
    # Cited verdicts from PAPER_SUMMARY (paired bootstrap on test abs-error).
    cited = {10: "p=0.015", 25: "p=0.002", 50: "p=0.226", 100: "p=0.029"}
    missing = [pct for pct in fractions
               if not _exists(f"results/sample_efficiency/qlnn_h3_pct{pct}")]
    if missing:
        print(f"SKIP fig_paired_bootstrap: sample_efficiency fractions "
              f"missing {missing}")
        return
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, pct in zip(axes, fractions):
        crun = f"results/sample_efficiency/classical_h4_h3_pct{pct}"
        qrun = f"results/sample_efficiency/qlnn_h3_pct{pct}"
        cseeds = _seed_dirs(crun)
        qseeds = _seed_dirs(qrun)
        cp = [p for p in (_preds(crun, s) for s in cseeds) if p is not None]
        qp = [p for p in (_preds(qrun, s) for s in qseeds) if p is not None]
        if not cp or not qp:
            continue
        yt = cp[0]["test_y_true_norm"]
        c_err = np.abs(np.stack([p["test_y_pred_norm"] for p in cp]).mean(0) - yt)
        q_err = np.abs(np.stack([p["test_y_pred_norm"] for p in qp]).mean(0) - yt)
        diff = q_err - c_err  # < 0 ⇒ QLNN better
        n = len(diff)
        boot = np.array([rng.choice(diff, n, replace=True).mean()
                         for _ in range(5000)])
        ax.hist(boot, bins=40, color=C_QLNN, alpha=0.65, edgecolor="black",
                linewidth=0.3)
        ax.axvline(0, color="black", linewidth=1.4, label="null (no diff)")
        ax.axvline(boot.mean(), color=C_CLASSICAL, linewidth=1.6,
                   linestyle="--", label=f"mean Δ={boot.mean():+.4f}")
        side = "QLNN better ←" if boot.mean() < 0 else "→ classical better"
        ax.set_title(f"{pct}% data  ({cited[pct]})\n{side}")
        ax.set_xlabel("bootstrap mean (|QLNN|−|classical|)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fontsize=7)
    axes[0].set_ylabel("bootstrap count")
    fig.suptitle("Paired-bootstrap of per-sample test error "
                 "(QLNN − classical); mass left of 0 ⇒ QLNN wins", y=1.03)
    fig.tight_layout()
    _save(fig, "fig_paired_bootstrap")


# ---------------------------------------------------------------------------
# T1.6 — per-seed strip plot
# ---------------------------------------------------------------------------
def fig_seed_strip():
    if not (_exists(CLASSICAL_RUN) and _exists(QLNN_RUN)):
        print("SKIP fig_seed_strip: reference runs missing")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, label in zip(
        axes, ("mae_raw", "r2_raw"), ("Test MAE (raw OD)", "Test R²")
    ):
        for x, (run, name, col) in enumerate(
            [(CLASSICAL_RUN, "Classical\nH=4", C_CLASSICAL),
             (QLNN_RUN, "QLNN\n4q/3L", C_QLNN)],
        ):
            vals = []
            for s in _seed_dirs(run):
                m = _load_json(f"{run}/seed_{s}/metrics.json")
                vals.append(m["test"][metric])
            vals = np.asarray(vals)
            jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.12
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=60, color=col,
                       edgecolor="black", linewidth=0.5, zorder=3,
                       label=f"{name.strip()} seeds")
            mu, sd = vals.mean(), vals.std()
            ci = 1.96 * sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ax.errorbar(x, mu, yerr=ci, fmt="_", color="black", markersize=28,
                        capsize=8, elinewidth=2, zorder=4)
            ax.text(x, mu, f"  {mu:.4f}\n  ±{sd:.4f}", va="center",
                    fontsize=8, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Classical H=4", "QLNN 4q/3L"])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-seed transparency — every seed shown (not just "
                 "mean±CI); QLNN's tight MAE cluster IS Claim 1", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_seed_strip")


# ---------------------------------------------------------------------------
# T1.7 — all 4 ansatz circuit diagrams
# ---------------------------------------------------------------------------
def fig_all_circuit_diagrams():
    try:
        import pennylane as qml
        from qlnn_.circuits import AnsatzConfig, build
    except Exception as e:
        print(f"SKIP fig_all_circuit_diagrams: import failed ({e})")
        return
    families = ["data_reuploading", "hardware_efficient",
                "strongly_entangling", "brickwall"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    rng = np.random.default_rng(0)
    for ax, fam in zip(axes.flatten(), families):
        circ = build(AnsatzConfig(name=fam, num_qubits=4, num_layers=2))
        w = 0.1 * rng.standard_normal(circ.weight_shape)
        x = np.linspace(-0.5, 0.5, 4)
        try:
            qml.draw_mpl(circ._qnode, style="pennylane")(x, w)
            tmp = OUT / f"_tmp_{fam}.png"
            plt.savefig(tmp, dpi=150, bbox_inches="tight")
            plt.close()
            ax.imshow(plt.imread(tmp))
            tmp.unlink()
        except Exception as e:  # pragma: no cover
            ax.text(0.5, 0.5, f"{fam}\n(draw failed: {e})", ha="center")
        ax.axis("off")
        ax.set_title(f"{fam} (4 qubits, 2 layers)", fontsize=10)
    fig.suptitle("The four searched ansätze — gate structure comparison",
                 y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_all_circuit_diagrams")


# ===========================================================================
# T2 — Option-B narrative (renders once the O-2 sweep + summarizer land)
# ===========================================================================
#
# Inputs:
#   results/option_b/option_b_table.json   (scripts/summarize_option_b.py)
#   results/baseline_lock.json             (the G1/G2 gate values)
# Optional context overlay:
#   results/circuit_search/circuit_search_table.json (prior search points)

_REGIME_ORDER = ["R0_control", "R1_weight_decay", "R2_physics_prior",
                 "R3_smooth_convergence"]
_CIRCUIT_COLOR = {
    "se_6q3l": "#009E73", "dr_4q3l": C_QLNN, "he_4q3l": "#E69F00",
}


def _option_b_rows() -> list[dict] | None:
    p = ROOT / "results" / "option_b" / "option_b_table.json"
    if not p.exists():
        return None
    with p.open() as f:
        rows = json.load(f)
    return rows or None


def _gates() -> tuple[float, float]:
    """(G1 MAE bar, G2 σ gate) from the frozen baseline lock."""
    lock = _load_json("results/baseline_lock.json")
    g1 = lock["classical"]["matched_param_H4"]["test_mae"]["mean"]
    g2 = 0.5 * lock["claim1_sigma_ratio"]["sigma_classical_H4"]
    return g1, g2


# ---------------------------------------------------------------------------
# T2.1 — accuracy↔variance frontier with the G1/G2 feasible box
# ---------------------------------------------------------------------------
def fig_accuracy_variance_frontier():
    rows = _option_b_rows()
    if rows is None:
        print("SKIP fig_accuracy_variance_frontier: run the O-2 sweep + "
              "scripts/summarize_option_b.py first")
        return
    g1, g2 = _gates()

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # Prior-search context (faint grey) if available.
    prior = ROOT / "results" / "circuit_search" / "circuit_search_table.json"
    if prior.exists():
        with prior.open() as f:
            for r in json.load(f):
                mu = r["test_mae_raw"]["mean"]
                sd = r["test_mae_raw"].get("std")
                if mu is None or sd is None:
                    continue
                ax.scatter(mu, sd, s=22, color=C_NULL, alpha=0.35, zorder=1)
        ax.scatter([], [], s=22, color=C_NULL, alpha=0.35,
                   label="prior search (proxy / promoted)")

    # Option-B points, colored by circuit, regime as marker.
    regime_marker = {"R0_control": "o", "R1_weight_decay": "s",
                     "R2_physics_prior": "^", "R3_smooth_convergence": "D"}
    for r in rows:
        ax.scatter(r["test_mae_mean"], r["test_mae_sigma"],
                   s=95, color=_CIRCUIT_COLOR.get(r["circuit"], "grey"),
                   marker=regime_marker.get(r["regime"], "o"),
                   edgecolor="black", linewidth=0.6, zorder=4)

    # Feasible box: MAE < g1 AND σ ≤ g2.
    ax.axvline(g1, color=C_CLASSICAL, linestyle="--", linewidth=1.4,
               label=f"G1: MAE = {g1:.4f} (classical H=4)")
    ax.axhline(g2, color="purple", linestyle="--", linewidth=1.4,
               label=f"G2: σ = {g2:.5f} (½·σ_classical)")
    xlo = min(min(r["test_mae_mean"] for r in rows), g1) - 0.003
    ylo = -0.001
    ax.add_patch(plt.Rectangle((xlo, ylo), g1 - xlo, g2 - ylo,
                               facecolor="green", alpha=0.10, zorder=0))
    ax.text(xlo + 0.0015, g2 * 0.5, "FEASIBLE\n(Option-B)", fontsize=9,
            color="green", fontweight="bold", va="center")

    # Legend proxies for circuit color + regime marker.
    for ck, col in _CIRCUIT_COLOR.items():
        ax.scatter([], [], s=95, color=col, edgecolor="black",
                   linewidth=0.6, label=f"circuit: {ck}")
    for rk, mk in regime_marker.items():
        ax.scatter([], [], s=95, color="white", marker=mk,
                   edgecolor="black", linewidth=0.6,
                   label=f"regime: {rk}")

    ax.set_xlabel("Test MAE at h=3 (raw OD, 3-seed proxy)")
    ax.set_ylabel("σ of test MAE across seeds")
    ax.set_title("Accuracy↔variance frontier — does any circuit×regime "
                 "land in the Option-B feasible box?")
    ax.legend(loc="upper right", fontsize=7, frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_accuracy_variance_frontier")


# ---------------------------------------------------------------------------
# T2.2 — regularization-regime arrows (R0 → {R1,R2,R3}) per circuit
# ---------------------------------------------------------------------------
def fig_regularization_arrows():
    rows = _option_b_rows()
    if rows is None:
        print("SKIP fig_regularization_arrows: run the O-2 sweep + "
              "summarizer first")
        return
    g1, g2 = _gates()
    by_circuit: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_circuit.setdefault(r["circuit"], {})[r["regime"]] = r

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for ck, regimes in by_circuit.items():
        col = _CIRCUIT_COLOR.get(ck, "grey")
        r0 = regimes.get("R0_control")
        if r0 is None:
            continue
        x0, y0 = r0["test_mae_mean"], r0["test_mae_sigma"]
        ax.scatter(x0, y0, s=120, color=col, edgecolor="black",
                   linewidth=0.8, zorder=4, marker="*")
        ax.annotate(f"{ck} R0", (x0, y0), fontsize=8, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")
        for rk in ("R1_weight_decay", "R2_physics_prior",
                   "R3_smooth_convergence"):
            rr = regimes.get(rk)
            if rr is None:
                continue
            x1, y1 = rr["test_mae_mean"], rr["test_mae_sigma"]
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.6,
                                alpha=0.8))
            ax.scatter(x1, y1, s=55, color=col, edgecolor="black",
                       linewidth=0.4, zorder=4)
            ax.annotate(rk.split("_")[0], (x1, y1), fontsize=7,
                        xytext=(3, -9), textcoords="offset points")

    ax.axvline(g1, color=C_CLASSICAL, linestyle="--", linewidth=1.2)
    ax.axhline(g2, color="purple", linestyle="--", linewidth=1.2)
    ax.add_patch(plt.Rectangle(
        (ax.get_xlim()[0], -0.001), g1 - ax.get_xlim()[0], g2 + 0.001,
        facecolor="green", alpha=0.10, zorder=0))
    ax.set_xlabel("Test MAE at h=3 (raw OD, 3-seed proxy)")
    ax.set_ylabel("σ of test MAE across seeds")
    ax.set_title("Does regularization pull circuits toward the feasible "
                 "box? R0→{R1,R2,R3} per circuit")
    ax.legend(handles=[plt.Line2D([], [], color=c, marker="*", linestyle="",
                                  markersize=12, label=ck)
                       for ck, c in _CIRCUIT_COLOR.items()],
              loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_regularization_arrows")


# ---------------------------------------------------------------------------
# T2.3 — circuit × regime penalized-objective heatmap
# ---------------------------------------------------------------------------
def fig_circuit_regime_heatmap():
    rows = _option_b_rows()
    if rows is None:
        print("SKIP fig_circuit_regime_heatmap: run the O-2 sweep + "
              "summarizer first")
        return
    circuits = sorted({r["circuit"] for r in rows})
    regimes = [rk for rk in _REGIME_ORDER
               if any(r["regime"] == rk for r in rows)]
    grid = np.full((len(circuits), len(regimes)), np.nan)
    g1pass = np.zeros_like(grid, dtype=bool)
    g2pass = np.zeros_like(grid, dtype=bool)
    for r in rows:
        i = circuits.index(r["circuit"])
        j = regimes.index(r["regime"]) if r["regime"] in regimes else None
        if j is None:
            continue
        grid[i, j] = r["penalized"]
        g1pass[i, j] = r["g1_accuracy_pass"]
        g2pass[i, j] = r["g2_sigma_pass"]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.set_yticks(range(len(circuits)))
    ax.set_yticklabels(circuits)
    for i in range(len(circuits)):
        for j in range(len(regimes)):
            if np.isnan(grid[i, j]):
                continue
            marks = ("✓G1" if g1pass[i, j] else "✗G1") + " " + \
                    ("✓G2" if g2pass[i, j] else "✗G2")
            ax.text(j, i, f"{grid[i, j]:.4f}\n{marks}", ha="center",
                    va="center", fontsize=8,
                    color="white" if grid[i, j] > np.nanmean(grid) else "black")
    fig.colorbar(im, ax=ax, label="penalized objective (lower = better)")
    ax.set_title("Circuit × regime — penalized objective "
                 "(MAE + 5·relu(σ−gate)); ✓G1✓G2 = Option-B candidate")
    fig.tight_layout()
    _save(fig, "fig_circuit_regime_heatmap")


# ---------------------------------------------------------------------------
# T2.4 (capstone) — master all-vs-all comparison
# ---------------------------------------------------------------------------
def fig_master_comparison():
    """Every classical + quantum configuration on shared (MAE, σ) axes
    with the G1/G2 feasible box. Reads results/master_comparison.json
    (scripts/build_master_comparison.py). One glance: where the whole
    population sits vs the Option-B target.
    """
    p = ROOT / "results" / "master_comparison.json"
    if not p.exists():
        print("SKIP fig_master_comparison: run "
              "scripts/build_master_comparison.py first")
        return
    with p.open() as f:
        rows = json.load(f)
    g1, g2 = _gates()

    style = {
        "classical_sweep":  (C_CLASSICAL, "o", "classical sweep"),
        "qlnn_reference":   ("black", "*", "QLNN reference"),
        "option_b":         (C_QLNN, "D", "Option-B circuit×regime"),
    }

    def _bucket(src: str) -> str:
        if src.startswith("prior_search"):
            return "prior_search"
        return src

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    seen = set()
    for r in rows:
        mae, sd = r["test_mae_mean"], r["test_mae_sigma"]
        if mae is None:
            continue
        b = _bucket(r["source"])
        col, mk, lab = style.get(
            b, (C_NULL, ".", "prior search (proxy/promoted)"))
        y = sd if sd is not None else 0.0
        ax.scatter(mae, y, s=(70 if b != "prior_search" else 26),
                   color=col, marker=mk,
                   alpha=(0.9 if b != "prior_search" else 0.35),
                   edgecolor="black" if b != "prior_search" else "none",
                   linewidth=0.5, zorder=(4 if b != "prior_search" else 1),
                   label=lab if lab not in seen else None)
        seen.add(lab)
        if sd is None:
            # 1-seed proxy: mark σ as undefined on a baseline rug.
            ax.scatter(mae, 0.0, s=18, color=col, marker="|",
                       alpha=0.4, zorder=1)

    ax.axvline(g1, color=C_CLASSICAL, linestyle="--", linewidth=1.4,
               label=f"G1: MAE={g1:.4f}")
    ax.axhline(g2, color="purple", linestyle="--", linewidth=1.4,
               label=f"G2: σ={g2:.5f}")
    x0 = ax.get_xlim()[0]
    ax.add_patch(plt.Rectangle((x0, -0.0005), g1 - x0, g2 + 0.0005,
                               facecolor="green", alpha=0.10, zorder=0))
    ax.text(x0 + 0.001, g2 * 0.5, "FEASIBLE\n(Option-B)", fontsize=9,
            color="green", fontweight="bold", va="center")

    n_pass = sum(r["G1_accuracy_pass"] and r["G2_reproducibility_pass"]
                 for r in rows)
    ax.set_xlabel("Test MAE at h=3 (raw OD)")
    ax.set_ylabel("σ of test MAE across seeds (0 = single-seed proxy)")
    ax.set_title(f"Master comparison — {len(rows)} configs; "
                 f"{n_pass} in the Option-B feasible box")
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_master_comparison")


# ===========================================================================
# SUPPLEMENT — full circuit-topology gallery (all 28 distinct topologies)
# ===========================================================================
#
# Standard QML main-text practice = family templates + a parametric
# search-space table (scripts/build_circuit_search_space.py). This is the
# COMPLETENESS supplement: every distinct topology drawn via qml.draw_mpl,
# one figure per ansatz family. Reads
# results/circuit_search_space/topologies.json (regime is a training knob,
# not topology — it does not appear here).

def _gallery_one_family(family: str, topos: list[dict]):
    try:
        import pennylane as qml
        from qlnn_.circuits import AnsatzConfig, build
    except Exception as e:
        print(f"SKIP fig_circuit_gallery_{family}: import failed ({e})")
        return
    rng = np.random.default_rng(0)
    n = len(topos)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n), squeeze=False)
    for ax, t in zip(axes[:, 0], topos):
        params: dict = {"encoding": t["encoding"]}
        if t["entanglement"] != "template":
            params["entanglement"] = t["entanglement"]
        if family == "brickwall":
            params["reupload"] = False
        try:
            circ = build(AnsatzConfig(name=family,
                                      num_qubits=t["num_qubits"],
                                      num_layers=t["num_layers"],
                                      params=params))
            w = 0.1 * rng.standard_normal(circ.weight_shape)
            x = np.linspace(-0.5, 0.5, t["num_qubits"])
            qml.draw_mpl(circ._qnode, style="pennylane")(x, w)
            tmp = OUT / f"_tmp_gal_{family}_{t['num_qubits']}_{t['num_layers']}.png"
            plt.savefig(tmp, dpi=130, bbox_inches="tight")
            plt.close()
            ax.imshow(plt.imread(tmp))
            tmp.unlink()
        except Exception as e:  # pragma: no cover
            ax.text(0.5, 0.5, f"draw failed: {e}", ha="center")
        ax.axis("off")
        ent = (t["entanglement"] if t["entanglement"] != "template"
               else "template-fixed")
        ax.set_title(f"{family}  ·  {t['num_qubits']}q / {t['num_layers']}L  "
                     f"·  encode={t['encoding']}  ·  ent={ent}", fontsize=9)
    fig.suptitle(f"Circuit gallery — {family} "
                 f"({n} distinct topologies)", y=1.005, fontsize=12)
    fig.tight_layout()
    _save(fig, f"fig_circuit_gallery_{family}")


def fig_circuit_gallery():
    """Supplement: render every distinct circuit topology, one figure per
    ansatz family. Gracefully skips if the search-space table is absent."""
    p = ROOT / "results" / "circuit_search_space" / "topologies.json"
    if not p.exists():
        print("SKIP fig_circuit_gallery: run "
              "scripts/build_circuit_search_space.py first")
        return
    with p.open() as f:
        payload = json.load(f)
    by_fam: dict[str, list[dict]] = {}
    for t in payload["topologies"]:
        by_fam.setdefault(t["family"], []).append(t)
    for fam in sorted(by_fam):
        _gallery_one_family(fam, sorted(
            by_fam[fam],
            key=lambda t: (t["num_qubits"], t["num_layers"],
                           t["encoding"], t["entanglement"])))
    print(f"circuit gallery: {payload['n_distinct_topologies']} topologies "
          f"across {len(by_fam)} families")


# ===========================================================================
# T3 — quantum-trainability / expressivity (reads results/quantum_trainability)
# ===========================================================================
def _qt(name: str):
    p = ROOT / "results" / "quantum_trainability" / f"{name}.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def fig_expressibility():
    rows = _qt("expressibility")
    if rows is None:
        print("SKIP fig_expressibility: run "
              "scripts/analyze_quantum_trainability.py first")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    rows = sorted(rows, key=lambda r: (r["family"], r["num_qubits"],
                                       r["num_layers"]))
    labels = [f"{r['family'][:4]} {r['num_qubits']}q{r['num_layers']}L "
              f"{r['encoding']}/{r['entanglement'][:3]}" for r in rows]
    vals = [r["expressibility_kl_to_haar"] for r in rows]
    fam_col = {"data": C_QLNN, "hard": "#E69F00",
               "stro": "#009E73", "bric": "#CC79A7"}
    cols = [fam_col.get(r["family"][:4], C_NULL) for r in rows]
    ax.bar(range(len(rows)), vals, color=cols, edgecolor="black",
           linewidth=0.3)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("KL( P_circuit || P_Haar )  — lower = more expressive")
    ax.set_title("Expressibility (Sim et al. 2019) per circuit topology")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_expressibility")


def fig_entangling_capability():
    rows = _qt("entangling")
    if rows is None:
        print("SKIP fig_entangling_capability: run T3 analysis first")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    rows = sorted(rows, key=lambda r: (r["family"], r["num_qubits"],
                                       r["num_layers"]))
    labels = [f"{r['family'][:4]} {r['num_qubits']}q{r['num_layers']}L"
              for r in rows]
    mu = [r["meyer_wallach_Q_mean"] for r in rows]
    sd = [r["meyer_wallach_Q_std"] for r in rows]
    ax.bar(range(len(rows)), mu, yerr=sd, color=C_QLNN,
           edgecolor="black", linewidth=0.3, capsize=2)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Meyer-Wallach Q (0 = product, 1 = max entangled)")
    ax.set_title("Entangling capability per circuit topology")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_entangling_capability")


def fig_barren_plateau():
    rows = _qt("barren_plateau")
    if rows is None:
        print("SKIP fig_barren_plateau: run T3 analysis first")
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fams = sorted({r["family"] for r in rows})
    fam_col = {"data_reuploading": C_QLNN, "hardware_efficient": "#E69F00",
               "strongly_entangling": "#009E73", "brickwall": "#CC79A7"}
    for fam in fams:
        for depth in sorted({r["num_layers"] for r in rows
                             if r["family"] == fam}):
            pts = sorted([r for r in rows if r["family"] == fam
                          and r["num_layers"] == depth],
                         key=lambda r: r["num_qubits"])
            xs = [p["num_qubits"] for p in pts]
            ys = [p["grad_var"] for p in pts]
            ax.plot(xs, ys, marker="o", color=fam_col.get(fam, C_NULL),
                    alpha=0.4 + 0.15 * depth, linewidth=1.3,
                    label=f"{fam} L={depth}")
    ax.set_yscale("log")
    ax.set_xlabel("num_qubits")
    ax.set_ylabel("Var[∂⟨Z₀⟩/∂θ]  (log) — exp. decay = barren plateau")
    ax.set_title("Barren-plateau scaling — can these circuits be scaled?")
    ax.legend(fontsize=7, frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_barren_plateau")


def fig_fisher_spectrum():
    fs = _qt("fisher_spectrum")
    if fs is None:
        print("SKIP fig_fisher_spectrum: run T3 analysis first")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    c = fs.get("classical_H4") or {}
    q = fs.get("qlnn_h3") or {}
    labels, means, stds, cols = [], [], [], []
    if c:
        labels.append("Classical H=4"); means.append(c["mean"])
        stds.append(c["std"]); cols.append(C_CLASSICAL)
    if q:
        labels.append("QLNN h=3"); means.append(q["mean"])
        stds.append(q["std"]); cols.append(C_QLNN)
    ax.bar(range(len(labels)), means, yerr=stds, color=cols,
           edgecolor="black", capsize=6)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("d_norm (effective dimension proxy)")
    ax.set_title("Fisher / effective-dimension — expressivity vs variance\n"
                 "(full eigenspectrum pending --emit-spectrum recompute)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_fisher_spectrum")


T1 = [
    fig_learning_curves, fig_forecast_trajectory, fig_pred_vs_actual,
    fig_residual_analysis, fig_paired_bootstrap, fig_seed_strip,
    fig_all_circuit_diagrams,
]

T3 = [
    fig_expressibility, fig_entangling_capability,
    fig_barren_plateau, fig_fisher_spectrum,
]

SUPP = [fig_circuit_gallery]

T2 = [
    fig_accuracy_variance_frontier, fig_regularization_arrows,
    fig_circuit_regime_heatmap, fig_master_comparison,
]


if __name__ == "__main__":
    for fn in T1 + T2 + T3 + SUPP:
        fn()
    print(f"\nDiagnostic figures written to {OUT}/")
