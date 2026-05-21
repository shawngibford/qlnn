"""Render the P5 H1 verdict figure — the paper's headline.

Reads:
  results/p4_forecaster_rollout/{system}_{family}/seed_N/metrics.json
    (QLNN best-ansatz per cell)
  results/p5_matched_baselines/{system}_{family}/seed_N/metrics.json
    (plain_neuralode, plain_mlp, skyline)
  results/p5_h1_verdict/h1_analysis.json
    (THE outcome — CONFIRMED / FALSIFIED / INCONCLUSIVE)

Emits paper/figures/fig_p5_h1_verdict.{png,pdf} as a 3-panel figure:

  Top:    per-system relative-L2 bar chart — QLNN best vs Neural-ODE
          vs Classical MLP vs Skyline. Persistence floor as dashed
          gray reference. 3 systems, 4 bars per system.

  Middle: per-cell Δ = NeuralODE − QLNN scatter, colored by regime.
          Δ_smooth_mean + Δ_broad_mean drawn as dashed reference lines.
          The KEY visual: smooth and broad scatter separately on
          the y-axis; H1 predicts smooth above broad.

  Bottom: H1 verdict bar — Δ_smooth − Δ_broad with 95% CI as error
          bar. Annotated with the verdict outcome
          (CONFIRMED / FALSIFIED / INCONCLUSIVE) and the bootstrap
          reasoning verbatim.

Handles partial data gracefully (hatched placeholders for missing).
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
P4_IN = REPO_ROOT / "results" / "p4_forecaster_rollout"
P5_IN = REPO_ROOT / "results" / "p5_matched_baselines"
VERDICT_IN = REPO_ROOT / "results" / "p5_h1_verdict"
OUT = REPO_ROOT / "paper" / "figures"

SYSTEMS = ["lotka_volterra", "van_der_pol", "lorenz"]
QLNN_FAMILIES = [
    "data_reuploading", "hardware_efficient",
    "strongly_entangling", "brickwall",
]
BASELINE_FAMILIES = ["plain_neuralode", "plain_mlp", "skyline"]

REGIME = {
    "lotka_volterra": "smooth_periodic",
    "van_der_pol":    "smooth_periodic",
    "lorenz":         "broadband_multiscale",
}

# Wong palette.
COLOR = {
    "qlnn_best":       "#0072B2",
    "plain_neuralode": "#D55E00",
    "plain_mlp":       "#009E73",
    "skyline":         "#CC79A7",
    "persistence":     "gray",
}

SYSTEM_PRETTY = {
    "lotka_volterra": "Lotka-Volterra (smooth/periodic)",
    "van_der_pol":    "Van der Pol μ=5 (smooth/periodic)",
    "lorenz":         "Lorenz '63 (broadband/chaotic)",
}


def _load_p4_qlnn_best(system: str, seed: int) -> tuple[float | None, str | None]:
    """Return (best_relL2, best_family) for QLNN at this cell."""
    best = None
    best_fam = None
    for family in QLNN_FAMILIES:
        p = P4_IN / f"{system}_{family}" / f"seed_{seed}" / "metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        v = float(m["relative_l2"])
        if best is None or v < best:
            best = v
            best_fam = family
    return best, best_fam


def _load_baseline(system: str, family: str, seed: int) -> float | None:
    p = P5_IN / f"{system}_{family}" / f"seed_{seed}" / "metrics.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["relative_l2"])


def _load_baseline_summary(system: str, family: str) -> dict | None:
    p = P5_IN / f"{system}_{family}" / "seeds_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _persistence_floor(system: str, seed: int = 0) -> float | None:
    """Read any P4 cell's persistence_floor_relative_l2 — same per system."""
    for family in QLNN_FAMILIES:
        p = P4_IN / f"{system}_{family}" / f"seed_{seed}" / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            return float(m.get("persistence_floor_relative_l2", np.nan))
    return None


# ---------------------------------------------------------------------------
# Top: per-system relative-L2 bars
# ---------------------------------------------------------------------------


def _top_relL2_bars(ax) -> None:
    bar_w = 0.18
    xpos = np.arange(len(SYSTEMS))
    categories = ["qlnn_best", "plain_neuralode", "plain_mlp", "skyline"]
    pretty = {
        "qlnn_best": "QLNN best-ansatz",
        "plain_neuralode": "Neural-ODE (H1 contrast)",
        "plain_mlp": "Classical MLP",
        "skyline": "Skyline (upper bound)",
    }
    seeds = [0, 1, 2]

    for i, cat in enumerate(categories):
        means, errs, missing = [], [], []
        for system in SYSTEMS:
            vals = []
            for s in seeds:
                if cat == "qlnn_best":
                    v, _ = _load_p4_qlnn_best(system, s)
                else:
                    v = _load_baseline(system, cat, s)
                if v is not None:
                    vals.append(v)
            if not vals:
                means.append(np.nan)
                errs.append(0.0)
                missing.append(True)
            else:
                means.append(float(np.mean(vals)))
                errs.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                missing.append(False)

        offsets = (i - (len(categories) - 1) / 2) * bar_w
        x = xpos + offsets
        means_arr = np.asarray(means)
        plot_means = np.where(np.isnan(means_arr), 1e-3, means_arr)
        bars = ax.bar(x, plot_means, bar_w, yerr=errs, capsize=2,
                      color=COLOR[cat], label=pretty[cat],
                      edgecolor="black", linewidth=0.4)
        for j, miss in enumerate(missing):
            if miss:
                bars[j].set_hatch("////")
                bars[j].set_alpha(0.3)

    # Persistence floor reference lines per system.
    for j, system in enumerate(SYSTEMS):
        pf = _persistence_floor(system)
        if pf is not None and np.isfinite(pf):
            ax.hlines(pf, xpos[j] - 0.5, xpos[j] + 0.5,
                      colors=COLOR["persistence"], linestyles="--",
                      linewidth=1.0,
                      label="persistence floor" if j == 0 else None)

    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([SYSTEM_PRETTY[s] for s in SYSTEMS],
                       rotation=12, ha="right")
    ax.set_ylabel("relative-L² (log)")
    ax.set_title("Top: forecaster rollout relative-L² — QLNN best vs "
                 "matched baselines (3 seeds)", fontsize=11)
    ax.set_ylim(1e-3, 5)
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    ax.grid(True, axis="y", which="major", alpha=0.3)


# ---------------------------------------------------------------------------
# Middle: per-cell Δ scatter (the H1 visual)
# ---------------------------------------------------------------------------


def _middle_delta_scatter(ax) -> None:
    """Per-cell Δ = NeuralODE − QLNN, scatter by regime."""
    smooth_x, smooth_y, broad_x, broad_y = [], [], [], []
    smooth_label_set = set()
    broad_label_set = set()

    for system in SYSTEMS:
        for seed in (0, 1, 2):
            qlnn_v, _ = _load_p4_qlnn_best(system, seed)
            no_v = _load_baseline(system, "plain_neuralode", seed)
            if qlnn_v is None or no_v is None:
                continue
            delta = no_v - qlnn_v
            regime = REGIME[system]
            if regime == "smooth_periodic":
                smooth_x.append(f"{system[:3]}_{seed}")
                smooth_y.append(delta)
                smooth_label_set.add(system)
            else:
                broad_x.append(f"{system[:3]}_{seed}")
                broad_y.append(delta)
                broad_label_set.add(system)

    # Plot smooth on x positions 0..n_smooth-1, broad after a gap.
    n_s = len(smooth_y)
    n_b = len(broad_y)
    xs = list(range(n_s))
    xb = list(range(n_s + 1, n_s + 1 + n_b))

    ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    if smooth_y:
        ax.scatter(xs, smooth_y, s=80, color="#0072B2",
                   edgecolor="black", linewidth=0.6,
                   label=f"smooth/periodic (n={n_s})")
        mean_s = float(np.mean(smooth_y))
        ax.hlines(mean_s, -0.5, n_s - 0.5,
                  colors="#0072B2", linestyles="--", linewidth=1.5,
                  label=f"Δ_smooth = {mean_s:.3f}")
    if broad_y:
        ax.scatter(xb, broad_y, s=80, color="#D55E00",
                   edgecolor="black", linewidth=0.6,
                   label=f"broadband/chaotic (n={n_b})")
        mean_b = float(np.mean(broad_y))
        ax.hlines(mean_b, n_s + 0.5, n_s + 0.5 + n_b,
                  colors="#D55E00", linestyles="--", linewidth=1.5,
                  label=f"Δ_broad = {mean_b:.3f}")

    all_x = xs + xb
    all_labels = smooth_x + broad_x
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Δ = Neural-ODE relL² − QLNN relL²\n(positive ⇒ QLNN better)")
    ax.set_title("Middle: per-cell Δ scatter — H1 predicts smooth > broad",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Bottom: H1 verdict bar + outcome annotation
# ---------------------------------------------------------------------------


def _bottom_h1_verdict(ax) -> None:
    p = VERDICT_IN / "h1_analysis.json"
    if not p.exists():
        ax.text(0.5, 0.5, "H1 verdict not yet computed\n"
                "(awaiting sweep completion)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="grey", style="italic")
        ax.set_title("Bottom: H1 verdict", fontsize=10)
        return

    verdict = json.loads(p.read_text())
    outcome = verdict["outcome"]
    color = {"CONFIRMED": "#009E73",
             "FALSIFIED": "#D55E00",
             "INCONCLUSIVE": "#CC79A7"}.get(outcome, "gray")

    if verdict["bootstrap"] is None:
        ax.text(0.5, 0.5,
                f"H1 OUTCOME: {outcome}\n\n{verdict['reasoning']}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=color, weight="bold",
                wrap=True)
        ax.set_title("Bottom: H1 verdict", fontsize=10)
        return

    b = verdict["bootstrap"]
    point = b["delta_diff_mean"]
    ci_lo, ci_hi = b["ci_low"], b["ci_high"]
    err_lo = point - ci_lo
    err_hi = ci_hi - point

    ax.errorbar([0], [point], yerr=[[err_lo], [err_hi]],
                fmt="o", color=color, markersize=14,
                ecolor=color, capsize=8, capthick=2, linewidth=2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Δ_smooth − Δ_broad\n(with 95% paired-bootstrap CI)")

    title = (f"Bottom: H1 verdict — {outcome}\n"
             f"Δ_smooth − Δ_broad = {point:.4f}, "
             f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    ax.set_title(title, fontsize=10, color=color)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotation: which side of zero the CI is on.
    if ci_lo > 0:
        annot = "CI > 0: smooth-regime advantage IS larger (H1 prediction holds)"
    elif ci_hi < 0:
        annot = "CI < 0: broadband-regime advantage IS larger (anti-H1)"
    else:
        annot = "CI contains 0: no significant regime-dependent gap"
    ax.text(0, ci_hi + 0.05 * abs(ci_hi - ci_lo) if ci_hi != ci_lo else ci_hi,
            annot, ha="center", va="bottom", fontsize=8, color=color,
            style="italic")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    fig = plt.figure(figsize=(13, 14))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55,
                           height_ratios=[1.0, 1.0, 0.9])

    ax_top = fig.add_subplot(gs[0])
    _top_relL2_bars(ax_top)

    ax_mid = fig.add_subplot(gs[1])
    _middle_delta_scatter(ax_mid)

    ax_bot = fig.add_subplot(gs[2])
    _bottom_h1_verdict(ax_bot)

    # Read verdict to populate suptitle.
    verdict_path = VERDICT_IN / "h1_analysis.json"
    if verdict_path.exists():
        v = json.loads(verdict_path.read_text())
        outcome = v["outcome"]
    else:
        outcome = "PENDING"
    fig.suptitle(
        f"P5 — H1 VERDICT: {outcome}\n"
        f"Pre-reg §7: Δ = NeuralODE_relL² − QLNN_best_relL² "
        f"(positive ⇒ QLNN better);  H1 = (Δ_smooth − Δ_broad) "
        f"95% CI excludes 0",
        y=0.995, fontsize=12, weight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fp = OUT / f"fig_p5_h1_verdict.{ext}"
        fig.savefig(fp)
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
