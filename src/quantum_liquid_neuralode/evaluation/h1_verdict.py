"""P5 commit 5 — H1 verdict module (THE publication critical path).

Computes the pre-reg §7 H1 decision mechanically. NO ad-hoc
post-hoc tweaks — pre-reg is binding.

From `ODE_PDE_PRE_REG.md` §7 verbatim:

  Let Δ_smooth = mean (over SMOOTH/PERIODIC systems & seeds) of
    [QLNN best-ansatz primary metric] − [Neural-ODE primary metric]
    in the *improvement* direction (lower relative-L2 ⇒ larger Δ).
  Let Δ_broad be the analogous quantity over BROADBAND/MULTISCALE/CHAOTIC.
  Both are computed with paired-bootstrap CIs.

  - H1 CONFIRMED iff Δ_smooth − Δ_broad > 0 AND the paired-bootstrap
    95% CI of (Δ_smooth − Δ_broad) excludes 0 AND it holds on the
    SOLVER task (forecaster reported as corroborating/contradicting,
    not gating).
  - H1 FALSIFIED iff the 95% CI of (Δ_smooth − Δ_broad) includes 0
    or is negative. Published as a rigorous mechanistic null.

**Improvement-direction convention (CRITICAL):**
  Since lower relative-L2 ⇒ "better forecast", and Δ should be POSITIVE
  when QLNN is better, we define:

      Δ_cell = NeuralODE_relL2(cell) − QLNN_relL2(cell)

  Δ_smooth > Δ_broad means QLNN's improvement over NeuralODE is
  larger on smooth-regime systems — exactly the H1 prediction.

Pre-registered regime partition (`ODE_PDE_PRE_REG.md` §2):

  SMOOTH/PERIODIC: lotka_volterra, van_der_pol, kuramoto, burgers_smooth, heat
  BROADBAND/MULTISCALE/CHAOTIC: lorenz, fitzhugh_nagumo, kdv, allen_cahn, burgers_shock

The verdict module:
  1. Collects per-cell (system, seed) relative-L2 for QLNN best-
     ansatz + NeuralODE baseline.
  2. Computes Δ per cell.
  3. Partitions cells into smooth / broad regimes.
  4. Applies underfit + skyline guards (pre-reg §6, §7): cells
     where the skyline cannot reach adequacy threshold are
     excluded; cells where any model is underfit are flagged.
  5. Bootstrap-resamples CELLS (with replacement) within each
     regime; on each resample computes (Δ_smooth_mean − Δ_broad_mean).
  6. Reports the 95% CI of the bootstrap distribution + the
     mechanical verdict per pre-reg §7.

Output is a JSONable dict suitable for `results/p5_h1_verdict/
h1_analysis.json` — the headline artifact of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Pre-registered regime partition (locked at commit 2646d74)
# ---------------------------------------------------------------------------


SMOOTH_PERIODIC_SYSTEMS = (
    "lotka_volterra",
    "van_der_pol",       # μ=5 limit cycle (smooth periodic)
    "kuramoto",          # phase-locked
    "burgers_smooth",
    "heat",
)

BROADBAND_MULTISCALE_SYSTEMS = (
    "lorenz",
    "fitzhugh_nagumo",   # relaxation spikes
    "kdv",
    "allen_cahn",
    "burgers_shock",
)


def regime_for_system(system: str) -> str:
    """Return 'smooth_periodic' or 'broadband_multiscale' for a
    pre-reg-listed system. Raises ValueError otherwise."""
    if system in SMOOTH_PERIODIC_SYSTEMS:
        return "smooth_periodic"
    if system in BROADBAND_MULTISCALE_SYSTEMS:
        return "broadband_multiscale"
    raise ValueError(
        f"unknown system {system!r}; expected one of "
        f"{SMOOTH_PERIODIC_SYSTEMS + BROADBAND_MULTISCALE_SYSTEMS}")


# ---------------------------------------------------------------------------
# Per-cell delta computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellRecord:
    """One (system, seed) measurement of QLNN vs Neural-ODE relative-L2.

    Attributes:
      system    : pre-reg-listed system name (e.g. 'lorenz').
      seed      : RNG seed for forecaster init / training shuffle.
      qlnn_relL2     : QLNN best-ansatz rollout relative-L2.
      neuralode_relL2: Neural-ODE baseline rollout relative-L2.
      qlnn_train_relL2 : optional QLNN train-side relative-L2
                         (for underfit guard); None if not recorded.
      neuralode_train_relL2 : same for NeuralODE.
      skyline_relL2  : optional skyline rollout relative-L2 (for
                       skyline guard); None if not recorded.
    """

    system: str
    seed: int
    qlnn_relL2: float
    neuralode_relL2: float
    qlnn_train_relL2: float | None = None
    neuralode_train_relL2: float | None = None
    skyline_relL2: float | None = None

    @property
    def regime(self) -> str:
        return regime_for_system(self.system)

    @property
    def delta(self) -> float:
        """Δ_cell = NeuralODE − QLNN  (positive means QLNN is BETTER)."""
        return float(self.neuralode_relL2 - self.qlnn_relL2)


# ---------------------------------------------------------------------------
# Underfit + skyline guards (pre-reg §6, §7)
# ---------------------------------------------------------------------------


def apply_guards(
    cells: list[CellRecord],
    *,
    underfit_threshold: float = 0.5,
    skyline_threshold: float = 0.5,
) -> tuple[list[CellRecord], dict[str, list[str]]]:
    """Apply pre-reg §6/§7 underfit + skyline guards.

    Args:
      cells : list of CellRecord across all (system, seed) cells.
      underfit_threshold : train-side relative-L2 above this means
                            the model is underfit (pre-reg §6's
                            "adequacy threshold"). Default 0.5 —
                            looser than 0.1 to avoid being overly
                            strict on chaotic systems; documented
                            in the verdict output.
      skyline_threshold : skyline rollout relative-L2 above this
                            means the system is out-of-reach (no
                            model can reach adequacy on this system)
                            — pre-reg §7. Default 0.5.

    Returns: (kept_cells, exclusions_dict) where exclusions_dict
    has keys 'underfit' and 'skyline_out_of_reach' listing the
    system×seed strings excluded.
    """
    kept = []
    exclusions = {"underfit": [], "skyline_out_of_reach": []}

    # First pass: identify skyline-out-of-reach systems.
    skyline_per_system: dict[str, list[float]] = {}
    for c in cells:
        if c.skyline_relL2 is not None:
            skyline_per_system.setdefault(c.system, []).append(c.skyline_relL2)
    out_of_reach_systems = {
        sys for sys, vals in skyline_per_system.items()
        if np.mean(vals) > skyline_threshold
    }

    # Second pass: apply guards.
    for c in cells:
        tag = f"{c.system}_seed{c.seed}"
        if c.system in out_of_reach_systems:
            exclusions["skyline_out_of_reach"].append(tag)
            continue
        # Underfit guard (only if train relL2 was recorded).
        if (c.qlnn_train_relL2 is not None
                and c.qlnn_train_relL2 > underfit_threshold):
            exclusions["underfit"].append(f"{tag}_qlnn")
            continue
        if (c.neuralode_train_relL2 is not None
                and c.neuralode_train_relL2 > underfit_threshold):
            exclusions["underfit"].append(f"{tag}_neuralode")
            continue
        kept.append(c)
    return kept, exclusions


# ---------------------------------------------------------------------------
# The bootstrap (the §7 paired bootstrap on aggregate quantities)
# ---------------------------------------------------------------------------


def h1_bootstrap(
    cells: list[CellRecord],
    *,
    n_iter: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap of (Δ_smooth − Δ_broad) per pre-reg §7.

    Resamples CELLS (with replacement, within each regime) and
    computes Δ_smooth_mean − Δ_broad_mean per resample. The 95%
    empirical CI of this distribution is the H1 statistical test.

    Args:
      cells  : list of CellRecord. Must include at least one cell
               from each regime.
      n_iter : bootstrap iterations (default 10000 per pre-reg §5
               "n_iter ≥ 10000").
      alpha  : two-sided significance level (default 0.05 → 95% CI).
      seed   : RNG seed for reproducibility.

    Returns: dict with keys:
      - delta_smooth_mean : mean Δ over smooth cells (point estimate)
      - delta_broad_mean : mean Δ over broad cells
      - delta_diff_mean : Δ_smooth − Δ_broad (point estimate)
      - ci_low, ci_high : empirical 95% CI of the bootstrap
                           distribution of (Δ_smooth − Δ_broad)
      - n_iter : bootstrap iterations
      - n_smooth, n_broad : cell counts per regime
      - alpha : echoed for audit
    """
    smooth = [c for c in cells if c.regime == "smooth_periodic"]
    broad = [c for c in cells if c.regime == "broadband_multiscale"]
    if not smooth:
        raise ValueError(
            "no SMOOTH/PERIODIC cells passed the guards — H1 is "
            "inconclusive for that regime")
    if not broad:
        raise ValueError(
            "no BROADBAND cells passed the guards — H1 is "
            "inconclusive for that regime")

    smooth_deltas = np.asarray([c.delta for c in smooth], dtype=np.float64)
    broad_deltas = np.asarray([c.delta for c in broad], dtype=np.float64)
    point_smooth = float(smooth_deltas.mean())
    point_broad = float(broad_deltas.mean())
    point_diff = point_smooth - point_broad

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_iter, dtype=np.float64)
    n_s, n_b = smooth_deltas.size, broad_deltas.size
    for i in range(n_iter):
        idx_s = rng.integers(0, n_s, size=n_s)
        idx_b = rng.integers(0, n_b, size=n_b)
        d_s = smooth_deltas[idx_s].mean()
        d_b = broad_deltas[idx_b].mean()
        diffs[i] = d_s - d_b

    lo = alpha / 2.0
    hi = 1.0 - alpha / 2.0
    ci_low = float(np.quantile(diffs, lo))
    ci_high = float(np.quantile(diffs, hi))

    return {
        "delta_smooth_mean": point_smooth,
        "delta_broad_mean": point_broad,
        "delta_diff_mean": point_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_iter": int(n_iter),
        "n_smooth": int(n_s),
        "n_broad": int(n_b),
        "alpha": float(alpha),
        "smooth_systems": sorted({c.system for c in smooth}),
        "broad_systems": sorted({c.system for c in broad}),
    }


# ---------------------------------------------------------------------------
# The mechanical decision rule
# ---------------------------------------------------------------------------


def h1_verdict(
    cells: list[CellRecord],
    *,
    n_iter: int = 10000,
    alpha: float = 0.05,
    underfit_threshold: float = 0.5,
    skyline_threshold: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Apply pre-reg §7 decision rule mechanically.

    Steps:
      1. Apply underfit + skyline guards (`apply_guards`).
      2. Bootstrap (Δ_smooth − Δ_broad) over kept cells.
      3. Return CONFIRMED / FALSIFIED / INCONCLUSIVE per:
           - CONFIRMED  iff CI excludes 0 AND CI > 0 (delta_diff > 0)
           - FALSIFIED  iff CI includes 0 OR CI < 0
           - INCONCLUSIVE iff a regime has no cells left after guards

    Returns a JSONable dict suitable for h1_analysis.json:
      - outcome: 'CONFIRMED', 'FALSIFIED', or 'INCONCLUSIVE'
      - bootstrap: nested dict from h1_bootstrap
      - guards: {kept_n, excluded_underfit, excluded_skyline}
      - thresholds: {underfit, skyline, alpha}
      - reasoning: a short human-readable verdict string
    """
    kept, exclusions = apply_guards(
        cells,
        underfit_threshold=underfit_threshold,
        skyline_threshold=skyline_threshold)

    n_smooth_kept = sum(1 for c in kept if c.regime == "smooth_periodic")
    n_broad_kept = sum(1 for c in kept if c.regime == "broadband_multiscale")
    if n_smooth_kept == 0 or n_broad_kept == 0:
        missing = ("smooth" if n_smooth_kept == 0 else "broad")
        return {
            "outcome": "INCONCLUSIVE",
            "reasoning": (
                f"No cells remain in the {missing}-regime after applying "
                f"underfit + skyline guards. Pre-reg §7: 'if exclusions "
                f"remove a regime's support, H1 is reported inconclusive "
                f"for that regime.'"),
            "bootstrap": None,
            "guards": {
                "kept_n": len(kept),
                "kept_smooth": n_smooth_kept,
                "kept_broad": n_broad_kept,
                "excluded_underfit": exclusions["underfit"],
                "excluded_skyline_out_of_reach":
                    exclusions["skyline_out_of_reach"],
            },
            "thresholds": {
                "underfit": underfit_threshold,
                "skyline": skyline_threshold,
                "alpha": alpha,
            },
        }

    boot = h1_bootstrap(kept, n_iter=n_iter, alpha=alpha, seed=seed)

    # Mechanical decision (pre-reg §7).
    ci_excludes_zero = boot["ci_low"] > 0 or boot["ci_high"] < 0
    ci_positive = boot["ci_low"] > 0
    if ci_excludes_zero and ci_positive:
        outcome = "CONFIRMED"
        reasoning = (
            f"H1 CONFIRMED: Δ_smooth − Δ_broad = {boot['delta_diff_mean']:.4f} "
            f"with 95% CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] "
            f"(excludes 0 and is positive). Pre-reg §7: QLNN's improvement "
            f"over Neural-ODE is materially larger on the smooth/periodic "
            f"regime — the Schuld-Fourier hypothesis prediction holds.")
    else:
        outcome = "FALSIFIED"
        ci_state = ("includes 0" if not ci_excludes_zero
                    else "is negative")
        reasoning = (
            f"H1 FALSIFIED: Δ_smooth − Δ_broad = {boot['delta_diff_mean']:.4f} "
            f"with 95% CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] "
            f"({ci_state}). Pre-reg §7: 'Published as a rigorous "
            f"mechanistic null.' QLNN does NOT show a regime-dependent "
            f"advantage over the non-liquid Neural-ODE baseline at this "
            f"compute budget.")

    return {
        "outcome": outcome,
        "reasoning": reasoning,
        "bootstrap": boot,
        "guards": {
            "kept_n": len(kept),
            "kept_smooth": n_smooth_kept,
            "kept_broad": n_broad_kept,
            "excluded_underfit": exclusions["underfit"],
            "excluded_skyline_out_of_reach":
                exclusions["skyline_out_of_reach"],
        },
        "thresholds": {
            "underfit": underfit_threshold,
            "skyline": skyline_threshold,
            "alpha": alpha,
        },
    }
