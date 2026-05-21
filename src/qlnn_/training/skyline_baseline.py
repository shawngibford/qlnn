"""P5 commit 3 — Known-structure skyline baseline (pre-reg §6 upper bound).

Per ODE_PDE_PRE_REG.md §6 binding row: "Skyline | known-structure
model (true RHS, fit only free constants) | Upper bound —
contextualizes every gap" and pre-reg §7 decision rule extension:
"if *no* model (including the known-structure skyline) achieves the
adequacy threshold on a system, that system is declared
out-of-reach" and excluded from H1 aggregation.

Per-system known structure (the analytic ODE form, free constants
to fit):

  lotka_volterra:
    u' = u·(α − β·v)
    v' = v·(δ·u − γ)
    free params: (α, β, δ, γ).

  van_der_pol:
    u' = v
    v' = μ·(1 − u²)·v − u
    free params: (μ,).

  lorenz '63:
    u' = σ·(v − u)
    v' = u·(ρ − w) − v
    w' = u·v − β·w
    free params: (σ, ρ, β).

Fitting procedure (per system):
  1. Estimate `dy/dt` along the training trajectory via central
     finite differences (the canonical RK4 step is small enough
     that central diff approximates `dy/dt` well — the synthetic_ode
     reference uses RK4 at dt=0.01-0.02 sampled every 10-12 steps,
     so the effective sampled-dt is small).
  2. Build the feature matrix `Φ(y)` whose columns are the
     structural basis functions (e.g. for LV: `[u, u·v]` for u'
     and `[v, u·v]` for v').
  3. Solve for the coefficients via least-squares:
     `coeffs = (Φᵀ Φ)⁻¹ Φᵀ dy/dt`.

Rollout: integrate the fitted ODE forward via fixed-step RK4 from
the test-trajectory's initial state, predicting the trajectory tail.

The skyline is **the upper bound** for the system — no
data-driven forecaster (QLNN, NeuralODE, MLP, classical PINN) can
beat the skyline if the data perfectly conforms to the structural
form. In practice, finite-difference noise in `dy/dt` + the
discrete RK4 rollout step lead to a small but nonzero gap; the
skyline's residual relL2 is the "structural floor."

For H1 aggregation, systems where the skyline fails to reach the
pre-registered adequacy threshold are **declared out-of-reach** and
excluded from the verdict. Documented in
`results/p5_h1_verdict/excluded_systems.json` when P5 commit 5 lands.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Per-system known structure: features + RHS predictor
# ---------------------------------------------------------------------------


def _lv_features_u(y: np.ndarray) -> np.ndarray:
    """LV u' structure: [u, u·v]. Coefficients yield α, −β."""
    u, v = y[..., 0], y[..., 1]
    return np.stack([u, u * v], axis=-1)


def _lv_features_v(y: np.ndarray) -> np.ndarray:
    """LV v' structure: [v, u·v]. Coefficients yield −γ, δ."""
    u, v = y[..., 0], y[..., 1]
    return np.stack([v, u * v], axis=-1)


def _vdp_features_u(y: np.ndarray) -> np.ndarray:
    """VdP u' = v. Single feature [v]; coefficient = 1 (locked)."""
    return y[..., 1:2]


def _vdp_features_v(y: np.ndarray) -> np.ndarray:
    """VdP v' = μ·(1 − u²)·v − u. Features [(1−u²)·v, u]; coefficients
    yield μ, −1 (the second locks to −1)."""
    u, v = y[..., 0], y[..., 1]
    return np.stack([(1.0 - u ** 2) * v, u], axis=-1)


def _lorenz_features_u(y: np.ndarray) -> np.ndarray:
    """Lorenz u' = σ(v − u). Single feature [v − u]; coefficient = σ."""
    u, v = y[..., 0], y[..., 1]
    return (v - u)[..., None]


def _lorenz_features_v(y: np.ndarray) -> np.ndarray:
    """Lorenz v' = u(ρ − w) − v = ρ·u − u·w − v.
    Features [u, u·w, v]; coefficients ρ, −1, −1."""
    u, v, w = y[..., 0], y[..., 1], y[..., 2]
    return np.stack([u, u * w, v], axis=-1)


def _lorenz_features_w(y: np.ndarray) -> np.ndarray:
    """Lorenz w' = u·v − β·w. Features [u·v, w]; coefficients 1, −β."""
    u, v, w = y[..., 0], y[..., 1], y[..., 2]
    return np.stack([u * v, w], axis=-1)


_FEATURE_BUILDERS: dict[str, list[Callable[[np.ndarray], np.ndarray]]] = {
    "lotka_volterra": [_lv_features_u, _lv_features_v],
    "van_der_pol":    [_vdp_features_u, _vdp_features_v],
    "lorenz":         [_lorenz_features_u, _lorenz_features_v,
                       _lorenz_features_w],
}

_FEATURE_DIMS: dict[str, list[int]] = {
    "lotka_volterra": [2, 2],
    "van_der_pol":    [1, 2],
    "lorenz":         [1, 3, 2],
}


# ---------------------------------------------------------------------------
# Skyline fit
# ---------------------------------------------------------------------------


def _central_diff(trajectory: np.ndarray, dt: float) -> np.ndarray:
    """Central finite difference for the interior; one-sided at the
    endpoints. Returns the same shape as `trajectory` (n_points, d)."""
    dydt = np.zeros_like(trajectory)
    dydt[1:-1] = (trajectory[2:] - trajectory[:-2]) / (2.0 * dt)
    # Endpoints: forward / backward Euler differences.
    dydt[0] = (trajectory[1] - trajectory[0]) / dt
    dydt[-1] = (trajectory[-1] - trajectory[-2]) / dt
    return dydt


def fit_skyline(
    system: str, trajectory: np.ndarray, dt: float,
    *, drop_endpoints: int = 2,
) -> dict:
    """Fit the per-system known-structure coefficients via least squares.

    Args:
      system        : one of 'lotka_volterra', 'van_der_pol', 'lorenz'.
      trajectory    : (n_points, d) training reference trajectory.
      dt            : sample-spaced physical-time step (sys.dt × sys.sample_every).
      drop_endpoints: drop this many points from each end of the
                      finite-difference estimate (default 2) to avoid
                      the one-sided endpoint estimates polluting the fit.

    Returns:
      A dict with:
        coeffs_per_component : list of (n_features,) coefficient arrays,
                                one per state component.
        rhs_fn               : callable rhs(y) → (d,) the fitted RHS.
        system               : echoed for provenance.
    """
    if system not in _FEATURE_BUILDERS:
        raise ValueError(
            f"unknown skyline system {system!r}; "
            f"expected one of {list(_FEATURE_BUILDERS)}")
    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory must be 2-D (n_points, d), got {trajectory.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    n = trajectory.shape[0]
    d = trajectory.shape[1]
    if d != len(_FEATURE_BUILDERS[system]):
        raise ValueError(
            f"system {system!r} expects state-dim {len(_FEATURE_BUILDERS[system])}, "
            f"got d={d}")

    dydt = _central_diff(trajectory, dt)
    # Drop endpoint diff samples (they're one-sided, lower order).
    drop = max(int(drop_endpoints), 1)
    interior_y = trajectory[drop:n - drop]
    interior_dydt = dydt[drop:n - drop]

    coeffs_per_component = []
    builders = _FEATURE_BUILDERS[system]
    for i, build_feat in enumerate(builders):
        Phi = build_feat(interior_y).astype(np.float64)   # (M, n_feat)
        target = interior_dydt[:, i]
        # np.linalg.lstsq is more numerically stable than (ΦᵀΦ)⁻¹ ΦᵀY
        # for poorly-conditioned features.
        coef, _resid, _rank, _sv = np.linalg.lstsq(
            Phi, target, rcond=None)
        coeffs_per_component.append(np.asarray(coef, dtype=np.float64))

    def rhs_fn(y: np.ndarray) -> np.ndarray:
        """The fitted RHS — predict dy/dt at state y."""
        dy = np.zeros(d, dtype=np.float64)
        for i, build_feat in enumerate(builders):
            Phi_y = build_feat(y[None, :]).astype(np.float64)  # (1, n_feat)
            # The matmul yields a (1,) array; squeeze to scalar.
            dy[i] = (Phi_y @ coeffs_per_component[i]).item()
        return dy

    return {
        "system": system,
        "coeffs_per_component": coeffs_per_component,
        "rhs_fn": rhs_fn,
    }


# ---------------------------------------------------------------------------
# Skyline rollout (RK4 integration of the fitted RHS)
# ---------------------------------------------------------------------------


def rollout_skyline(
    rhs_fn: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray, n_steps: int, dt: float,
) -> np.ndarray:
    """Fixed-step RK4 rollout of the fitted RHS from `y0` for `n_steps`.

    Args:
      rhs_fn  : callable rhs(y) → (d,). Same convention as
                synthetic_ode.simulate's per-system RHS.
      y0      : (d,) initial state.
      n_steps : number of rollout steps.
      dt      : physical-time step (must match the sampled-dt of the
                trajectory the model was fit on).

    Returns: (n_steps, d) predicted trajectory.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    y0 = np.asarray(y0, dtype=np.float64).flatten()
    d = y0.shape[0]
    traj = np.empty((n_steps, d), dtype=np.float64)
    y = y0.copy()
    for i in range(n_steps):
        k1 = rhs_fn(y)
        k2 = rhs_fn(y + 0.5 * dt * k1)
        k3 = rhs_fn(y + 0.5 * dt * k2)
        k4 = rhs_fn(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i] = y
    return traj


# ---------------------------------------------------------------------------
# Convenience: full skyline pipeline (fit + rollout)
# ---------------------------------------------------------------------------


def skyline_predict(
    system: str, train_trajectory: np.ndarray, y0: np.ndarray,
    n_steps: int, dt: float,
) -> tuple[np.ndarray, dict]:
    """Fit + rollout in one call.

    Args:
      system           : 'lotka_volterra' / 'van_der_pol' / 'lorenz'.
      train_trajectory : (n_train, d) — the training reference.
      y0               : (d,) — rollout initial state (typically the
                          first state of the test trajectory).
      n_steps          : rollout horizon.
      dt               : sampled-dt.

    Returns: (trajectory (n_steps, d), fit_info dict).
    """
    fit_info = fit_skyline(system, train_trajectory, dt)
    traj = rollout_skyline(fit_info["rhs_fn"], y0, n_steps, dt)
    return traj, fit_info
