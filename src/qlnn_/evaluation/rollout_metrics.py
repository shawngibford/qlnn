"""P4 commit 2 — rollout metric suite (the pre-reg §5 locked set).

Every metric here consumes a `(T, d)` trajectory predicted by an
autoregressive forecaster (see `rollout.py:autoregressive_rollout`)
and a `(T, d)` reference trajectory of the same shape (the
numerical-RK4 ground truth produced by `synthetic_ode.simulate(...)`
or the spectral PDE solver `pde_systems.*`).

The four metric primitives (pre-reg §5 verbatim):

  1. **relative-L2 error** — the PRIMARY headline endpoint.
     `‖û − u‖₂ / ‖u‖₂` over the full rollout horizon. For ODE state
     vectors the field norm is the Euclidean state norm; for PDE
     fields it is the spatial-grid L2 over the flat field. This is
     the established PDEBench / FNO standard, NOT 1-step MAE
     (banned per Risk R3 — the persistence trap).

  2. **valid-prediction-time (VPT)** — first rollout step (or time)
     at which the running relative-L2 first exceeds ε = 0.3
     (pre-reg locked). For chaotic systems (Lorenz) we report VPT
     in **Lyapunov times** — normalized by the system's largest
     Lyapunov exponent so the number is dynamics-meaningful.

  3. **spectral error** — L2 norm of the difference between the
     predicted and reference power-spectral-densities (PSD), via
     FFT. Directly probes the Fourier-bias mechanism (H1 / H3): a
     Fourier-biased model should track low-k power and lose
     high-k power on broadband systems.

  4. **invariant drift** — for systems with a known conserved
     quantity (KdV mass + energy, Lotka-Volterra Hamiltonian-like
     H = u + v − ln(uv)), the time-series of |I(û(t)) − I(û(0))|
     normalized by |I(û(0))|. A faithful rollout has drift ≪ 1;
     a diverging rollout has drift → ∞.

Plus a per-system Lyapunov-exponent table for the chaotic-regime
VPT normalization. Locked values (computed externally for the
canonical configs in `synthetic_ode.py`):

  Lorenz (σ=10, ρ=28, β=8/3) : λ₁ ≈ 0.906 nat/sec
  (Other systems are non-chaotic; LE undefined / negative.)

1-step / h-step MAE is INTENTIONALLY ABSENT from this module —
Risk R3, the persistence trap (HANDOFF + pre-reg §5). A reviewer
who wants MAE can compute it themselves from the raw field.npz.
We deliberately do not expose it as a metric primitive to avoid
accidental headline use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Per-system Lyapunov exponents (chaotic-regime VPT normalization)
# ---------------------------------------------------------------------------


LYAPUNOV_EXPONENT = {
    # The canonical Lorenz '63 largest-LE — computed numerically from a
    # long trajectory; locked to 3 sig figs for reproducibility.
    "lorenz": 0.906,
    # Non-chaotic systems: LE undefined for VPT-in-Lyapunov reporting.
    "lotka_volterra": None,
    "van_der_pol": None,
    "fitzhugh_nagumo": None,
    "kuramoto": None,
}


# ---------------------------------------------------------------------------
# Primary endpoint: relative-L2 error
# ---------------------------------------------------------------------------


def relative_l2_error(
    u_pred: np.ndarray, u_ref: np.ndarray, *, eps: float = 1e-12,
) -> float:
    """Compute the rollout relative-L2 error over the full horizon.

    `‖û − u‖₂ / ‖u‖₂` on the flattened arrays. Robust to division by
    zero via `eps` floor on the denominator.

    Args:
      u_pred : (T, d) or (T, ..., d) predicted trajectory.
      u_ref  : same shape as `u_pred`, the numerical reference.
      eps    : floor on the denominator to avoid 0/0 (default 1e-12).

    Returns: scalar relative-L2 error.

    This is THE primary endpoint per pre-reg §5. The headline number
    of every cell in the H1 verdict aggregation.
    """
    if u_pred.shape != u_ref.shape:
        raise ValueError(
            f"u_pred shape {u_pred.shape} != u_ref shape {u_ref.shape}")
    a = np.asarray(u_pred, dtype=np.float64).ravel()
    b = np.asarray(u_ref, dtype=np.float64).ravel()
    num = float(np.linalg.norm(a - b))
    den = max(float(np.linalg.norm(b)), eps)
    return num / den


def relative_l2_over_time(
    u_pred: np.ndarray, u_ref: np.ndarray, *, eps: float = 1e-12,
) -> np.ndarray:
    """Per-timestep relative-L2 — the running version used by VPT.

    For each time index t, compute `‖û(t) − u(t)‖₂ / ‖u(t)‖₂` over
    the trailing state dims. Returns shape `(T,)`.

    Used downstream by `valid_prediction_time` to find the first t
    where this exceeds the pre-reg threshold ε = 0.3.
    """
    if u_pred.shape != u_ref.shape:
        raise ValueError(
            f"u_pred shape {u_pred.shape} != u_ref shape {u_ref.shape}")
    a = np.asarray(u_pred, dtype=np.float64)
    b = np.asarray(u_ref, dtype=np.float64)
    # Flatten everything except the leading time axis.
    a_t = a.reshape(a.shape[0], -1)
    b_t = b.reshape(b.shape[0], -1)
    diff = np.linalg.norm(a_t - b_t, axis=1)
    norm = np.maximum(np.linalg.norm(b_t, axis=1), eps)
    return diff / norm


# ---------------------------------------------------------------------------
# Valid-prediction-time (VPT) — when the rollout first diverges
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VPTResult:
    """Container for the VPT report."""
    vpt_step: int             # first step index where relL2 > threshold;
                               # -1 if never (the whole rollout is valid)
    vpt_time: float            # vpt_step · dt (physical time)
    vpt_lyapunov: float | None # vpt_time × λ₁ (chaotic systems) or None
    rel_l2_curve: np.ndarray   # the running relative-L2 per step
    threshold: float           # the threshold used (recorded for audit)


def valid_prediction_time(
    u_pred: np.ndarray, u_ref: np.ndarray,
    *,
    dt: float,
    threshold: float = 0.3,
    lyapunov_exponent: float | None = None,
) -> VPTResult:
    """Compute the valid-prediction-time per pre-reg §5.

    Args:
      u_pred  : (T, d) or (T, ..., d) predicted trajectory.
      u_ref   : same-shape reference.
      dt      : physical-time step between consecutive rollout indices.
      threshold : relative-L2 ceiling; defaults to 0.3 (pre-reg locked).
      lyapunov_exponent : if provided, vpt_time is also reported
                          in Lyapunov times = vpt_time × λ₁.
                          For chaotic systems (Lorenz) the pre-reg
                          asks for this normalization explicitly.

    Returns a `VPTResult` with the step + time + (optional) Lyapunov
    time, plus the full per-step relative-L2 curve for plotting.

    If the rollout never exceeds the threshold (the model tracked
    the reference all the way to the horizon), `vpt_step = -1` and
    `vpt_time` is the full rollout duration. This semantic
    distinction is important: VPT = T (rollout length) means "still
    valid at horizon end"; VPT < T means "diverged at this point."
    """
    rel_l2 = relative_l2_over_time(u_pred, u_ref)
    above = np.where(rel_l2 > threshold)[0]
    if above.size == 0:
        # Never exceeded — the entire rollout is valid.
        T = rel_l2.size
        vpt_step = -1
        vpt_time = float(T * dt)
    else:
        vpt_step = int(above[0])
        vpt_time = float(vpt_step * dt)

    vpt_lyap: float | None = None
    if lyapunov_exponent is not None:
        if lyapunov_exponent <= 0:
            raise ValueError(
                f"lyapunov_exponent must be > 0, got {lyapunov_exponent}")
        vpt_lyap = vpt_time * float(lyapunov_exponent)

    return VPTResult(
        vpt_step=vpt_step, vpt_time=vpt_time,
        vpt_lyapunov=vpt_lyap, rel_l2_curve=rel_l2,
        threshold=float(threshold),
    )


# ---------------------------------------------------------------------------
# Spectral error — Fourier-bias probe
# ---------------------------------------------------------------------------


def spectral_error(
    u_pred: np.ndarray, u_ref: np.ndarray,
    *, axis: int = 0, eps: float = 1e-12,
) -> float:
    """L2 error between the PSDs of predicted vs reference rollout.

    Uses `np.fft.rfft` along the temporal axis (default axis=0), then
    `|fft|²` averaged over the state-dim axis. Returns scalar
    `‖PSD(û) − PSD(u)‖₂ / ‖PSD(u)‖₂`.

    Per pre-reg §5: "directly probes the Fourier-bias mechanism
    (H1/H3): a Fourier-biased model should track low-k power and
    lose high-k power on broadband systems."

    Args:
      u_pred, u_ref : (T, d) or (T, ..., d) trajectories.
      axis  : temporal axis (default 0). The FFT is taken along this
              axis and the PSD is averaged over the remaining axes
              after flattening them.
      eps   : floor on denominator (PSD(u_ref) might be ~0 for
              perfectly periodic signals at non-spectral frequencies).
    """
    if u_pred.shape != u_ref.shape:
        raise ValueError(
            f"u_pred shape {u_pred.shape} != u_ref shape {u_ref.shape}")
    a = np.asarray(u_pred, dtype=np.float64)
    b = np.asarray(u_ref, dtype=np.float64)

    # Move time axis to front, flatten the trailing state axes.
    if axis != 0:
        a = np.moveaxis(a, axis, 0)
        b = np.moveaxis(b, axis, 0)
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)

    # PSD per channel via |rfft|^2 (one-sided spectrum).
    psd_a = np.abs(np.fft.rfft(a, axis=0)) ** 2
    psd_b = np.abs(np.fft.rfft(b, axis=0)) ** 2

    # Average over the trailing (state-dim) axes so multi-channel
    # ODE states return one summary scalar.
    psd_a = psd_a.mean(axis=1)
    psd_b = psd_b.mean(axis=1)

    num = float(np.linalg.norm(psd_a - psd_b))
    den = max(float(np.linalg.norm(psd_b)), eps)
    return num / den


# ---------------------------------------------------------------------------
# Invariant drift — conserved-quantity tracking
# ---------------------------------------------------------------------------


def invariant_drift(
    trajectory: np.ndarray,
    invariant_fn: Callable[[np.ndarray], float],
    *, eps: float = 1e-12,
) -> np.ndarray:
    """Compute per-step relative drift of a conserved quantity.

    `drift(t) = |I(û(t)) − I(û(0))| / max(|I(û(0))|, eps)`.

    For KdV, plug in `kdv_mass` or `kdv_energy` (functions of the
    field at a single time slice). For Lotka-Volterra plug in the
    Hamiltonian-like H(u, v) = u + v − ln(u·v). For Lorenz / other
    dissipative systems no invariant exists; the metric is undefined
    and should NOT be reported (per pre-reg §5).

    Args:
      trajectory  : (T, d) or (T, ..., d).
      invariant_fn : callable that takes one time slice (shape (d,)
                     or (..., d)) and returns a scalar invariant.

    Returns: (T,) per-step drift; drift[0] = 0 by construction.

    The pre-reg's headline number for this metric is typically the
    FINAL drift `drift[-1]` (how far it walked off conservation by
    the end of the rollout); the per-step curve is for diagnosis.
    """
    a = np.asarray(trajectory, dtype=np.float64)
    invariants = np.asarray(
        [invariant_fn(a[t]) for t in range(a.shape[0])], dtype=np.float64)
    I0 = float(invariants[0])
    den = max(abs(I0), eps)
    return np.abs(invariants - I0) / den


# ---------------------------------------------------------------------------
# Built-in invariants for the canonical systems
# ---------------------------------------------------------------------------


def lotka_volterra_invariant(state: np.ndarray) -> float:
    """Hamiltonian-like conserved quantity for canonical LV ODE.

    For the canonical form `u' = u(1 − v), v' = v(u − 1)`, the
    quantity `H(u, v) = u + v − ln(u·v)` is exactly conserved along
    every orbit. Both `u, v > 0` is required for the ln; positivity
    is structural for the LV model and the synthetic_ode initial
    conditions.
    """
    u, v = float(state[0]), float(state[1])
    if u <= 0 or v <= 0:
        raise ValueError(
            f"LV invariant requires u, v > 0; got u={u}, v={v}")
    return u + v - np.log(u * v)


def kdv_mass(field: np.ndarray, *, dx: float = 1.0) -> float:
    """KdV conserved quantity 1: total mass = ∫ u dx ≈ Σ u Δx.

    Args:
      field : (n_x,) spatial field at one time slice.
      dx    : grid spacing for the Riemann sum.

    Per pre-reg §5: "for KdV, relative drift of conserved mass and
    energy over the rollout."
    """
    return float(np.sum(field) * dx)


def kdv_energy(field: np.ndarray, *, dx: float = 1.0) -> float:
    """KdV conserved quantity 2: total energy = ∫ u² dx ≈ Σ u² Δx.

    Same shape contract as `kdv_mass`.
    """
    return float(np.sum(np.asarray(field) ** 2) * dx)


# ---------------------------------------------------------------------------
# Utilities — Lyapunov-time normalization
# ---------------------------------------------------------------------------


def normalize_to_lyapunov_time(
    t_physical: float | np.ndarray, lyapunov_exponent: float,
) -> float | np.ndarray:
    """Convert a physical time (seconds, or units of the ODE) into
    Lyapunov times: τ_λ = t · λ₁.

    Per pre-reg §5: "For chaotic systems (Lorenz) VPT is reported
    in Lyapunov times (normalized by the system's largest Lyapunov
    exponent...) so the number is dynamics-meaningful."
    """
    if lyapunov_exponent <= 0:
        raise ValueError(
            f"lyapunov_exponent must be > 0, got {lyapunov_exponent}")
    return t_physical * lyapunov_exponent


# ---------------------------------------------------------------------------
# A bundle dataclass for the per-cell rollout-metrics output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutMetrics:
    """The pre-reg §5 four-metric bundle, plus diagnostics.

    Constructed by the P4 sweep CLI once per (model, system, seed)
    cell. Serialized to per-seed metrics.json alongside the field
    field.npz; aggregated into seeds_summary.json by the
    `summarize` function from `p3_8_review_demo`.

    `vpt_lyapunov` is None for non-chaotic systems (per
    `LYAPUNOV_EXPONENT`). `invariant_drift_final` is None for
    dissipative systems with no invariant. The serializer skips
    None fields cleanly.
    """
    relative_l2: float
    vpt_step: int
    vpt_time: float
    vpt_lyapunov: float | None
    spectral_error: float
    invariant_drift_final: float | None
    threshold: float        # the VPT threshold used (audit hook)
