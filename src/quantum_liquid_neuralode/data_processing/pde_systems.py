"""Synthetic 1-D PDE benchmark systems — the *hardness ladder* the pivot
needs to defeat the persistence trap that sank the bioreactor-OD task.

Rationale (see `ODE_PDE_PRE_REG.md`): a 1-step / h-step forecast on a
near-persistent scalar is trivially solved by copy-forward. Autoregressive
rollout on a *nonlinear field* has no trivial solution — that is the whole
point of introducing PDEs. These three 1-D systems are the pre-registered
ladder, with the regime split that hypothesis **H1** is stated over:

  - viscous Burgers, SMOOTH regime (large nu, no shock)   -> H1 "smooth_periodic"
  - viscous Burgers, SHOCK  regime (small nu, shock forms) -> H1 "broadband_multiscale"
  - Allen-Cahn (sharp bistable fronts, multiscale)          -> H1 "broadband_multiscale"
  - KdV (dispersive solitons; conserved mass & momentum)    -> H1 "broadband_multiscale"

Numerics — chosen to match `synthetic_ode.py`'s discipline (pure numpy,
zero new deps, fully deterministic, no adaptive solver hiding dynamics)
while being *correct* for stiff semilinear PDEs:

  - Fourier-spectral spatial derivatives on a periodic domain (numpy.fft).
  - Cox-Matthews **integrating-factor RK4** (Kassam & Trefethen 2005,
    "Fourth-order time-stepping for stiff PDEs"): the linear operator L
    (the heat term nu*u_xx, the Allen-Cahn ε²u_xx+u, the KdV dispersion
    u_xxx) is integrated *exactly* via the integrating factor e^{L*dt};
    classical RK4 acts only on the mild nonlinear term. Naive explicit
    RK4 on KdV needs dt ~ dx^3 and does not conserve invariants — IF-RK4
    does, and stays stable at a moderate dt.
  - 2/3-rule dealiasing on the quadratic/cubic nonlinearities, the
    standard guard against aliasing-driven blow-up.

The emitted artifact is an **npz field** (`u[t, x]`, the `x`/`t` grids,
the initial condition, the periodic BC tag, the conserved invariants,
full params + a provenance hash) — NOT the scalar qZETA-CSV schema. The
scalar-target CSV seam is deliberately blocked for PDEs; downstream code
consumes the field directly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Semilinear spectral form:  u_t = L̂ u + N(u)
#   L̂ is diagonal in Fourier space (an array over wavenumbers).
#   N(u) is evaluated in physical space and returned in Fourier space.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDESystem:
    """Canonical configuration for one 1-D periodic PDE benchmark.

    `linear(k)` returns the (possibly complex) Fourier multiplier L̂(k).
    `nonlinear(uhat, k, mask)` takes Fourier coeffs, returns the Fourier
    coeffs of the nonlinear term N (it does its own ifft/fft and applies
    the dealiasing `mask`). `ic(x)` returns the initial field u(x, 0).
    `invariants` names the conserved/monotone quantities the validation
    tests check (computed by `compute_invariants`).
    """

    name: str
    equation: str                       # human-readable PDE, for the npz
    regime: str                         # H1 bin: smooth_periodic | broadband_multiscale
    domain_length: float                # periodic domain [0, L)
    n_x: int                            # spatial grid points
    dt: float                           # IF-RK4 time step (accuracy/stability)
    n_steps: int                        # total integration steps
    sample_every: int                   # keep every k-th step in the field
    params: dict
    ic: Callable[[np.ndarray], np.ndarray]
    linear: Callable[[np.ndarray], np.ndarray]
    nonlinear: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    invariants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.n_x < 8 or (self.n_x & (self.n_x - 1)) != 0:
            raise ValueError("n_x must be a power of two >= 8 (FFT grid)")
        if self.sample_every < 1:
            raise ValueError("sample_every must be >= 1")
        if self.n_steps < self.sample_every:
            raise ValueError("n_steps must be >= sample_every")

    @property
    def n_frames(self) -> int:
        """Number of time slices kept in the field (incl. t=0)."""
        return self.n_steps // self.sample_every + 1


def _grid(sys: PDESystem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, k, dealias_mask) for a periodic spectral grid."""
    n = sys.n_x
    dx = sys.domain_length / n
    x = dx * np.arange(n)
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    # 2/3 rule: zero the top third of the spectrum on nonlinear products.
    cutoff = (n // 3)
    mask = np.ones(n, dtype=np.float64)
    kf = np.fft.fftfreq(n) * n  # integer mode index, signed
    mask[np.abs(kf) > cutoff] = 0.0
    return x, k, mask


def _ifrk4(sys: PDESystem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cox-Matthews integrating-factor RK4 for u_t = L̂ u + N(u).

    Returns (t, x, U) with U shape (n_frames, n_x), real-valued.
    """
    x, k, mask = _grid(sys)
    L = np.asarray(sys.linear(k))
    h = sys.dt
    E = np.exp(h * L)
    E2 = np.exp(h * L / 2.0)
    Nf = sys.nonlinear

    u0 = np.asarray(sys.ic(x), dtype=np.float64)
    v = np.fft.fft(u0)

    frames = np.empty((sys.n_frames, sys.n_x), dtype=np.float64)
    frames[0] = u0
    fi = 1
    for step in range(1, sys.n_steps + 1):
        Nv = Nf(v, k, mask)
        a = E2 * v + (h / 2.0) * E2 * Nv
        Na = Nf(a, k, mask)
        b = E2 * v + (h / 2.0) * Na
        Nb = Nf(b, k, mask)
        c = E * v + h * E2 * Nb
        Nc = Nf(c, k, mask)
        v = E * v + (h / 6.0) * (E * Nv + 2.0 * E2 * (Na + Nb) + Nc)
        if step % sys.sample_every == 0:
            u = np.real(np.fft.ifft(v))
            if not np.all(np.isfinite(u)):
                raise FloatingPointError(
                    f"{sys.name}: non-finite field at step {step} — "
                    f"reduce dt or increase viscosity/dealiasing")
            frames[fi] = u
            fi += 1
    t = np.arange(sys.n_frames, dtype=np.float64) * h * sys.sample_every
    return t, x, frames


# ---------------------------------------------------------------------------
# Nonlinear terms (conservative form where a conservation law must hold)
# ---------------------------------------------------------------------------


def _burgers_nonlinear(vhat, k, mask):
    # -u u_x  written conservatively as  -1/2 d/dx(u^2)  so that the
    # discrete mass ∫u dx is conserved to spectral round-off.
    u = np.real(np.fft.ifft(vhat))
    return -0.5j * k * (np.fft.fft(u * u) * mask)


def _allen_cahn_nonlinear(vhat, k, mask):
    # reaction -u^3 (the linear +u lives in L̂); cubic → 2/3 dealias.
    u = np.real(np.fft.ifft(vhat))
    return -np.fft.fft(u ** 3) * mask


def _kdv_nonlinear(vhat, k, mask):
    # -6 u u_x = -3 d/dx(u^2), conservative form (preserves ∫u, ∫u^2).
    u = np.real(np.fft.ifft(vhat))
    return -3.0j * k * (np.fft.fft(u * u) * mask)


# ---------------------------------------------------------------------------
# Analytic references (used by the validation tests — the PDE analogue of
# synthetic_ode's "conserved quantity / boundedness / chaos" checks)
# ---------------------------------------------------------------------------


def burgers_inviscid_shock_time(du0_dx_min: float) -> float:
    """First gradient-catastrophe time of inviscid Burgers for a smooth
    IC: t* = -1 / min_x u0'(x). For u0 = sin(x), min u0' = -1 → t* = 1."""
    if du0_dx_min >= 0:
        return np.inf
    return -1.0 / du0_dx_min


def allen_cahn_front_width(eps: float) -> float:
    """Equilibrium tanh-front interface width of u_t = ε²u_xx + u − u³:
    u ~ tanh((x - x0) / (√2 ε)); the characteristic width is √2 ε."""
    return np.sqrt(2.0) * eps


def kdv_soliton(x: np.ndarray, t: float, c: float, x0: float,
                domain_length: float) -> np.ndarray:
    """Exact 1-soliton of u_t + 6 u u_x + u_xxx = 0:
    u = (c/2) sech²( (√c/2)(x − c t − x0) ), translating at speed c.
    Argument is wrapped into the periodic domain."""
    arg = x - c * t - x0
    arg = (arg + domain_length / 2.0) % domain_length - domain_length / 2.0
    return 0.5 * c / np.cosh(0.5 * np.sqrt(c) * arg) ** 2


# ---------------------------------------------------------------------------
# Canonical configurations
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * np.pi


def _burgers(regime: str) -> PDESystem:
    # IC u0 = sin(x) on [0, 2π): min u0' = -1 → inviscid shock at t*=1.
    nu = {"smooth": 0.12, "shock": 4.0e-3}[regime]
    return PDESystem(
        name=f"burgers_{regime}",
        equation="u_t + u u_x = nu u_xx  (periodic)",
        regime=("smooth_periodic" if regime == "smooth"
                else "broadband_multiscale"),
        domain_length=_TWO_PI,
        n_x=256,
        dt=2.0e-4,
        n_steps=10000,           # T = 2.0  (past the inviscid shock t*=1)
        sample_every=50,         # 201 frames
        params={"nu": nu, "ic": "sin(x)", "shock_time_inviscid": 1.0},
        ic=lambda x: np.sin(x),
        linear=lambda k, nu=nu: -nu * k ** 2,
        nonlinear=_burgers_nonlinear,
        invariants=("mass",),    # ∫u dx conserved (conservative flux form)
    )


def _allen_cahn() -> PDESystem:
    eps = 0.06
    Lx = _TWO_PI

    def ic(x, Lx=Lx, eps=eps):
        # Periodic two-front config (up-front at L/4, down-front at 3L/4,
        # plateaus u≈±1). The fronts start DELIBERATELY TOO NARROW
        # (0.6·√2ε): equal-well Allen-Cahn must relax the interface to
        # its equilibrium width √2ε while keeping the (symmetric) fronts
        # stationary — that relaxation is what the validation test
        # checks, so the integrator is exercised on real dynamics, not
        # just steady-state preservation.
        w = 0.6 * np.sqrt(2.0) * eps
        return (np.tanh((x - Lx / 4.0) / w)
                - np.tanh((x - 3.0 * Lx / 4.0) / w) - 1.0)

    return PDESystem(
        name="allen_cahn",
        equation="u_t = eps^2 u_xx + u - u^3  (periodic)",
        regime="broadband_multiscale",
        domain_length=Lx,
        n_x=1024,                # ~10 pts across a √2·eps≈0.085 front
        dt=1.0e-3,
        n_steps=8000,            # T = 8.0
        sample_every=40,         # 201 frames
        params={"eps": eps, "ic": "periodic tanh two-front",
                "front_width": float(np.sqrt(2.0) * eps)},
        ic=ic,
        linear=lambda k, eps=eps: 1.0 - (eps ** 2) * k ** 2,
        nonlinear=_allen_cahn_nonlinear,
        invariants=("ginzburg_landau_energy",),  # monotone non-increasing
    )


def _kdv() -> PDESystem:
    Lx = 40.0
    c = 4.0                       # soliton speed; amplitude c/2 = 2
    x0 = Lx / 2.0
    return PDESystem(
        name="kdv",
        equation="u_t + 6 u u_x + u_xxx = 0  (periodic)",
        regime="broadband_multiscale",
        domain_length=Lx,
        n_x=512,
        dt=4.0e-4,
        n_steps=12500,            # T = 5.0 → soliton travels c·T = 20 = L/2
        sample_every=125,         # 101 frames
        params={"c": c, "x0": x0, "ic": "exact 1-soliton sech^2"},
        ic=lambda x, Lx=Lx, c=c, x0=x0: kdv_soliton(x, 0.0, c, x0, Lx),
        linear=lambda k: 1j * k ** 3,        # -u_xxx integrated exactly
        nonlinear=_kdv_nonlinear,
        invariants=("mass", "momentum"),     # ∫u dx, ∫u^2 dx conserved
    )


def get_pde_system(name: str) -> PDESystem:
    """Return the canonical configuration for one PDE benchmark."""
    if name == "burgers_smooth":
        return _burgers("smooth")
    if name == "burgers_shock":
        return _burgers("shock")
    if name == "allen_cahn":
        return _allen_cahn()
    if name == "kdv":
        return _kdv()
    raise ValueError(
        f"unknown PDE system {name!r}; choose from burgers_smooth, "
        f"burgers_shock, allen_cahn, kdv")


PDE_SYSTEMS = ["burgers_smooth", "burgers_shock", "allen_cahn", "kdv"]


# ---------------------------------------------------------------------------
# Invariants / diagnostics
# ---------------------------------------------------------------------------


def compute_invariants(name: str, x: np.ndarray,
                        U: np.ndarray) -> dict[str, np.ndarray]:
    """Time series of each named invariant for a simulated field U.

    Periodic-grid integral = sum * dx (trapezoid == midpoint here).
    - mass     : ∫ u dx                 (Burgers, KdV — conserved)
    - momentum : ∫ u² dx                (KdV — conserved)
    - ginzburg_landau_energy : ∫ [ (eps²/2) u_x² + (1/4)(1-u²)² ] dx
                 (Allen-Cahn Lyapunov functional — non-increasing)
    """
    sys = get_pde_system(name)
    dx = sys.domain_length / sys.n_x
    out: dict[str, np.ndarray] = {}
    if "mass" in sys.invariants:
        out["mass"] = U.sum(axis=1) * dx
    if "momentum" in sys.invariants:
        out["momentum"] = (U ** 2).sum(axis=1) * dx
    if "ginzburg_landau_energy" in sys.invariants:
        eps = sys.params["eps"]
        k = 2.0 * np.pi * np.fft.fftfreq(sys.n_x, d=dx)
        ux = np.real(np.fft.ifft(1j * k[None, :] * np.fft.fft(U, axis=1)))
        dens = 0.5 * eps ** 2 * ux ** 2 + 0.25 * (1.0 - U ** 2) ** 2
        out["ginzburg_landau_energy"] = dens.sum(axis=1) * dx
    return out


# ---------------------------------------------------------------------------
# Simulate / emit npz field artifact
# ---------------------------------------------------------------------------


def simulate_pde(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                      PDESystem]:
    """Integrate one PDE system. Returns (t, x, U, system); U is
    (n_frames, n_x), real, deterministic given the canonical config."""
    sys = get_pde_system(name)
    t, x, U = _ifrk4(sys)
    return t, x, U, sys


def _provenance_hash(name: str, t, x, U, params: dict) -> str:
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    for arr in (t, x, U):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


def make_pde_npz(name: str, path: str) -> dict:
    """Simulate and write the field artifact to `path` (.npz).

    Saved keys: u (n_frames,n_x) float64, x (n_x,), t (n_frames,),
    u0 (n_x,), bc='periodic', equation, regime, params (json),
    invariants (json: each named series), sha256 provenance hash.
    Returns the metadata dict (also embedded in the npz).
    """
    t, x, U, sys = simulate_pde(name)
    inv = compute_invariants(name, x, U)
    meta = {
        "name": sys.name,
        "equation": sys.equation,
        "regime": sys.regime,
        "bc": "periodic",
        "domain_length": sys.domain_length,
        "n_x": sys.n_x,
        "dt": sys.dt,
        "n_steps": sys.n_steps,
        "sample_every": sys.sample_every,
        "params": sys.params,
        "invariant_names": list(sys.invariants),
    }
    meta["sha256"] = _provenance_hash(name, t, x, U, sys.params)
    np.savez_compressed(
        path,
        u=U.astype(np.float64),
        x=x.astype(np.float64),
        t=t.astype(np.float64),
        u0=U[0].astype(np.float64),
        meta_json=json.dumps(meta, default=str),
        invariants_json=json.dumps(
            {kk: vv.tolist() for kk, vv in inv.items()}),
    )
    return meta
