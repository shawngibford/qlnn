"""Synthetic ODE benchmark systems — the controlled testbed for
characterizing *what the (Q)LNN forecasters are actually good at*.

The single 778-point bioreactor run cannot exercise the continuous-time
inductive bias these Neural-ODE models are built around (OD is ~0.99
autocorrelated; persistence is nearly unbeatable at h=3). A controlled
ODE suite isolates the dynamical regime — stiff / oscillatory / chaotic /
coupled / high-dimensional — so we can make mechanistic claims instead of
dataset-bound ones, with unlimited clean data and no seed-variance /
data-scarcity confound.

Canonical 5 (the standard Neural-ODE benchmark set):
  - lotka_volterra   2D coupled nonlinear, has a conserved quantity
  - fitzhugh_nagumo  2D excitable relaxation oscillator
  - van_der_pol      2D stiff nonlinear oscillator (stiffness ↑ with mu)
  - lorenz           3D chaotic (sensitive dependence — long-horizon test)
  - kuramoto         N-D high-dimensional coupled phase oscillators

Design choices that keep this methodologically comparable to the
bioreactor study (locked eval protocol must transfer unchanged):
  - ONE long trajectory per system (mirrors the single-run qZETA setup),
    chronologically split 70/15/15 downstream by the existing pipeline.
  - Fixed-step RK4 in pure numpy — zero new dependencies, fully
    deterministic given a seed, completely under our control (no adaptive
    solver hiding dynamics). Step size is chosen per system so RK4 is
    accurate even for the stiff Van der Pol regime we use.
  - Optional observation noise (Gaussian, std in state units) so the
    forecasting task is non-trivial and the reproducibility/σ story
    remains measurable.

`make_ode_dataframe` emits a pandas DataFrame with a synthetic `DATE`
column + state-variable feature columns, i.e. exactly the schema
`load_qzeta` already accepts — so `train_baseline.py` / `train_qlnn.py` /
the ansatz registry / the Option-B gate machinery all consume synthetic
ODE data with **no trainer code changes**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Right-hand sides:  f(t, y, p) -> dy/dt   (y is the full state vector)
# ---------------------------------------------------------------------------


def _lotka_volterra(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    x, z = y
    return np.array([
        p["alpha"] * x - p["beta"] * x * z,
        p["delta"] * x * z - p["gamma"] * z,
    ])


def _fitzhugh_nagumo(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    v, w = y
    return np.array([
        v - v**3 / 3.0 - w + p["I"],
        p["eps"] * (v + p["a"] - p["b"] * w),
    ])


def _van_der_pol(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    x, z = y
    return np.array([z, p["mu"] * (1.0 - x**2) * z - x])


def _lorenz(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    x, z, u = y
    return np.array([
        p["sigma"] * (z - x),
        x * (p["rho"] - u) - z,
        x * z - p["beta"] * u,
    ])


def _kuramoto(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    # y = phases theta_i ; coupling K, natural frequencies omega.
    theta = y
    diff = theta[None, :] - theta[:, None]          # theta_j - theta_i
    coupling = (p["K"] / len(theta)) * np.sin(diff).sum(axis=1)
    return p["omega"] + coupling


@dataclass(frozen=True)
class ODESystem:
    name: str
    rhs: Callable[[float, np.ndarray, dict], np.ndarray]
    y0: np.ndarray
    params: dict
    dt: float                       # RK4 integration step (accuracy)
    state_names: list[str]
    target: str                     # which state column is the forecast target
    # transient steps to discard so we sample the attractor / limit cycle
    burn_in: int = 0
    # Optional observable transform applied to the raw state before the
    # DataFrame is built (the integration + physics tests still see raw
    # state). Used for Kuramoto: a raw unwrapped phase drifts ~linearly
    # (persistence-trivial — the exact pathology this suite exists to
    # avoid), so we observe sin(theta) which is bounded and oscillatory.
    observe: Callable[[np.ndarray], np.ndarray] | None = None
    observe_prefix: str | None = None     # renames columns if observed
    # Sampling stride: integrate at fine `dt` for accuracy but keep every
    # `sample_every`-th step, so the SAMPLED series spans many cycles /
    # Lyapunov times — otherwise a slow oscillator (e.g. stiff Van der
    # Pol) barely completes one period over the whole dataset and the
    # forecasting task degenerates to "extrapolate a slow ramp".
    sample_every: int = 1

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.sample_every < 1:
            raise ValueError("sample_every must be >= 1")
        if len(self.y0) != len(self.state_names):
            raise ValueError("y0 length must equal len(state_names)")
        if self.target not in self.output_names:
            raise ValueError(
                f"target {self.target!r} not in output columns "
                f"{self.output_names}")

    @property
    def output_names(self) -> list[str]:
        """Column names after the optional observable transform — these are
        the DataFrame feature columns the trainer sees."""
        if self.observe is not None and self.observe_prefix is not None:
            return [f"{self.observe_prefix}{i}"
                    for i in range(len(self.state_names))]
        return list(self.state_names)


def _kuramoto_system(n: int = 12, seed: int = 0) -> ODESystem:
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 1.0, size=n)
    theta0 = rng.uniform(-np.pi, np.pi, size=n)
    return ODESystem(
        name="kuramoto",
        rhs=_kuramoto,
        y0=theta0,
        params={"K": 2.0, "omega": omega},
        dt=0.02,
        state_names=[f"theta{i}" for i in range(n)],
        target="sin_theta0",
        burn_in=2000,
        sample_every=4,   # ~40 phase cycles over 4000 rows
        observe=np.sin,
        observe_prefix="sin_theta",
    )


def get_system(name: str) -> ODESystem:
    """Return the canonical configuration for one benchmark system."""
    if name == "lotka_volterra":
        return ODESystem(
            name="lotka_volterra", rhs=_lotka_volterra,
            y0=np.array([10.0, 5.0]),
            params={"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4},
            dt=0.01, state_names=["prey", "predator"], target="prey",
            burn_in=0, sample_every=10,   # ~40 predator-prey cycles
        )
    if name == "fitzhugh_nagumo":
        return ODESystem(
            name="fitzhugh_nagumo", rhs=_fitzhugh_nagumo,
            y0=np.array([-1.0, 1.0]),
            params={"I": 0.5, "eps": 0.08, "a": 0.7, "b": 0.8},
            dt=0.02, state_names=["v", "w"], target="v", burn_in=1000,
            sample_every=12,   # ~30 relaxation cycles over 4000 rows
        )
    if name == "van_der_pol":
        return ODESystem(
            name="van_der_pol", rhs=_van_der_pol,
            y0=np.array([2.0, 0.0]),
            # mu=5 → genuinely stiff relaxation oscillation; dt small
            # enough that fixed-step RK4 stays accurate through the
            # fast switching layers.
            params={"mu": 5.0}, dt=0.005,
            state_names=["x", "v"], target="x", burn_in=2000,
            sample_every=20,   # ~50 stiff relaxation cycles over 4000 rows
        )
    if name == "lorenz":
        return ODESystem(
            name="lorenz", rhs=_lorenz,
            y0=np.array([1.0, 1.0, 1.0]),
            params={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
            dt=0.01, state_names=["x", "y", "z"], target="x",
            burn_in=1000, sample_every=5,   # ~180 Lyapunov times
        )
    if name == "kuramoto":
        return _kuramoto_system()
    raise ValueError(
        f"unknown system {name!r}; choose from lotka_volterra, "
        f"fitzhugh_nagumo, van_der_pol, lorenz, kuramoto")


SYSTEMS = ["lotka_volterra", "fitzhugh_nagumo", "van_der_pol",
           "lorenz", "kuramoto"]


def _rk4(sys: ODESystem, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step RK4 integrated at `sys.dt`, sampled every
    `sys.sample_every` steps. Returns (t, Y) with Y shape
    (n_points, state_dim) after discarding the burn-in transient. `t` is
    the true continuous time of each kept sample (for provenance plots).
    """
    se = sys.sample_every
    total_steps = (n_points + sys.burn_in) * se
    y = np.asarray(sys.y0, dtype=np.float64).copy()
    dt = sys.dt
    kept = np.empty((n_points + sys.burn_in, y.size), dtype=np.float64)
    ki = 0
    for k in range(total_steps):
        if k % se == 0:
            kept[ki] = y
            ki += 1
        t = k * dt
        k1 = sys.rhs(t, y, sys.params)
        k2 = sys.rhs(t + dt / 2, y + dt / 2 * k1, sys.params)
        k3 = sys.rhs(t + dt / 2, y + dt / 2 * k2, sys.params)
        k4 = sys.rhs(t + dt, y + dt * k3, sys.params)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(
                f"{sys.name}: non-finite state at step {k} — reduce dt")
    Y = kept[sys.burn_in:n_points + sys.burn_in]
    t = np.arange(n_points, dtype=np.float64) * dt * se
    return t, Y


def simulate(
    name: str,
    *,
    n_points: int = 4000,
    noise_std: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, ODESystem]:
    """Integrate one system. Returns (t, Y, system).

    noise_std adds i.i.d. Gaussian observation noise (state units) so the
    forecasting task is non-trivial; the integration itself stays
    deterministic given the system definition.
    """
    sys = get_system(name)
    t, Y = _rk4(sys, n_points)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        Y = Y + rng.normal(0.0, noise_std, size=Y.shape)
    return t, Y, sys


def make_ode_dataframe(
    name: str,
    *,
    n_points: int = 4000,
    noise_std: float = 0.0,
    seed: int = 0,
    start: str = "2020-01-01",
) -> tuple[pd.DataFrame, str]:
    """Build a qZETA-schema DataFrame for one synthetic system.

    Columns: DATE (synthetic timestamps spaced by the system dt, in
    seconds) + one column per state variable. This is exactly what
    `load_qzeta` expects, so the existing trainer pipeline consumes it
    unchanged. Returns (df, target_col).
    """
    t, Y, sys = simulate(name, n_points=n_points, noise_std=noise_std,
                         seed=seed)
    # Forecasting-step semantics are decoupled from integration accuracy:
    # the ODE `dt` governs RK4 fidelity, but each *sampled row* is spaced
    # exactly 1 hour apart so the locked protocol's `horizon_hours=H`
    # means "H steps ahead" and `window_size=24` means "24 steps of
    # history" — identical discrete-step semantics to the ~hourly qZETA
    # run. (`t` from simulate() is retained only for provenance plots.)
    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.arange(len(Y)), unit="h")
    # CRITICAL: load_qzeta parses DATE with dayfirst=True (the qZETA CSV
    # uses DD/MM/YYYY). Emitting ISO YYYY-MM-DD would be silently
    # misparsed+reordered by that loader. Serialize in the exact
    # day-first format qZETA uses so the round-trip is bit-exact and the
    # whole trainer pipeline stays unchanged.
    date_str = dates.strftime("%d/%m/%Y %H:%M:%S")
    data = {"DATE": date_str}
    obs = sys.observe(Y) if sys.observe is not None else Y
    for j, col in enumerate(sys.output_names):
        data[col] = obs[:, j]
    return pd.DataFrame(data), sys.target
