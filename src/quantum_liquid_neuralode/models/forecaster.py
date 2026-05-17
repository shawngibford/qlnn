from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .liquid_cell import LiquidCell


ODEMethod = Literal["euler", "rk4", "dopri5"]


@dataclass(frozen=True)
class LiquidODForecasterConfig:
    input_size: int
    hidden_size: int
    horizon_hours: float
    forecast_steps: int  # used for fixed-step solvers (euler/rk4); ignored for dopri5
    od_index: int
    # Initial value of the learnable delta-scale (raw OD units). The forecaster
    # exposes ``delta_scale_init`` as the user-facing knob; the trained value
    # is read back via ``model.delta_scale().item()``. ``delta_scale_min`` is
    # the soft floor (softplus + min) preventing the parameter from collapsing
    # to zero.
    delta_scale_init: float = 1.0
    delta_scale_min: float = 0.01
    tau_min: float = 0.1
    ode_method: ODEMethod = "euler"
    # torchdiffeq adaptive tolerances (only used when ode_method == "dopri5")
    rtol: float = 1e-3
    atol: float = 1e-4


def _inv_softplus(y: float) -> float:
    """Inverse of softplus: x such that log(1 + exp(x)) = y, for y > 0.

    Used to initialize the unconstrained parameter so that
    ``softplus(x) + min == delta_scale_init`` (when feasible).
    """
    if y <= 0:
        raise ValueError(f"inverse softplus requires y > 0, got {y}")
    # Numerically: log(expm1(y)) — stable for moderate y. For very large y the
    # result is approximately y itself; for very small y it's log(y).
    return float(math.log(math.expm1(y)))


class _ConstantInputVectorField(nn.Module):
    """Wraps a LiquidCell so torchdiffeq.odeint can call it as f(t, h).

    The input x is held constant over the integration interval. We rebind .x
    via .set_x(x) before each odeint call.
    """

    def __init__(self, cell: LiquidCell) -> None:
        super().__init__()
        self.cell = cell
        self._x: Tensor | None = None

    def set_x(self, x: Tensor) -> None:
        self._x = x

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        if self._x is None:
            raise RuntimeError("_ConstantInputVectorField.set_x must be called before forward")
        return self.cell(h, self._x, t=t)


class LiquidODForecaster(nn.Module):
    """Forecast OD(t + horizon) given a history window up to time t.

    Modeling choice:
    - Encode the first window step to the initial hidden state h_0.
    - Evolve h forward over the *history* using the observed per-step inputs
      (one ODE solve per step, with the step's input held constant; this
      preserves the asynchronous-sampling property of a true neural ODE).
    - Evolve h over the *forecast horizon* with the last observed input held
      constant.
    - Predict a residual delta around persistence:
          OD(t+h) = OD(t) + tanh(delta_head(h)) * delta_scale
      where ``delta_scale = softplus(delta_scale_unconstrained) + delta_scale_min``
      is a LEARNABLE positive scalar. This removes the hard cap that previously
      bottle-necked the model near persistence: with the old fixed
      ``delta_scale=0.1`` the model could not represent 1-h deltas larger than
      ±0.1 OD even though the log-phase regime exhibits larger jumps. The
      softplus + floor parameterization lets the model expand its delta
      headroom while staying strictly positive.

    ODE methods:
    - "euler"  — explicit Euler with `forecast_steps` sub-steps per integration interval.
                 Fast, MPS-friendly. (default)
    - "rk4"    — classical Runge-Kutta 4, also fixed-step.
    - "dopri5" — Dormand-Prince adaptive via torchdiffeq.odeint. More principled
                 for a Neural-ODE paper claim, but heavier per-step.
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        horizon_hours: float,
        forecast_steps: int,
        od_index: int,
        delta_scale_init: float | None = None,
        delta_scale_min: float = 0.01,
        tau_min: float = 0.1,
        ode_method: ODEMethod = "euler",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        # Legacy alias kept so existing scripts/YAML configs (`delta_scale: 0.1`)
        # keep working. When supplied, treated as the init value of the now-
        # learnable parameter.
        delta_scale: float | None = None,
    ) -> None:
        super().__init__()

        if delta_scale_init is None and delta_scale is None:
            delta_scale_init = 1.0
        elif delta_scale is not None:
            if delta_scale_init is not None:
                raise ValueError(
                    "pass either `delta_scale_init` (new) or `delta_scale` (legacy alias), not both"
                )
            delta_scale_init = float(delta_scale)
        # By here delta_scale_init is a float.
        delta_scale_init = float(delta_scale_init)  # type: ignore[arg-type]

        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be > 0")
        if forecast_steps <= 0:
            raise ValueError("forecast_steps must be positive")
        if not (0 <= od_index < input_size):
            raise ValueError("od_index out of range")
        if delta_scale_init <= 0:
            raise ValueError("delta_scale_init must be > 0")
        if delta_scale_min <= 0:
            raise ValueError("delta_scale_min must be > 0")
        if delta_scale_init <= delta_scale_min:
            raise ValueError(
                f"delta_scale_init ({delta_scale_init}) must exceed delta_scale_min "
                f"({delta_scale_min}) so the softplus pre-image is well defined"
            )
        if ode_method not in ("euler", "rk4", "dopri5"):
            raise ValueError(f"unknown ode_method: {ode_method}")

        self.horizon_hours = float(horizon_hours)
        self.forecast_steps = int(forecast_steps)
        self.od_index = int(od_index)
        self.delta_scale_min = float(delta_scale_min)
        self.ode_method = ode_method
        self.rtol = float(rtol)
        self.atol = float(atol)

        # Initialize the unconstrained parameter so that at init
        # softplus(x) + min == delta_scale_init.
        init_unconstrained = _inv_softplus(float(delta_scale_init) - float(delta_scale_min))
        self.delta_scale_unconstrained = nn.Parameter(
            torch.tensor(init_unconstrained, dtype=torch.float32)
        )

        self.encoder = nn.Linear(input_size, hidden_size)
        self.cell = LiquidCell(input_size=input_size, hidden_size=hidden_size, tau_min=tau_min)
        self.delta_head = nn.Linear(hidden_size, 1)

        if ode_method == "dopri5":
            self._vf = _ConstantInputVectorField(self.cell)
        else:
            self._vf = None

    @classmethod
    def from_config(cls, cfg: LiquidODForecasterConfig) -> "LiquidODForecaster":
        return cls(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            horizon_hours=cfg.horizon_hours,
            forecast_steps=cfg.forecast_steps,
            od_index=cfg.od_index,
            delta_scale_init=cfg.delta_scale_init,
            delta_scale_min=cfg.delta_scale_min,
            tau_min=cfg.tau_min,
            ode_method=cfg.ode_method,
            rtol=cfg.rtol,
            atol=cfg.atol,
        )

    def delta_scale(self) -> Tensor:
        """Current value of the (learnable, strictly-positive) delta scale.

        Use ``.item()`` or ``float(...)`` for a Python scalar, or call this
        inside a forward pass to keep the gradient connection.
        """
        return F.softplus(self.delta_scale_unconstrained) + self.delta_scale_min

    def _integrate(self, *, h: Tensor, x: Tensor, dt: Tensor | float, n_substeps: int) -> Tensor:
        """Integrate dh/dt = cell(h, x) over an interval of length `dt`.

        Args:
            h: (batch, hidden_size)
            x: (batch, input_size) — held constant over the interval
            dt: scalar (float) or (batch,) tensor giving interval length in hours
            n_substeps: number of solver sub-steps within the interval (fixed-step solvers)

        Returns h after the interval.
        """
        if self.ode_method == "euler":
            return self._integrate_euler(h=h, x=x, dt=dt, n_substeps=n_substeps)
        if self.ode_method == "rk4":
            return self._integrate_rk4(h=h, x=x, dt=dt, n_substeps=n_substeps)
        if self.ode_method == "dopri5":
            return self._integrate_dopri5(h=h, x=x, dt=dt)
        raise RuntimeError(f"unhandled ode_method: {self.ode_method}")

    def _scalar_or_batched_dt(self, dt: Tensor | float, batch: int, device: torch.device) -> Tensor:
        if isinstance(dt, Tensor):
            if dt.ndim == 0:
                return dt.expand(batch).to(device)
            if dt.shape == (batch,):
                return dt.to(device)
            raise ValueError(f"unexpected dt shape: {tuple(dt.shape)}")
        return torch.full((batch,), float(dt), device=device, dtype=torch.float32)

    def _integrate_euler(self, *, h: Tensor, x: Tensor, dt: Tensor | float, n_substeps: int) -> Tensor:
        dt_t = self._scalar_or_batched_dt(dt, batch=h.shape[0], device=h.device).unsqueeze(-1)
        sub = dt_t / float(n_substeps)
        for _ in range(n_substeps):
            h = h + sub * self.cell(h, x)
        return h

    def _integrate_rk4(self, *, h: Tensor, x: Tensor, dt: Tensor | float, n_substeps: int) -> Tensor:
        dt_t = self._scalar_or_batched_dt(dt, batch=h.shape[0], device=h.device).unsqueeze(-1)
        sub = dt_t / float(n_substeps)
        for _ in range(n_substeps):
            k1 = self.cell(h, x)
            k2 = self.cell(h + 0.5 * sub * k1, x)
            k3 = self.cell(h + 0.5 * sub * k2, x)
            k4 = self.cell(h + sub * k3, x)
            h = h + (sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return h

    def _integrate_dopri5(self, *, h: Tensor, x: Tensor, dt: Tensor | float) -> Tensor:
        # torchdiffeq.odeint needs scalar timesteps. We support per-batch dt by
        # normalizing the integration interval to [0, 1] and rescaling the
        # vector field by dt — equivalent change of variables.
        from torchdiffeq import odeint

        dt_t = self._scalar_or_batched_dt(dt, batch=h.shape[0], device=h.device).unsqueeze(-1)

        # Rescaled vector field: dh/du = dt * f(h, x), where u in [0, 1].
        assert self._vf is not None  # set in __init__ when ode_method == "dopri5"
        vf = self._vf
        vf.set_x(x)

        class _Scaled(nn.Module):
            def __init__(self, base: nn.Module, scale: Tensor) -> None:
                super().__init__()
                self.base = base
                self.scale = scale  # (batch, 1)

            def forward(self, t: Tensor, h_: Tensor) -> Tensor:
                return self.scale * self.base(t, h_)

        scaled = _Scaled(vf, dt_t)

        u = torch.tensor([0.0, 1.0], device=h.device, dtype=h.dtype)
        sol = odeint(scaled, h, u, method="dopri5", rtol=self.rtol, atol=self.atol)
        # sol shape: (2, batch, hidden) — take the endpoint.
        return sol[-1]

    def forward(self, x: Tensor, t_hours: Tensor) -> Tensor:
        """Forecast OD at t_end + horizon.

        Args:
            x: (batch, T, input_size)
            t_hours: (batch, T) monotonically increasing, in hours (window-relative ok)

        Returns:
            od_pred: (batch,)
        """
        if x.ndim != 3:
            raise ValueError(f"x must be 3D (batch, T, input_size), got {tuple(x.shape)}")
        if t_hours.ndim != 2:
            raise ValueError(f"t_hours must be 2D (batch, T), got {tuple(t_hours.shape)}")
        if x.shape[0] != t_hours.shape[0] or x.shape[1] != t_hours.shape[1]:
            raise ValueError("x and t_hours must match in batch and T")
        if x.shape[1] < 2:
            raise ValueError("Need at least 2 time points in the history window")

        _, T, _ = x.shape

        # History: evolve over each observed sub-interval with that step's input.
        h = self.encoder(x[:, 0, :])
        for i in range(T - 1):
            dt = t_hours[:, i + 1] - t_hours[:, i]
            # During history we use 1 sub-step per observed interval — matches
            # the original Euler script's semantics. The actual dt is data-driven.
            h = self._integrate(h=h, x=x[:, i, :], dt=dt, n_substeps=1)

        # Forecast horizon: evolve with constant inputs at t_end.
        x_last = x[:, -1, :]
        h = self._integrate(
            h=h,
            x=x_last,
            dt=self.horizon_hours,
            n_substeps=self.forecast_steps,
        )

        # Residual delta around persistence (delta_scale is now learnable).
        od_last = x_last[:, self.od_index]
        delta = torch.tanh(self.delta_head(h).squeeze(-1)) * self.delta_scale()
        return od_last + delta
