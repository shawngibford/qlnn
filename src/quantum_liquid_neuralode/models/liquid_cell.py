from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LiquidCellConfig:
    input_size: int
    hidden_size: int
    tau_min: float = 0.1


class LiquidCell(nn.Module):
    """Continuous-time RNN cell with learnable time constants.

    Dynamics:
        dh/dt = -h / tau + tanh(W_h h + W_x x)

    Notes:
    - tau is constrained positive via softplus and a small tau_min for stability.
    - This module computes the *derivative* (dh/dt). An external ODE solver
      performs time integration.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        tau_min: float = 0.1,
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if tau_min <= 0:
            raise ValueError(f"tau_min must be positive, got {tau_min}")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.tau_min = float(tau_min)

        # Learnable time constants (unconstrained); mapped to positive via softplus.
        self.tau_unconstrained = nn.Parameter(torch.randn(self.hidden_size) * 0.5)

        # Synaptic weights
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_x = nn.Linear(self.input_size, self.hidden_size, bias=True)

        self.activation = nn.Tanh()

    def tau(self) -> Tensor:
        """Positive time constants, shape: (hidden_size,)."""
        return F.softplus(self.tau_unconstrained) + self.tau_min

    def forward(self, h: Tensor, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """Compute dh/dt.

        Args:
            h: (batch, hidden_size)
            x: (batch, input_size)
            t: optional time (unused, kept for ODE-solver compatibility)

        Returns:
            dh_dt: (batch, hidden_size)
        """
        if h.ndim != 2:
            raise ValueError(f"h must have shape (batch, hidden_size), got {tuple(h.shape)}")
        if x.ndim != 2:
            raise ValueError(f"x must have shape (batch, input_size), got {tuple(x.shape)}")
        if h.shape[1] != self.hidden_size:
            raise ValueError(
                f"h.shape[1] must be hidden_size={self.hidden_size}, got {h.shape[1]}"
            )
        if x.shape[1] != self.input_size:
            raise ValueError(
                f"x.shape[1] must be input_size={self.input_size}, got {x.shape[1]}"
            )
        if h.shape[0] != x.shape[0]:
            raise ValueError(
                f"batch sizes must match: h.shape[0]={h.shape[0]} vs x.shape[0]={x.shape[0]}"
            )

        tau = self.tau()  # (hidden_size,)
        pre_activation = self.W_h(h) + self.W_x(x)
        f_h = self.activation(pre_activation)

        # Broadcast tau across batch dimension.
        dh_dt = (-h / tau) + f_h
        return dh_dt
