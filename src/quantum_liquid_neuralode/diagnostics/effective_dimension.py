"""Empirical-Fisher effective dimension (Abbas et al. 2021), PyTorch side.

Mirrors ``qlnn_/diagnostics/effective_dimension.py`` so the classical
Liquid-ODE and the QLNN are scored on the same metric. See that module's
docstring for the mathematical statement; this file is the torch/autograd
implementation.

Two interface conventions:

1. The low-level numeric helpers ``empirical_fisher`` and
   ``normalized_effective_dimension`` operate on numpy / torch tensors and
   are framework-agnostic — you give them a ``forward_scalar(theta, idx)``
   callable that returns a 0-dim torch tensor, and they take care of the
   per-sample Jacobian + Fisher accumulation. Useful for tiny synthetic
   tests.

2. The model-specific helper ``classical_forward_from_flat`` knows how to
   flatten the parameters of a ``LiquidODForecaster``, build a
   ``forward_scalar(theta, idx)`` via ``torch.func.functional_call``, and
   return both. This is what the analysis script
   (``scripts/run_effective_dimension.py``) uses.

Numerical note: Jacobians are computed on CPU regardless of the model's
training device. The QLNN side uses JAX (CPU); aligning the classical side
on CPU keeps the Fisher numerically comparable and avoids known MPS gaps.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.func import functional_call, jacrev


# ---------------------------------------------------------------------------
# Generic API
# ---------------------------------------------------------------------------
def empirical_fisher(
    forward_scalar: Callable[[Tensor, int], Tensor],
    theta_flat: Tensor,
    sample_indices: Sequence[int],
) -> Tensor:
    """Empirical Fisher = (1/n) sum_i J_i^T J_i (Gaussian-output regression).

    Args:
        forward_scalar: callable (theta_flat, sample_idx) -> 0-dim Tensor.
            Must be autograd-traceable in ``theta_flat``.
        theta_flat: (D,) parameter vector (requires_grad not required —
            ``jacrev`` will handle it).
        sample_indices: iterable of integer sample indices, length n.

    Returns:
        (D, D) empirical Fisher matrix as a float64 CPU Tensor.
    """
    theta_flat = theta_flat.detach().to(torch.float64).cpu()
    D = int(theta_flat.shape[0])
    fisher = torch.zeros((D, D), dtype=torch.float64)
    n = 0
    for idx in sample_indices:
        # We bake the integer sample index into a thunk so jacrev only sees
        # the parameter argument. For a scalar output, jacrev gives the
        # gradient (shape (D,)) — exactly what we want per sample.
        def _f(theta: Tensor, i: int = int(idx)) -> Tensor:
            return forward_scalar(theta, i)

        g = jacrev(_f)(theta_flat)  # (D,)
        g = g.to(torch.float64)
        fisher = fisher + torch.outer(g, g)
        n += 1
    if n == 0:
        raise ValueError("sample_indices must contain at least one index")
    return fisher / float(n)


def _slogdet_psd(matrix: Tensor) -> Tensor:
    """log det of a numerically-SPD matrix via eigvalsh.

    See the JAX-side twin for the rationale (eigvalsh + clip floor avoids
    spurious negative-det branches from float roundoff).
    """
    matrix = 0.5 * (matrix + matrix.T)
    eigs = torch.linalg.eigvalsh(matrix)
    eigs = torch.clamp(eigs, min=1e-30)
    return torch.sum(torch.log(eigs))


def normalized_effective_dimension(
    fisher: Tensor,
    n: int,
    gamma: float = 1.0,
) -> float:
    """Trained-theta normalized effective dimension (Abbas et al. 2021).

    Computes

        d_hat = log det( I_D + (gamma*n / (2*pi*log n)) * F_norm ) / log( n / (2*pi*log n) )

    with F_norm = F * D / trace(F) (trace-normalized).

    Args:
        fisher: (D, D) empirical Fisher (torch Tensor or array-like).
        n: number of samples used to estimate F.
        gamma: scale factor (default 1.0).

    Returns:
        Scalar effective dimension (Python float). Bounded above by D.
    """
    fisher_t = torch.as_tensor(fisher, dtype=torch.float64)
    if fisher_t.ndim != 2 or fisher_t.shape[0] != fisher_t.shape[1]:
        raise ValueError(f"fisher must be (D, D); got {tuple(fisher_t.shape)}")
    D = int(fisher_t.shape[0])
    if n < 2:
        raise ValueError(f"n must be >= 2; got {n}")

    tr = torch.trace(fisher_t)
    if float(tr) > 0:
        fisher_norm = fisher_t * (D / tr)
    else:
        fisher_norm = fisher_t  # zero Fisher → log det of I = 0

    log_n = float(np.log(n))
    kappa = (gamma * n) / (2.0 * np.pi * log_n)
    eye = torch.eye(D, dtype=torch.float64)

    log_det = _slogdet_psd(eye + kappa * fisher_norm)
    denom = float(np.log(n / (2.0 * np.pi * log_n)))
    return float(log_det.item() / denom)


def effective_dimension_curve(
    fisher_fn: Callable[[int], Tensor],
    n_values: Sequence[int],
    gamma: float = 1.0,
) -> dict:
    """Compute d_hat for several sample counts and check monotonicity."""
    n_list = [int(v) for v in n_values]
    if any(b - a < 0 for a, b in zip(n_list, n_list[1:])):
        raise ValueError("n_values must be sorted in non-decreasing order")
    d_norms: list[float] = []
    for n in n_list:
        F = fisher_fn(n)
        d_norms.append(normalized_effective_dimension(F, n=n, gamma=gamma))
    monotone = all(b - a > -1e-6 for a, b in zip(d_norms, d_norms[1:]))
    return {
        "n_values": n_list,
        "d_norms": d_norms,
        "monotonic_increasing": bool(monotone),
    }


# ---------------------------------------------------------------------------
# Classical model helpers
# ---------------------------------------------------------------------------
def flatten_model_params(model: torch.nn.Module):
    """Flatten the trainable parameters of an nn.Module into a (D,) tensor.

    Returns:
        theta_flat: (D,) float64 CPU Tensor of concatenated trainable params.
        names: list of parameter names (in flatten order).
        shapes: list of original shapes (one per name).
        unflatten: callable theta_flat -> dict[name, Tensor] suitable for
            ``functional_call``.
    """
    names: list[str] = []
    shapes: list[torch.Size] = []
    sizes: list[int] = []
    flats: list[Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        names.append(name)
        shapes.append(p.shape)
        sizes.append(int(p.numel()))
        flats.append(p.detach().to(torch.float64).cpu().reshape(-1))
    theta_flat = torch.cat(flats, dim=0)

    def unflatten(theta: Tensor) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        offset = 0
        for name, shape, size in zip(names, shapes, sizes):
            out[name] = theta[offset : offset + size].reshape(shape)
            offset += size
        return out

    return theta_flat, names, shapes, unflatten


def classical_forward_from_flat(
    model: torch.nn.Module,
    x_all: np.ndarray,    # (N, T, F)
    t_all: np.ndarray,    # (N, T)
):
    """Build (theta_flat, forward_scalar) for a LiquidODForecaster on CPU.

    The returned ``forward_scalar(theta, i)`` rebuilds the parameter dict
    from ``theta`` and calls ``model(x_i, t_i)`` via
    ``torch.func.functional_call``, returning a 0-dim tensor (the scalar
    OD-prediction for sample i).
    """
    model_cpu = model.to("cpu").eval()
    theta_flat, _names, _shapes, unflatten = flatten_model_params(model_cpu)

    # Keep buffers (none for this model, but pass through anyway via
    # functional_call's "merged" dict).
    buffers = {name: buf.detach().clone() for name, buf in model_cpu.named_buffers()}

    x_t = torch.from_numpy(x_all.astype(np.float32))  # (N, T, F)
    t_t = torch.from_numpy(t_all.astype(np.float32))  # (N, T)

    def forward_scalar(theta: Tensor, idx: int) -> Tensor:
        params = unflatten(theta.to(torch.float64))
        # functional_call wants float dtypes consistent with the model
        # (float32 in practice). Cast each param at call time.
        params32 = {k: v.to(torch.float32) for k, v in params.items()}
        merged = {**params32, **buffers}
        # Add a batch dim of 1, evaluate, then strip it back to a scalar.
        xb = x_t[idx : idx + 1]
        tb = t_t[idx : idx + 1]
        y = functional_call(model_cpu, merged, (xb, tb))
        return y.reshape(())

    return theta_flat, forward_scalar
