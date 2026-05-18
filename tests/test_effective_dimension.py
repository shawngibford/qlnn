"""Tests for the Abbas-2021 empirical-Fisher effective-dimension estimator.

Covers both the JAX-side (``qlnn_.diagnostics.effective_dimension``) and
PyTorch-side (``quantum_liquid_neuralode.diagnostics.effective_dimension``)
implementations.

What's NOT covered here is the end-to-end run on the trained checkpoints —
that's the analysis script. These tests pin down the numeric properties of
the estimator itself.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# Both modules expose the same generic API; we test them in parallel where
# possible so the two sides stay in lock-step.
import jax
import jax.numpy as jnp
from qlnn_.diagnostics import effective_dimension as ed_jax

import torch
from quantum_liquid_neuralode.diagnostics import effective_dimension as ed_torch


# ---------------------------------------------------------------------------
# 1. d_norm(F=0) == 0
# ---------------------------------------------------------------------------
def test_normalized_effective_dimension_zero_for_zero_fisher_jax() -> None:
    D = 5
    F = jnp.zeros((D, D))
    d = ed_jax.normalized_effective_dimension(F, n=500)
    # log det I = 0 ⇒ d_hat = 0.
    assert abs(d) < 1e-10


def test_normalized_effective_dimension_zero_for_zero_fisher_torch() -> None:
    D = 5
    F = torch.zeros((D, D))
    d = ed_torch.normalized_effective_dimension(F, n=500)
    assert abs(d) < 1e-10


# ---------------------------------------------------------------------------
# 2. d_norm increases (non-strictly, but for a well-conditioned F monotonically)
#    with n.
# ---------------------------------------------------------------------------
def _random_psd(D: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, D))
    return (A @ A.T) / D + 1e-3 * np.eye(D)  # SPD, well-conditioned


# NOTE on bounds: the original five tests in this section assumed d_norm
# is bounded by D and monotonically increasing in n. Both assumptions are
# WRONG for the trained-θ specialization of Abbas et al. (2021) Eq. (4) at
# finite n. The d_norm can exceed D at small n (the D bound is asymptotic),
# and monotonicity in n depends on log det(F_norm): isotropic F approaches
# D from ABOVE (decreasing), highly anisotropic F approaches D from BELOW
# (increasing). The correct behavioral property is asymptotic convergence
# to D — tested below.


def test_d_norm_isotropic_fisher_approaches_D_from_above() -> None:
    """For F = I, d_hat(n) → D as n → ∞, approaching from above.

    With F_norm = I and γ=1: d_hat(n) = D * log(1+κ) / log(κ) where
    κ = n/(2π log n). Since log(1+κ) > log(κ), d_hat(n) > D for finite n
    and d_hat(n) → D monotonically (decreasing) as n → ∞.
    """
    D = 4
    F = jnp.eye(D)
    d_500 = ed_jax.normalized_effective_dimension(F, n=500)
    d_50000 = ed_jax.normalized_effective_dimension(F, n=50000)
    # Both should exceed D for finite n; large n is closer to D.
    assert d_500 > D
    assert d_50000 > D
    assert abs(d_50000 - D) < abs(d_500 - D), (
        f"d_500={d_500}, d_50000={d_50000}: expected larger n to approach D"
    )


def test_d_norm_anisotropic_fisher_approaches_D_from_below() -> None:
    """For an anisotropic trace-normalized F with det < 1, d_hat → D from below.

    Pre-built F_norm with trace=D=4 and spread eigenvalues so det << 1.
    """
    D = 4
    # Eigenvalues 0.1, 0.2, 1.3, 2.4 sum to 4.0 (trace = D); product ≪ 1.
    eigs = jnp.asarray([0.1, 0.2, 1.3, 2.4])
    F = jnp.diag(eigs)
    d_500 = ed_jax.normalized_effective_dimension(F, n=500)
    d_50000 = ed_jax.normalized_effective_dimension(F, n=50000)
    assert d_500 < D
    assert d_50000 < D
    # The larger n is closer to D.
    assert abs(d_50000 - D) < abs(d_500 - D), (
        f"d_500={d_500}, d_50000={d_50000}: expected larger n to approach D"
    )


# ---------------------------------------------------------------------------
# 4. JAX empirical Fisher matches a finite-difference Fisher on a tiny model.
# ---------------------------------------------------------------------------
def test_empirical_fisher_jax_matches_finite_difference() -> None:
    """f(theta, x) = theta[0]^2 + theta[1] * x.

    Per-sample gradient w.r.t. theta = [2*theta[0], x].
    Empirical Fisher = (1/n) sum_i [[4*theta[0]^2, 2*theta[0]*x_i],
                                    [2*theta[0]*x_i, x_i^2]].
    """
    theta = jnp.array([0.7, -1.3])
    xs = jnp.array([0.5, 1.5, -2.0, 0.1])

    def forward_scalar(th: jnp.ndarray, i: int) -> jnp.ndarray:
        return th[0] ** 2 + th[1] * xs[i]

    F_auto = ed_jax.empirical_fisher(forward_scalar, theta, list(range(xs.shape[0])))
    F_auto = np.asarray(F_auto)

    # Analytical Fisher.
    t0 = float(theta[0])
    x_np = np.asarray(xs)
    n = x_np.shape[0]
    f00 = (1.0 / n) * np.sum(4.0 * t0 * t0 * np.ones_like(x_np))
    f01 = (1.0 / n) * np.sum(2.0 * t0 * x_np)
    f11 = (1.0 / n) * np.sum(x_np * x_np)
    F_ref = np.array([[f00, f01], [f01, f11]], dtype=np.float64)

    # Loosened from 1e-7 to 1e-4 because JAX is float32 here (we deliberately
    # do NOT enable jax_enable_x64 to avoid Diffrax dtype contagion in the
    # forecaster). Float32 max-precision is ~7 sig figs; on this small
    # synthetic the FD gradient and autodiff gradient agree to ~5 sig figs.
    np.testing.assert_allclose(F_auto, F_ref, atol=1e-4)

    # Finite-difference cross-check on a single sample to catch sign errors.
    i = 1
    eps = 1e-4
    g_fd = np.zeros(2)
    for k in range(2):
        th_plus = np.asarray(theta).copy()
        th_minus = np.asarray(theta).copy()
        th_plus[k] += eps
        th_minus[k] -= eps
        f_plus = th_plus[0] ** 2 + th_plus[1] * float(xs[i])
        f_minus = th_minus[0] ** 2 + th_minus[1] * float(xs[i])
        g_fd[k] = (f_plus - f_minus) / (2.0 * eps)
    g_auto = jax.jacfwd(lambda th: forward_scalar(th, i))(theta)
    # float32 limits FD agreement to ~1e-3 absolute on this synthetic.
    np.testing.assert_allclose(np.asarray(g_auto), g_fd, atol=1e-3)


def test_empirical_fisher_torch_matches_finite_difference() -> None:
    theta = torch.tensor([0.7, -1.3], dtype=torch.float64)
    xs = torch.tensor([0.5, 1.5, -2.0, 0.1], dtype=torch.float64)

    def forward_scalar(th: torch.Tensor, i: int) -> torch.Tensor:
        return th[0] ** 2 + th[1] * xs[i]

    F_auto = ed_torch.empirical_fisher(forward_scalar, theta, list(range(xs.shape[0])))
    F_auto_np = F_auto.numpy()

    t0 = float(theta[0])
    x_np = xs.numpy()
    n = x_np.shape[0]
    f00 = (1.0 / n) * np.sum(4.0 * t0 * t0 * np.ones_like(x_np))
    f01 = (1.0 / n) * np.sum(2.0 * t0 * x_np)
    f11 = (1.0 / n) * np.sum(x_np * x_np)
    F_ref = np.array([[f00, f01], [f01, f11]], dtype=np.float64)

    np.testing.assert_allclose(F_auto_np, F_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. JAX and torch implementations agree on the same Fisher.
# ---------------------------------------------------------------------------
def test_jax_and_torch_normalized_effective_dimension_agree() -> None:
    F_np = _random_psd(D=10, seed=7)
    n = 500
    d_jax = ed_jax.normalized_effective_dimension(jnp.asarray(F_np), n=n)
    d_torch = ed_torch.normalized_effective_dimension(torch.from_numpy(F_np), n=n)
    # Loosened from 1e-8 because the JAX side computes the Fisher in float32
    # (deliberate, to keep Diffrax happy) while the Torch side uses float64.
    # Both then convert to numpy float64 for the eigendecomp inside
    # _slogdet_psd, so the residual difference is just the Fisher-accumulation
    # precision gap.
    assert math.isclose(d_jax, d_torch, rel_tol=1e-6, abs_tol=1e-6), (d_jax, d_torch)


# ---------------------------------------------------------------------------
# 6. Trace normalization actually removes parameter rescaling.
# ---------------------------------------------------------------------------
def test_trace_normalization_invariance_torch() -> None:
    """If theta -> alpha*theta, the Fisher scales by 1/alpha^2 (since gradient
    scales by 1/alpha for chain rule); after trace-normalization the
    effective dimension should be invariant.

    We just verify trace-normalization is in effect by scaling F by an
    arbitrary positive constant — d_norm should not move.
    """
    F = torch.from_numpy(_random_psd(D=7, seed=3))
    d_a = ed_torch.normalized_effective_dimension(F, n=500)
    d_b = ed_torch.normalized_effective_dimension(123.4 * F, n=500)
    assert abs(d_a - d_b) < 1e-9


# ---------------------------------------------------------------------------
# 7. flatten/unflatten round-trips a tiny classical model.
# ---------------------------------------------------------------------------
def test_flatten_model_params_roundtrip_torch() -> None:
    """The (flatten, unflatten) pair must reconstruct identical parameters."""
    from quantum_liquid_neuralode.models import LiquidODForecaster

    model = LiquidODForecaster(
        input_size=3,
        hidden_size=4,
        horizon_hours=1.0,
        forecast_steps=2,
        od_index=0,
        delta_scale=0.1,
    )
    theta_flat, names, shapes, unflatten = ed_torch.flatten_model_params(model)
    d = unflatten(theta_flat)

    # Names match the trainable parameter set.
    expected = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names == expected

    # Reconstructed values match the originals.
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        np.testing.assert_allclose(
            d[name].detach().numpy(),
            p.detach().to(torch.float64).cpu().numpy(),
            atol=1e-12,
        )
