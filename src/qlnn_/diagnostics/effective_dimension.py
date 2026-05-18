"""Empirical-Fisher effective dimension (Abbas et al. 2021) for JAX models.

Following Abbas, Sutter, Zoufal, Lucchi, Figalli, Woerner,
"The power of quantum neural networks", *Nature Computational Science* 1,
403–409 (2021), Eq. (4):

    d_{n,gamma}(M)
        = 2 * (log( (1/V_Theta) * integral_Theta sqrt(det(I_d + (gamma*n/(2*pi*log n)) * F_hat(theta))) dtheta ))
              / log( n / (2*pi*log n) )

where F_hat is the (normalized) empirical Fisher information matrix.

For a regression model with a Gaussian output and a fixed observation noise
the empirical Fisher F_hat(theta_hat) at the trained parameter theta_hat is

    F_hat(theta_hat) = (1/n) * sum_i J_i^T J_i,
    J_i = d y_hat_i / d theta  in R^D                      (D = #params).

This is the standard Gaussian-likelihood simplification (Abbas et al.,
"Methods → Empirical Fisher" and Eq. (3); see also Karakida, Akaho, Amari
2019 for the same form).

Single-theta (trained-model) specialization. Abbas et al. § "Effective
dimension of QNNs" use the trained-parameter version of d_{n,gamma}, i.e.
they drop the parameter-volume average (the integral over Theta is
intractable for any non-trivial QNN) and report

    d_hat_{n,gamma}(M, theta_hat)
        = log det( I_d + (gamma * n / (2*pi*log n)) * F_norm(theta_hat) )
          / log( n / (2*pi*log n) )

with `F_norm(theta_hat) = F_hat(theta_hat) * D / trace(F_hat(theta_hat))` —
i.e. the empirical Fisher rescaled to have trace equal to the number of
parameters. The trace normalization removes the arbitrary global scale that
a reparametrization of theta would otherwise introduce; without it the
"effective dimension" would depend on, e.g., whether you parametrize an
angle in radians or in turns. The same convention is what the Abbas et al.
reference implementation (Qiskit Machine Learning's
``EffectiveDimension`` / ``LocalEffectiveDimension``) uses.

This module computes both: the raw empirical Fisher and the normalized
effective dimension, plus a small helper for the n-sweep monotonicity
sanity check.

Numerical note: we use `slogdet` (log of det) and clip its sign to avoid
the spurious "negative determinant" branch that finite-precision eigenvalue
estimates of an SPD matrix can occasionally produce.
"""
from __future__ import annotations

# NOTE: we intentionally do NOT call jax.config.update("jax_enable_x64", True)
# here, because doing so corrupts Diffrax dtype promotion in the QLNN
# forecaster (RuntimeError: buffer.at[i].set with mismatched dtypes). The
# Fisher computation is done in float32 on the JAX side; numerical stability
# of the eigendecomposition is recovered by casting to numpy float64 only
# inside _slogdet_psd.

from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Generic API (matches the PyTorch-side mirror in
# quantum_liquid_neuralode/diagnostics/effective_dimension.py).
# ---------------------------------------------------------------------------
def empirical_fisher(
    forward_scalar: Callable[[jnp.ndarray, int], jnp.ndarray],
    theta_flat: jnp.ndarray,
    sample_indices: Sequence[int],
) -> jnp.ndarray:
    """Empirical Fisher = (1/n) sum_i J_i^T J_i (Gaussian-output regression).

    Args:
        forward_scalar: callable (theta_flat, sample_idx) -> scalar prediction.
            Must be jax-traceable in `theta_flat`.
        theta_flat: (D,) flattened parameter vector at which to evaluate F.
        sample_indices: list / sequence of integer sample indices (length n).
            Each is passed (as a Python int / static value) to `forward_scalar`.

    Returns:
        (D, D) empirical Fisher matrix.
    """
    # IMPORTANT (H-01): we intentionally accumulate in numpy float64, NOT
    # in JAX. Enabling jax_enable_x64 globally breaks Diffrax dtype
    # promotion in the QLNN forecaster (RuntimeError: buffer.at[i].set
    # with mismatched dtypes), so the module keeps JAX in float32. Per-
    # sample gradients are computed via jacrev in float32, then promoted
    # to numpy float64 for the outer-product accumulation. The headline
    # analysis driver (scripts/run_effective_dimension.py) does this
    # same pattern.
    theta_flat = jnp.asarray(theta_flat)
    D = int(theta_flat.shape[0])
    fisher_np = np.zeros((D, D), dtype=np.float64)
    n = 0
    for idx in sample_indices:
        # jacrev: reverse-mode is required for Diffrax-using models (custom_vjp).
        # For scalar outputs jacrev is also more efficient than jacfwd.
        g = jax.jacrev(forward_scalar)(theta_flat, idx)  # (D,) float32
        g_np = np.asarray(g, dtype=np.float64)
        fisher_np = fisher_np + np.outer(g_np, g_np)
        n += 1
    if n == 0:
        raise ValueError("sample_indices must contain at least one index")
    return jnp.asarray(fisher_np / float(n))


def _slogdet_psd(matrix: jnp.ndarray) -> float:
    """log det of a (numerically) SPD matrix via numpy float64 eigendecomp.

    Cast to numpy float64 ONLY for the eigenvalue extraction — keeps the JAX
    side in its native dtype (which is critical for not breaking Diffrax dtype
    promotion when the model is being evaluated to produce the Fisher).

    Falls back gracefully on tiny negative eigenvalues (which a true SPD
    matrix cannot have; they are pure roundoff) by clipping to a small
    positive floor before logging.
    """
    m = np.asarray(matrix, dtype=np.float64)
    m = 0.5 * (m + m.T)
    eigs = np.linalg.eigvalsh(m)
    eigs = np.clip(eigs, 1e-30, None)
    return float(np.sum(np.log(eigs)))


def normalized_effective_dimension(
    fisher: jnp.ndarray,
    n: int,
    gamma: float = 1.0,
) -> float:
    """Trained-theta normalized effective dimension (Abbas et al. 2021).

    Computes

        d_hat = log det( I_D + (gamma * n / (2*pi*log n)) * F_norm ) / log( n / (2*pi*log n) )

    where F_norm = F * D / trace(F) is the trace-normalized empirical Fisher.

    Args:
        fisher: (D, D) empirical Fisher matrix.
        n: number of samples used to estimate the Fisher.
        gamma: scaling parameter from Eq. (4). Default 1.0 (Abbas et al.
            visual convention).

    Returns:
        Scalar effective dimension (Python float). Bounded by D for any F.
    """
    # Work in numpy float64 throughout — keeps eigendecomp stable while
    # avoiding JAX x64 contagion (which corrupts Diffrax dtype handling).
    fisher = np.asarray(fisher, dtype=np.float64)
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError(f"fisher must be square (D, D); got shape {tuple(fisher.shape)}")
    D = int(fisher.shape[0])
    if n < 2:
        raise ValueError(f"n must be >= 2 (need log(n) > 0); got {n}")

    # Trace-normalize (removes parameter-scale arbitrariness). All-numpy.
    tr = float(np.trace(fisher))
    fisher_norm = fisher * (D / tr) if tr > 0 else fisher

    log_n = float(np.log(n))
    kappa = (gamma * n) / (2.0 * np.pi * log_n)
    eye = np.eye(D, dtype=np.float64)

    log_det = _slogdet_psd(eye + kappa * fisher_norm)
    denom = float(np.log(n / (2.0 * np.pi * log_n)))
    return float(log_det / denom)


def effective_dimension_curve(
    fisher_fn: Callable[[int], jnp.ndarray],
    n_values: Sequence[int],
    gamma: float = 1.0,
) -> dict:
    """Compute d_hat for several sample counts and check monotonicity.

    Args:
        fisher_fn: callable n -> empirical Fisher computed on n samples.
        n_values: list of n values, in increasing order.
        gamma: passed through to ``normalized_effective_dimension``.

    Returns:
        {
            "n_values": list[int],
            "d_norms": list[float],
            "monotonic_increasing": bool  (strictly increasing within 1e-6 tol),
        }
    """
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
# QLNN-specific helpers: flatten parameters of an Equinox model and build the
# scalar forward(theta_flat, sample_idx) needed by `empirical_fisher`.
# ---------------------------------------------------------------------------
def flatten_model_params(model: eqx.Module):
    """Flatten the trainable (array) leaves of an Equinox model to a vector.

    Returns:
        theta_flat: (D,) jnp.ndarray of concatenated array leaves.
        unflatten: callable theta_flat -> model with leaves repopulated.
    """
    params, static = eqx.partition(model, eqx.is_array)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = [leaf.shape for leaf in leaves]
    sizes = [int(np.prod(s)) for s in shapes]
    theta_flat = jnp.concatenate([leaf.reshape(-1) for leaf in leaves])

    def unflatten(theta: jnp.ndarray) -> eqx.Module:
        # Slice theta back into per-leaf arrays and recombine with the static
        # (non-array) parts of the model.
        new_leaves = []
        offset = 0
        for shape, size in zip(shapes, sizes):
            chunk = theta[offset : offset + size].reshape(shape)
            new_leaves.append(chunk)
            offset += size
        new_params = jax.tree_util.tree_unflatten(treedef, new_leaves)
        return eqx.combine(new_params, static)

    return theta_flat, unflatten


def qlnn_forward_from_flat(
    model: eqx.Module,
    x_all: jnp.ndarray,   # (N, T, F)
    t_all: jnp.ndarray,   # (N, T)
) -> tuple[jnp.ndarray, Callable[[jnp.ndarray, int], jnp.ndarray]]:
    """Build the (theta_flat, forward_scalar) pair an Abbas-style estimator needs.

    The returned ``forward_scalar(theta_flat, i)`` rebuilds the model from
    ``theta_flat`` and evaluates it on the i-th sample, returning a scalar.
    """
    theta_flat, unflatten = flatten_model_params(model)

    x_all = jnp.asarray(x_all)
    t_all = jnp.asarray(t_all)

    def forward_scalar(theta: jnp.ndarray, idx: int) -> jnp.ndarray:
        m = unflatten(theta)
        return m(x_all[idx], t_all[idx])

    return theta_flat, forward_scalar
