"""Tests for QLNNForecaster.

Mirrors the test style used in tests/qlnn_/test_quantum_feature_encoder.py.
Uses small T, num_qubits, num_layers so each test runs in well under 30 s.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from qlnn_.models import QLNNForecaster, QLNNForecasterConfig


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------
INPUT_DIM = 5
NUM_QUBITS = 3
NUM_LAYERS = 2
T = 4
OD_INDEX = 0


def _cfg(**overrides) -> QLNNForecasterConfig:
    base = dict(
        input_dim=INPUT_DIM,
        num_qubits=NUM_QUBITS,
        num_layers=NUM_LAYERS,
        horizon_hours=1.0,
        od_index=OD_INDEX,
        delta_scale=0.1,
    )
    base.update(overrides)
    return QLNNForecasterConfig(**base)


def _model(seed: int = 0, **cfg_overrides) -> QLNNForecaster:
    cfg = _cfg(**cfg_overrides)
    return QLNNForecaster(cfg, key=jax.random.PRNGKey(seed))


def _sample(seed: int = 0):
    """Return (x, t_hours) for a single window."""
    key = jax.random.PRNGKey(seed + 100)
    x = 0.5 * jax.random.normal(key, (T, INPUT_DIM))
    # ~10-min spaced history then horizon stitched on by the model.
    t_hours = jnp.linspace(0.0, 0.5, T)
    return x, t_hours


# ----------------------------------------------------------------------
# 1. Output is a scalar
# ----------------------------------------------------------------------
def test_output_is_scalar():
    m = _model()
    x, t = _sample()
    y = m(x, t)
    assert y.shape == ()
    assert jnp.isfinite(y)


# ----------------------------------------------------------------------
# 2. Residual anchors to OD at initialization with tiny delta_scale
# ----------------------------------------------------------------------
def test_residual_anchors_to_od_last_at_init():
    # With delta_scale = 1e-3 and untrained params, prediction must be
    # within delta_scale + a small slack of the persistence baseline.
    m = _model(delta_scale=0.001)
    x, t = _sample()
    y = m(x, t)
    persistence = x[-1, OD_INDEX]
    assert float(jnp.abs(y - persistence)) < 0.0015


# ----------------------------------------------------------------------
# 3. Gradients flow to every trainable array leaf
# ----------------------------------------------------------------------
def test_gradients_flow_to_all_leaves():
    m = _model()
    x, t = _sample()

    def loss_fn(model, x, t):
        return (model(x, t)) ** 2

    grads = eqx.filter_grad(loss_fn)(m, x, t)
    leaves = jtu.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert len(leaves) >= 6  # cell W,b,circuit_weights,tau,A + forecaster heads
    # Every trainable leaf must receive a non-zero gradient signal.
    for i, leaf in enumerate(leaves):
        s = float(jnp.abs(leaf).sum())
        assert s > 0.0, f"leaf #{i} has zero gradient (shape {leaf.shape})"


# ----------------------------------------------------------------------
# 4. JIT round-trip matches eager
# ----------------------------------------------------------------------
def test_jit_round_trip():
    m = _model()
    x, t = _sample()

    @eqx.filter_jit
    def fwd(model, x, t):
        return model(x, t)

    y_eager = m(x, t)
    y_jit = fwd(m, x, t)
    assert jnp.allclose(y_eager, y_jit, atol=1e-4)


# ----------------------------------------------------------------------
# 5. vmap over batch
# ----------------------------------------------------------------------
def test_vmap_over_batch_of_samples():
    m = _model()
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    xs = jnp.stack([0.5 * jax.random.normal(k, (T, INPUT_DIM)) for k in keys])
    ts = jnp.stack([jnp.linspace(0.0, 0.5, T) for _ in range(3)])

    ys = jax.vmap(m)(xs, ts)
    assert ys.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(ys)))


# ----------------------------------------------------------------------
# 6. Bad input shapes
# ----------------------------------------------------------------------
def test_rejects_bad_input_shapes():
    m = _model()
    x, t = _sample()

    # 1-D x
    with pytest.raises(ValueError):
        m(x[0], t)

    # Mismatched t_hours length
    with pytest.raises(ValueError):
        m(x, t[:-1])

    # T < 2
    with pytest.raises(ValueError):
        m(x[:1], t[:1])

    # Wrong input_dim trailing
    with pytest.raises(ValueError):
        m(jnp.zeros((T, INPUT_DIM + 1)), t)


# ----------------------------------------------------------------------
# 7. Solver swap (tsit5 vs dopri5)
# ----------------------------------------------------------------------
def test_solver_swap_runs_for_both_tsit5_and_dopri5():
    x, t = _sample()
    for solver_name in ("tsit5", "dopri5"):
        m = _model(solver=solver_name)
        y = m(x, t)
        assert y.shape == ()
        assert bool(jnp.isfinite(y))


def test_solver_swap_numerical_equivalence():
    """Tsit5 and Dopri5 are both 5th-order RK methods with adaptive stepsize.

    On the same integration problem with the same rtol/atol they should agree
    to well within the controller tolerance. We use a slightly loose
    rtol=1e-2, atol=1e-3 because the adaptive step sequences differ between
    the two solvers and small round-off-level discrepancies accumulate over
    the multi-segment history+horizon integration.
    """
    x, t = _sample()

    # Use the SAME random key so the two models have byte-identical params.
    cfg_t = _cfg(solver="tsit5")
    cfg_d = _cfg(solver="dopri5")
    key = jax.random.PRNGKey(0)
    m_t = QLNNForecaster(cfg_t, key=key)
    m_d = QLNNForecaster(cfg_d, key=key)

    y_t = m_t(x, t)
    y_d = m_d(x, t)

    assert jnp.allclose(y_t, y_d, rtol=1e-2, atol=1e-3), (
        f"tsit5 vs dopri5 disagree: {float(y_t)} vs {float(y_d)}"
    )


# ----------------------------------------------------------------------
# 8. Reproducibility under same key
# ----------------------------------------------------------------------
def test_reproducible_under_same_key():
    m1 = _model(seed=42)
    m2 = _model(seed=42)

    leaves1 = jtu.tree_leaves(eqx.filter(m1, eqx.is_array))
    leaves2 = jtu.tree_leaves(eqx.filter(m2, eqx.is_array))

    assert len(leaves1) == len(leaves2)
    for a, b in zip(leaves1, leaves2):
        assert jnp.allclose(a, b)

    m3 = _model(seed=1)
    leaves3 = jtu.tree_leaves(eqx.filter(m3, eqx.is_array))
    # At least one leaf must differ between distinct seeds.
    any_diff = any(
        a.shape == c.shape and not jnp.allclose(a, c) for a, c in zip(leaves1, leaves3)
    )
    assert any_diff


# ----------------------------------------------------------------------
# 9. Config validation
# ----------------------------------------------------------------------
def test_rejects_non_monotone_t_hours():
    """t_hours must be strictly increasing.

    A duplicate or out-of-order timestamp leaves a dt <= 0 segment, which
    silently no-ops (dt == 0) or produces NaN (dt < 0). The model guards
    via `eqx.error_if`, which raises both eager and under JIT.
    """
    m = _model()
    x, _ = _sample()

    # Duplicate timestamp.
    t_dup = jnp.array([0.0, 0.0, 0.2, 0.4])
    with pytest.raises(eqx.EquinoxRuntimeError):
        m(x, t_dup)

    # Out-of-order timestamp.
    t_back = jnp.array([0.0, 0.2, 0.1, 0.4])
    with pytest.raises(eqx.EquinoxRuntimeError):
        m(x, t_back)

    # Same guard must trip under JIT.
    @eqx.filter_jit
    def fwd(model, x, t):
        return model(x, t)

    with pytest.raises(eqx.EquinoxRuntimeError):
        fwd(m, x, t_dup).block_until_ready()


def test_config_post_init_validation():
    with pytest.raises(ValueError):
        QLNNForecasterConfig(input_dim=INPUT_DIM, solver="bogus")
    with pytest.raises(ValueError):
        QLNNForecasterConfig(input_dim=INPUT_DIM, od_index=INPUT_DIM)  # out of range
    with pytest.raises(ValueError):
        QLNNForecasterConfig(input_dim=INPUT_DIM, od_index=-1)
    with pytest.raises(ValueError):
        QLNNForecasterConfig(input_dim=INPUT_DIM, delta_scale=0.0)
    with pytest.raises(ValueError):
        QLNNForecasterConfig(input_dim=INPUT_DIM, delta_scale=-0.1)
    with pytest.raises(ValueError):
        QLNNForecasterConfig(input_dim=INPUT_DIM, horizon_hours=0.0)
    with pytest.raises(ValueError):
        QLNNForecasterConfig(input_dim=INPUT_DIM, horizon_hours=-1.0)


# ----------------------------------------------------------------------
# Optional: timing diagnostic (not strict — printed for the report)
# ----------------------------------------------------------------------
def test_timing_smoke(capsys):
    m = _model()
    x, t = _sample()

    # Eager
    t0 = time.perf_counter()
    y = m(x, t)
    y.block_until_ready()
    eager_s = time.perf_counter() - t0

    @eqx.filter_jit
    def fwd(model, x, t):
        return model(x, t)

    # JIT warm-up + cached call
    fwd(m, x, t).block_until_ready()
    t0 = time.perf_counter()
    y2 = fwd(m, x, t)
    y2.block_until_ready()
    jit_s = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\n[timing] eager={eager_s*1000:.1f} ms  jit_cached={jit_s*1000:.2f} ms")
    assert jnp.isfinite(y) and jnp.isfinite(y2)
