"""Tests for the classical LTC forecaster (P7.10 commit 1).

Smoke + invariants for the missing fourth-quadrant baseline. The
LTC cell must:
  - Forward cleanly on a single sample (T, d) → (d,)
  - Maintain tau() > tau_min for all hidden units at init AND after
    parameter updates (the softplus + min pattern is non-negotiable)
  - Drop into the existing forecaster_adapter unchanged
  - Be capacity-matched to PlainNeuralODEForecaster (within Q params,
    where Q is the hidden dimension)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from qlnn_.models.classical_ltc_forecaster import (
    ClassicalLTCCell,
    ClassicalLTCCellConfig,
    ClassicalLTCForecaster,
    ClassicalLTCForecasterConfig,
)
from qlnn_.models.plain_neuralode_forecaster import (
    PlainNeuralODEForecaster,
    PlainNeuralODEForecasterConfig,
)


@pytest.fixture
def cell_cfg():
    return ClassicalLTCCellConfig(
        input_dim=2, hidden_dim=4, activation="tanh")


@pytest.fixture
def forecaster_cfg():
    return ClassicalLTCForecasterConfig(
        input_dim=2, hidden_dim=4, activation="tanh")


@pytest.fixture
def cell(cell_cfg):
    return ClassicalLTCCell(cell_cfg, key=jax.random.PRNGKey(0))


@pytest.fixture
def forecaster(forecaster_cfg):
    return ClassicalLTCForecaster(
        forecaster_cfg, key=jax.random.PRNGKey(0))


# --- Cell-level tests -------------------------------------------------------


def test_cell_forward_shape(cell):
    h = jnp.zeros((4,))
    x = jnp.array([0.1, -0.2])
    out = cell(0.0, h, x)
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))


def test_cell_tau_positive_at_init(cell):
    tau = cell.tau()
    assert tau.shape == (4,)
    assert jnp.all(tau >= cell.config.tau_min)
    # At init, tau() should equal tau_init (within float tolerance).
    assert jnp.allclose(tau, cell.config.tau_init, atol=1e-6)


def test_cell_tau_min_holds_under_perturbation():
    """Even if tau_unconstrained is pushed very negative, tau >= tau_min."""
    cfg = ClassicalLTCCellConfig(input_dim=2, hidden_dim=4, tau_min=0.1)
    cell = ClassicalLTCCell(cfg, key=jax.random.PRNGKey(0))
    # Push tau_unconstrained to -1e6 (would yield softplus ≈ 0).
    import equinox as eqx
    cell = eqx.tree_at(
        lambda c: c.tau_unconstrained, cell,
        jnp.full((4,), -1e6))
    tau = cell.tau()
    assert jnp.all(tau >= cfg.tau_min - 1e-7)
    assert jnp.allclose(tau, cfg.tau_min, atol=1e-5)


def test_cell_capacity_matched_to_plain_neuralode(cell):
    """The LTC cell adds exactly Q parameters over PlainNeuralODECell."""
    from qlnn_.models.plain_neuralode_forecaster import (
        NeuralODECell, NeuralODECellConfig)
    plain_cfg = NeuralODECellConfig(input_dim=2, hidden_dim=4)
    plain_cell = NeuralODECell(plain_cfg, key=jax.random.PRNGKey(0))
    delta = cell.num_parameters() - plain_cell.num_parameters()
    assert delta == cell.config.hidden_dim, (
        f"LTC should add exactly hidden_dim={cell.config.hidden_dim} "
        f"params (tau_unconstrained); got delta={delta}")


def test_cell_validates_shapes(cell):
    with pytest.raises(ValueError, match="h must have shape"):
        cell(0.0, jnp.zeros((3,)), jnp.zeros((2,)))
    with pytest.raises(ValueError, match="x must have shape"):
        cell(0.0, jnp.zeros((4,)), jnp.zeros((3,)))


def test_cell_config_validates_tau():
    with pytest.raises(ValueError, match="tau_init must be > tau_min"):
        ClassicalLTCCellConfig(
            input_dim=2, hidden_dim=4, tau_init=0.1, tau_min=0.1)
    with pytest.raises(ValueError, match="tau_min must be > 0"):
        ClassicalLTCCellConfig(
            input_dim=2, hidden_dim=4, tau_init=1.0, tau_min=0.0)


# --- Forecaster-level tests -------------------------------------------------


def test_forecaster_forward_shape(forecaster):
    x = jnp.linspace(0.0, 1.0, 30).reshape(15, 2)
    y = forecaster(x)
    assert y.shape == (2,)
    assert jnp.all(jnp.isfinite(y))


def test_forecaster_residual_around_persistence(forecaster):
    """y = x[-1] + delta — delta should be bounded by delta_scale_init."""
    x = jnp.linspace(0.0, 1.0, 30).reshape(15, 2)
    y = forecaster(x)
    diff = jnp.abs(y - x[-1])
    # delta = tanh(...) * delta_scale, so |delta| <= delta_scale_init.
    assert jnp.all(diff <= forecaster.config.delta_scale_init + 1e-5)


def test_forecaster_adapter_compat(forecaster):
    """Drop-in to the standard vector-forecaster adapter."""
    from qlnn_.evaluation.forecaster_adapters import (
        make_vector_forecaster_adapter,
    )
    adapter = make_vector_forecaster_adapter(forecaster)
    history = jnp.linspace(0.0, 1.0, 30).reshape(15, 2)
    out = adapter(history, dt=0.05)
    assert out.shape == (2,)


def test_forecaster_capacity_close_to_plain_neuralode(forecaster):
    """LTC adds only the τ-unconstrained vector relative to plain NeuralODE.

    Capacity ratio must stay within factor of 2 per pre-reg §6.
    """
    plain_cfg = PlainNeuralODEForecasterConfig(
        input_dim=2, hidden_dim=4, activation="tanh")
    plain_fc = PlainNeuralODEForecaster(
        plain_cfg, key=jax.random.PRNGKey(0))
    import equinox as eqx

    def count_params(model):
        leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        return sum(int(jnp.size(leaf)) for leaf in leaves)

    n_ltc = count_params(forecaster)
    n_plain = count_params(plain_fc)
    delta = n_ltc - n_plain
    assert delta == forecaster.config.hidden_dim, (
        f"LTC forecaster should add exactly hidden_dim={forecaster.config.hidden_dim} "
        f"params over plain Neural-ODE; got {n_ltc} vs {n_plain} (delta={delta})")
    assert n_ltc <= 2 * n_plain, "Capacity ratio violates pre-reg §6"


def test_forecaster_validates_input(forecaster):
    with pytest.raises(ValueError, match="x must have shape"):
        forecaster(jnp.zeros((10, 3)))
    with pytest.raises(ValueError, match="need at least 2 time points"):
        forecaster(jnp.zeros((1, 2)))


def test_tau_is_trainable():
    """Confirm tau_unconstrained is a trainable PyTree leaf."""
    cfg = ClassicalLTCCellConfig(input_dim=2, hidden_dim=4)
    cell = ClassicalLTCCell(cfg, key=jax.random.PRNGKey(0))
    import equinox as eqx
    leaves = jax.tree_util.tree_leaves(eqx.filter(cell, eqx.is_array))
    # tau_unconstrained should be one of the leaves (shape (Q,)).
    shapes = [leaf.shape for leaf in leaves]
    assert (4,) in shapes, (
        f"tau_unconstrained (shape (4,)) not in trainable leaves: {shapes}")


def test_forecaster_seeded_determinism(forecaster_cfg):
    """Same seed → same params + same output."""
    a = ClassicalLTCForecaster(forecaster_cfg, key=jax.random.PRNGKey(42))
    b = ClassicalLTCForecaster(forecaster_cfg, key=jax.random.PRNGKey(42))
    x = jnp.linspace(0.0, 1.0, 30).reshape(15, 2)
    assert jnp.allclose(a(x), b(x))
