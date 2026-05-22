"""Tests for the non-liquid quantum cell (P7.11 commit 1).

Smoke + invariants for the τ-ablated variant of LiquidQuantumCell.
The cell must:
  - Forward (t, h, x) → dh/dt without error, shape (Q,)
  - Have NO `tau_unconstrained` parameter exposed in the PyTree
  - Have exactly Q fewer trainable parameters than LiquidQuantumCell at
    matching config (encoder + A but no τ vector)
  - Drop into Diffrax integration unchanged (same call signature)
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from qlnn_.cells import (
    LiquidQuantumCell, LiquidQuantumCellConfig,
    NonLiquidQuantumCell, NonLiquidQuantumCellConfig,
)


@pytest.fixture
def cfg():
    return NonLiquidQuantumCellConfig(
        input_dim=3, num_qubits=4, num_layers=2)


@pytest.fixture
def cell(cfg):
    return NonLiquidQuantumCell(cfg, key=jax.random.PRNGKey(0))


def test_forward_shape(cell):
    h = jnp.zeros((4,))
    x = jnp.array([0.1, -0.2, 0.3])
    out = cell(0.0, h, x)
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))


def test_no_tau_attribute(cell):
    """Non-liquid cell must NOT expose tau_unconstrained or tau()."""
    assert not hasattr(cell, "tau_unconstrained"), (
        "Non-liquid cell leaked tau_unconstrained from LiquidQuantumCell")
    assert not hasattr(cell, "tau"), (
        "Non-liquid cell leaked tau() method from LiquidQuantumCell")


def test_param_count_is_liquid_minus_Q(cell):
    """NonLiquid has exactly Q fewer params than Liquid at matching config.

    The Q-element delta is the `tau_unconstrained` vector that lives in
    the liquid cell but not in the non-liquid one.
    """
    liquid_cfg = LiquidQuantumCellConfig(
        input_dim=cell.config.input_dim,
        num_qubits=cell.config.num_qubits,
        num_layers=cell.config.num_layers,
        ring_entanglement=cell.config.ring_entanglement,
        init_w_std=cell.config.init_w_std,
        init_circuit_std=cell.config.init_circuit_std,
        ansatz=cell.config.ansatz,
    )
    liquid_cell = LiquidQuantumCell(liquid_cfg, key=jax.random.PRNGKey(0))
    delta = liquid_cell.num_parameters() - cell.num_parameters()
    assert delta == cell.config.num_qubits, (
        f"Liquid - NonLiquid param count should be exactly num_qubits"
        f" ({cell.config.num_qubits}); got delta={delta}")


def test_A_init_to_ones(cell):
    """A initializes to ones — same as LiquidQuantumCell convention."""
    assert jnp.allclose(cell.A, jnp.ones((cell.config.num_qubits,)))


def test_call_validates_shapes(cell):
    with pytest.raises(ValueError, match="h must have shape"):
        cell(0.0, jnp.zeros((3,)), jnp.zeros((3,)))
    with pytest.raises(ValueError, match="x must have shape"):
        cell(0.0, jnp.zeros((4,)), jnp.zeros((5,)))


def test_config_validates(cfg):
    """Config validation rejects invalid shapes."""
    with pytest.raises(ValueError, match="input_dim must be >= 1"):
        NonLiquidQuantumCellConfig(input_dim=0)
    with pytest.raises(ValueError, match="num_qubits must be >= 1"):
        NonLiquidQuantumCellConfig(input_dim=3, num_qubits=0)
    with pytest.raises(ValueError, match="num_layers must be >= 1"):
        NonLiquidQuantumCellConfig(input_dim=3, num_layers=0)


def test_vmap_batching(cell):
    """The cell vmaps correctly over a batch axis."""
    B = 8
    h_batch = jnp.zeros((B, cell.config.num_qubits))
    x_batch = jax.random.normal(
        jax.random.PRNGKey(1), (B, cell.config.input_dim))
    out = jax.vmap(cell, in_axes=(None, 0, 0))(0.0, h_batch, x_batch)
    assert out.shape == (B, cell.config.num_qubits)
    assert jnp.all(jnp.isfinite(out))


def test_dynamics_equals_liquid_minus_tau_leak(cell):
    """Mathematical-identity check: dh/dt of the non-liquid cell equals
    the liquid cell's dh/dt PLUS the `1/τ ⊙ h` leak that we removed.

    NonLiquid: dh/dt = -q⊙h + A⊙q
    Liquid:    dh/dt = -(1/τ + q)⊙h + A⊙q
    →   Liquid - NonLiquid = -(1/τ)⊙h.

    Building a liquid cell with matched encoder + A and verifying the
    identity at a random point.
    """
    liquid_cfg = LiquidQuantumCellConfig(
        input_dim=cell.config.input_dim,
        num_qubits=cell.config.num_qubits,
        num_layers=cell.config.num_layers,
        ring_entanglement=cell.config.ring_entanglement,
        init_w_std=cell.config.init_w_std,
        init_circuit_std=cell.config.init_circuit_std,
        ansatz=cell.config.ansatz,
    )
    liquid_cell = LiquidQuantumCell(liquid_cfg, key=jax.random.PRNGKey(0))
    # Force liquid_cell's encoder + A to match the non-liquid cell
    # exactly (same key was used in __init__, but assert by overriding).
    liquid_cell = eqx.tree_at(
        lambda c: (c.encoder, c.A),
        liquid_cell,
        (cell.encoder, cell.A))

    h = jax.random.normal(
        jax.random.PRNGKey(2), (cell.config.num_qubits,))
    x = jax.random.normal(
        jax.random.PRNGKey(3), (cell.config.input_dim,))

    dh_nl = cell(0.0, h, x)
    dh_l = liquid_cell(0.0, h, x)
    diff = dh_l - dh_nl
    expected_diff = -(1.0 / liquid_cell.tau()) * h

    assert jnp.allclose(diff, expected_diff, atol=1e-6), (
        f"NonLiquid + (-1/τ ⊙ h) should equal Liquid dh/dt. "
        f"diff={diff}, expected={expected_diff}")


def test_diffrax_integration_compat(cell):
    """The non-liquid cell drops into Diffrax with the same signature."""
    import diffrax

    def vf(t, y, args):
        return cell(t, y, args)

    term = diffrax.ODETerm(vf)
    solver = diffrax.Tsit5()
    h0 = jnp.zeros((cell.config.num_qubits,))
    x_const = jnp.array([0.1, -0.2, 0.3])
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=1.0, dt0=0.1,
        y0=h0, args=x_const,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=1024)
    assert sol.ys.shape == (1, cell.config.num_qubits)
    assert jnp.all(jnp.isfinite(sol.ys))


def test_seeded_determinism(cfg):
    """Same seed → same encoder + A → same output."""
    a = NonLiquidQuantumCell(cfg, key=jax.random.PRNGKey(42))
    b = NonLiquidQuantumCell(cfg, key=jax.random.PRNGKey(42))
    h = jnp.array([0.1, 0.2, 0.3, 0.4])
    x = jnp.array([0.5, 0.6, 0.7])
    assert jnp.allclose(a(0.0, h, x), b(0.0, h, x))
