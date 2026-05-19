"""End-to-end test: QLNNForecaster trains a forward pass through each
registered ansatz, and the legacy (ansatz=None) path produces identical
behavior to before the refactor (= the registered data_reuploading factory).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from qlnn_ import QLNNForecaster, QLNNForecasterConfig
from qlnn_.circuits import AnsatzConfig


BUILTINS = ("data_reuploading", "hardware_efficient", "strongly_entangling", "brickwall")


@pytest.fixture
def fake_window():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (12, 7))
    t = jnp.linspace(0.0, 2.0, 12)
    return x, t


@pytest.mark.parametrize("name", BUILTINS)
def test_forecaster_forward_pass_through_each_ansatz(name, fake_window):
    x, t = fake_window
    cfg = QLNNForecasterConfig(
        input_dim=7, num_qubits=3, num_layers=2, horizon_hours=1.0, od_index=0,
        ansatz=AnsatzConfig(name=name, num_qubits=3, num_layers=2),
    )
    m = QLNNForecaster(cfg, key=jax.random.PRNGKey(0))
    y = m(x, t)
    assert y.shape == ()
    assert jnp.isfinite(y)


def test_legacy_default_path_matches_data_reuploading_factory(fake_window):
    """`ansatz=None` should produce the SAME prediction as
    `ansatz=AnsatzConfig(name='data_reuploading', ...)` with the same seed,
    since the encoder's `resolved_ansatz()` returns exactly that config.
    """
    x, t = fake_window
    cfg_default = QLNNForecasterConfig(
        input_dim=7, num_qubits=4, num_layers=3, horizon_hours=1.0, od_index=0,
    )
    cfg_explicit = QLNNForecasterConfig(
        input_dim=7, num_qubits=4, num_layers=3, horizon_hours=1.0, od_index=0,
        ansatz=AnsatzConfig(
            name="data_reuploading", num_qubits=4, num_layers=3,
            params={"ring_entanglement": True},
        ),
    )
    m_default = QLNNForecaster(cfg_default, key=jax.random.PRNGKey(0))
    m_explicit = QLNNForecaster(cfg_explicit, key=jax.random.PRNGKey(0))
    y1 = m_default(x, t)
    y2 = m_explicit(x, t)
    # Same seed + same circuit ⇒ identical weights and identical forward pass.
    assert jnp.allclose(y1, y2, atol=1e-6)


def test_forecaster_gradient_flows_through_alternative_ansatz(fake_window):
    """Gradients of a simple loss must reach the ansatz's circuit_weights."""
    import equinox as eqx
    x, t = fake_window
    cfg = QLNNForecasterConfig(
        input_dim=7, num_qubits=3, num_layers=2, horizon_hours=1.0, od_index=0,
        ansatz=AnsatzConfig(name="hardware_efficient", num_qubits=3, num_layers=2),
    )
    m = QLNNForecaster(cfg, key=jax.random.PRNGKey(0))

    def loss_fn(model):
        return (model(x, t) - 0.3) ** 2

    g = eqx.filter_grad(loss_fn)(m)
    cw_grad = g.cell.encoder.circuit_weights
    assert cw_grad.shape == m.cell.encoder.circuit_weights.shape
    assert jnp.all(jnp.isfinite(cw_grad))
    # Non-trivial: at least one element of the gradient should be non-zero.
    assert jnp.abs(cw_grad).max() > 0.0


def test_mismatched_ansatz_qubits_raises():
    """If the YAML accidentally puts an ansatz with wrong num_qubits, the
    encoder should refuse to construct (rather than silently masking it).
    """
    a = AnsatzConfig(name="data_reuploading", num_qubits=5, num_layers=3)
    cfg = QLNNForecasterConfig(
        input_dim=7, num_qubits=4, num_layers=3, horizon_hours=1.0, od_index=0,
        ansatz=a,
    )
    with pytest.raises(ValueError, match="ansatz.num_qubits"):
        QLNNForecaster(cfg, key=jax.random.PRNGKey(0))
