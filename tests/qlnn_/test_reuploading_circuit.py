import jax
import jax.numpy as jnp
import pytest

from qlnn_.circuits.reuploading import DataReuploadingCircuit, DataReuploadingConfig


def test_circuit_output_shape():
    cfg = DataReuploadingConfig(num_qubits=4, num_layers=2)
    c = DataReuploadingCircuit(cfg)
    inputs = jnp.zeros(4)
    weights = jnp.zeros(c.weight_shape)
    out = c(inputs, weights)
    assert out.shape == (4,)
    # Identity-init circuit + zero inputs => every qubit is |0>, ⟨Z⟩=+1.
    assert jnp.allclose(out, 1.0, atol=1e-5)


def test_circuit_expvals_in_bounds():
    cfg = DataReuploadingConfig(num_qubits=3, num_layers=3)
    c = DataReuploadingCircuit(cfg)
    inputs = jnp.array([0.5, -0.3, 1.0])
    weights = 0.1 * jax.random.normal(jax.random.PRNGKey(0), c.weight_shape)
    out = c(inputs, weights)
    assert out.shape == (3,)
    assert bool(jnp.all(out >= -1.0 - 1e-6))
    assert bool(jnp.all(out <= 1.0 + 1e-6))


def test_circuit_grad_through_inputs_and_weights():
    cfg = DataReuploadingConfig(num_qubits=3, num_layers=2)
    c = DataReuploadingCircuit(cfg)

    def loss(inputs, weights):
        return jnp.sum(c(inputs, weights) ** 2)

    inputs = jnp.array([0.5, -0.3, 1.0])
    weights = 0.1 * jax.random.normal(jax.random.PRNGKey(0), c.weight_shape)

    gx = jax.grad(loss, argnums=0)(inputs, weights)
    gw = jax.grad(loss, argnums=1)(inputs, weights)

    assert gx.shape == inputs.shape
    assert gw.shape == weights.shape
    # Non-trivial gradient mass.
    assert float(jnp.abs(gx).sum()) > 0.0
    assert float(jnp.abs(gw).sum()) > 0.0


def test_circuit_rejects_bad_shapes():
    cfg = DataReuploadingConfig(num_qubits=4, num_layers=2)
    c = DataReuploadingCircuit(cfg)
    with pytest.raises(ValueError):
        c(jnp.zeros(3), jnp.zeros(c.weight_shape))
    with pytest.raises(ValueError):
        c(jnp.zeros(4), jnp.zeros((1, 4, 3)))


def test_config_validates():
    with pytest.raises(ValueError):
        DataReuploadingConfig(num_qubits=0)
    with pytest.raises(ValueError):
        DataReuploadingConfig(num_layers=0)
