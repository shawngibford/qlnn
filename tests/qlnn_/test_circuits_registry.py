"""Tests for the ansatz registry / build / protocol contract."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from qlnn_.circuits import (
    AnsatzConfig,
    AnsatzProtocol,
    available,
    build,
    register,
)


BUILTINS = ("data_reuploading", "hardware_efficient", "strongly_entangling", "brickwall")


def test_all_builtins_registered():
    names = set(available())
    for name in BUILTINS:
        assert name in names, f"built-in ansatz {name!r} missing from registry"


def test_build_unknown_raises_with_listing():
    with pytest.raises(KeyError, match="unknown ansatz"):
        build(AnsatzConfig(name="not_a_real_ansatz"))


def test_register_duplicate_without_overwrite_raises():
    def fake_factory(cfg):  # pragma: no cover - the raise happens first
        raise RuntimeError
    with pytest.raises(ValueError, match="already registered"):
        register("data_reuploading", fake_factory)


def test_register_validates_name_and_factory():
    with pytest.raises(ValueError, match="non-empty string"):
        register("", lambda cfg: None)
    with pytest.raises(ValueError, match="callable"):
        register("temp_bad_factory", "not callable")  # type: ignore[arg-type]


def test_ansatz_config_validates_inputs():
    with pytest.raises(ValueError, match="non-empty string"):
        AnsatzConfig(name="")
    with pytest.raises(ValueError, match="num_qubits"):
        AnsatzConfig(name="data_reuploading", num_qubits=0)
    with pytest.raises(ValueError, match="num_layers"):
        AnsatzConfig(name="data_reuploading", num_layers=0)
    with pytest.raises(ValueError, match="params"):
        AnsatzConfig(name="data_reuploading", params="oops")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", BUILTINS)
def test_builtin_ansatz_honors_protocol_contract(name: str):
    """Every registered built-in must satisfy AnsatzProtocol:
       - output shape (num_qubits,)
       - values in [-1, 1]
       - weight_shape matches the array consumed by __call__
    """
    cfg = AnsatzConfig(name=name, num_qubits=4, num_layers=2)
    ansatz = build(cfg)

    assert isinstance(ansatz, AnsatzProtocol)
    assert ansatz.output_dim == 4

    rng = jax.random.PRNGKey(0)
    weights = 0.05 * jax.random.normal(rng, ansatz.weight_shape)
    inputs = jnp.linspace(-0.5, 0.5, 4)

    out = ansatz(inputs, weights)
    assert out.shape == (4,)
    assert jnp.all(out >= -1.0 - 1e-5)
    assert jnp.all(out <= 1.0 + 1e-5)


@pytest.mark.parametrize("name", BUILTINS)
def test_builtin_ansatz_is_differentiable(name: str):
    """Every built-in must produce finite gradients through jax.grad."""
    cfg = AnsatzConfig(name=name, num_qubits=3, num_layers=2)
    ansatz = build(cfg)

    rng = jax.random.PRNGKey(1)
    weights = 0.05 * jax.random.normal(rng, ansatz.weight_shape)
    inputs = jnp.array([0.1, -0.2, 0.3])

    def scalar(w):
        return ansatz(inputs, w).sum()

    g = jax.grad(scalar)(weights)
    assert g.shape == ansatz.weight_shape
    assert jnp.all(jnp.isfinite(g))


def test_data_reuploading_entanglement_params_change_output():
    """Changing the entanglement pattern should change the circuit output."""
    inputs = jnp.array([0.4, -0.3, 0.2, 0.1])
    rng = jax.random.PRNGKey(2)
    base = AnsatzConfig(name="data_reuploading", num_qubits=4, num_layers=2,
                        params={"entanglement": "linear"})
    alt = AnsatzConfig(name="data_reuploading", num_qubits=4, num_layers=2,
                      params={"entanglement": "all_to_all"})
    a1 = build(base)
    a2 = build(alt)
    w = 0.3 * jax.random.normal(rng, a1.weight_shape)
    assert not jnp.allclose(a1(inputs, w), a2(inputs, w), atol=1e-4)
