import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from qlnn_ import QuantumFeatureEncoder, QuantumFeatureEncoderConfig
from qlnn_.encoders.quantum_feature_encoder import encoder_apply_batched


def _enc(input_dim=7, num_qubits=4, num_layers=3, seed=0) -> QuantumFeatureEncoder:
    cfg = QuantumFeatureEncoderConfig(input_dim=input_dim, num_qubits=num_qubits, num_layers=num_layers)
    return QuantumFeatureEncoder(cfg, key=jax.random.PRNGKey(seed))


def test_encoder_single_sample_shape_and_bounds():
    enc = _enc()
    x = jnp.arange(7.0)
    y = enc(x)
    assert y.shape == (4,)
    # Each expectation must be in [-1, 1].
    assert bool(jnp.all(y >= -1.0 - 1e-6))
    assert bool(jnp.all(y <= 1.0 + 1e-6))


def test_encoder_batched_via_vmap():
    enc = _enc()
    X = jnp.tile(jnp.arange(7.0), (5, 1))
    Y = encoder_apply_batched(enc, X)
    assert Y.shape == (5, 4)
    # All rows should be identical since the inputs are identical.
    assert jnp.allclose(Y - Y[0:1], 0.0, atol=1e-6)


def test_encoder_vmap_with_heterogeneous_inputs():
    """Different inputs must produce different outputs under vmap.

    Catches the vmap-collapse failure mode where all rows accidentally
    return the same value regardless of input (e.g. if the batch axis was
    dropped or the circuit was inadvertently broadcast).
    """
    enc = _enc()
    # 3 deliberately different feature vectors.
    X = jnp.stack([
        jnp.arange(7.0),
        jnp.linspace(-1.0, 1.0, 7),
        jnp.array([0.5, -0.5, 0.7, -0.7, 0.3, -0.3, 0.1]),
    ])
    Y = encoder_apply_batched(enc, X)
    assert Y.shape == (3, 4)

    # Every pair of rows must differ in at least one component by > 1e-6.
    for i in range(3):
        for j in range(i + 1, 3):
            row_diff = float(jnp.max(jnp.abs(Y[i] - Y[j])))
            assert row_diff > 1e-6, (
                f"rows {i} and {j} are identical to 1e-6: "
                f"Y[{i}]={Y[i]} Y[{j}]={Y[j]} (vmap collapse?)"
            )


def test_encoder_gradients_flow_through_all_leaves():
    enc = _enc()
    x = jnp.arange(7.0)
    target = jnp.zeros(4)

    def loss_fn(enc, x, target):
        return jnp.mean((enc(x) - target) ** 2)

    grads = eqx.filter_grad(loss_fn)(enc, x, target)
    leaves = jtu.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert len(leaves) >= 2  # at least W and circuit_weights
    total = sum(float(jnp.abs(g).sum()) for g in leaves)
    assert total > 0.0


def test_encoder_jit_round_trip():
    enc = _enc()
    x = jnp.arange(7.0)

    @eqx.filter_jit
    def fwd(enc, x):
        return enc(x)

    y_eager = enc(x)
    y_jit = fwd(enc, x)
    # JIT should reproduce eager forward to numerical precision.
    assert jnp.allclose(y_eager, y_jit, atol=1e-5)


def test_encoder_reproducible_under_same_key():
    e1 = _enc(seed=42)
    e2 = _enc(seed=42)
    assert jnp.allclose(e1.W, e2.W)
    assert jnp.allclose(e1.b, e2.b)
    assert jnp.allclose(e1.circuit_weights, e2.circuit_weights)

    e3 = _enc(seed=1)
    assert not jnp.allclose(e1.W, e3.W)


def test_encoder_rejects_bad_input_shape():
    enc = _enc()
    with pytest.raises(ValueError):
        enc(jnp.arange(6.0))   # wrong input_dim
    with pytest.raises(ValueError):
        encoder_apply_batched(enc, jnp.arange(7.0))  # not 2D
    with pytest.raises(ValueError):
        encoder_apply_batched(enc, jnp.zeros((3, 5)))  # wrong last dim


def test_encoder_param_count_matches_expectation():
    # input_dim=7, num_qubits=4, num_layers=3
    # W: 7*4 = 28, b: 4, circuit_weights: 3*4*3 = 36, total = 68
    enc = _enc(input_dim=7, num_qubits=4, num_layers=3)
    assert enc.num_parameters() == 68
