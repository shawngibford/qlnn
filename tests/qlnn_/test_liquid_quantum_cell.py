import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from qlnn_.cells import LiquidQuantumCell, LiquidQuantumCellConfig


def _cell(
    input_dim: int = 7,
    num_qubits: int = 4,
    num_layers: int = 2,
    tau_min: float = 0.1,
    tau_init: float = 1.0,
    seed: int = 0,
) -> LiquidQuantumCell:
    cfg = LiquidQuantumCellConfig(
        input_dim=input_dim,
        num_qubits=num_qubits,
        num_layers=num_layers,
        tau_min=tau_min,
        tau_init=tau_init,
    )
    return LiquidQuantumCell(cfg, key=jax.random.PRNGKey(seed))


def test_output_shape_and_finite():
    cell = _cell()
    h = jnp.linspace(-0.5, 0.5, 4)
    x = jnp.arange(7.0)
    dh = cell(0.0, h, x)
    assert dh.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(dh)))


def test_tau_is_positive_and_above_min():
    tau_min = 0.25
    cell = _cell(tau_min=tau_min, tau_init=1.5)
    tau = cell.tau()
    assert tau.shape == (cell.config.num_qubits,)
    assert bool(jnp.all(tau > 0.0))
    assert float(tau.min()) >= tau_min - 1e-7


def test_tau_init_matches_request():
    tau_min, tau_init = 0.1, 1.0
    cell = _cell(tau_min=tau_min, tau_init=tau_init)
    tau = cell.tau()
    assert jnp.allclose(tau, jnp.full_like(tau, tau_init), atol=1e-4)

    # Also check a non-default tau_init.
    cell2 = _cell(tau_min=0.2, tau_init=2.5)
    assert jnp.allclose(cell2.tau(), jnp.full((4,), 2.5), atol=1e-4)


def test_gradients_flow_to_all_trainable_params():
    cell = _cell()
    h = jnp.linspace(-0.5, 0.5, 4)
    x = jnp.arange(7.0)

    def loss_fn(cell, h, x):
        return jnp.mean(cell(0.0, h, x))

    grads = eqx.filter_grad(loss_fn)(cell, h, x)

    # tau_unconstrained
    g_tau = grads.tau_unconstrained
    assert g_tau is not None
    assert float(jnp.abs(g_tau).sum()) > 0.0

    # A
    g_A = grads.A
    assert g_A is not None
    assert float(jnp.abs(g_A).sum()) > 0.0

    # encoder.W, encoder.b, encoder.circuit_weights
    g_W = grads.encoder.W
    g_b = grads.encoder.b
    g_cw = grads.encoder.circuit_weights

    assert float(jnp.abs(g_W).sum()) > 0.0
    # b starts at zero and is wrapped through tanh — gradient should still flow
    # because tanh has nonzero derivative around 0.
    assert float(jnp.abs(g_b).sum()) > 0.0
    assert float(jnp.abs(g_cw).sum()) > 0.0


def test_vmap_over_batch():
    cell = _cell()
    B, Q, F = 5, cell.config.num_qubits, cell.config.input_dim

    # Different x per row, so outputs should differ across the batch.
    key = jax.random.PRNGKey(123)
    k_h, k_x = jax.random.split(key, 2)
    h_batch = 0.1 * jax.random.normal(k_h, (B, Q))
    x_batch = jax.random.normal(k_x, (B, F))

    batched = jax.vmap(lambda h_, x_: cell(0.0, h_, x_))
    dh = batched(h_batch, x_batch)
    assert dh.shape == (B, Q)
    # Rows must differ (different x ⇒ different q ⇒ different dh).
    diffs = jnp.abs(dh - dh[0:1]).sum(axis=1)
    assert float(diffs[1:].max()) > 1e-6


def test_rejects_bad_h_shape():
    cell = _cell()
    x = jnp.arange(7.0)
    with pytest.raises(ValueError):
        cell(0.0, jnp.zeros(3), x)  # wrong hidden dim
    with pytest.raises(ValueError):
        cell(0.0, jnp.zeros((1, 4)), x)  # wrong rank


def test_rejects_bad_x_shape():
    cell = _cell()
    h = jnp.zeros(4)
    with pytest.raises(ValueError):
        cell(0.0, h, jnp.zeros(6))  # wrong input_dim
    with pytest.raises(ValueError):
        cell(0.0, h, jnp.zeros((1, 7)))  # wrong rank


def test_reproducible_under_same_key():
    c1 = _cell(seed=42)
    c2 = _cell(seed=42)
    assert jnp.allclose(c1.encoder.W, c2.encoder.W)
    assert jnp.allclose(c1.encoder.circuit_weights, c2.encoder.circuit_weights)
    assert jnp.allclose(c1.tau_unconstrained, c2.tau_unconstrained)
    assert jnp.allclose(c1.A, c2.A)

    c3 = _cell(seed=7)
    # Encoder leaves differ across keys; tau/A are deterministic so they match.
    assert not jnp.allclose(c1.encoder.W, c3.encoder.W)


def test_config_validates_tau_init_above_tau_min():
    with pytest.raises(ValueError):
        LiquidQuantumCellConfig(input_dim=7, tau_init=0.05, tau_min=0.1)
    # Equality is also invalid (softplus has no real preimage of 0).
    with pytest.raises(ValueError):
        LiquidQuantumCellConfig(input_dim=7, tau_init=0.1, tau_min=0.1)


def test_config_rejects_tau_min_above_one():
    """Stability guard: tau_min > 1 lets the leak coefficient flip sign.

    Leak coefficient is (1/tau + q(x)) with q(x) ∈ [-1, 1]. If tau_min > 1
    then 1/tau_min < 1 and q(x) = -1 makes the leak negative — cell goes
    from contractive to exponentially growing.
    """
    # tau_min just above 1 is rejected.
    with pytest.raises(ValueError, match="tau_min"):
        LiquidQuantumCellConfig(input_dim=7, tau_min=1.5, tau_init=2.0)
    with pytest.raises(ValueError, match="tau_min"):
        LiquidQuantumCellConfig(input_dim=7, tau_min=1.01, tau_init=2.0)
    # The boundary tau_min == 1.0 is accepted (1/tau_min == 1, still safe).
    cfg = LiquidQuantumCellConfig(input_dim=7, tau_min=1.0, tau_init=2.0)
    assert cfg.tau_min == 1.0


def test_jit_round_trip():
    cell = _cell()
    h = jnp.linspace(-0.3, 0.4, 4)
    x = jnp.arange(7.0)

    @eqx.filter_jit
    def fwd(cell, h, x):
        return cell(0.0, h, x)

    y_eager = cell(0.0, h, x)
    y_jit = fwd(cell, h, x)
    assert jnp.allclose(y_eager, y_jit, atol=1e-5)
