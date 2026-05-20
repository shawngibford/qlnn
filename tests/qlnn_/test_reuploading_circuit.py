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


def test_default_omits_terminal_block_preserves_legacy_shape():
    """Backward-compat anchor: default (terminal_block=False) keeps the
    historical (L, Q, 3) weight shape so every committed OD checkpoint
    keeps loading. The dual-check (CIRCUIT_SPECS.md §7) confirmed this
    reduced form is spectrum-equivalent to canonical Schuld Eq. 4 — the
    accessible Ω H1 depends on is unchanged."""
    cfg = DataReuploadingConfig(num_qubits=4, num_layers=3)
    assert cfg.terminal_block is False
    assert cfg.total_variational_layers == 3
    assert DataReuploadingCircuit(cfg).weight_shape == (3, 4, 3)


def test_terminal_block_adds_canonical_Schuld_Eq4_variational_layer():
    """Opt-in canonical Eq. 4 form: one more W^{(L+1)} block AFTER the
    last entangler (no data re-upload after it). Weight shape grows
    (L, Q, 3) → (L+1, Q, 3)."""
    cfg = DataReuploadingConfig(
        num_qubits=4, num_layers=3, terminal_block=True)
    assert cfg.total_variational_layers == 4
    c = DataReuploadingCircuit(cfg)
    assert c.weight_shape == (4, 4, 3)
    out = c(jnp.array([0.3, -0.2, 0.5, 0.1]),
            0.1 * jax.random.normal(jax.random.PRNGKey(0), c.weight_shape))
    assert out.shape == (4,)
    assert bool(jnp.all(out >= -1.0 - 1e-6) & jnp.all(out <= 1.0 + 1e-6))


def test_terminal_block_changes_output_vs_default():
    """The terminal W^{(L+1)} is a real circuit difference (not a
    cosmetic flag): with non-zero weights, output should differ from the
    default (terminal_block=False) at the same `weights[:L]` slice."""
    L, Q = 2, 3
    cfg_off = DataReuploadingConfig(num_qubits=Q, num_layers=L,
                                     terminal_block=False)
    cfg_on  = DataReuploadingConfig(num_qubits=Q, num_layers=L,
                                     terminal_block=True)
    c_off = DataReuploadingCircuit(cfg_off)
    c_on  = DataReuploadingCircuit(cfg_on)
    key = jax.random.PRNGKey(0)
    w_off = 0.3 * jax.random.normal(key, c_off.weight_shape)
    # Same first L W-blocks, plus a non-trivial terminal block.
    w_on = jnp.concatenate(
        [w_off, 0.3 * jax.random.normal(jax.random.PRNGKey(1),
                                         (1, Q, 3))], axis=0)
    inputs = jnp.array([0.4, -0.1, 0.7])
    o_off = c_off(inputs, w_off)
    o_on = c_on(inputs, w_on)
    # The two circuits must produce materially different per-wire
    # expectations once the W^{(L+1)} block is non-trivial.
    assert float(jnp.max(jnp.abs(o_on - o_off))) > 1e-3
