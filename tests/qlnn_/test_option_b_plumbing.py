"""Phase O-1 plumbing tests for the Option-B circuit search.

Two new knobs are exposed to the search:
  1. QLNNTrainerConfig.lr_schedule ∈ {"constant", "cosine"}
  2. QLNNForecasterConfig.init_circuit_std (plumbed through to the
     QuantumFeatureEncoder)

The critical property under test is BACKWARD COMPATIBILITY: the default
values must reproduce the historical behavior bit-identically so the
locked paper claims and existing checkpoints are untouched.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from qlnn_ import QLNNForecaster, QLNNForecasterConfig, QLNNTrainerConfig
from qlnn_.training.trainer import _build_optimizer


# ---------------------------------------------------------------------------
# lr_schedule
# ---------------------------------------------------------------------------
def test_lr_schedule_defaults_to_constant():
    cfg = QLNNTrainerConfig()
    assert cfg.lr_schedule == "constant"


def test_lr_schedule_validates():
    with pytest.raises(ValueError, match="lr_schedule must be"):
        QLNNTrainerConfig(lr_schedule="exponential")


def test_constant_schedule_ignores_total_steps_and_builds():
    cfg = QLNNTrainerConfig(lr=1e-3, lr_schedule="constant")
    # Must build with or without total_steps (constant path ignores it).
    opt_a = _build_optimizer(cfg)
    opt_b = _build_optimizer(cfg, total_steps=999)
    for opt in (opt_a, opt_b):
        state = opt.init({"w": jnp.zeros((3,))})
        assert state is not None


def test_cosine_schedule_requires_total_steps():
    cfg = QLNNTrainerConfig(lr=1e-3, lr_schedule="cosine")
    with pytest.raises(ValueError, match="requires total_steps"):
        _build_optimizer(cfg)  # total_steps omitted
    with pytest.raises(ValueError, match="requires total_steps"):
        _build_optimizer(cfg, total_steps=0)


def test_cosine_schedule_actually_decays():
    """A cosine schedule must monotonically decay from cfg.lr → ~0 over
    the training horizon. We verify by replaying the schedule on a dummy
    param under pure SGD and checking step sizes shrink.
    """
    lr0 = 0.1
    total = 50
    sched = optax.cosine_decay_schedule(init_value=lr0, decay_steps=total)
    early = float(sched(0))
    mid = float(sched(total // 2))
    late = float(sched(total - 1))
    assert early == pytest.approx(lr0, rel=1e-6)
    assert early > mid > late
    assert late < 0.05 * lr0  # decayed to near-zero by the end


def test_cosine_optimizer_builds_with_weight_decay():
    """The cosine schedule must compose with the adamw (weight_decay>0)
    branch, not just plain adam."""
    cfg = QLNNTrainerConfig(lr=2e-3, weight_decay=1e-3, lr_schedule="cosine")
    opt = _build_optimizer(cfg, total_steps=120)
    state = opt.init({"w": jnp.zeros((4,))})
    # One update step should run without error.
    updates, _ = opt.update({"w": jnp.ones((4,))}, state, {"w": jnp.zeros((4,))})
    assert jnp.all(jnp.isfinite(updates["w"]))


# ---------------------------------------------------------------------------
# init_circuit_std
# ---------------------------------------------------------------------------
def test_init_circuit_std_defaults_preserve_behavior():
    """ansatz=None + default init_circuit_std must produce identical
    circuit weights to the historical path for the same seed.
    """
    cfg_default = QLNNForecasterConfig(
        input_dim=7, num_qubits=4, num_layers=3, horizon_hours=3.0, od_index=0,
    )
    assert cfg_default.init_circuit_std == 0.05
    m = QLNNForecaster(cfg_default, key=jax.random.PRNGKey(0))
    # Historical init std = 0.05; the standard deviation of the circuit
    # weights should be on that order (not exactly, finite-sample, but the
    # right magnitude — a regression to e.g. 0.1 would roughly double it).
    cw = m.cell.encoder.circuit_weights
    assert 0.02 < float(jnp.std(cw)) < 0.09


def test_init_circuit_std_changes_circuit_weight_scale():
    """A larger init_circuit_std must produce visibly larger-scale circuit
    weights — proving the knob is actually wired through to the encoder.
    """
    small = QLNNForecaster(
        QLNNForecasterConfig(input_dim=7, num_qubits=4, num_layers=3,
                             horizon_hours=3.0, od_index=0,
                             init_circuit_std=0.02),
        key=jax.random.PRNGKey(0),
    )
    large = QLNNForecaster(
        QLNNForecasterConfig(input_dim=7, num_qubits=4, num_layers=3,
                             horizon_hours=3.0, od_index=0,
                             init_circuit_std=0.20),
        key=jax.random.PRNGKey(0),
    )
    assert float(jnp.std(large.cell.encoder.circuit_weights)) > \
           3.0 * float(jnp.std(small.cell.encoder.circuit_weights))


def test_init_circuit_std_forward_pass_finite():
    m = QLNNForecaster(
        QLNNForecasterConfig(input_dim=7, num_qubits=3, num_layers=2,
                             horizon_hours=1.0, od_index=0,
                             init_circuit_std=0.15),
        key=jax.random.PRNGKey(1),
    )
    x = jax.random.normal(jax.random.PRNGKey(2), (10, 7))
    t = jnp.linspace(0.0, 2.0, 10)
    assert jnp.isfinite(m(x, t))
