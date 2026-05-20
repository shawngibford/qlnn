"""Faithfulness tests for `qcpinn` (Farea, Khan, Celebi 2025).

The strongest unit-test hooks per CIRCUIT_SPECS §3 are the closed-form
per-topology counts of Table 2 (paper p.7), independently corroborated
by the worked example at n=5, L=1 (paper p.15) and by the P3a
dual-check. All 4 topologies are asserted on both `params` AND
`two-qubit gates`; depth is asserted on Cascade and Cross-mesh (the
double-verified pair).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from qlnn_.circuits.qcpinn import (
    QCPINNConfig,
    build_qcpinn,
    build_qcpinn_circuit,
    init_qcpinn_weights,
    n_trainable_pqc_params,
)


# ---------- Hook 1: closed-form trainable PQC param count (Table 2) --------

@pytest.mark.parametrize("topology,expected", [
    ("Alternate",  4 * (5 - 1) * 1),    # = 16
    ("Cascade",    3 * 5 * 1),          # = 15  (paper p.15 worked anchor)
    ("Cross-mesh", (5 * 5 + 4 * 5) * 1),  # = 45  (paper p.15 worked anchor)
    ("Layered",    4 * 5 * 1),          # = 20
])
def test_pqc_param_count_matches_Table_2(topology, expected):
    cfg = QCPINNConfig(num_qubits=5, num_layers=1, topology=topology)
    assert cfg.n_pqc_params == expected
    w = init_qcpinn_weights(cfg, seed=0)
    assert n_trainable_pqc_params(w, cfg) == expected


# ---------- Hook 2: closed-form two-qubit-gate count (Table 2) -------------

@pytest.mark.parametrize("topology,expected_2q", [
    ("Alternate",  (5 - 1) * 1),         # = 4
    ("Cascade",    5 * 1),               # = 5
    ("Cross-mesh", (5 * 5 - 5) * 1),     # = 20
    ("Layered",    (5 - 1) * 1),         # = 4
])
def test_two_qubit_gate_count_matches_Table_2(topology, expected_2q):
    cfg = QCPINNConfig(num_qubits=5, num_layers=1, topology=topology)
    assert cfg.n_two_qubit_gates == expected_2q

    # Static gate-level check on the tape.
    circ = build_qcpinn_circuit(cfg)
    w = init_qcpinn_weights(cfg, seed=0)
    theta = jnp.zeros((cfg.num_qubits,))
    tape = qml.workflow.construct_tape(circ)(theta, w)
    two_q = sum(1 for op in tape.operations if len(op.wires) == 2)
    assert two_q == expected_2q


# ---------- Hook 3: paper p.15 worked anchors ------------------------------

def test_paper_p15_worked_anchor_cascade():
    """Paper p.15: 'Angle-Cascade ... five qubits and a single quantum
    layer (L=1) ... depth (n+2)L = 7 ... about five entangling gates
    ... roughly 15 trainable parameters.'"""
    cfg = QCPINNConfig(num_qubits=5, num_layers=1, topology="Cascade")
    assert cfg.n_pqc_params == 15
    assert cfg.n_two_qubit_gates == 5
    assert cfg.expected_depth == 7


def test_paper_p15_worked_anchor_crossmesh():
    """Paper p.15: 'Angle-Cross-mesh ... depth (n²−n+4)L = 24 ...
    approximately 45 trainable parameters ... ≈20 entangling gates.'"""
    cfg = QCPINNConfig(num_qubits=5, num_layers=1, topology="Cross-mesh")
    assert cfg.n_pqc_params == 45
    assert cfg.n_two_qubit_gates == 20
    assert cfg.expected_depth == 24


# ---------- Hook 4: embedding rotations are NOT trained --------------------

def test_embedding_rotations_are_not_in_pqc_param_count():
    """Per §App.A 'Open-question resolution' the angle-embedding RX(θ)
    gates carry data-dependent (NOT trained) angles — Table 2 counts
    only the variational ansatz."""
    cfg = QCPINNConfig(num_qubits=5, num_layers=1, topology="Cascade")
    w = init_qcpinn_weights(cfg, seed=0)
    # The trainable PQC scalars match the closed-form formula exactly,
    # i.e. the embedding gates' 5 angles are NOT in this count.
    assert n_trainable_pqc_params(w, cfg) == cfg.n_pqc_params
    # And there is no key in the weight pytree representing trainable
    # embedding angles (the embedding is data-driven via the pre-NN).
    embed_keys = [k for k in w if "embed" in k.lower()]
    assert embed_keys == []


# ---------- Hook 5: per-qubit ⟨Z⟩ readout shape & range --------------------

def test_circuit_readout_is_n_qubit_z_in_minus1_to_1():
    cfg = QCPINNConfig(num_qubits=4, num_layers=1, topology="Cascade")
    circ = build_qcpinn_circuit(cfg)
    w = init_qcpinn_weights(cfg, seed=0)
    theta = jnp.asarray([0.3, -0.2, 0.5, 0.1])
    out = circ(theta, w)
    arr = jnp.stack(out) if isinstance(out, tuple) else out
    assert arr.shape == (4,)
    assert bool(jnp.all(arr >= -1.0 - 1e-6))
    assert bool(jnp.all(arr <= 1.0 + 1e-6))


# ---------- Hook 6: full pipeline is a scalar solver-style function --------

def test_full_pipeline_is_scalar_function_of_scalar_coordinate():
    cfg = QCPINNConfig(num_qubits=4, num_layers=1, topology="Cascade",
                       pre_hidden=8, post_hidden=8)
    f = build_qcpinn(cfg)
    w = init_qcpinn_weights(cfg, seed=0)
    y = f(jnp.asarray(0.3), w)
    assert jnp.ndim(y) == 0
    assert np.isfinite(float(y))


# ---------- Hook 7: jax.jacrev input-coord derivative is finite ------------

def test_jacrev_input_derivative_finite_for_solver_use():
    cfg = QCPINNConfig(num_qubits=4, num_layers=1, topology="Cascade",
                       pre_hidden=8, post_hidden=8)
    f = build_qcpinn(cfg)
    w = init_qcpinn_weights(cfg, seed=0)
    du_dx = jax.jacrev(lambda x: f(x, w))(jnp.asarray(0.3))
    assert np.isfinite(float(du_dx))


# ---------- Hook 8: config validates --------------------------------------

def test_config_validation():
    with pytest.raises(ValueError):
        QCPINNConfig(num_qubits=1)
    with pytest.raises(ValueError):
        QCPINNConfig(num_layers=0)
    with pytest.raises(ValueError):
        QCPINNConfig(topology="Bogus")
