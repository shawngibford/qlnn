"""Faithfulness tests for `rf_qrc` (Recurrence-Free QRC, QRC-C4).

Each test corresponds to a specific unit-test hook from
`refs/CIRCUIT_SPECS.md` §4 (paper Ahmed et al., PRR 6 043082, 2024).
"""
from __future__ import annotations

import numpy as np
import pytest

from qlnn_.circuits.rf_qrc import RFQRCConfig, RFQRCForecaster


def _make(n_qubits: int = 4, input_dim: int = 3, seed: int = 0):
    cfg = RFQRCConfig(num_qubits=n_qubits, input_dim=input_dim,
                      alpha_seed=seed, beta=1e-9)
    return cfg, RFQRCForecaster(cfg)


# ---------- HOOK 1: trained params = size(W_out) only -----------------------

def test_only_readout_is_trainable():
    cfg, fc = _make(n_qubits=4, input_dim=2)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, cfg.input_dim))
    Y = rng.standard_normal((40, 3))
    fc.fit(X, Y)
    expected = (2 ** cfg.num_qubits + 1) * Y.shape[1]
    assert fc.n_trained_params == expected
    assert fc.W_out.shape == (cfg.feature_dim, Y.shape[1])


# ---------- HOOK 2: V(α) shape + range, seed-deterministic ------------------

def test_alpha_is_uniform_0_to_4pi_and_seed_deterministic():
    cfg1, fc1 = _make(n_qubits=5, seed=7)
    cfg2, fc2 = _make(n_qubits=5, seed=7)
    cfg3, fc3 = _make(n_qubits=5, seed=8)
    assert fc1.alpha.shape == (cfg1.num_qubits,)
    a1 = np.asarray(fc1.alpha)
    assert np.all(a1 >= 0.0) and np.all(a1 <= 4.0 * np.pi + 1e-9)
    np.testing.assert_array_equal(np.asarray(fc1.alpha),
                                   np.asarray(fc2.alpha))
    assert not np.allclose(np.asarray(fc1.alpha), np.asarray(fc3.alpha))


# ---------- HOOK 3: recurrence absent (no feedback into the circuit) --------

def test_recurrence_absent_quantum_step_independent_of_history():
    """The quantum reservoir at step t depends ONLY on x_in(t) and the
    frozen α — NEVER on r(t−1). Probe by reservoir-running an input
    twice in two contexts: in isolation vs after a long lead-in series.
    The raw quantum measurement r̂(t) for that input must be identical.
    (RF-QRC: leak_rate=1 ⇒ r=r̂, so the full feature vector matches.)
    """
    cfg, fc = _make(n_qubits=4, input_dim=2)
    rng = np.random.default_rng(0)
    # Fit only to populate the input-rescale (we never use W_out here).
    fc.fit(rng.standard_normal((20, cfg.input_dim)),
           rng.standard_normal((20, 1)))

    probe = np.array([[0.3, -0.4]])
    lead = rng.standard_normal((30, cfg.input_dim))
    feat_alone = fc.reservoir_features(probe)
    feat_after_lead = fc.reservoir_features(
        np.concatenate([lead, probe], axis=0))[-1:]
    np.testing.assert_allclose(feat_alone, feat_after_lead,
                                rtol=0.0, atol=1e-10)


# ---------- HOOK 4: Φ applied exactly twice per step ------------------------

def test_feature_map_applied_exactly_twice_per_step():
    """Static gate-level check: the reservoir tape, when expanded, must
    contain exactly two Hadamard gates per qubit per step (one from
    each Φ block) and no third (would mean Φ ×3 or a remaining P)."""
    import pennylane as qml
    cfg, fc = _make(n_qubits=3, input_dim=3)
    fc._fit_rescale(np.zeros((1, cfg.input_dim)))  # populate rescale
    theta = fc._theta_for_input(np.zeros(cfg.input_dim))
    tape = qml.workflow.construct_tape(fc._reservoir)(theta, fc.alpha)
    h_count = sum(1 for op in tape.operations if op.name == "Hadamard")
    assert h_count == 2 * cfg.num_qubits, (
        f"Φ should be applied twice (⇒ {2*cfg.num_qubits} H gates), "
        f"got {h_count}")


# ---------- HOOK 5: closed-form ridge — normal equation residual = 0 --------

def test_ridge_solution_satisfies_normal_equation():
    cfg, fc = _make(n_qubits=4, input_dim=2)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, cfg.input_dim))
    Y = rng.standard_normal((50, 2))
    fc.fit(X, Y)

    R = fc.reservoir_features(X).T              # (D, T)
    D = R.shape[0]
    lhs = (R @ R.T + cfg.beta * np.eye(D)) @ fc.W_out
    rhs = R @ Y
    # Eq. 3 must hold to machine precision (closed-form solve).
    np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-7)


# ---------- HOOK 6: encoding angles in [0, 2π] ------------------------------

def test_data_encoding_angles_in_zero_to_2pi():
    cfg, fc = _make(n_qubits=4, input_dim=3)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, cfg.input_dim))
    fc._fit_rescale(X)
    for x in X:
        theta = np.asarray(fc._theta_for_input(x))
        assert np.all(theta >= 0.0)
        assert np.all(theta <= 2.0 * np.pi + 1e-9)


# ---------- additional sanity: fit returns finite weights, predict shapes ---

def test_fit_and_predict_end_to_end():
    cfg, fc = _make(n_qubits=4, input_dim=2)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, cfg.input_dim))
    Y = (X[:, :1] ** 2 + 0.1 * rng.standard_normal((60, 1)))
    fc.fit(X, Y)
    Y_hat = fc.predict(X)
    assert Y_hat.shape == Y.shape
    assert np.all(np.isfinite(Y_hat))
    # And the reservoir state vector is a valid probability vector at
    # every step (paper §II C p.5).
    R = fc.reservoir_features(X)
    probs = R[:, :-1]                               # drop bias
    # JAX float32 (global x64 forbidden, locked gotcha #2): sum-to-1
    # tolerance must accommodate ~1e-6 float32 drift on 2^n entries.
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=5e-6)
    assert np.all(probs >= -1e-6)
