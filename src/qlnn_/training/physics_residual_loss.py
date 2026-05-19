"""Solver path — physics-informed residual training of a quantum circuit.

This is the QLNN's *solver* task (ODE_PDE_PRE_REG.md §3.1): the circuit
is trained to satisfy a differential equation by minimizing a physics
residual + IC/BC penalty, with NO supervised trajectory targets. The
held-out reference solution is used only for evaluation.

Architecture lineage — Chebyshev-feature DQC (Kyriienko, Paine, Elfving,
PRA 103 052416 (2021), arXiv:2011.10395), per the dual-verified spec
card in `refs/CIRCUIT_SPECS.md` §5 `chebyshev_dqc`:

  - Feature map: Chebyshev tower ⊗_j R_y(2·j·arccos x)  (paper Eq. 15;
    tower degree linear in qubit index). DESIGN CHOICE: qubits are
    1-indexed (j = 1..n) so the angle is 2·j·arccos x and no qubit gets
    the trivial R_y(0)=I — faithful to the "tower" construction of
    Eq. 15, which is defined up to the qubit-index origin.
  - Variational ansatz: HEA layers Rz–Rx–Rz + ring CNOT  (paper Fig. 5a).
  - Readout: total magnetization Ĉ = Σ_j ⟨Z_j⟩  (paper §III.3) — a
    scalar, exactly what a scalar ODE/PDE solution field needs.
  - DQC residual loss: the differential equation is enforced by the
    circuit-derivative residual (paper Eqs. 19–22). The paper takes the
    feature derivative d⟨Ĉ⟩/dx analytically via the shift rule
    (Eqs. 9–10, = ¼·(dφ/dx)·(⟨Ĉ⟩⁺−⟨Ĉ⟩⁻)). **We obtain the identical
    derivative via `jax.jacrev`** (reverse-mode autodiff through the
    PennyLane JAX QNode) instead of the manual shift rule. This is a
    mathematically-equivalent substitution, disclosed here per the P3a
    discipline, and it is the LOCKED convention (HANDOFF gotcha #1:
    `jax.jacrev` only — the eventual Diffrax-coupled solver forbids
    forward-mode through its `custom_vjp`; using jacrev now validates
    the exact nested-autodiff pattern the scaled solver will need).

`solver_prototype_ode` is the P3 acceptance gate: train the circuit to
solve u' = −u, u(0)=1 (exact u = e^{−t}) purely by physics residual,
and verify it recovers the analytic solution. Nothing scales until this
nested autodiff (grad_θ of a loss containing jacrev_t of the QNode)
is shown to work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optax
import pennylane as qml

from ..circuits.reuploading import _entangle


# ---------------------------------------------------------------------------
# Chebyshev-feature DQC circuit  (scalar coordinate -> scalar field value)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChebyshevDQCConfig:
    """Config for the Chebyshev-tower DQC solver circuit.

    Faithful to refs/CIRCUIT_SPECS.md §5 (Kyriienko et al. 2011.10395).
    Weight shape: (num_layers, num_qubits, 3) — the Rz,Rx,Rz HEA angles.
    """

    num_qubits: int = 4
    num_layers: int = 4
    entanglement: str = "ring"            # paper Fig. 5a nearest-neighbour
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.entanglement not in ("linear", "ring", "all_to_all"):
            raise ValueError(
                f"entanglement must be linear/ring/all_to_all, "
                f"got {self.entanglement!r}")

    @property
    def weight_shape(self) -> tuple[int, int, int]:
        return (self.num_layers, self.num_qubits, 3)


def build_chebyshev_dqc(
    cfg: ChebyshevDQCConfig | None = None,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return a JAX-interfaced solver circuit  f(x, weights) -> ⟨Ĉ⟩.

    `x` is the SCALAR solver coordinate, pre-mapped into [-1, 1] (so
    arccos is well-defined — the Chebyshev domain). Output is the total
    magnetization Σ_j ⟨Z_j⟩ (paper §III.3), a scalar.
    """
    cfg = cfg or ChebyshevDQCConfig()
    dev = qml.device(cfg.device_name, wires=cfg.num_qubits)

    @qml.qnode(dev, interface="jax")
    def circuit(x: jnp.ndarray, weights: jnp.ndarray):
        # Chebyshev tower feature map (paper Eq. 15), 1-indexed qubits.
        xc = jnp.clip(x, -1.0 + 1e-7, 1.0 - 1e-7)
        phi = jnp.arccos(xc)
        for j in range(cfg.num_qubits):
            qml.RY(2.0 * (j + 1) * phi, wires=j)
        # HEA variational layers: Rz–Rx–Rz + entangler (paper Fig. 5a).
        for layer in range(cfg.num_layers):
            for i in range(cfg.num_qubits):
                qml.RZ(weights[layer, i, 0], wires=i)
                qml.RX(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            _entangle(cfg.num_qubits, cfg.entanglement)
        # Total-magnetization readout Ĉ = Σ_j ⟨Z_j⟩ (paper §III.3).
        return qml.expval(qml.sum(*(qml.PauliZ(i)
                                    for i in range(cfg.num_qubits))))

    return circuit


# ---------------------------------------------------------------------------
# Physics-residual loss + training  (DQC, paper Eqs. 19–22)
# ---------------------------------------------------------------------------


def _affine_to_chebyshev(t, t0, t1):
    """Map the solver coordinate t ∈ [t0, t1] onto x ∈ [-1, 1]."""
    return 2.0 * (t - t0) / (t1 - t0) - 1.0


def init_solver_params(weight_shape, *, seed: int = 0) -> dict:
    """Trainable pytree: circuit weights + an output affine (scale,bias).

    The affine readout maps the measured observable Ĉ ∈ [−n, n] onto the
    target solution range. Kyriienko et al. fit a function f(x); the
    affine map of the cost observable to f is an explicit, standard part
    of the DQC universal-approximation usage (paper §III.3 / Eqs. 19–22)
    — disclosed here per the P3a discipline.
    """
    key = jax.random.PRNGKey(seed)
    return {
        "w": 0.1 * jax.random.normal(key, weight_shape),
        "s": jnp.asarray(1.0),
        "b": jnp.asarray(0.0),
    }


def make_residual_loss(
    circuit: Callable,
    rhs: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    t0: float,
    t1: float,
    u0: float,
    ic_weight: float = 0.0,          # unused: IC is hard-constrained
):
    """Build the DQC residual loss for a 1st-order ODE  du/dt = rhs(t, u).

    **Hard-constrained trial solution** (Lagaris, Likas & Fotiadis,
    IEEE TNN 1998; the standard PINN/DQC boundary technique, also how
    Kyriienko et al. 2011.10395 §IV impose boundary data):

        u(t) = u0 + (t − t0) · [ s·circuit(x(t), w) + b ]

    so u(t0) = u0 *exactly* — no soft IC penalty, and (critically here)
    no IC term anchored at the Chebyshev-singular endpoint x=−1, whose
    degenerate (data-free) feature map would otherwise pin a shared
    trivial value and corrupt the fit. The network only learns the
    correction. The input-coordinate derivative du/dt is taken with
    **jax.jacrev** (the locked convention). Loss = mean_t residual^2
    (Eqs. 19–22); `ic_weight` is retained in the signature for
    back-compat but is unused (the IC is structural).
    """

    def u_of_t(t, p):
        x = _affine_to_chebyshev(t, t0, t1)
        n = p["s"] * circuit(x, p["w"]) + p["b"]
        return u0 + (t - t0) * n

    # du/dt via reverse-mode autodiff w.r.t. the scalar coordinate t.
    du_dt = jax.jacrev(u_of_t, argnums=0)

    def loss(p, t_colloc):
        u = jax.vmap(lambda tt: u_of_t(tt, p))(t_colloc)
        ut = jax.vmap(lambda tt: du_dt(tt, p))(t_colloc)
        res = ut - rhs(t_colloc, u)
        return jnp.mean(res ** 2)

    return loss, u_of_t


@dataclass(frozen=True)
class SolverResult:
    params: dict
    final_loss: float
    t: jnp.ndarray
    u_pred: jnp.ndarray


def train_solver(
    circuit: Callable,
    rhs: Callable,
    *,
    t0: float,
    t1: float,
    u0: float,
    weight_shape: tuple[int, ...],
    n_colloc: int = 60,
    steps: int = 800,
    lr: float = 0.02,
    seed: int = 0,
    ic_weight: float = 20.0,
) -> SolverResult:
    """Train (weights, scale, bias) to satisfy the ODE by physics residual.

    This is the nested autodiff under test: `grad` (w.r.t. the param
    pytree) of a loss that itself contains `jacrev` (w.r.t. t) of the
    PennyLane QNode.
    """
    p = init_solver_params(weight_shape, seed=seed)
    # Interior collocation for the physics residual (the IC is a
    # separate explicit term at t0); excludes the Chebyshev-singular
    # endpoints x=±1 — standard DQC practice (see note in train_solver's
    # eval grid and the chebyshev_dqc spec card).
    t_colloc = jnp.linspace(t0, t1, n_colloc + 2)[1:-1]

    loss_fn, u_of_t = make_residual_loss(
        circuit, rhs, t0=t0, t1=t1, u0=u0, ic_weight=ic_weight)
    opt = optax.adam(lr)
    opt_state = opt.init(p)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    last = 0.0
    for _ in range(steps):
        last, grads = loss_and_grad(p, t_colloc)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)

    # Accuracy is assessed on the Chebyshev INTERIOR. At x=±1 the tower
    # angle 2j·arccos x degenerates (arccos(1)=0, arccos(−1)=π ⇒
    # 2jπ ≡ 0 mod 2π ⇒ R_y=I): the feature map carries no data at the
    # bare endpoints. This is an inherent property of the Chebyshev map
    # (Kyriienko et al. 2011.10395, Eq. 15) — DQC collocates/evaluates
    # on interior Chebyshev-type nodes, never bare ±1. We exclude the
    # two singular endpoints from the eval grid accordingly.
    t_eval = jnp.linspace(t0, t1, 102)[1:-1]
    u_pred = jax.vmap(lambda tt: u_of_t(tt, p))(t_eval)
    return SolverResult(params=p, final_loss=float(last),
                        t=t_eval, u_pred=u_pred)


def solver_prototype_ode(
    *, num_qubits: int = 4, num_layers: int = 5, steps: int = 1200,
    seed: int = 0,
) -> tuple[SolverResult, float]:
    # L5/1200 with the Lagaris hard-IC trial solution recovers e^{−t} to
    # interior MAE ≈ 0.003–0.007 across seeds {0,1,2} (~22s/run).
    """P3 ACCEPTANCE GATE.

    Solve u' = −u, u(0)=1 on t ∈ [0, 2] purely by physics residual
    (exact solution u = e^{−t}). Returns (result, test_MAE_vs_exact).
    If this converges, the nested input-coordinate autodiff through the
    PennyLane+JAX QNode works and the solver path can scale.
    """
    cfg = ChebyshevDQCConfig(num_qubits=num_qubits, num_layers=num_layers)
    circuit = build_chebyshev_dqc(cfg)

    def rhs(t, u):                       # du/dt = −u
        return -u

    res = train_solver(
        circuit, rhs, t0=0.0, t1=2.0, u0=1.0,
        weight_shape=cfg.weight_shape, steps=steps, seed=seed)
    exact = jnp.exp(-res.t)
    mae = float(jnp.mean(jnp.abs(res.u_pred - exact)))
    return res, mae
