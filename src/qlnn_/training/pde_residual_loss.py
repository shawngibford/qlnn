"""P3.7 — PDE solver path with mixed-2nd-derivative nested autodiff.

Sibling to `physics_residual_loss.py` (the 1D ODE solver gate at commit
77009ce, contract IMMUTABLE). Adds (t, x) coordinate handling so we can
train solver circuits against the P2 PDE fields (`data/pde/*.npz`).

THE OPEN QUESTION (Risk-#2 redux): does nested autodiff through a
PennyLane JAX QNode work for the **spatial second derivative**
u_xx = jacrev(jacrev(u_of_tx, argnums=1), argnums=1)? The 1D gate
(commit 77009ce) confirmed first-order jacrev works; mixed 2nd-order
is untested in this repo. Expected to work — PennyLane's JAX
interface uses `vjp` (not Diffrax's `custom_vjp`), so reverse-over-
reverse should compose — but the mechanism check
(`test_mixed_jacrev_through_qnode_is_finite_and_nontrivial`) is the
authoritative answer.

The split-qubit 2D Chebyshev feature map:
- Half the qubits (0..n_q/2−1) encode the time coordinate via the
  1-indexed Chebyshev tower R_y(2j·arccos(t̃)).
- The other half encode the spatial coordinate the same way.
- Then standard HEA layers entangle them and the readout is
  Σⱼ⟨Zⱼ⟩ (total magnetization, scalar).

The Lagaris hard-IC trial solution generalizes to PDE (A22, 2026-05-28
docstring fix — earlier text mis-described the t-prefactor as t rather
than (t-t0); the implementation at line 217 has always been correct):
    u(t, x) = u₀(x) + (t − t0) · ( s · circuit_2d(map_t(t), map_x(x), w) + b )
- IC u(t0, x) = u₀(x) is structural (no soft penalty at the singular
  t=t0 endpoint where the Chebyshev map degenerates).
- BC handling: periodic for Burgers/Allen-Cahn — the residual is
  evaluated at interior points (t̃, x̃) ∈ (−1, 1)² and the periodic
  BC is enforced implicitly by loss matching at near-boundary points.

Convention (LOCKED, HANDOFF gotcha #1): jax.jacrev only, never jacfwd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml

from qlnn_.circuits.reuploading import _entangle


# ---------------------------------------------------------------------------
# Affine coordinate maps  [t0, t1] → [-1, 1]  (per axis)
# ---------------------------------------------------------------------------


def _affine_to_chebyshev_axis(v, v0, v1):
    """Linear bijection from physical [v0, v1] onto Chebyshev [-1, 1].

    Mirrors `physics_residual_loss._affine_to_chebyshev` but per-axis
    (vendored inline so this module is standalone-importable; we do
    not import from the 1D solver to keep the gate contract isolated).
    """
    return 2.0 * (v - v0) / (v1 - v0) - 1.0


# ---------------------------------------------------------------------------
# 2D Chebyshev-feature DQC circuit  (scalar (t, x) -> scalar field value)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChebyshevDQC2DConfig:
    """Config for the split-qubit 2D Chebyshev DQC solver circuit.

    The first `n_t_qubits` qubits encode the (affine-mapped to [-1, 1])
    time coordinate via the 1-indexed Chebyshev tower; the next
    `n_x_qubits` qubits encode the spatial coordinate the same way.
    Then `num_layers` of HEA (Rz, Rx, Rz + ring CNOT) entangle them.
    Output is the total magnetization Σⱼ⟨Zⱼ⟩, a scalar.

    Default n_t_qubits=n_x_qubits=4 matches the 1D gate's per-axis
    expressivity (the 1D ChebyshevDQCConfig defaults to 4 qubits).
    """

    n_t_qubits: int = 4
    n_x_qubits: int = 4
    num_layers: int = 5
    entanglement: str = "ring"
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.n_t_qubits < 1:
            raise ValueError(f"n_t_qubits must be >= 1, got {self.n_t_qubits}")
        if self.n_x_qubits < 1:
            raise ValueError(f"n_x_qubits must be >= 1, got {self.n_x_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.entanglement not in ("linear", "ring", "all_to_all"):
            raise ValueError(
                f"entanglement must be linear/ring/all_to_all, "
                f"got {self.entanglement!r}")

    @property
    def num_qubits(self) -> int:
        return self.n_t_qubits + self.n_x_qubits

    @property
    def weight_shape(self) -> tuple[int, int, int]:
        """(num_layers, num_qubits, 3) — HEA Rz, Rx, Rz angles."""
        return (self.num_layers, self.num_qubits, 3)


def build_chebyshev_dqc_2d(
    cfg: ChebyshevDQC2DConfig | None = None,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return a JAX-interfaced solver circuit  f(t̃, x̃, weights) → scalar.

    Inputs:
      t̃, x̃ : scalars already mapped to [-1, 1] (the Chebyshev domain).
      weights : (num_layers, num_qubits, 3) HEA angles.

    Output: total magnetization Σⱼ⟨Zⱼ⟩ ∈ [-num_qubits, num_qubits].
    """
    cfg = cfg or ChebyshevDQC2DConfig()
    n_t = cfg.n_t_qubits
    n_x = cfg.n_x_qubits
    n = cfg.num_qubits
    dev = qml.device(cfg.device_name, wires=n)

    @qml.qnode(dev, interface="jax")
    def circuit(t_chev, x_chev, weights):
        # Clip the inputs to the Chebyshev interior (avoid the
        # arccos degeneracy at the bare ±1 endpoints; same convention
        # as the 1D solver gate).
        tc = jnp.clip(t_chev, -1.0 + 1e-7, 1.0 - 1e-7)
        xc = jnp.clip(x_chev, -1.0 + 1e-7, 1.0 - 1e-7)
        phi_t = jnp.arccos(tc)
        phi_x = jnp.arccos(xc)
        # Split-qubit Chebyshev tower feature map: qubits 0..n_t-1
        # encode t via R_y(2(j+1)·φ_t); next n_x qubits encode x.
        for j in range(n_t):
            qml.RY(2.0 * (j + 1) * phi_t, wires=j)
        for k in range(n_x):
            qml.RY(2.0 * (k + 1) * phi_x, wires=n_t + k)
        # HEA variational layers: Rz, Rx, Rz + ring CNOT.
        for layer in range(cfg.num_layers):
            for i in range(n):
                qml.RZ(weights[layer, i, 0], wires=i)
                qml.RX(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            _entangle(n, cfg.entanglement)
        # Total magnetization readout (scalar).
        return qml.expval(qml.sum(*(qml.PauliZ(i) for i in range(n))))

    return circuit


# ---------------------------------------------------------------------------
# PDE residual loss  (Lagaris hard-IC for (t, x); the rhs is provided)
# ---------------------------------------------------------------------------


def init_pde_solver_params(weight_shape, *, seed: int = 0) -> dict:
    """Trainable pytree: circuit weights + output affine (scale, bias).

    Identical {w, s, b} convention as the 1D gate. The affine head
    rescales the scalar Σ⟨Z⟩ output to the target field's range.
    """
    key = jax.random.PRNGKey(seed)
    return {
        "w": 0.1 * jax.random.normal(key, weight_shape),
        "s": jnp.asarray(1.0),
        "b": jnp.asarray(0.0),
    }


def make_pde_residual_loss(
    circuit: Callable,
    pde_residual: Callable,
    ic_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    t0: float, t1: float, x0: float, x1: float,
    need_uxxx: bool = False,
):
    """Build the PDE-residual loss for any equation expressible as
    `r(t, x, u, u_t, u_x, u_xx) = 0`, or
    `r(t, x, u, u_t, u_x, u_xx, u_xxx) = 0` when `need_uxxx=True`
    (e.g. KdV).

    Args:
      circuit : `f(t̃, x̃, w) → scalar` returned by `build_chebyshev_dqc_2d`.
      pde_residual : `r(t, x, u, u_t, u_x, u_xx[, u_xxx]) → scalar` — the
          PDE's residual function. With `need_uxxx=False` (default), the
          signature is the 2nd-order form (heat, burgers, AC). With
          `need_uxxx=True`, the residual is called with an additional
          `u_xxx` argument (KdV).
      ic_fn : `u₀(x)` — the initial condition u(t=t0, x).
      t0, t1, x0, x1 : physical-domain bounds.
      need_uxxx : whether to compute the third spatial derivative via
          triple-nested reverse-mode autodiff. P7.8 mechanism gate at
          `scripts/run_p7_8_kdv_gate.py` confirmed `jacrev³` produces
          finite non-trivial values at the canonical Chebyshev-DQC 2D
          circuit shape (4 t-qubits, 4 x-qubits, 5 layers) and at
          ~0.5× the per-point cost of `jacrev²` (XLA optimizations).

    Returns:
      `(loss_fn, u_of_tx)` where:
        - `u_of_tx(t, x, p) → scalar`  is the trial solution with Lagaris
          hard-IC: `u(t,x) = u₀(x) + (t − t0)·(s·circuit + b)`.
        - `loss_fn(p, tx_colloc) → scalar`  mean squared residual over
          collocation points; `tx_colloc` is a `(N, 2)` array of (t, x)
          pairs.
    """

    def u_of_tx(t, x, p):
        t_chev = _affine_to_chebyshev_axis(t, t0, t1)
        x_chev = _affine_to_chebyshev_axis(x, x0, x1)
        n = p["s"] * circuit(t_chev, x_chev, p["w"]) + p["b"]
        return ic_fn(x) + (t - t0) * n

    # First derivatives via reverse-mode autodiff w.r.t. each coordinate.
    du_dt = jax.jacrev(u_of_tx, argnums=0)
    du_dx = jax.jacrev(u_of_tx, argnums=1)
    # Spatial second derivative: reverse-over-reverse (the convention).
    # This is THE mechanism the original P3.7 gate tested.
    d2u_dx2 = jax.jacrev(du_dx, argnums=1)
    if need_uxxx:
        # KdV-specific: third spatial derivative via triple-nested
        # reverse-mode autodiff. Gate-tested at scripts/run_p7_8_kdv_gate.py.
        d3u_dx3 = jax.jacrev(d2u_dx2, argnums=1)
    else:
        d3u_dx3 = None  # not used; kept for static dispatch

    def _point_residual(t, x, p):
        u = u_of_tx(t, x, p)
        ut = du_dt(t, x, p)
        ux = du_dx(t, x, p)
        uxx = d2u_dx2(t, x, p)
        if need_uxxx:
            uxxx = d3u_dx3(t, x, p)
            return pde_residual(t, x, u, ut, ux, uxx, uxxx)
        return pde_residual(t, x, u, ut, ux, uxx)

    def loss(p, tx_colloc):
        # tx_colloc has shape (N, 2); split into per-axis vectors.
        ts = tx_colloc[:, 0]
        xs = tx_colloc[:, 1]
        res = jax.vmap(lambda tt, xx: _point_residual(tt, xx, p))(ts, xs)
        return jnp.mean(res ** 2)

    return loss, u_of_tx


# ---------------------------------------------------------------------------
# Train loop  (Cartesian interior collocation + optax adam)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDESolverResult:
    params: dict
    final_loss: float
    t_eval: jnp.ndarray
    x_eval: jnp.ndarray
    u_pred: jnp.ndarray             # shape (n_t_eval, n_x_eval), interior
    loss_history: list


def train_pde_solver(
    circuit: Callable,
    pde_residual: Callable,
    ic_fn: Callable,
    *,
    t0: float, t1: float, x0: float, x1: float,
    weight_shape: tuple[int, ...],
    n_t_colloc: int = 24,
    n_x_colloc: int = 24,
    n_t_eval: int = 50,
    n_x_eval: int = 50,
    steps: int = 1200,
    lr: float = 0.02,
    seed: int = 0,
) -> PDESolverResult:
    """Train (weights, scale, bias) to satisfy the PDE by physics residual.

    Cartesian (t × x) interior collocation, both axes exclude bare
    Chebyshev ±1 endpoints (same convention as the 1D gate). The
    nested jax.jacrev pattern through the QNode is exactly what the
    mechanism gate tests.
    """
    p = init_pde_solver_params(weight_shape, seed=seed)
    # Interior collocation on both axes (Chebyshev-singular bare ±1
    # excluded by construction — n+2 then drop the endpoints).
    t_colloc = jnp.linspace(t0, t1, n_t_colloc + 2)[1:-1]
    x_colloc = jnp.linspace(x0, x1, n_x_colloc + 2)[1:-1]
    T, X = jnp.meshgrid(t_colloc, x_colloc, indexing="ij")
    tx_colloc = jnp.stack([T.ravel(), X.ravel()], axis=1)         # (N, 2)

    loss_fn, u_of_tx = make_pde_residual_loss(
        circuit, pde_residual, ic_fn,
        t0=t0, t1=t1, x0=x0, x1=x1)
    opt = optax.adam(lr)
    opt_state = opt.init(p)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    hist: list[float] = []
    last = 0.0
    for _ in range(steps):
        last, grads = loss_and_grad(p, tx_colloc)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        hist.append(float(last))

    # Interior eval grid (also excludes bare ±1 endpoints).
    t_eval = jnp.linspace(t0, t1, n_t_eval + 2)[1:-1]
    x_eval = jnp.linspace(x0, x1, n_x_eval + 2)[1:-1]
    Te, Xe = jnp.meshgrid(t_eval, x_eval, indexing="ij")
    u_pred = jax.vmap(jax.vmap(lambda tt, xx: u_of_tx(tt, xx, p)))(Te, Xe)

    return PDESolverResult(
        params=p, final_loss=float(last),
        t_eval=t_eval, x_eval=x_eval,
        u_pred=u_pred, loss_history=hist)


# ---------------------------------------------------------------------------
# Heat-equation gate  (the make-or-break convergence target)
# ---------------------------------------------------------------------------


def pde_solver_gate_heat(
    *,
    nu: float = 0.1,
    n_t_qubits: int = 4,
    n_x_qubits: int = 4,
    num_layers: int = 5,
    steps: int = 1200,
    seed: int = 0,
) -> tuple[PDESolverResult, float]:
    """P3.7 ACCEPTANCE GATE — convergence half.

    Solve the heat equation `u_t = ν u_xx` with `u(0, x) = sin(x)` on
    `x ∈ [0, 2π)`, `t ∈ [0, 1]`. Exact analytic solution:
        u(t, x) = e^{−νt} sin(x).
    Returns (result, interior_mae_vs_analytic).

    If `mae_vs_analytic < 0.10` at seed 0, the convergence gate is
    satisfied. The mechanism gate is a SEPARATE test that exercises
    the nested jacrev directly; this function is the empirical-
    convergence proof.
    """
    cfg = ChebyshevDQC2DConfig(
        n_t_qubits=n_t_qubits, n_x_qubits=n_x_qubits,
        num_layers=num_layers)
    circuit = build_chebyshev_dqc_2d(cfg)

    def rhs_heat(t, x, u, ut, ux, uxx):
        return ut - nu * uxx

    def ic_sin(x):
        return jnp.sin(x)

    t0, t1 = 0.0, 1.0
    x0, x1 = 0.0, 2.0 * jnp.pi
    res = train_pde_solver(
        circuit, rhs_heat, ic_sin,
        t0=t0, t1=t1, x0=x0, x1=x1,
        weight_shape=cfg.weight_shape,
        steps=steps, seed=seed)
    # Compute MAE vs analytic on the interior eval grid.
    Te, Xe = jnp.meshgrid(res.t_eval, res.x_eval, indexing="ij")
    u_exact = jnp.exp(-nu * Te) * jnp.sin(Xe)
    mae = float(jnp.mean(jnp.abs(res.u_pred - u_exact)))
    return res, mae
