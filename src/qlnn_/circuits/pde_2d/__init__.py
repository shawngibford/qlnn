"""P3.9 — PDE 2D-port circuit builders.

Sibling subpackage to `src/qlnn_/circuits/{te_qpinn,qcpinn,reuploading,...}`.
Each module in here exposes a `build_<family>_2d(cfg)` factory that
returns a `(t_chev, x_chev, weights) → scalar` circuit compatible with
`qlnn_.training.pde_residual_loss.make_pde_residual_loss`.

The 1D builders in the sibling files remain unchanged and continue to
serve the ODE-side solver path. The 2D ports are deliberate parallels —
they expose the same `{w, s, b}` Lagaris-hard-IC pytree outer contract
even when the inner `w` is a multi-leaf dict (e.g. qcpinn's pre-NN +
PQC + post-NN combined).

P3a faithfulness discipline: each 2D port carries a declared
design-choice subsection in `refs/CIRCUIT_SPECS.md` documenting the
extension from 1D scalar input to 2D (t, x) input.
"""

from qlnn_.circuits.pde_2d.qcpinn_2d import (
    QCPINN2DConfig,
    build_qcpinn_2d,
    init_qcpinn_2d_solver_params,
)

__all__ = [
    "QCPINN2DConfig",
    "build_qcpinn_2d",
    "init_qcpinn_2d_solver_params",
]
