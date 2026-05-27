"""Smoke-test the quantum feature encoder on real qZETA windows.

End-to-end check that the JAX-side encoder consumes data produced by the
PyTorch-side data pipeline (numpy hand-off at the boundary) and produces
sensible latents. Not a training run — just a forward pass + grad to confirm
the cross-stack plumbing works.

Usage:
    .venv/bin/python scripts/qlnn_smoke_encoder.py
"""
from __future__ import annotations

import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from qlnn_ import QuantumFeatureEncoder, QuantumFeatureEncoderConfig
from qlnn_.encoders.quantum_feature_encoder import encoder_apply_batched
from quantum_liquid_neuralode.data_processing import (
    apply_minmax,
    fit_minmax,
    load_qzeta,
    make_horizon_windows,
    split_indices,
    time_hours_from_date,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CSV = REPO_ROOT / "data" / "raw" / "qZETA_data_copy.csv"


def main() -> None:
    df = load_qzeta(CSV)
    n = len(df)
    sp = split_indices(n, train_ratio=0.70, val_ratio=0.15)
    time_hours = time_hours_from_date(df)

    feature_cols = list(DEFAULT_FEATURE_COLS)
    scalers = fit_minmax(
        df,
        feature_cols,
        fit_end=sp.train_end,
        fixed_bounds={DEFAULT_TARGET_COL: (0.0, 3.8)},
    )
    df_n = apply_minmax(df, feature_cols, scalers)

    feat = df_n[feature_cols].iloc[: sp.train_end].to_numpy(dtype=np.float32)
    od = df_n[DEFAULT_TARGET_COL].iloc[: sp.train_end].to_numpy(dtype=np.float32)
    t = time_hours[: sp.train_end].astype(np.float64)

    win = make_horizon_windows(
        features=feat, od=od, time_hours=t,
        window_size=24, stride=1,
        horizon_hours=1.0, horizon_tolerance_hours=1e-3,
    )

    # Take the final time-step of every window as our feature vector for encoding.
    x_last_np = win.x[:, -1, :]  # (n_windows, F)
    print(f"qZETA windows : {win.x.shape},  using final step: {x_last_np.shape}")

    # Convert to JAX and route through encoder.
    X = jnp.asarray(x_last_np)
    cfg = QuantumFeatureEncoderConfig(
        input_dim=X.shape[-1], num_qubits=4, num_layers=3,
    )
    encoder = QuantumFeatureEncoder(cfg, key=jax.random.PRNGKey(0))
    print(f"encoder       : {encoder.num_parameters()} parameters, output_dim={encoder.output_dim}")

    # JIT-compiled batched apply.
    fwd = eqx.filter_jit(encoder_apply_batched)
    t0 = time.time()
    Y = fwd(encoder, X)
    t1 = time.time()
    Y2 = fwd(encoder, X)
    t2 = time.time()
    Y.block_until_ready()
    print(f"first call (compile+run): {1000*(t1-t0):.1f} ms")
    print(f"second call (cached jit): {1000*(t2-t1):.2f} ms  for {X.shape[0]} windows")
    print(f"latent shape  : {Y.shape}")
    print(f"latent stats  : mean={float(Y.mean()):+.4f}  std={float(Y.std()):.4f}  "
          f"min={float(Y.min()):+.4f}  max={float(Y.max()):+.4f}")

    # Confirm gradient flows through a downstream scalar loss against a dummy target.
    def loss_fn(enc, X, target):
        return jnp.mean((encoder_apply_batched(enc, X) - target) ** 2)

    target = jnp.zeros((X.shape[0], encoder.output_dim))
    grads = eqx.filter_grad(loss_fn)(encoder, X, target)
    import jax.tree_util as jtu
    total = sum(float(jnp.abs(g).sum()) for g in jtu.tree_leaves(eqx.filter(grads, eqx.is_array)))
    print(f"total grad mass through encoder over real data: {total:.4f}")

    print("\nOK — quantum feature encoder consumes qZETA windows and is differentiable end-to-end.")


if __name__ == "__main__":
    main()
