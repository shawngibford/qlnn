from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from quantum_liquid_neuralode.models import LiquidCell
from quantum_liquid_neuralode.utils import select_device


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "qZETA_data_copy.csv"


def _load_qzeta(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # Normalize the one column name that contains a space.
    if "TEMP EXT" in df.columns and "TEMP_EXT" not in df.columns:
        df = df.rename(columns={"TEMP EXT": "TEMP_EXT"})

    if "DATE" not in df.columns:
        raise ValueError("Expected a DATE column in the CSV")

    dt = pd.to_datetime(df["DATE"], format="mixed", dayfirst=True, errors="raise")
    df = df.assign(DATE=dt).sort_values("DATE").reset_index(drop=True)

    return df


@dataclass(frozen=True)
class SplitIdx:
    train_end: int
    val_end: int


def _split_indices(n: int, *, train_ratio: float, val_ratio: float) -> SplitIdx:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Ensure each split can produce at least one window.
    if train_end < 3:
        raise ValueError("train split too small")
    if val_end <= train_end:
        raise ValueError("val split too small")

    return SplitIdx(train_end=train_end, val_end=val_end)


def _fit_minmax(
    df: pd.DataFrame,
    cols: list[str],
    *,
    fit_end: int,
    fixed_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, MinMaxScaler]:
    """Fit per-column MinMax scalers.

    If fixed_bounds is provided for a column, the scaler is fit on those bounds
    instead of the training slice. This is useful for variables like OD where a
    train-only MinMax can produce extreme out-of-range values later in the run.
    """
    scalers: dict[str, MinMaxScaler] = {}
    fixed_bounds = fixed_bounds or {}

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

        sc = MinMaxScaler()

        if col in fixed_bounds:
            vmin, vmax = fixed_bounds[col]
            if vmax <= vmin:
                raise ValueError(f"Invalid fixed bounds for {col}: min={vmin}, max={vmax}")
            bounds_df = pd.DataFrame({col: [vmin, vmax]})
            sc.fit(bounds_df)
        else:
            sc.fit(df[[col]].iloc[:fit_end])

        scalers[col] = sc

    return scalers


def _apply_minmax(df: pd.DataFrame, cols: list[str], scalers: dict[str, MinMaxScaler]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = scalers[col].transform(out[[col]])
    return out


def _make_horizon_windows(
    *,
    features: np.ndarray,  # (N, F)
    od: np.ndarray,  # (N,)
    time_hours: np.ndarray,  # (N,)
    window_size: int,
    stride: int,
    horizon_hours: float,
    horizon_tolerance_hours: float,
    index_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create (history window) -> (OD at +horizon) supervised examples.

    We choose the label index as the first timestamp >= t_end + horizon_hours.
    To keep the task a true fixed-horizon forecast, we drop windows whose realized
    horizon differs from the requested horizon by more than horizon_tolerance_hours.

    Returns:
        x_windows: (n_windows, window_size, n_features)
        t_windows: (n_windows, window_size) window-relative times in hours
        y_targets: (n_windows,) OD at +horizon (normalized)
        od_last: (n_windows,) OD at t_end (normalized)
        od_prev: (n_windows,) OD at t_end-1 step (normalized)
        dt_last: (n_windows,) last time delta within the window (hours)
        end_idx: (n_windows,) global row index of t_end in the original dataframe
        target_idx: (n_windows,) global row index of the target (t_end + horizon)
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be > 0")
    if horizon_tolerance_hours < 0:
        raise ValueError("horizon_tolerance_hours must be >= 0")

    if not (features.shape[0] == od.shape[0] == time_hours.shape[0]):
        raise ValueError("features/od/time_hours must have matching first dimension")

    xs: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    ys: list[float] = []

    od_last_list: list[float] = []
    od_prev_list: list[float] = []
    dt_last_list: list[float] = []

    end_idx_list: list[int] = []
    target_idx_list: list[int] = []

    n = features.shape[0]
    for start in range(0, n - window_size, stride):
        end = start + window_size  # exclusive
        end_idx = end - 1

        target_time = time_hours[end_idx] + horizon_hours
        target_idx = int(np.searchsorted(time_hours, target_time, side="left"))

        if target_idx >= n:
            continue

        realized = float(time_hours[target_idx] - time_hours[end_idx])
        if abs(realized - horizon_hours) > horizon_tolerance_hours:
            continue

        xw = features[start:end]
        tw = time_hours[start:end]
        tw = tw - tw[0]

        xs.append(xw)
        ts.append(tw.astype(np.float32))
        ys.append(float(od[target_idx]))

        od_last_list.append(float(od[end_idx]))
        od_prev_list.append(float(od[end_idx - 1]))
        dt_last_list.append(float(tw[-1] - tw[-2]))

        end_idx_list.append(int(index_offset + end_idx))
        target_idx_list.append(int(index_offset + target_idx))

    if not xs:
        raise ValueError("No windows created (check window_size/stride/horizon vs data length)")

    return (
        np.stack(xs, axis=0),
        np.stack(ts, axis=0),
        np.asarray(ys, dtype=np.float32),
        np.asarray(od_last_list, dtype=np.float32),
        np.asarray(od_prev_list, dtype=np.float32),
        np.asarray(dt_last_list, dtype=np.float32),
        np.asarray(end_idx_list, dtype=np.int64),
        np.asarray(target_idx_list, dtype=np.int64),
    )


def _metrics_from_arrays(*, y_true: np.ndarray, y_pred: np.ndarray, od_scaler: MinMaxScaler) -> dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    mse_norm = float(np.mean((y_pred - y_true) ** 2))

    y_true_raw = od_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
    y_pred_raw = od_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)

    return {
        "mse_norm": mse_norm,
        "r2_raw": float(r2_score(y_true_raw, y_pred_raw)),
    }


class LiquidODForecaster(nn.Module):
    """Forecast OD(t + horizon) given a history window up to time t.

    Modeling choice:
    - We evolve a hidden state over the observed history using the per-step dt.
    - We then evolve the hidden state over the forecast horizon assuming inputs
      remain constant at their last observed value (x_last).
    - The final prediction is *residual* around persistence:
          OD(t+h) = OD(t) + delta
      where delta is bounded via tanh for stability.
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        horizon_hours: float,
        forecast_steps: int,
        od_index: int,
        delta_scale: float,
        tau_min: float = 0.1,
    ) -> None:
        super().__init__()

        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be > 0")
        if forecast_steps <= 0:
            raise ValueError("forecast_steps must be positive")
        if not (0 <= od_index < input_size):
            raise ValueError("od_index out of range")
        if delta_scale <= 0:
            raise ValueError("delta_scale must be > 0")

        self.horizon_hours = float(horizon_hours)
        self.forecast_steps = int(forecast_steps)
        self.od_index = int(od_index)
        self.delta_scale = float(delta_scale)

        self.encoder = nn.Linear(input_size, hidden_size)
        self.cell = LiquidCell(input_size=input_size, hidden_size=hidden_size, tau_min=tau_min)
        self.delta_head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor, t_hours: Tensor) -> Tensor:
        """Forecast OD at t_end + horizon.

        Args:
            x: (batch, T, input_size)
            t_hours: (batch, T) monotonically increasing, in hours (window-relative ok)

        Returns:
            od_pred: (batch,)
        """
        if x.ndim != 3:
            raise ValueError(f"x must be 3D (batch, T, input_size), got {tuple(x.shape)}")
        if t_hours.ndim != 2:
            raise ValueError(f"t_hours must be 2D (batch, T), got {tuple(t_hours.shape)}")
        if x.shape[0] != t_hours.shape[0] or x.shape[1] != t_hours.shape[1]:
            raise ValueError("x and t_hours must match in batch and T")
        if x.shape[1] < 2:
            raise ValueError("Need at least 2 time points in the history window")

        _, T, _ = x.shape

        # Evolve hidden state over the observed history.
        h = self.encoder(x[:, 0, :])
        for i in range(T - 1):
            dt = (t_hours[:, i + 1] - t_hours[:, i]).unsqueeze(-1)
            dh_dt = self.cell(h, x[:, i, :])
            h = h + dt * dh_dt

        # Forecast horizon with constant inputs at their last observed value.
        x_last = x[:, -1, :]
        dt_f = self.horizon_hours / float(self.forecast_steps)
        for _ in range(self.forecast_steps):
            h = h + dt_f * self.cell(h, x_last)

        # Residual forecast around persistence.
        od_last = x_last[:, self.od_index]
        delta = torch.tanh(self.delta_head(h).squeeze(-1)) * self.delta_scale
        return od_last + delta


@torch.no_grad()
def _eval_metrics(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    od_scaler: MinMaxScaler,
) -> dict[str, float]:
    model.eval()

    mse_sum = 0.0
    n_elems = 0

    y_true_raw: list[np.ndarray] = []
    y_pred_raw: list[np.ndarray] = []

    for xb, tb, yb in loader:
        xb = xb.to(device)
        tb = tb.to(device)
        yb = yb.to(device)

        yp = model(xb, tb)

        se = (yp - yb).pow(2)
        mse_sum += float(se.sum().detach().cpu().item())
        n_elems += int(yb.numel())

        yp_np = yp.detach().cpu().numpy().reshape(-1, 1)
        yb_np = yb.detach().cpu().numpy().reshape(-1, 1)

        y_pred_raw.append(od_scaler.inverse_transform(yp_np).reshape(-1))
        y_true_raw.append(od_scaler.inverse_transform(yb_np).reshape(-1))

    y_true_raw_all = np.concatenate(y_true_raw, axis=0)
    y_pred_raw_all = np.concatenate(y_pred_raw, axis=0)

    return {
        "mse_norm": float(mse_sum / max(n_elems, 1)),
        "r2_raw": float(r2_score(y_true_raw_all, y_pred_raw_all)),
    }


@torch.no_grad()
def _predict_numpy(
    *,
    model: nn.Module,
    x: np.ndarray,
    t_hours: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    preds: list[np.ndarray] = []
    n = x.shape[0]

    for i in range(0, n, batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(device)
        tb = torch.from_numpy(t_hours[i : i + batch_size]).to(device)
        yp = model(xb, tb).detach().cpu().numpy()
        preds.append(yp)

    return np.concatenate(preds, axis=0)


def _build_predictions_df(
    *,
    split: str,
    df_raw: pd.DataFrame,
    time_hours_full: np.ndarray,
    end_idx: np.ndarray,
    target_idx: np.ndarray,
    od_last_norm: np.ndarray,
    od_prev_norm: np.ndarray,
    dt_last_hours: np.ndarray,
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    baseline_linear_norm: np.ndarray,
    od_scaler: MinMaxScaler,
) -> pd.DataFrame:
    if not (
        len(end_idx)
        == len(target_idx)
        == len(od_last_norm)
        == len(od_prev_norm)
        == len(dt_last_hours)
        == len(y_true_norm)
        == len(y_pred_norm)
    ):
        raise ValueError("Index and prediction arrays must have matching lengths")

    # Convert to numpy arrays to avoid index-alignment surprises when mixing Series and ndarrays.
    date_end = df_raw["DATE"].iloc[end_idx].to_numpy()
    date_target = df_raw["DATE"].iloc[target_idx].to_numpy()

    out = pd.DataFrame(
        {
            "split": split,
            "end_idx": end_idx,
            "target_idx": target_idx,
            "date_end": date_end,
            "date_target": date_target,
            "t_end_hours": time_hours_full[end_idx],
            "t_target_hours": time_hours_full[target_idx],
            "od_t_norm": od_last_norm,
            "od_prev_norm": od_prev_norm,
            "dt_last_hours": dt_last_hours,
            "y_true_norm": y_true_norm,
            "y_pred_norm": y_pred_norm,
            "baseline_persist_norm": od_last_norm,
            "baseline_linear_norm": baseline_linear_norm,
        }
    )

    def inv(arr: np.ndarray) -> np.ndarray:
        return od_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(-1)

    out["od_t_raw"] = inv(out["od_t_norm"].to_numpy(dtype=np.float32))
    out["od_prev_raw"] = inv(out["od_prev_norm"].to_numpy(dtype=np.float32))

    out["y_true_raw"] = inv(out["y_true_norm"].to_numpy(dtype=np.float32))
    out["y_pred_raw"] = inv(out["y_pred_norm"].to_numpy(dtype=np.float32))
    out["baseline_persist_raw"] = inv(out["baseline_persist_norm"].to_numpy(dtype=np.float32))
    out["baseline_linear_raw"] = inv(out["baseline_linear_norm"].to_numpy(dtype=np.float32))

    out["residual_norm"] = out["y_pred_norm"] - out["y_true_norm"]
    out["abs_error_norm"] = out["residual_norm"].abs()

    out["residual_raw"] = out["y_pred_raw"] - out["y_true_raw"]
    out["abs_error_raw"] = out["residual_raw"].abs()

    out["abs_error_persist_raw"] = (out["baseline_persist_raw"] - out["y_true_raw"]).abs()
    out["abs_error_linear_raw"] = (out["baseline_linear_raw"] - out["y_true_raw"]).abs()

    out["delta_true_raw"] = out["y_true_raw"] - out["od_t_raw"]
    out["delta_pred_raw"] = out["y_pred_raw"] - out["od_t_raw"]

    # Recent slope proxy at forecast origin (OD change per hour over last in-window step).
    dt = out["dt_last_hours"].to_numpy(dtype=np.float64)
    safe_dt = np.where(dt > 0, dt, np.nan)
    out["slope_last_raw_per_hr"] = (out["od_t_raw"].to_numpy(dtype=np.float64) - out["od_prev_raw"].to_numpy(dtype=np.float64)) / safe_dt

    return out.sort_values("date_target").reset_index(drop=True)


def _permutation_importance(
    *,
    model: nn.Module,
    x: np.ndarray,
    t_hours: np.ndarray,
    y_true: np.ndarray,
    feature_names: list[str],
    device: torch.device,
    batch_size: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Permutation importance via per-window feature-trajectory shuffling.

    For each feature, we permute the entire (T,) history for that feature across
    windows. This preserves within-window temporal structure while breaking the
    association with the target.

    Returns a long-form DataFrame with per-repeat delta-MSE.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    rng = np.random.default_rng(seed)
    baseline_pred = _predict_numpy(model=model, x=x, t_hours=t_hours, device=device, batch_size=batch_size)
    baseline_mse = float(np.mean((baseline_pred - y_true) ** 2))

    rows: list[dict[str, float | str | int]] = []
    n = x.shape[0]

    for j, name in enumerate(feature_names):
        for r in range(repeats):
            perm = rng.permutation(n)
            x_perm = x.copy()
            x_perm[:, :, j] = x[perm, :, j]

            yp = _predict_numpy(model=model, x=x_perm, t_hours=t_hours, device=device, batch_size=batch_size)
            mse = float(np.mean((yp - y_true) ** 2))

            rows.append(
                {
                    "feature": str(name),
                    "feature_index": int(j),
                    "repeat": int(r),
                    "mse_norm": mse,
                    "delta_mse_norm": mse - baseline_mse,
                }
            )

    return pd.DataFrame(rows)


def _plot_perm_importance(df: pd.DataFrame, *, split: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    summary = (
        df.groupby("feature", as_index=False)
        .agg(delta_mse_mean=("delta_mse_norm", "mean"), delta_mse_std=("delta_mse_norm", "std"))
        .sort_values("delta_mse_mean", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=summary, x="feature", y="delta_mse_mean", ax=ax, color="C0")
    ax.errorbar(
        x=np.arange(len(summary)),
        y=summary["delta_mse_mean"].to_numpy(),
        yerr=summary["delta_mse_std"].fillna(0.0).to_numpy(),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(f"Permutation importance (Δ MSE_norm) — {split}")
    ax.set_xlabel("Feature (history permuted across windows)")
    ax.set_ylabel("Δ MSE_norm")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_binned_diagnostics(
    pred: pd.DataFrame,
    *,
    split: str,
    plots_dir: Path,
    n_bins_1d: int,
    n_bins_2d: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    def _bin_mid(series: pd.Series, bins: pd.Series) -> pd.Series:
        # bins is a Categorical of pd.Interval.
        mids = {iv: 0.5 * (iv.left + iv.right) for iv in bins.cat.categories}
        return bins.map(mids).astype(float)

    # --- 1D bins: OD(t) ---
    od_bin = pd.qcut(pred["od_t_raw"], q=n_bins_1d, duplicates="drop")
    od_mid = _bin_mid(pred["od_t_raw"], od_bin)

    df_od = pred.assign(od_bin=od_bin, od_mid=od_mid)
    df_od_long = pd.concat(
        [
            df_od[["od_mid", "od_bin"]].assign(method="model", abs_error=df_od["abs_error_raw"]),
            df_od[["od_mid", "od_bin"]].assign(method="persistence", abs_error=df_od["abs_error_persist_raw"]),
            df_od[["od_mid", "od_bin"]].assign(method="linear", abs_error=df_od["abs_error_linear_raw"]),
        ],
        axis=0,
        ignore_index=True,
    )

    od_summary = (
        df_od_long.groupby(["method", "od_mid"], as_index=False)
        .agg(mae=("abs_error", "mean"))
        .sort_values("od_mid")
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=od_summary, x="od_mid", y="mae", hue="method", marker="o", ax=ax)
    ax.set_title(f"MAE by OD(t) bin (1h-ahead) — {split}")
    ax.set_xlabel("OD(t) bin midpoint")
    ax.set_ylabel("MAE (raw OD)")
    fig.tight_layout()
    fig.savefig(plots_dir / f"mae_by_od_bin_{split}.png", dpi=150)
    plt.close(fig)

    # --- 1D bins: slope proxy ---
    slope = pred["slope_last_raw_per_hr"].replace([np.inf, -np.inf], np.nan).dropna()
    # If slope has too few unique values, qcut can fail; guard with nunique.
    if slope.nunique() >= 3:
        slope_bin = pd.qcut(pred["slope_last_raw_per_hr"], q=n_bins_1d, duplicates="drop")
        slope_mid = _bin_mid(pred["slope_last_raw_per_hr"], slope_bin)

        df_sl = pred.assign(slope_bin=slope_bin, slope_mid=slope_mid)
        df_sl_long = pd.concat(
            [
                df_sl[["slope_mid", "slope_bin"]].assign(method="model", abs_error=df_sl["abs_error_raw"]),
                df_sl[["slope_mid", "slope_bin"]].assign(method="persistence", abs_error=df_sl["abs_error_persist_raw"]),
                df_sl[["slope_mid", "slope_bin"]].assign(method="linear", abs_error=df_sl["abs_error_linear_raw"]),
            ],
            axis=0,
            ignore_index=True,
        )

        sl_summary = (
            df_sl_long.groupby(["method", "slope_mid"], as_index=False)
            .agg(mae=("abs_error", "mean"))
            .sort_values("slope_mid")
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=sl_summary, x="slope_mid", y="mae", hue="method", marker="o", ax=ax)
        ax.set_title(f"MAE by recent slope bin (OD/hr) — {split}")
        ax.set_xlabel("Slope bin midpoint (OD/hr)")
        ax.set_ylabel("MAE (raw OD)")
        fig.tight_layout()
        fig.savefig(plots_dir / f"mae_by_slope_bin_{split}.png", dpi=150)
        plt.close(fig)

        # --- 2D heatmaps: OD(t) vs slope ---
        od_bin2 = pd.qcut(pred["od_t_raw"], q=n_bins_2d, duplicates="drop")
        slope_bin2 = pd.qcut(pred["slope_last_raw_per_hr"], q=n_bins_2d, duplicates="drop")

        heat_abs = pred.assign(od_bin2=od_bin2, slope_bin2=slope_bin2).pivot_table(
            values="abs_error_raw", index="od_bin2", columns="slope_bin2", aggfunc="mean"
        )

        heat_improve = pred.assign(od_bin2=od_bin2, slope_bin2=slope_bin2)
        heat_improve["improve_vs_persist"] = heat_improve["abs_error_persist_raw"] - heat_improve["abs_error_raw"]
        heat_improve = heat_improve.pivot_table(
            values="improve_vs_persist", index="od_bin2", columns="slope_bin2", aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heat_abs, ax=ax, cmap="viridis", cbar_kws={"label": "Mean |error| (raw OD)"})
        ax.set_title(f"Mean absolute error heatmap: OD(t) vs slope — {split}")
        ax.set_xlabel("Slope bin (OD/hr)")
        ax.set_ylabel("OD(t) bin")
        fig.tight_layout()
        fig.savefig(plots_dir / f"heatmap_abs_error_od_vs_slope_{split}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heat_improve, ax=ax, cmap="coolwarm", center=0.0, cbar_kws={"label": "Persist |error| - Model |error|"})
        ax.set_title(f"Improvement vs persistence heatmap: OD(t) vs slope — {split}")
        ax.set_xlabel("Slope bin (OD/hr)")
        ax.set_ylabel("OD(t) bin")
        fig.tight_layout()
        fig.savefig(plots_dir / f"heatmap_improve_vs_persist_od_vs_slope_{split}.png", dpi=150)
        plt.close(fig)


def _save_plots(
    *,
    history: pd.DataFrame,
    pred_val: pd.DataFrame,
    pred_test: pd.DataFrame,
    baseline_persist_val_mse: float,
    baseline_linear_val_mse: float,
    perm_val: pd.DataFrame | None,
    perm_test: pd.DataFrame | None,
    plots_dir: Path,
    n_bins_1d: int,
    n_bins_2d: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=history, x="epoch", y="train_mse_norm", ax=ax, label="train")
    sns.lineplot(data=history, x="epoch", y="val_mse_norm", ax=ax, label="val")
    ax.axhline(baseline_persist_val_mse, color="black", linestyle="--", linewidth=1, label="persistence (val)")
    ax.axhline(baseline_linear_val_mse, color="gray", linestyle=":", linewidth=1, label="linear (val)")
    ax.set_title("Training vs validation MSE (normalized OD)")
    ax.set_ylabel("MSE (norm)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "training_mse.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=history, x="epoch", y="val_r2_raw", ax=ax, label="val")
    ax.set_title("Validation R² (raw OD)")
    ax.set_ylabel("R² (raw)")
    fig.tight_layout()
    fig.savefig(plots_dir / "training_val_r2.png", dpi=150)
    plt.close(fig)

    def plot_split(pred: pd.DataFrame, *, split: str) -> None:
        # Pred vs true
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(data=pred, x="y_true_raw", y="y_pred_raw", s=18, alpha=0.6, ax=ax)
        lo = min(pred["y_true_raw"].min(), pred["y_pred_raw"].min())
        hi = max(pred["y_true_raw"].max(), pred["y_pred_raw"].max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_title(f"Predicted vs true OD (1h ahead) — {split}")
        ax.set_xlabel("OD true")
        ax.set_ylabel("OD predicted")
        fig.tight_layout()
        fig.savefig(plots_dir / f"pred_vs_true_{split}.png", dpi=150)
        plt.close(fig)

        # Residual histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(pred["residual_raw"], bins=50, kde=True, ax=ax)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(f"Residuals (pred - true) — {split}")
        ax.set_xlabel("Residual (raw OD)")
        fig.tight_layout()
        fig.savefig(plots_dir / f"residual_hist_{split}.png", dpi=150)
        plt.close(fig)

        # Residual vs OD(t)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=pred, x="od_t_raw", y="residual_raw", s=18, alpha=0.6, ax=ax)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"Residual vs OD(t) — {split}")
        ax.set_xlabel("OD at forecast origin (t)")
        ax.set_ylabel("Residual (raw OD)")
        fig.tight_layout()
        fig.savefig(plots_dir / f"residual_vs_odt_{split}.png", dpi=150)
        plt.close(fig)

        # Error over time
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=pred, x="date_target", y="abs_error_raw", ax=ax)
        ax.set_title(f"Absolute error over time (target timestamps) — {split}")
        ax.set_xlabel("Target time")
        ax.set_ylabel("|error| (raw OD)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plots_dir / f"abs_error_over_time_{split}.png", dpi=150)
        plt.close(fig)

        # Time series overlay at target timestamps
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(pred["date_target"], pred["y_true_raw"], label="true", linewidth=1.5)
        ax.plot(pred["date_target"], pred["baseline_persist_raw"], label="persistence", linewidth=1.0, alpha=0.9)
        ax.plot(pred["date_target"], pred["y_pred_raw"], label="model", linewidth=1.0, alpha=0.9)
        ax.set_title(f"1h-ahead OD forecast (aligned at target time) — {split}")
        ax.set_xlabel("Target time")
        ax.set_ylabel("OD")
        ax.legend(loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plots_dir / f"timeseries_{split}.png", dpi=150)
        plt.close(fig)

        # Delta distribution (true vs pred)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(pred["delta_true_raw"], bins=50, kde=True, ax=ax, color="C0", label="true")
        sns.histplot(pred["delta_pred_raw"], bins=50, kde=True, ax=ax, color="C1", label="pred", alpha=0.6)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(f"ΔOD over 1h: true vs predicted — {split}")
        ax.set_xlabel("ΔOD (raw OD)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / f"delta_hist_{split}.png", dpi=150)
        plt.close(fig)

    plot_split(pred_val, split="val")
    plot_split(pred_test, split="test")

    # Permutation importance
    if perm_val is not None:
        _plot_perm_importance(perm_val, split="val", out_path=plots_dir / "perm_importance_val.png")
    if perm_test is not None:
        _plot_perm_importance(perm_test, split="test", out_path=plots_dir / "perm_importance_test.png")

    # Binned error diagnostics + heatmaps
    _plot_binned_diagnostics(pred_val, split="val", plots_dir=plots_dir, n_bins_1d=n_bins_1d, n_bins_2d=n_bins_2d)
    _plot_binned_diagnostics(pred_test, split="test", plots_dir=plots_dir, n_bins_1d=n_bins_1d, n_bins_2d=n_bins_2d)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LiquidNN experiment: forecast OD 1-hour ahead from a history window (includes OD(t) as an input)."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)

    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=["OD", "PRE", "TEMP_EXT", "TEMP_CULTURE", "PAR_LIGHT", "PH", "DO"],
    )
    parser.add_argument("--target-col", type=str, default="OD")

    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--horizon-hours", type=float, default=1.0)
    parser.add_argument("--horizon-tol-hours", type=float, default=1e-3)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--tau-min", type=float, default=0.1)

    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=6,
        help="Euler steps used to evolve the hidden state over the forecast horizon.",
    )

    # Bound delta via tanh: OD(t+h)=OD(t)+tanh(delta_raw)*delta_scale.
    parser.add_argument("--delta-scale", type=float, default=0.05)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    # Training control
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)

    # Optional: fixed scaling bounds for OD (recommended for within-run time splits).
    parser.add_argument("--od-min", type=float, default=0.0)
    parser.add_argument("--od-max", type=float, default=3.8)

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "od_forecast_1h",
        help="Directory to write artifacts (metrics, predictions, plots).",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not write any artifacts to disk.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation (still saves CSV/JSON).")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Save only config/metrics/history and skip predictions tables, perm importance, and plots (useful for HPO).",
    )

    # Diagnostics
    parser.add_argument("--perm-repeats", type=int, default=5)
    parser.add_argument("--no-perm", action="store_true", help="Skip permutation importance computation.")
    parser.add_argument("--bins-1d", type=int, default=10, help="Quantile bins for 1D breakdown plots.")
    parser.add_argument("--bins-2d", type=int, default=6, help="Quantile bins per axis for 2D heatmaps.")

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = _load_qzeta(args.csv)

    n = len(df)
    split = _split_indices(n, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # time in hours from start
    time_hours = (df["DATE"] - df["DATE"].iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64) / 3600.0

    cols_to_scale = list(dict.fromkeys(list(args.feature_cols) + [args.target_col]))

    fixed_bounds: dict[str, tuple[float, float]] = {}
    if args.od_max is not None:
        fixed_bounds["OD"] = (float(args.od_min), float(args.od_max))

    scalers = _fit_minmax(df, cols_to_scale, fit_end=split.train_end, fixed_bounds=fixed_bounds)
    df_n = _apply_minmax(df, cols_to_scale, scalers)

    def segment_windows(
        start: int, end: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feat = df_n[args.feature_cols].iloc[start:end].to_numpy(dtype=np.float32)
        od = df_n[args.target_col].iloc[start:end].to_numpy(dtype=np.float32)
        t = time_hours[start:end].astype(np.float64)
        return _make_horizon_windows(
            features=feat,
            od=od,
            time_hours=t,
            window_size=args.window_size,
            stride=args.stride,
            horizon_hours=args.horizon_hours,
            horizon_tolerance_hours=args.horizon_tol_hours,
            index_offset=start,
        )

    (
        x_train,
        t_train,
        y_train,
        od_last_train,
        od_prev_train,
        dt_last_train,
        end_idx_train,
        target_idx_train,
    ) = segment_windows(0, split.train_end)
    (
        x_val,
        t_val,
        y_val,
        od_last_val,
        od_prev_val,
        dt_last_val,
        end_idx_val,
        target_idx_val,
    ) = segment_windows(split.train_end, split.val_end)
    (
        x_test,
        t_test,
        y_test,
        od_last_test,
        od_prev_test,
        dt_last_test,
        end_idx_test,
        target_idx_test,
    ) = segment_windows(split.val_end, n)

    device = select_device(prefer_mps=True)
    print(f"device: {device}")
    print(f"rows: {n} (train_end={split.train_end}, val_end={split.val_end})")
    if args.od_max is not None:
        print(f"OD scaling: fixed MinMax with od_min={args.od_min}, od_max={args.od_max}")
    else:
        print("OD scaling: train-only MinMax")

    print(
        f"forecast horizon: {args.horizon_hours}h (tol={args.horizon_tol_hours}h), "
        f"window_size={args.window_size}, stride={args.stride}"
    )
    print(f"windows: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(t_train.astype(np.float32)),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(t_val.astype(np.float32)),
        torch.from_numpy(y_val),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(t_test.astype(np.float32)),
        torch.from_numpy(y_test),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.target_col not in args.feature_cols:
        raise ValueError("For residual forecasting, target_col must be included in feature_cols (need OD(t)).")

    od_index = args.feature_cols.index(args.target_col)

    model = LiquidODForecaster(
        input_size=len(args.feature_cols),
        hidden_size=args.hidden_size,
        horizon_hours=args.horizon_hours,
        forecast_steps=args.forecast_steps,
        od_index=od_index,
        delta_scale=args.delta_scale,
        tau_min=args.tau_min,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Baselines (all in normalized OD units).
    persist_val = _metrics_from_arrays(y_true=y_val, y_pred=od_last_val, od_scaler=scalers[args.target_col])
    persist_test = _metrics_from_arrays(y_true=y_test, y_pred=od_last_test, od_scaler=scalers[args.target_col])

    safe_dt_val = np.where(dt_last_val > 0, dt_last_val, np.nan)
    safe_dt_test = np.where(dt_last_test > 0, dt_last_test, np.nan)

    linear_pred_val = od_last_val + (od_last_val - od_prev_val) * (args.horizon_hours / safe_dt_val)
    linear_pred_test = od_last_test + (od_last_test - od_prev_test) * (args.horizon_hours / safe_dt_test)

    # Replace any nan/infs (should be rare) with persistence.
    linear_pred_val = np.where(np.isfinite(linear_pred_val), linear_pred_val, od_last_val)
    linear_pred_test = np.where(np.isfinite(linear_pred_test), linear_pred_test, od_last_test)

    linear_val = _metrics_from_arrays(y_true=y_val, y_pred=linear_pred_val, od_scaler=scalers[args.target_col])
    linear_test = _metrics_from_arrays(y_true=y_test, y_pred=linear_pred_test, od_scaler=scalers[args.target_col])

    print("baselines:")
    print(f"  persistence | val:  mse_norm={persist_val['mse_norm']:.6f}, r2_raw={persist_val['r2_raw']:.4f}")
    print(f"  persistence | test: mse_norm={persist_test['mse_norm']:.6f}, r2_raw={persist_test['r2_raw']:.4f}")
    print(f"  linear      | val:  mse_norm={linear_val['mse_norm']:.6f}, r2_raw={linear_val['r2_raw']:.4f}")
    print(f"  linear      | test: mse_norm={linear_test['mse_norm']:.6f}, r2_raw={linear_test['r2_raw']:.4f}")

    if args.eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if args.patience <= 0:
        raise ValueError("patience must be positive")

    history_rows: list[dict[str, float]] = []

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        mse_sum = 0.0
        n_elems = 0

        for xb, tb, yb in train_loader:
            xb = xb.to(device)
            tb = tb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            yp = model(xb, tb)

            se = (yp - yb).pow(2)
            loss = se.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            mse_sum += float(se.sum().detach().cpu().item())
            n_elems += int(yb.numel())

        if epoch % args.eval_every == 0 or epoch == 1:
            train_mse = float(mse_sum / max(n_elems, 1))
            val_metrics = _eval_metrics(model=model, loader=val_loader, device=device, od_scaler=scalers[args.target_col])

            improved = val_metrics["mse_norm"] < best_val
            if improved:
                best_val = val_metrics["mse_norm"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            history_rows.append(
                {
                    "epoch": float(epoch),
                    "train_mse_norm": float(train_mse),
                    "val_mse_norm": float(val_metrics["mse_norm"]),
                    "val_r2_raw": float(val_metrics["r2_raw"]),
                    "best_val_mse_norm": float(best_val),
                    "best_epoch": float(best_epoch),
                }
            )

            print(
                f"epoch {epoch:4d} | train_mse_norm={train_mse:.6f} | "
                f"val_mse_norm={val_metrics['mse_norm']:.6f} | val_r2_raw={val_metrics['r2_raw']:.4f} | "
                f"best_val_mse={best_val:.6f} (epoch {best_epoch})"
            )

            if bad_epochs >= args.patience:
                print(f"early stopping at epoch {epoch} (no val improvement for {args.patience} evals)")
                break

    if best_state is None:
        raise RuntimeError("No best_state captured (unexpected)")

    model.load_state_dict(best_state)

    val_metrics = _eval_metrics(model=model, loader=val_loader, device=device, od_scaler=scalers[args.target_col])
    test_metrics = _eval_metrics(model=model, loader=test_loader, device=device, od_scaler=scalers[args.target_col])

    print("\nfinal metrics (best val checkpoint):")
    print(f"  best_epoch: {best_epoch}")
    print(f"  val:  mse_norm={val_metrics['mse_norm']:.6f}, r2_raw={val_metrics['r2_raw']:.4f}")
    print(f"  test: mse_norm={test_metrics['mse_norm']:.6f}, r2_raw={test_metrics['r2_raw']:.4f}")

    history_df = pd.DataFrame(history_rows)

    if args.no_save:
        return

    output_dir: Path = args.output_dir
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config/metrics.
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    metrics_summary = {
        "best_epoch": int(best_epoch),
        "baselines": {
            "persistence": {"val": persist_val, "test": persist_test},
            "linear": {"val": linear_val, "test": linear_test},
        },
        "model": {"val": val_metrics, "test": test_metrics},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")

    # Save model state.
    torch.save(best_state, output_dir / "best_state.pt")

    # Save training curve.
    history_df.to_csv(output_dir / "history.csv", index=False)

    if args.metrics_only:
        print(f"\nsaved metrics-only artifacts to: {output_dir}")
        return

    # Build prediction tables (val/test) for diagnostics.
    y_pred_val = _predict_numpy(model=model, x=x_val, t_hours=t_val, device=device, batch_size=args.batch_size).astype(
        np.float32
    )
    y_pred_test = _predict_numpy(
        model=model, x=x_test, t_hours=t_test, device=device, batch_size=args.batch_size
    ).astype(np.float32)

    pred_val_df = _build_predictions_df(
        split="val",
        df_raw=df,
        time_hours_full=time_hours,
        end_idx=end_idx_val,
        target_idx=target_idx_val,
        od_last_norm=od_last_val,
        od_prev_norm=od_prev_val,
        dt_last_hours=dt_last_val,
        y_true_norm=y_val,
        y_pred_norm=y_pred_val,
        baseline_linear_norm=linear_pred_val.astype(np.float32),
        od_scaler=scalers[args.target_col],
    )
    pred_test_df = _build_predictions_df(
        split="test",
        df_raw=df,
        time_hours_full=time_hours,
        end_idx=end_idx_test,
        target_idx=target_idx_test,
        od_last_norm=od_last_test,
        od_prev_norm=od_prev_test,
        dt_last_hours=dt_last_test,
        y_true_norm=y_test,
        y_pred_norm=y_pred_test,
        baseline_linear_norm=linear_pred_test.astype(np.float32),
        od_scaler=scalers[args.target_col],
    )

    pred_val_df.to_csv(output_dir / "predictions_val.csv", index=False)
    pred_test_df.to_csv(output_dir / "predictions_test.csv", index=False)

    perm_val_df: pd.DataFrame | None = None
    perm_test_df: pd.DataFrame | None = None

    if not args.no_perm:
        if args.perm_repeats <= 0:
            raise ValueError("perm-repeats must be positive")

        perm_val_df = _permutation_importance(
            model=model,
            x=x_val,
            t_hours=t_val,
            y_true=y_val,
            feature_names=list(args.feature_cols),
            device=device,
            batch_size=args.batch_size,
            repeats=args.perm_repeats,
            seed=args.seed,
        )
        perm_test_df = _permutation_importance(
            model=model,
            x=x_test,
            t_hours=t_test,
            y_true=y_test,
            feature_names=list(args.feature_cols),
            device=device,
            batch_size=args.batch_size,
            repeats=args.perm_repeats,
            seed=args.seed + 1,
        )

        perm_val_df.to_csv(output_dir / "perm_importance_val.csv", index=False)
        perm_test_df.to_csv(output_dir / "perm_importance_test.csv", index=False)

    if not args.no_plots:
        _save_plots(
            history=history_df,
            pred_val=pred_val_df,
            pred_test=pred_test_df,
            baseline_persist_val_mse=float(persist_val["mse_norm"]),
            baseline_linear_val_mse=float(linear_val["mse_norm"]),
            perm_val=perm_val_df,
            perm_test=perm_test_df,
            plots_dir=plots_dir,
            n_bins_1d=int(args.bins_1d),
            n_bins_2d=int(args.bins_2d),
        )

    print(f"\nsaved artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
