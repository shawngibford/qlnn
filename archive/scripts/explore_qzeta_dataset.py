from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quantum_liquid_neuralode.data_processing import BioreactorDataPreprocessor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "qZETA_data_copy.csv"


def _load_and_sort(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize column names minimally (strip whitespace).
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # The dataset uses "TEMP EXT"; our code/tests typically use "TEMP_EXT".
    if "TEMP EXT" in df.columns and "TEMP_EXT" not in df.columns:
        df = df.rename(columns={"TEMP EXT": "TEMP_EXT"})

    if "DATE" in df.columns:
        dt = pd.to_datetime(df["DATE"], format="mixed", dayfirst=True, errors="raise")
        df = df.assign(DATE=dt).sort_values("DATE").reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore and preprocess the qZETA bioreactor CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=["PRE", "TEMP_EXT", "TEMP_CULTURE", "PAR_LIGHT", "PH", "DO"],
        help="Feature columns to use (after column renaming).",
    )
    parser.add_argument("--target-col", type=str, default="OD")
    args = parser.parse_args()

    df = _load_and_sort(args.csv)

    print(f"csv: {args.csv}")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")

    print("\nmissing per col:")
    print(df.isna().sum().to_string())

    if "DATE" in df.columns and pd.api.types.is_datetime64_any_dtype(df["DATE"]):
        d = df["DATE"].diff().dropna()
        print("\nDATE delta summary:")
        print(f"  min:    {d.min()}")
        print(f"  median: {d.median()}")
        print(f"  max:    {d.max()}")
        print("  top delta counts:")
        print(d.value_counts().head(10).to_string())

    if args.target_col in df.columns:
        corr = df.corr(numeric_only=True)[args.target_col].sort_values(ascending=False)
        print(f"\ncorrelation with {args.target_col} (numeric columns only):")
        print(corr.to_string())

    # Basic preprocessing → sequences.
    pre = BioreactorDataPreprocessor(df=df.copy(), feature_cols=args.feature_cols, target_col=args.target_col)
    pre.normalize_minmax()
    sequences, targets = pre.create_sequences(window_size=args.window_size, stride=args.stride)

    print("\npreprocessor outputs:")
    print(f"  feature_cols: {args.feature_cols}")
    print(f"  target_col:   {args.target_col}")
    print(f"  sequences:    {sequences.shape} (n_windows, window, n_features)")
    print(f"  targets:      {targets.shape} (n_windows, window)")
    print(f"  sequences min/max: {sequences.min():.6f} / {sequences.max():.6f}")
    print(f"  targets   min/max: {targets.min():.6f} / {targets.max():.6f}")


if __name__ == "__main__":
    main()
