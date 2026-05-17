from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLS: tuple[str, ...] = (
    "OD",
    "PRE",
    "TEMP_EXT",
    "TEMP_CULTURE",
    "PAR_LIGHT",
    "PH",
    "DO",
)
DEFAULT_TARGET_COL = "OD"


def load_qzeta(csv_path: str | Path) -> pd.DataFrame:
    """Load the qZETA bioreactor CSV with the column/date normalization
    every downstream script depends on.

    Normalizes:
    - strips whitespace from column names
    - renames "TEMP EXT" -> "TEMP_EXT"
    - parses DATE column and sorts ascending
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    if "TEMP EXT" in df.columns and "TEMP_EXT" not in df.columns:
        df = df.rename(columns={"TEMP EXT": "TEMP_EXT"})

    if "DATE" not in df.columns:
        raise ValueError("Expected a DATE column in the CSV")

    dt = pd.to_datetime(df["DATE"], format="mixed", dayfirst=True, errors="raise")
    df = df.assign(DATE=dt).sort_values("DATE").reset_index(drop=True)

    return df


def time_hours_from_date(df: pd.DataFrame, *, date_col: str = "DATE") -> np.ndarray:
    """Return time-from-start in hours as a float64 ndarray of length len(df)."""
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    secs = (df[date_col] - df[date_col].iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
    return secs / 3600.0
