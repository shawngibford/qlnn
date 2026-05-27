from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class BioreactorDataPreprocessor:
    """Prepare bioreactor time series for modeling.

    This class is intentionally minimal and testable:
    - No implicit feature selection; caller provides feature/target columns.
    - Normalization uses per-column MinMaxScaler to map to [0, 1], which is
      commonly convenient for angle-based quantum feature encodings.
    """

    df: pd.DataFrame
    feature_cols: List[str]
    target_col: str

    scalers: Dict[str, MinMaxScaler] | None = None

    @classmethod
    def from_csv(
        cls,
        data_path: str | Path,
        *,
        feature_cols: Sequence[str],
        target_col: str,
    ) -> "BioreactorDataPreprocessor":
        df = pd.read_csv(data_path)
        return cls(df=df, feature_cols=list(feature_cols), target_col=target_col)

    def normalize_minmax(self) -> "BioreactorDataPreprocessor":
        scalers: Dict[str, MinMaxScaler] = {}

        cols = list(self.feature_cols) + [self.target_col]
        for col in cols:
            if col not in self.df.columns:
                raise ValueError(f"Column not found in dataframe: {col}")

            scaler = MinMaxScaler()
            self.df[col] = scaler.fit_transform(self.df[[col]])
            scalers[col] = scaler

        self.scalers = scalers
        return self

    def create_sequences(
        self,
        *,
        window_size: int,
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences.

        Returns:
            sequences: (n_windows, window_size, n_features)
            targets: (n_windows, window_size)
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        n_rows = len(self.df)
        if n_rows < window_size + 1:
            raise ValueError(
                f"Not enough rows ({n_rows}) for window_size={window_size}. Need at least {window_size + 1}."
            )

        features = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        target = self.df[self.target_col].to_numpy(dtype=np.float32)

        sequences: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        # `+ 1` so the final valid window (start = n_rows - window_size) is
        # included. Off-by-one bug fix.
        for i in range(0, n_rows - window_size + 1, stride):
            seq = features[i : i + window_size]
            tgt = target[i : i + window_size]
            sequences.append(seq)
            targets.append(tgt)

        return np.stack(sequences, axis=0), np.stack(targets, axis=0)
