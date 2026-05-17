import numpy as np
import pandas as pd
import pytest

from quantum_liquid_neuralode.data_processing import BioreactorDataPreprocessor


def test_preprocessor_normalize_and_create_sequences_shapes():
    n = 10
    df = pd.DataFrame(
        {
            "PRE": np.linspace(0.0, 1.0, n),
            "TEMP_EXT": np.linspace(10.0, 20.0, n),
            "TEMP_CULTURE": np.linspace(30.0, 40.0, n),
            "PAR_LIGHT": np.linspace(100.0, 200.0, n),
            "PH": np.linspace(6.5, 7.5, n),
            "DO": np.linspace(0.2, 0.9, n),
            "OD": np.linspace(0.47, 3.8, n),
        }
    )

    feature_cols = ["PRE", "TEMP_EXT", "TEMP_CULTURE", "PAR_LIGHT", "PH", "DO"]
    target_col = "OD"

    pre = BioreactorDataPreprocessor(df=df, feature_cols=feature_cols, target_col=target_col)
    pre.normalize_minmax()

    sequences, targets = pre.create_sequences(window_size=4, stride=2)

    # n=10, window_size=4, stride=2 -> valid start indices [0, 2, 4, 6] = 4 windows.
    # (Was 3 under the off-by-one bug fixed in commit XXX.)
    assert sequences.shape == (4, 4, 6)
    assert targets.shape == (4, 4)

    assert sequences.min() >= -1e-6
    assert sequences.max() <= 1.0 + 1e-6

    assert targets.min() >= -1e-6
    assert targets.max() <= 1.0 + 1e-6


def test_create_sequences_includes_final_window():
    """Regression test for R1-B1 off-by-one in `create_sequences`.

    Boundary case: n_rows = window_size + 1. With stride=1 there are exactly
    2 valid windows (start=0 and start=1). The buggy code produced only 1.
    """
    window_size = 4
    n_rows = window_size + 1  # 5
    df = pd.DataFrame(
        {
            "A": np.arange(n_rows, dtype=np.float32),
            "OD": np.linspace(0.1, 0.9, n_rows, dtype=np.float32),
        }
    )

    pre = BioreactorDataPreprocessor(df=df, feature_cols=["A"], target_col="OD")
    sequences, targets = pre.create_sequences(window_size=window_size, stride=1)

    # Expected start indices: range(0, n_rows - window_size + 1, 1) = [0, 1] -> 2 windows.
    assert sequences.shape == (2, window_size, 1)
    assert targets.shape == (2, window_size)
    # The final window must start at index 1 (the off-by-one bug dropped this).
    np.testing.assert_allclose(sequences[-1, :, 0], np.arange(1, 1 + window_size, dtype=np.float32))


def test_preprocessor_raises_if_window_too_large():
    df = pd.DataFrame({"A": [1, 2, 3], "OD": [0.1, 0.2, 0.3]})
    pre = BioreactorDataPreprocessor(df=df, feature_cols=["A"], target_col="OD")

    with pytest.raises(ValueError):
        pre.create_sequences(window_size=10, stride=1)
