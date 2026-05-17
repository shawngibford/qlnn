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

    assert sequences.shape == (3, 4, 6)
    assert targets.shape == (3, 4)

    assert sequences.min() >= -1e-6
    assert sequences.max() <= 1.0 + 1e-6

    assert targets.min() >= -1e-6
    assert targets.max() <= 1.0 + 1e-6


def test_preprocessor_raises_if_window_too_large():
    df = pd.DataFrame({"A": [1, 2, 3], "OD": [0.1, 0.2, 0.3]})
    pre = BioreactorDataPreprocessor(df=df, feature_cols=["A"], target_col="OD")

    with pytest.raises(ValueError):
        pre.create_sequences(window_size=10, stride=1)
