"""Shared test fixtures for the SOH estimation benchmark.

Provides synthetic data fixtures that mimic the real dataset structure.
Most tests run entirely on synthetic data; the CALCE loader tests in
test_preprocessing.py are the exception and read the real .xlsx files
from data/raw/calce (skipped automatically if the directory is absent).
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def numpy_data():
    """Small synthetic numpy arrays for fast unit tests.

    Returns:
        Tuple of (X, y) with 20 samples and 5 features.
    """
    rng = np.random.RandomState(42)
    X = rng.randn(20, 5)
    y = 0.8 + 0.1 * X[:, 0] - 0.05 * X[:, 1] + rng.randn(20) * 0.01
    return X, y


@pytest.fixture
def feature_df_small():
    """Synthetic feature matrix mimicking feature_matrix parquet files.

    Creates a DataFrame with 3 cells, 20 cycles each, and 5 numeric
    features plus metadata columns.

    Returns:
        pd.DataFrame with columns matching the real feature matrix structure.
    """
    rng = np.random.RandomState(42)
    records = []
    for cell_id in ["cell_A", "cell_B", "cell_C"]:
        for cycle in range(1, 21):
            soh = 1.0 - 0.005 * cycle + rng.randn() * 0.01
            records.append(
                {
                    "cell_id": cell_id,
                    "dataset": "test",
                    "cycle_number": cycle,
                    "soh": soh,
                    "feat_1": rng.randn(),
                    "feat_2": rng.randn(),
                    "feat_3": rng.randn(),
                    "feat_4": rng.randn(),
                    "feat_5": rng.randn(),
                }
            )
    return pd.DataFrame(records)
