"""Shared test fixtures for the SOH estimation benchmark.

Provides synthetic data fixtures that mimic the real dataset structure
without requiring actual NASA/CALCE files. All fixtures are function-scoped
for isolation.
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
    """Synthetic feature matrix mimicking feature_matrix.parquet.

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


@pytest.fixture
def feature_df_with_nans():
    """Feature matrix with NaN values in some feature columns.

    Same structure as feature_df_small but with ~10% NaN in feat_3.

    Returns:
        pd.DataFrame with NaN values.
    """
    df = feature_df_small()
    rng = np.random.RandomState(99)
    nan_mask = rng.rand(len(df)) < 0.1
    df.loc[nan_mask, "feat_3"] = np.nan
    return df


@pytest.fixture
def mock_config():
    """Minimal config dict matching default.yaml structure.

    Returns:
        Dict with only the keys needed by model training code.
    """
    return {
        "models": {
            "classical": {
                "rf": {
                    "n_trials": 5,
                    "param_space": {
                        "n_estimators": [100, 200],
                        "max_depth": [10, None],
                        "min_samples_leaf": [1, 5],
                        "max_features": ["sqrt", 0.5],
                    },
                },
                "svr": {
                    "n_trials": 5,
                    "param_space": {
                        "C": [0.01, 100],
                        "epsilon": [0.001, 0.1],
                        "gamma": ["scale", "auto"],
                    },
                },
                "gpr": {
                    "max_train_samples": 500,
                    "n_restarts_optimizer": 1,
                },
            },
            "dl": {
                "batch_size": 16,
                "max_epochs": 3,
                "learning_rate": 0.001,
                "patience_early_stopping": 3,
                "patience_lr_reduce": 2,
                "lr_reduce_factor": 0.5,
                "n_seeds": 1,
                "sequence_window": 5,
            },
        },
        "evaluation": {
            "metrics": ["rmse", "mae", "maxae", "r2"],
        },
    }
