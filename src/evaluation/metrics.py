"""Evaluation metrics for SOH estimation.

Implements the four metrics used throughout the benchmark: RMSE, MAE,
MaxAE, and R² (coefficient of determination).
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground-truth SOH values.
        y_pred: Predicted SOH values.

    Returns:
        RMSE scalar.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Ground-truth SOH values.
        y_pred: Predicted SOH values.

    Returns:
        MAE scalar.
    """
    return float(mean_absolute_error(y_true, y_pred))


def maxae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum Absolute Error.

    Args:
        y_true: Ground-truth SOH values.
        y_pred: Predicted SOH values.

    Returns:
        MaxAE scalar.
    """
    return float(np.max(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²).

    Args:
        y_true: Ground-truth SOH values.
        y_pred: Predicted SOH values.

    Returns:
        R² scalar.
    """
    return float(r2_score(y_true, y_pred))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all four benchmark metrics at once.

    Args:
        y_true: Ground-truth SOH values.
        y_pred: Predicted SOH values.

    Returns:
        Dictionary with keys: rmse, mae, maxae, r2.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "maxae": maxae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
