"""Naive baseline model for SOH estimation.

The baseline predicts the mean SOH of the training set for every test
sample. This provides a lower bound on model performance.
"""

import numpy as np


class NaiveBaseline:
    """Mean-prediction baseline.

    Attributes:
        mean_soh: Mean SOH value computed from the training set.
    """

    mean_soh: float

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBaseline":
        """Fit by computing training mean.

        Args:
            X: Training features (unused, kept for API compatibility).
            y: Training SOH targets.

        Returns:
            self.
        """
        self.mean_soh = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the training mean for all samples.

        Args:
            X: Test features (unused, kept for API compatibility).

        Returns:
            Array of constant predictions.
        """
        return np.full(len(X), self.mean_soh)
