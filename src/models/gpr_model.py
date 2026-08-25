"""Gaussian Process Regression model.

Wraps sklearn GaussianProcessRegressor with Matérn(1.5) kernel and
optional subsampling for computational tractability.
"""

import logging

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

logger = logging.getLogger(__name__)


def subsample_train_set(
    X: np.ndarray, y: np.ndarray, max_train_samples: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically cap the training set for GP tractability.

    Args:
        X: Training features.
        y: Training targets.
        max_train_samples: Maximum number of rows to keep.
        seed: Subsampling seed.

    Returns:
        Possibly subsampled (X, y).
    """
    if len(X) > max_train_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), max_train_samples, replace=False)
        logger.info("Subsampled GPR training set: %d -> %d", len(X), max_train_samples)
        return X[idx], y[idx]
    return X, y


def build_gpr(n_restarts: int = 5, seed: int = 42) -> GaussianProcessRegressor:
    """Construct an unfitted GPR with the fixed Matérn(1.5)+White kernel.

    Args:
        n_restarts: Number of optimizer restarts for kernel hyperparameters.
        seed: Random seed.

    Returns:
        Unfitted model instance.
    """
    kernel = Matern(nu=1.5, length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(
        noise_level=0.01, noise_level_bounds=(1e-5, 1e1)
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=seed,
    )


def train_gpr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_train_samples: int = 5000,
    n_restarts: int = 5,
) -> tuple[GaussianProcessRegressor, dict[str, float]]:
    """Train GPR with Matérn(1.5) kernel and evaluate on test set.

    If training set exceeds max_train_samples, a random subsample is used.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        max_train_samples: Maximum training set size.
        n_restarts: Number of optimizer restarts for kernel hyperparameters.

    Returns:
        Tuple of (fitted model, test_metrics).
    """
    if len(X_train) > max_train_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), max_train_samples, replace=False)
        X_train_sub = X_train[idx]
        y_train_sub = y_train[idx]
        logger.info("Subsampled GPR training set: %d -> %d", len(X_train), max_train_samples)
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    kernel = Matern(nu=1.5, length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(
        noise_level=0.01, noise_level_bounds=(1e-5, 1e1)
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=42,
    )

    logger.info("Fitting GPR on %d samples...", len(X_train_sub))
    model.fit(X_train_sub, y_train_sub)
    logger.info("GPR fitted. Final kernel: %s", model.kernel_)

    y_pred = model.predict(X_test)

    from src.evaluation.metrics import compute_all_metrics

    test_metrics = compute_all_metrics(y_test, y_pred)

    logger.info("GPR test metrics: %s", test_metrics)
    return model, test_metrics
