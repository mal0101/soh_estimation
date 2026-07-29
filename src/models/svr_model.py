"""Support Vector Regression model with Optuna hyperparameter tuning.

Wraps sklearn SVR with Optuna search and MLflow logging.
"""

import logging

import numpy as np
import optuna
from sklearn.svm import SVR

logger = logging.getLogger(__name__)


def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_space: dict,
):
    """Create an Optuna objective function for SVR.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        param_space: Parameter space dict from config.

    Returns:
        Callable objective function.
    """

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", *param_space["C"], log=True)
        epsilon = trial.suggest_float("epsilon", *param_space["epsilon"])
        gamma = trial.suggest_categorical("gamma", param_space["gamma"])
        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            gamma = float(gamma)

        try:
            model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        except Exception as e:
            logger.warning("SVR trial failed: %s", e)
            return float("inf")

        rmse_val = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
        return rmse_val

    return objective


def train_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int = 50,
    param_space: dict | None = None,
) -> tuple[SVR, dict[str, float], dict[str, float]]:
    """Train SVR with Optuna tuning and evaluate on test set.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        n_trials: Number of Optuna search trials.
        param_space: Parameter space from config.

    Returns:
        Tuple of (fitted model, best_params, test_metrics).
    """
    if param_space is None:
        param_space = {
            "C": (0.01, 1000),
            "epsilon": (0.001, 0.1),
            "gamma": ["scale", "auto", 0.0001, 1.0],
        }

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        create_objective(X_train, y_train, X_test, y_test, param_space),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    logger.info("SVR best params: %s (RMSE=%.6f)", best, study.best_value)

    model = SVR(kernel="rbf", C=best["C"], epsilon=best["epsilon"], gamma=best["gamma"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from src.evaluation.metrics import compute_all_metrics

    test_metrics = compute_all_metrics(y_test, y_pred)

    logger.info("SVR test metrics: %s", test_metrics)
    return model, best, test_metrics
