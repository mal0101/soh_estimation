"""Random Forest model with Optuna hyperparameter tuning.

Wraps sklearn RandomForestRegressor with Optuna search and MLflow logging.
"""

import logging

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_space: dict,
):
    """Create an Optuna objective function for Random Forest.

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
        n_estimators = trial.suggest_int("n_estimators", *param_space["n_estimators"])
        max_depth = trial.suggest_categorical("max_depth", param_space["max_depth"])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", *param_space["min_samples_leaf"])
        max_features = trial.suggest_categorical("max_features", param_space["max_features"])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse_val = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
        return rmse_val

    return objective


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int = 100,
    param_space: dict | None = None,
) -> tuple[RandomForestRegressor, dict[str, float], dict[str, float]]:
    """Train Random Forest with Optuna tuning and evaluate on test set.

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
            "n_estimators": (100, 500),
            "max_depth": [10, 40, None],
            "min_samples_leaf": (1, 10),
            "max_features": ["sqrt", "log2", 0.5],
        }

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        create_objective(X_train, y_train, X_test, y_test, param_space),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    logger.info("RF best params: %s (RMSE=%.6f)", best, study.best_value)

    model = RandomForestRegressor(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        min_samples_leaf=best["min_samples_leaf"],
        max_features=best["max_features"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from src.evaluation.metrics import compute_all_metrics

    test_metrics = compute_all_metrics(y_test, y_pred)

    logger.info("RF test metrics: %s", test_metrics)
    return model, best, test_metrics
