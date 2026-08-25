"""Random Forest model with Optuna hyperparameter tuning.

Separation of concerns:
    - ``optimize_rf`` selects hyperparameters using an INNER validation
      split (never the outer LOOCV test cell).
    - ``build_rf`` constructs a fresh model from chosen parameters so the
      orchestrator can refit on the FULL training fold.
"""

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

DEFAULT_PARAM_SPACE: dict[str, Any] = {
    "n_estimators": (100, 500),
    "max_depth": [10, 40, None],
    "min_samples_leaf": (1, 10),
    "max_features": ["sqrt", "log2", 0.5],
}


def build_rf(params: dict[str, Any], seed: int = 42) -> RandomForestRegressor:
    """Construct an unfitted RandomForestRegressor from a params dict.

    Args:
        params: Hyperparameters (n_estimators, max_depth,
            min_samples_leaf, max_features).
        seed: Random seed for reproducibility.

    Returns:
        Unfitted model instance.
    """
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        random_state=seed,
        n_jobs=-1,
    )


def optimize_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    param_space: dict | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Select RF hyperparameters via Optuna against the validation split.

    Args:
        X_train: Inner-training features.
        y_train: Inner-training targets.
        X_val: Inner-validation features. Must NOT be the outer test cell.
        y_val: Inner-validation targets.
        n_trials: Number of Optuna search trials.
        param_space: Parameter space dict (bounds/categoricals).
        seed: Sampler and estimator seed.

    Returns:
        Best parameter dictionary.
    """
    space = param_space or DEFAULT_PARAM_SPACE

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_categorical("max_depth", space["max_depth"]),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", *space["min_samples_leaf"]
            ),
            "max_features": trial.suggest_categorical("max_features", space["max_features"]),
        }
        model = build_rf(params, seed=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return float(np.sqrt(np.mean((y_val - y_pred) ** 2)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("RF best params: %s (val RMSE=%.6f)", study.best_params, study.best_value)
    return study.best_params


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    param_space: dict | None = None,
    X_refit: np.ndarray | None = None,
    y_refit: np.ndarray | None = None,
    seed: int = 42,
):
    """Optimize + fit RF; evaluate on the validation split.

    Kept for API compatibility with existing tests/notebooks. The
    orchestrator uses optimize_rf/build_rf directly.

    Args:
        X_train: Inner-training features.
        y_train: Inner-training targets.
        X_val: Validation features used for selection AND reported metrics.
        y_val: Validation targets.
        n_trials: Number of Optuna trials.
        param_space: Parameter space from config.
        X_refit: Optional larger training set for the final refit
            (inner-train + inner-val of the outer fold).
        y_refit: Optional targets for the final refit.
        seed: Random seed.

    Returns:
        Tuple of (fitted model, best_params, val_metrics).
    """
    from src.evaluation.metrics import compute_all_metrics

    best = optimize_rf(X_train, y_train, X_val, y_val, n_trials=n_trials, param_space=param_space, seed=seed)
    model = build_rf(best, seed=seed)
    if X_refit is not None and y_refit is not None:
        model.fit(X_refit, y_refit)
    else:
        model.fit(X_train, y_train)
    metrics = compute_all_metrics(y_val, model.predict(X_val))
    return model, best, metrics
