"""Support Vector Regression model with Optuna hyperparameter tuning.

Separation of concerns:
    - ``optimize_svr`` selects hyperparameters using an INNER validation
      split (never the outer LOOCV test cell).
    - ``build_svr`` constructs a fresh model from chosen parameters so the
      orchestrator can refit on the FULL training fold.
"""

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

DEFAULT_PARAM_SPACE: dict[str, Any] = {
    "C": (0.01, 1000),
    "epsilon": (0.001, 0.1),
    "gamma": ["scale", "auto", 0.0001, 1.0],
}


def build_svr(params: dict[str, Any]) -> SVR:
    """Construct an unfitted SVR from a params dict.

    Args:
        params: Hyperparameters (C, epsilon, gamma).

    Returns:
        Unfitted model instance.
    """
    gamma = params["gamma"]
    if isinstance(gamma, str) and gamma not in ("scale", "auto"):
        gamma = float(gamma)
    return SVR(kernel="rbf", C=float(params["C"]), epsilon=float(params["epsilon"]), gamma=gamma)


def optimize_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    param_space: dict | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Select SVR hyperparameters via Optuna against the validation split.

    Args:
        X_train: Inner-training features.
        y_train: Inner-training targets.
        X_val: Inner-validation features. Must NOT be the outer test cell.
        y_val: Inner-validation targets.
        n_trials: Number of Optuna search trials.
        param_space: Parameter space dict (bounds/categoricals).
        seed: Sampler seed.

    Returns:
        Best parameter dictionary.
    """
    space = param_space or DEFAULT_PARAM_SPACE

    def objective(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", *space["C"], log=True),
            "epsilon": trial.suggest_float("epsilon", *space["epsilon"]),
            "gamma": trial.suggest_categorical("gamma", space["gamma"]),
        }
        try:
            model = build_svr(params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        except Exception as e:
            logger.warning("SVR trial failed: %s", e)
            return float("inf")

        return float(np.sqrt(np.mean((y_val - y_pred) ** 2)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("SVR best params: %s (val RMSE=%.6f)", study.best_params, study.best_value)
    return study.best_params


def train_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    param_space: dict | None = None,
    X_refit: np.ndarray | None = None,
    y_refit: np.ndarray | None = None,
    seed: int = 42,
):
    """Optimize + fit SVR; evaluate on the validation split.

    Kept for API compatibility with existing tests/notebooks. The
    orchestrator uses optimize_svr/build_svr directly.

    Args:
        X_train: Inner-training features.
        y_train: Inner-training targets.
        X_val: Validation features used for selection AND reported metrics.
        y_val: Validation targets.
        n_trials: Number of Optuna trials.
        param_space: Parameter space from config.
        X_refit: Optional larger training set for the final refit.
        y_refit: Optional targets for the final refit.
        seed: Random seed.

    Returns:
        Tuple of (fitted model, best_params, val_metrics).
    """
    from src.evaluation.metrics import compute_all_metrics

    best = optimize_svr(X_train, y_train, X_val, y_val, n_trials=n_trials, param_space=param_space, seed=seed)
    model = build_svr(best)
    if X_refit is not None and y_refit is not None:
        model.fit(X_refit, y_refit)
    else:
        model.fit(X_train, y_train)
    metrics = compute_all_metrics(y_val, model.predict(X_val))
    return model, best, metrics
