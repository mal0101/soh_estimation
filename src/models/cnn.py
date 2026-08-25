"""1D Convolutional Neural Network for SOH estimation.

Three-layer Conv1D with Global Average Pooling that takes windowed
per-cycle features and predicts the SOH of the final cycle.

Uses the same leakage-safe two-stage protocol as the LSTM:
inner-split Optuna selection, then fixed-epoch refit on the full
training fold.
"""

import logging
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics
from src.models.dl_base import SOHDataset, evaluate, train_loop
from src.utils.seeding import set_seed

logger = logging.getLogger(__name__)

DEFAULT_PARAM_SPACE: dict[str, Any] = {
    "filters_1": [16, 32, 64],
    "filters_2": [32, 64, 128],
    "filters_3": [64, 128, 256],
    "kernel_size": [3, 5, 7],
    "dropout": (0.1, 0.4),
    "learning_rate": (1e-4, 1e-2),
}


class CNNModel(nn.Module):
    """Three-layer 1D CNN with Global Average Pooling.

    Architecture:
        Conv1d + BN + ReLU × 3 → AdaptiveAvgPool1d(1) → Flatten
        → Linear → ReLU → Dropout → Linear → 1
    """

    def __init__(
        self,
        input_dim: int,
        filters_list: list[int],
        kernel_size: int,
        dense_dim: int,
        dropout: float,
    ) -> None:
        """Initialize the CNN model.

        Args:
            input_dim: Number of input features per time step.
            filters_list: List of 3 filter counts for each conv layer.
            kernel_size: Convolution kernel size.
            dense_dim: Width of the penultimate dense layer.
            dropout: Dropout probability.
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, filters_list[0], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters_list[0]),
            nn.ReLU(),
            nn.Conv1d(filters_list[0], filters_list[1], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters_list[1]),
            nn.ReLU(),
            nn.Conv1d(filters_list[1], filters_list[2], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters_list[2]),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(filters_list[2], dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, input_dim].

        Returns:
            Predictions of shape [batch, 1].
        """
        out = x.permute(0, 2, 1)
        out = self.conv_layers(out)
        out = self.pool(out).squeeze(-1)
        result: torch.Tensor = self.fc(out)
        return result


def build_cnn(input_dim: int, params: dict[str, Any], dense_units: int = 64) -> CNNModel:
    """Construct an unfitted CNNModel from a params dict.

    Args:
        input_dim: Number of input features per step.
        params: Hyperparameters incl. filters_{1,2,3}, kernel_size, dropout.
        dense_units: Dense layer width (config default).

    Returns:
        Unfitted model instance.
    """
    return CNNModel(
        input_dim,
        [int(params["filters_1"]), int(params["filters_2"]), int(params["filters_3"])],
        int(params["kernel_size"]),
        dense_dim=int(dense_units),
        dropout=float(params["dropout"]),
    )


def optimize_cnn(
    train_df,
    val_df,
    feature_cols: list[str],
    window_size: int,
    train_cfg: dict,
    device: torch.device,
    seed: int = 42,
    n_trials: int = 10,
    param_space: dict | None = None,
    dense_units: int = 64,
) -> dict[str, Any]:
    """Select CNN hyperparameters against an INNER validation split.

    Args:
        train_df: Inner-training DataFrame (cell-grouped, scaled features).
        val_df: Inner-validation DataFrame (held-out inner cell).
        feature_cols: Feature column names.
        window_size: Sequence window length.
        train_cfg: DL training config.
        device: Compute device.
        seed: Random seed.
        n_trials: Number of Optuna trials.
        param_space: Search space override.
        dense_units: Dense layer width (config default).

    Returns:
        Best hyperparameters plus 'best_epoch' and 'val_rmse'.
    """
    space = param_space or DEFAULT_PARAM_SPACE
    batch_size = train_cfg.get("batch_size", 64)
    input_dim = len(feature_cols)

    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "filters_1": trial.suggest_categorical("filters_1", space["filters_1"]),
            "filters_2": trial.suggest_categorical("filters_2", space["filters_2"]),
            "filters_3": trial.suggest_categorical("filters_3", space["filters_3"]),
            "kernel_size": trial.suggest_categorical("kernel_size", space["kernel_size"]),
            "dropout": trial.suggest_float("dropout", *space["dropout"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        }

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        set_seed(seed)
        model = build_cnn(input_dim, params, dense_units=dense_units)
        train_ds = SOHDataset(train_df, feature_cols, window_size)
        val_ds = SOHDataset(val_df, feature_cols, window_size)
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise optuna.TrialPruned("empty sequence dataset")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        trial_config = {**train_cfg, "learning_rate": params["learning_rate"]}
        history = train_loop(model, train_loader, val_loader, trial_config, device, seed)
        trial.set_user_attr("best_epoch", int(history["best_epoch"]))

        y_true, y_pred = evaluate(model, val_loader, device)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["best_epoch"] = int(study.best_trial.user_attrs.get("best_epoch", 0))
    best["val_rmse"] = float(study.best_value)
    logger.info("CNN best params: %s (val RMSE=%.6f)", best, best["val_rmse"])
    return best


def train_cnn_final(
    full_train_df,
    feature_cols: list[str],
    window_size: int,
    train_cfg: dict,
    device: torch.device,
    seed: int,
    params: dict[str, Any],
    dense_units: int = 64,
) -> tuple[CNNModel, dict]:
    """Fit the final CNN on the full training fold for fixed epochs.

    Args:
        full_train_df: Full fold-training DataFrame.
        feature_cols: Feature column names.
        window_size: Sequence window length.
        train_cfg: DL training config.
        device: Compute device.
        seed: Random seed.
        params: Selected hyperparameters.
        dense_units: Dense layer width (config default).

    Returns:
        Tuple of (trained model, history dict).
    """
    batch_size = train_cfg.get("batch_size", 64)
    set_seed(seed)
    model = build_cnn(len(feature_cols), params, dense_units=dense_units)

    train_ds = SOHDataset(full_train_df, feature_cols, window_size)
    if len(train_ds) == 0:
        raise ValueError("No trainable sequences in the full training fold")
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    shuffle_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    final_config = {
        **train_cfg,
        "learning_rate": params["learning_rate"],
        "fixed_epochs": max(int(params.get("best_epoch", 0)) + 1, 5),
    }
    history = train_loop(model, shuffle_loader, loader, final_config, device, seed)
    return model, history


def evaluate_cnn(
    model: CNNModel,
    test_df,
    feature_cols: list[str],
    window_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Evaluate a trained CNN on the held-out test cell.

    Returns:
        Tuple of (y_true, y_pred, metrics).
    """
    test_ds = SOHDataset(test_df, feature_cols, window_size)
    if len(test_ds) == 0:
        raise ValueError("No test sequences for this fold")
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = compute_all_metrics(y_true, y_pred)
    return y_true, y_pred, metrics
