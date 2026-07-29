"""1D Convolutional Neural Network for SOH estimation.

Three-layer Conv1D with Global Average Pooling that takes windowed
per-cycle features and predicts the SOH of the final cycle.
"""

import logging

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics
from src.models.dl_base import SOHDataset, evaluate, set_seed, train_loop

logger = logging.getLogger(__name__)


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
        return self.fc(out)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    device: torch.device,
    seed: int = 42,
    n_trials: int = 30,
) -> tuple[dict, dict[str, float]]:
    """Train CNN with Optuna hyperparameter search.

    Args:
        X_train: Training features [n, window, features].
        y_train: Training SOH targets.
        X_test: Test features.
        y_test: Test SOH targets.
        config: DL config dict.
        device: Compute device.
        seed: Random seed.
        n_trials: Number of Optuna trials.

    Returns:
        Tuple of (best_params, test_metrics).
    """
    input_dim = X_train.shape[2]
    batch_size = config.get("batch_size", 64)

    def objective(trial: optuna.Trial) -> float:
        f1 = trial.suggest_categorical("filters_1", [16, 32, 64])
        f2 = trial.suggest_categorical("filters_2", [32, 64, 128])
        f3 = trial.suggest_categorical("filters_3", [64, 128, 256])
        ks = trial.suggest_categorical("kernel_size", [3, 5, 7])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        model = CNNModel(input_dim, [f1, f2, f3], ks, dense_dim=64, dropout=dropout)

        set_seed(seed)
        train_dataset = SOHDataset(
            _arrays_to_df(X_train, y_train), list(range(X_train.shape[2])), X_train.shape[1]
        )
        val_dataset = SOHDataset(
            _arrays_to_df(X_test, y_test), list(range(X_test.shape[2])), X_test.shape[1]
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        trial_config = {**config, "learning_rate": lr}
        train_loop(model, train_loader, val_loader, trial_config, device, seed)

        y_true, y_pred = evaluate(model, val_loader, device)
        rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return rmse_val

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info("CNN best params: %s (RMSE=%.6f)", best, study.best_value)

    model = CNNModel(
        input_dim,
        [best["filters_1"], best["filters_2"], best["filters_3"]],
        best["kernel_size"],
        dense_dim=64,
        dropout=best["dropout"],
    )
    set_seed(seed)
    train_dataset = SOHDataset(
        _arrays_to_df(X_train, y_train), list(range(X_train.shape[2])), X_train.shape[1]
    )
    test_dataset = SOHDataset(
        _arrays_to_df(X_test, y_test), list(range(X_test.shape[2])), X_test.shape[1]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    final_config = {**config, "learning_rate": best["learning_rate"]}
    train_loop(model, train_loader, test_loader, final_config, device, seed)

    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = compute_all_metrics(y_true, y_pred)
    logger.info("CNN test metrics: %s", metrics)
    return best, metrics


def _arrays_to_df(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Convert windowed arrays to a flat DataFrame for SOHDataset.

    Args:
        X: Feature array [n_samples, window, n_features].
        y: Target array [n_samples].

    Returns:
        DataFrame with synthetic cell_id, cycle_number, soh, and feature columns.
    """
    n_samples = len(y)
    n_features = X.shape[2]
    records = []
    for i in range(n_samples):
        row = {
            "cell_id": "train",
            "cycle_number": i,
            "soh": float(y[i]),
        }
        for j in range(n_features):
            row[j] = float(X[i, -1, j])
        records.append(row)
    return pd.DataFrame(records)
