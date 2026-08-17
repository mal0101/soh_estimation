"""Transformer model for SOH estimation.

Encoder-only Transformer that takes windowed per-cycle features and
predicts the SOH of the final cycle in the window.
"""

import logging
import math
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics
from src.models.dl_base import SOHDataset, evaluate, train_loop
from src.utils.seeding import set_seed

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 50) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].

        Returns:
            Encoded tensor of same shape.
        """
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    """Encoder-only Transformer for sequence-to-scalar regression.

    Architecture:
        Linear(input_dim → d_model) + PositionalEncoding
        → N × TransformerEncoderLayer → Global Average Pooling
        → Linear(d_model → 1)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_blocks: int,
        ffn_dim: int,
        dropout: float,
        max_seq_len: int = 50,
    ) -> None:
        """Initialize the Transformer model.

        Args:
            input_dim: Number of input features per time step.
            d_model: Internal embedding dimension.
            n_heads: Number of attention heads.
            n_blocks: Number of encoder blocks.
            ffn_dim: Feed-forward network hidden dimension.
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, input_dim].

        Returns:
            Predictions of shape [batch, 1].
        """
        out = self.input_proj(x)
        out = self.pos_encoding(out)
        out = self.encoder(out)
        out = out.mean(dim=1)
        return self.fc(out)


def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    device: torch.device,
    seed: int = 42,
    n_trials: int = 30,
) -> tuple[dict, dict[str, float]]:
    """Train Transformer with Optuna hyperparameter search.

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
        d_model = trial.suggest_categorical("d_model", [32, 64])
        n_heads = trial.suggest_categorical("n_heads", [2, 4])
        n_blocks = trial.suggest_categorical("n_encoder_blocks", [1, 2, 3])
        dropout = trial.suggest_float("dropout", 0.1, 0.3)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        model = TransformerModel(
            input_dim, d_model, n_heads, n_blocks, ffn_dim=d_model * 2, dropout=dropout
        )

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

    best: dict[str, Any] = study.best_params
    logger.info("Transformer best params: %s (RMSE=%.6f)", best, study.best_value)

    model = TransformerModel(
        input_dim,
        best["d_model"],
        best["n_heads"],
        best["n_encoder_blocks"],
        ffn_dim=best["d_model"] * 2,
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
    logger.info("Transformer test metrics: %s", metrics)
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
