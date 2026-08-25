"""Transformer model for SOH estimation.

Encoder-only Transformer that takes windowed per-cycle features and
predicts the SOH of the final cycle in the window.

Uses the same leakage-safe two-stage protocol as the LSTM/CNN:
inner-split Optuna selection, then fixed-epoch refit on the full
training fold.
"""

import logging
import math
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
    "d_model": [32, 64],
    "n_heads": [2, 4],
    "n_encoder_blocks": [1, 2, 3],
    "dropout": (0.1, 0.3),
    "learning_rate": (1e-4, 1e-2),
}


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 200) -> None:
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
        encoded: torch.Tensor = x + self.pe[:, : x.size(1)]
        return encoded


class TransformerModel(nn.Module):
    """Encoder-only Transformer for sequence-to-scalar regression.

    Architecture:
        Linear(input_dim → d_model) + PositionalEncoding
        → N × TransformerEncoderLayer → Global Average Pooling
        → Linear(d_model → 1)

    The FFN width is fixed at ``2 * d_model`` to keep the parameter
    budget proportional to the embedding size.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_blocks: int,
        dropout: float,
        ffn_mult: int = 2,
        max_seq_len: int = 200,
    ) -> None:
        """Initialize the Transformer model.

        Args:
            input_dim: Number of input features per time step.
            d_model: Internal embedding dimension.
            n_heads: Number of attention heads.
            n_blocks: Number of encoder blocks.
            dropout: Dropout probability.
            ffn_mult: FFN hidden dim = ffn_mult * d_model.
            max_seq_len: Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_mult * d_model,
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
        projected: torch.Tensor = self.input_proj(x)
        encoded: torch.Tensor = self.pos_encoding(projected)
        ctx: torch.Tensor = self.encoder(encoded)
        pooled = ctx.mean(dim=1)
        result: torch.Tensor = self.fc(pooled)
        return result


def build_transformer(input_dim: int, params: dict[str, Any]) -> TransformerModel:
    """Construct an unfitted TransformerModel from a params dict.

    Args:
        input_dim: Number of input features per step.
        params: Hyperparameters (d_model, n_heads, n_encoder_blocks, dropout).

    Returns:
        Unfitted model instance with max_seq_len matched to the window.
    """
    return TransformerModel(
        input_dim,
        int(params["d_model"]),
        int(params["n_heads"]),
        int(params["n_encoder_blocks"]),
        float(params["dropout"]),
    )


def optimize_transformer(
    train_df,
    val_df,
    feature_cols: list[str],
    window_size: int,
    train_cfg: dict,
    device: torch.device,
    seed: int = 42,
    n_trials: int = 10,
    param_space: dict | None = None,
) -> dict[str, Any]:
    """Select Transformer hyperparameters against an INNER validation split.

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

    Returns:
        Best hyperparameters plus 'best_epoch' and 'val_rmse'.
    """
    space = param_space or DEFAULT_PARAM_SPACE
    batch_size = train_cfg.get("batch_size", 64)
    input_dim = len(feature_cols)

    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "d_model": trial.suggest_categorical("d_model", space["d_model"]),
            "n_heads": trial.suggest_categorical("n_heads", space["n_heads"]),
            "n_encoder_blocks": trial.suggest_categorical(
                "n_encoder_blocks", space["n_encoder_blocks"]
            ),
            "dropout": trial.suggest_float("dropout", *space["dropout"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        }

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        set_seed(seed)
        model = build_transformer(input_dim, params)
        # Positional encoding must cover the configured window length.
        model.pos_encoding = PositionalEncoding(int(params["d_model"]), max_len=max(window_size * 2, 64))
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
    logger.info("Transformer best params: %s (val RMSE=%.6f)", best, best["val_rmse"])
    return best


def train_transformer_final(
    full_train_df,
    feature_cols: list[str],
    window_size: int,
    train_cfg: dict,
    device: torch.device,
    seed: int,
    params: dict[str, Any],
) -> tuple[TransformerModel, dict]:
    """Fit the final Transformer on the full training fold for fixed epochs.

    Args:
        full_train_df: Full fold-training DataFrame.
        feature_cols: Feature column names.
        window_size: Sequence window length.
        train_cfg: DL training config.
        device: Compute device.
        seed: Random seed.
        params: Selected hyperparameters.

    Returns:
        Tuple of (trained model, history dict).
    """
    batch_size = train_cfg.get("batch_size", 64)
    set_seed(seed)
    model = build_transformer(len(feature_cols), params)
    model.pos_encoding = PositionalEncoding(int(params["d_model"]), max_len=max(window_size * 2, 64))

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


def evaluate_transformer(
    model: TransformerModel,
    test_df,
    feature_cols: list[str],
    window_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Evaluate a trained Transformer on the held-out test cell.

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
