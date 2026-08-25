"""LSTM model for SOH estimation.

Two-layer LSTM that takes windowed per-cycle features and predicts the
SOH of the final cycle in the window.

Protocol:
    - ``optimize_lstm`` tunes hyperparameters on an INNER cell split,
      early stopping on the inner-validation cell. The best trial's
      epoch count is recorded.
    - ``train_lstm_final`` refits on the FULL training fold for exactly
      the chosen number of epochs. The outer test cell is never seen
      during selection.
    - Datasets are built directly from cell-grouped DataFrames via
      SOHDataset, so sequence windows never cross cell boundaries and no
      double-windowing shift occurs.
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
    "lstm_1_units": [32, 64, 128],
    "lstm_2_units": [16, 32, 64],
    "dropout": (0.1, 0.4),
    "learning_rate": (1e-4, 1e-2),
}


class LSTMModel(nn.Module):
    """Two-layer LSTM for sequence-to-scalar regression.

    Architecture:
        LSTM(input_dim → hidden_1) → LSTM(hidden_1 → hidden_2)
        → last time-step hidden → Linear → ReLU → Dropout → Linear → 1
    """

    def __init__(
        self,
        input_dim: int,
        hidden_1: int,
        hidden_2: int,
        dense_dim: int,
        dropout: float,
    ) -> None:
        """Initialize the LSTM model.

        Args:
            input_dim: Number of input features per time step.
            hidden_1: Hidden size of the first LSTM layer.
            hidden_2: Hidden size of the second LSTM layer.
            dense_dim: Width of the penultimate dense layer.
            dropout: Dropout probability.
        """
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_1, hidden_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_2, dense_dim),
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
        out1, _ = self.lstm1(x)
        out2, (h_n, _) = self.lstm2(out1)
        last_hidden = self.dropout(h_n[-1])
        result: torch.Tensor = self.fc(last_hidden)
        return result


def build_lstm(input_dim: int, params: dict[str, Any], dense_units: int = 16) -> LSTMModel:
    """Construct an unfitted LSTMModel from a params dict.

    Args:
        input_dim: Number of input features per step.
        params: Hyperparameters (lstm_1_units, lstm_2_units, dropout).
        dense_units: Width of the penultimate dense layer (config default).

    Returns:
        Unfitted model instance.
    """
    return LSTMModel(
        input_dim,
        int(params["lstm_1_units"]),
        int(params["lstm_2_units"]),
        dense_dim=int(dense_units),
        dropout=float(params["dropout"]),
    )


def optimize_lstm(
    train_df,
    val_df,
    feature_cols: list[str],
    window_size: int,
    train_cfg: dict,
    device: torch.device,
    seed: int = 42,
    n_trials: int = 10,
    param_space: dict | None = None,
    dense_units: int = 16,
) -> dict[str, Any]:
    """Select LSTM hyperparameters against an INNER validation split.

    Args:
        train_df: Inner-training DataFrame (cell-grouped, scaled features).
        val_df: Inner-validation DataFrame (held-out inner cell).
        feature_cols: Feature column names.
        window_size: Sequence window length.
        train_cfg: DL training config (batch_size, max_epochs, ...).
        device: Compute device.
        seed: Random seed.
        n_trials: Number of Optuna trials.
        param_space: Search space override.
        dense_units: Dense layer width (from config defaults).

    Returns:
        Dict with best hyperparameters plus 'best_epoch' and
        'val_rmse' from the best trial.
    """
    space = param_space or DEFAULT_PARAM_SPACE
    batch_size = train_cfg.get("batch_size", 64)
    input_dim = len(feature_cols)

    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "lstm_1_units": trial.suggest_categorical("lstm_1_units", space["lstm_1_units"]),
            "lstm_2_units": trial.suggest_categorical("lstm_2_units", space["lstm_2_units"]),
            "dropout": trial.suggest_float("dropout", *space["dropout"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        }

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        set_seed(seed)
        model = build_lstm(input_dim, params, dense_units=dense_units)
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
    logger.info("LSTM best params: %s (val RMSE=%.6f)", best, best["val_rmse"])
    return best


def train_lstm_final(
    full_train_df,
    feature_cols: list[str],
    window_size: int,
    train_cfg: dict,
    device: torch.device,
    seed: int,
    params: dict[str, Any],
    dense_units: int = 16,
) -> tuple[LSTMModel, dict]:
    """Fit the final LSTM on the full training fold for fixed epochs.

    Args:
        full_train_df: Full fold-training DataFrame (all training cells).
        feature_cols: Feature column names.
        window_size: Sequence window length.
        train_cfg: DL training config.
        device: Compute device.
        seed: Random seed.
        params: Selected hyperparameters incl. 'best_epoch' and
            'learning_rate'.
        dense_units: Dense layer width (config default).

    Returns:
        Tuple of (trained model, history dict).
    """
    batch_size = train_cfg.get("batch_size", 64)
    set_seed(seed)
    model = build_lstm(len(feature_cols), params, dense_units=dense_units)

    train_ds = SOHDataset(full_train_df, feature_cols, window_size)
    if len(train_ds) == 0:
        raise ValueError("No trainable sequences in the full training fold")
    # Validation loader reuses training data purely to compute the loss
    # curve (no early stopping occurs under fixed_epochs).
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    shuffle_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    final_config = {
        **train_cfg,
        "learning_rate": params["learning_rate"],
        "fixed_epochs": max(int(params.get("best_epoch", 0)) + 1, 5),
    }
    history = train_loop(model, shuffle_loader, loader, final_config, device, seed)
    return model, history


def evaluate_lstm(
    model: LSTMModel,
    test_df,
    feature_cols: list[str],
    window_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Evaluate a trained LSTM on the held-out test cell.

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
