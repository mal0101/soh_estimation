"""Shared deep learning infrastructure for SOH estimation.

Provides the dataset class, training loop, evaluation utilities, device
management, and seeding needed by all DL models (LSTM, CNN, Transformer).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.utils.seeding import set_seed

logger = logging.getLogger(__name__)


class SOHDataset(Dataset):
    """Windowed sequence dataset for per-cycle SOH features.

    Each sample is a window of ``window_size`` consecutive cycles from a
    single cell, and the target is the SOH of the last cycle in the window.
    """

    def __init__(
        self,
        feature_matrix: pd.DataFrame,
        feature_cols: list[str] | list[int],
        window_size: int,
    ) -> None:
        """Initialize the dataset.

        Args:
            feature_matrix: DataFrame with columns ``cell_id``, ``cycle_number``,
                ``soh``, and all feature columns.
            feature_cols: List of feature column names to use as model input.
            window_size: Number of consecutive cycles per sample.
        """
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for cell_id in feature_matrix["cell_id"].unique():
            cell_df = feature_matrix[feature_matrix["cell_id"] == cell_id].sort_values(
                "cycle_number"
            )
            cell_df = cell_df.dropna(subset=feature_cols + ["soh"])
            features = cell_df[feature_cols].values.astype(np.float32)
            soh_values = cell_df["soh"].values.astype(np.float32)

            for i in range(len(features) - window_size + 1):
                x = torch.tensor(features[i : i + window_size], dtype=torch.float32)
                y = torch.tensor(soh_values[i + window_size - 1], dtype=torch.float32)
                self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)


def create_sequences(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create windowed sequences from a feature matrix.

    Convenience function that builds a SOHDataset and returns numpy arrays.

    Args:
        feature_df: Feature matrix DataFrame.
        feature_cols: Feature column names.
        window_size: Sequence window length.

    Returns:
        Tuple of (X, y) numpy arrays. X has shape [n_samples, window_size, n_features].
    """
    dataset = SOHDataset(feature_df, feature_cols, window_size)
    if len(dataset) == 0:
        return np.array([]), np.array([])
    X = torch.stack([s[0] for s in dataset]).numpy()
    y = torch.stack([s[1] for s in dataset]).numpy()
    return X, y


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a PyTorch model with early stopping and LR scheduling.

    The weights from the BEST validation-loss epoch are restored before
    returning: without restoration, early stopping reports metrics from
    up to ``patience_early_stopping`` stale epochs past the optimum.

    Args:
        model: The neural network module.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training config dict with keys: learning_rate, max_epochs,
            patience_early_stopping, patience_lr_reduce, lr_reduce_factor,
            and optionally fixed_epochs (train exactly this many epochs
            with no early stopping — used for final refits).
        device: Compute device.
        seed: Random seed.

    Returns:
        Dict with keys: train_loss (list per epoch), val_loss (list per
        epoch), best_epoch (int), best_val_loss (float).
    """
    import copy

    set_seed(seed)
    model = model.to(device)

    lr = config.get("learning_rate", 0.001)
    max_epochs = config.get("max_epochs", 100)
    patience_es = config.get("patience_early_stopping", 10)
    patience_lr = config.get("patience_lr_reduce", 5)
    lr_factor = config.get("lr_reduce_factor", 0.5)
    fixed_epochs = config.get("fixed_epochs")

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = (
        ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=patience_lr)
        if fixed_epochs is None
        else None
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    epochs_no_improve = 0

    total_epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    for epoch in range(total_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch).squeeze(-1)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch).squeeze(-1)
                loss = criterion(output, y_batch)
                val_losses.append(loss.item())

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)

        improved = mean_val < best_val_loss
        if improved:
            best_val_loss = mean_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step(mean_val)

        if fixed_epochs is None and epochs_no_improve >= patience_es:
            logger.info("Early stopping at epoch %d (best epoch %d)", epoch + 1, best_epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return true vs. predicted arrays.

    Args:
        model: Trained model.
        loader: Data loader.
        device: Compute device.

    Returns:
        Tuple of (y_true, y_pred) as numpy arrays.
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        output = model(X_batch).squeeze(-1)
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(output.cpu().numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return y_true, y_pred


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save model state dict to disk.

    Args:
        model: PyTorch model.
        path: File path for the checkpoint.
    """
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load model state dict from disk.

    Args:
        model: Uninitialized PyTorch model.
        path: File path of the checkpoint.
        device: Device to map tensors to.

    Returns:
        Model with loaded weights.
    """
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model
