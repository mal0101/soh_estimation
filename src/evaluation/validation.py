"""Cross-validation strategies for SOH estimation benchmark.

Provides cell-based Leave-One-Cell-Out Cross-Validation (LOOCV), sequence
windowing for temporal models, and per-fold feature scaling.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def cell_fold_splits(feature_df: pd.DataFrame) -> list[dict]:
    """Compute cell-based LOOCV index splits WITHOUT committing features.

    Feature-column selection happens per fold during training (fitted on
    training rows only), so the fold construction here deliberately does
    NOT touch any feature values — it only partitions row indices by
    cell. This keeps the split definition independent of whichever
    feature subset each fold later selects.

    Args:
        feature_df: Feature matrix with a 'cell_id' column.

    Returns:
        List of fold dicts with keys:
            - 'fold': int fold index (0-based)
            - 'test_cell': str cell ID held out
            - 'train_indices': np.ndarray of row indices for training
            - 'test_indices': np.ndarray of row indices for testing
            - 'train_cells': list of training cell IDs (sorted)
    """
    cell_ids = sorted(feature_df["cell_id"].unique())
    folds = []

    for fold_idx, test_cell in enumerate(cell_ids):
        test_mask = feature_df["cell_id"] == test_cell
        train_mask = ~test_mask
        folds.append(
            {
                "fold": fold_idx,
                "test_cell": test_cell,
                "train_indices": feature_df.index[train_mask].values,
                "test_indices": feature_df.index[test_mask].values,
                "train_cells": [c for c in cell_ids if c != test_cell],
            }
        )
        logger.info(
            "Fold %d: test_cell=%s, train_rows=%d, test_rows=%d",
            fold_idx,
            test_cell,
            int(train_mask.sum()),
            int(test_mask.sum()),
        )

    logger.info("Created %d LOOCV folds", len(folds))
    return folds


def materialize_fold(
    feature_df: pd.DataFrame,
    fold: dict,
    feature_cols: list[str],
) -> dict:
    """Materialize one fold's arrays for a specific feature-column subset.

    Rows containing NaN in ANY selected feature or in 'soh' are dropped
    separately per split (the NaN profile may differ between train and
    test cells for some features).

    Args:
        feature_df: Full candidate feature matrix.
        fold: Fold dict from cell_fold_splits.
        feature_cols: Selected feature columns for this fold.

    Returns:
        Dict with X_train/y_train/X_test/y_test plus NaN-filtered
        index arrays.
    """
    cols = list(feature_cols) + ["soh"]
    train_df = feature_df.loc[fold["train_indices"]].dropna(subset=cols)
    test_df = feature_df.loc[fold["test_indices"]].dropna(subset=cols)

    return {
        "fold": fold["fold"],
        "test_cell": fold["test_cell"],
        "feature_cols": list(feature_cols),
        "train_indices": train_df.index.values,
        "test_indices": test_df.index.values,
        "X_train": train_df[feature_cols].values,
        "y_train": train_df["soh"].values,
        "X_test": test_df[feature_cols].values,
        "y_test": test_df["soh"].values,
        "train_df": train_df,
        "test_df": test_df,
    }


def cell_based_loocv(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
) -> list[dict]:
    """Cell-based Leave-One-Cell-Out Cross-Validation.

    Each fold holds out all cycles from one cell for testing and trains
    on all remaining cells.

    Args:
        feature_df: Feature matrix with columns including 'cell_id' and 'soh'.
        feature_cols: List of feature column names to use for training.

    Returns:
        List of fold dicts, each with keys:
            - 'fold': int fold index (0-based)
            - 'test_cell': str cell ID held out
            - 'train_indices': np.ndarray of row indices for training
            - 'test_indices': np.ndarray of row indices for testing
            - 'X_train': np.ndarray of training features
            - 'y_train': np.ndarray of training SOH
            - 'X_test': np.ndarray of test features
            - 'y_test': np.ndarray of test SOH
    """
    cell_ids = feature_df["cell_id"].unique()
    folds = []

    for fold_idx, test_cell in enumerate(sorted(cell_ids)):
        test_mask = feature_df["cell_id"] == test_cell
        train_mask = ~test_mask

        train_df = feature_df.loc[train_mask].dropna(subset=feature_cols + ["soh"])
        test_df = feature_df.loc[test_mask].dropna(subset=feature_cols + ["soh"])

        X_train = train_df[feature_cols].values
        y_train = train_df["soh"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["soh"].values

        folds.append(
            {
                "fold": fold_idx,
                "test_cell": test_cell,
                "train_indices": train_df.index.values,
                "test_indices": test_df.index.values,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }
        )

        logger.info(
            "Fold %d: test_cell=%s, train=%d, test=%d",
            fold_idx,
            test_cell,
            len(X_train),
            len(X_test),
        )

    logger.info("Created %d LOOCV folds", len(folds))
    return folds


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on training data and transform both splits.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def save_fold_indices(
    folds: list[dict],
    output_dir: str | Path = "experiments",
    dataset: str = "all",
) -> Path:
    """Save fold indices to JSON for reproducibility.

    Args:
        folds: List of fold dicts from cell_based_loocv or cell_fold_splits.
        output_dir: Directory to write fold_indices{suffix}.json.
        dataset: Dataset name ('nasa', 'calce', or 'all'); every file
            carries an explicit suffix so runs on different datasets
            never overwrite one another.

    Returns:
        Path to the saved JSON file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fold_data = []
    for f in folds:
        entry = {
            "fold": f["fold"],
            "test_cell": f["test_cell"],
            "train_indices": (
                f["train_indices"].tolist() if hasattr(f["train_indices"], "tolist") else list(f["train_indices"])
            ),
            "test_indices": (
                f["test_indices"].tolist() if hasattr(f["test_indices"], "tolist") else list(f["test_indices"])
            ),
        }
        if "train_cells" in f:
            entry["train_cells"] = list(f["train_cells"])
        fold_data.append(entry)

    suffix = f"_{dataset}"
    filepath = out / f"fold_indices{suffix}.json"
    with open(filepath, "w") as fh:
        json.dump({"dataset": dataset, "folds": fold_data}, fh, indent=2)

    logger.info("Saved fold indices to %s", filepath)
    return filepath
