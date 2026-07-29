"""Visualization functions for model evaluation and comparison.

Produces publication-ready matplotlib figures for the benchmark study.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_soh_trajectory(
    soh_true: dict[str, np.ndarray],
    soh_pred: dict[str, np.ndarray],
    cycle_numbers: dict[str, np.ndarray],
    title: str = "SOH Trajectory",
) -> plt.Figure:
    """Plot true vs. predicted SOH trajectories for each cell.

    Args:
        soh_true: Dict mapping cell_id to true SOH array.
        soh_pred: Dict mapping cell_id to predicted SOH array.
        cycle_numbers: Dict mapping cell_id to cycle number array.
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    n_cells = len(soh_true)
    fig, axes = plt.subplots(1, n_cells, figsize=(5 * n_cells, 4), sharey=True)
    if n_cells == 1:
        axes = [axes]

    for ax, cell_id in zip(axes, sorted(soh_true.keys()), strict=False):
        ax.plot(cycle_numbers[cell_id], soh_true[cell_id], "b-", label="True", alpha=0.8)
        ax.plot(cycle_numbers[cell_id], soh_pred[cell_id], "r--", label="Predicted", alpha=0.8)
        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("SOH")
        ax.set_title(cell_id)
        ax.legend(fontsize=8)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    title: str | None = None,
) -> plt.Figure:
    """Plot predicted vs. true SOH scatter with identity line.

    Args:
        y_true: Ground-truth SOH values.
        y_pred: Predicted SOH values.
        model_name: Name of the model for the title.
        title: Optional custom title.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    lims = [min(y_true.min(), y_pred.min()) - 0.01, max(y_true.max(), y_pred.max()) + 0.01]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True SOH")
    ax.set_ylabel("Predicted SOH")
    ax.set_title(title or f"{model_name}: Predicted vs. True SOH")
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


def plot_comparison_bars(
    results: dict[str, dict[str, float]],
    metric: str = "rmse",
    title: str | None = None,
) -> plt.Figure:
    """Plot a grouped bar chart comparing models on a metric.

    Args:
        results: Dict mapping model names to metric dicts.
        metric: Base metric name (e.g. 'rmse', 'r2').
        title: Optional figure title.

    Returns:
        Matplotlib Figure object.
    """
    models = list(results.keys())
    means = [results[m].get(f"{metric}_mean", 0) for m in models]
    stds = [results[m].get(f"{metric}_std", 0) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], rotation=45)
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Model Comparison: {metric.upper()}")
    plt.tight_layout()
    return fig


def plot_fold_rmse(
    fold_results: dict[str, list[float]],
    title: str = "Fold-wise RMSE",
) -> plt.Figure:
    """Plot RMSE per fold for each model.

    Args:
        fold_results: Dict mapping model names to lists of per-fold RMSE values.
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    records = []
    for model, rmses in fold_results.items():
        for i, rmse in enumerate(rmses):
            records.append({"Model": model.upper(), "Fold": i, "RMSE": rmse})
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="Fold", y="RMSE", hue="Model", ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig
