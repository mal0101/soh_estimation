"""Model comparison tables and ranking.

Aggregates metrics across all models into a single comparison table
and provides rank-based analysis.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_comparison_table(
    results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Build a model × metric comparison DataFrame.

    Args:
        results: Dict mapping model names to aggregated metric dicts.
            Each value should contain keys like 'rmse_mean', 'rmse_std', etc.
        metrics: List of metric base names to include. Defaults to
            ['rmse', 'mae', 'maxae', 'r2'].

    Returns:
        DataFrame with models as rows and metrics as columns.
    """
    if metrics is None:
        metrics = ["rmse", "mae", "maxae", "r2"]

    rows = []
    for model_name, model_metrics in results.items():
        row: dict[str, Any] = {"Model": model_name}
        for m in metrics:
            mean_key = f"{m}_mean"
            std_key = f"{m}_std"
            if mean_key in model_metrics:
                mean_val = model_metrics[mean_key]
                std_val = model_metrics.get(std_key, 0)
                row[f"{m.upper()}"] = f"{mean_val:.6f} ± {std_val:.6f}"
                row[f"{m.upper()}_raw"] = mean_val
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def rank_models(
    results: dict[str, dict[str, float]],
    primary_metric: str = "rmse_mean",
) -> pd.DataFrame:
    """Rank models by a primary metric (ascending = better).

    Args:
        results: Dict mapping model names to aggregated metric dicts.
        primary_metric: Metric key to rank by.

    Returns:
        DataFrame with Rank, Model, and the primary metric value.
    """
    records = []
    for model_name, metrics in results.items():
        if primary_metric in metrics:
            records.append(
                {
                    "Model": model_name,
                    primary_metric: metrics[primary_metric],
                }
            )

    df = pd.DataFrame(records).sort_values(primary_metric, kind="mergesort").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df
