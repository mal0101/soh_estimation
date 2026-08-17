"""Feature matrix assembly and selection.

Combines all per-cycle features into a unified DataFrame, applies
correlation-based filtering and Random Forest importance ranking,
and saves the result as a parquet file.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.features.energy import (
    compute_coulombic_efficiency,
    compute_discharge_energy,
    compute_mean_discharge_voltage,
)
from src.features.ica import extract_ica_features
from src.features.internal_resistance import estimate_ir_from_discharge, extract_eis_features
from src.features.temperature import compute_temperature_features
from src.features.trend import compute_capacity_fade_rate

logger = logging.getLogger(__name__)


def _compute_charge_capacity(charge_cycle: dict[str, Any]) -> float:
    """Compute charge capacity from a charge cycle via current integration."""
    from src.preprocessing.capacity import compute_cumulative_capacity

    cap = compute_cumulative_capacity(charge_cycle["current"], charge_cycle["time"])
    return float(cap[-1]) if len(cap) > 0 else np.nan


def build_feature_matrix(
    processed_cells: dict[str, dict],
    soh_df: pd.DataFrame,
    min_peak_prominence: float = 0.01,
) -> pd.DataFrame:
    """Assemble the full feature matrix from preprocessed cell data.

    For each discharge cycle, computes: ICA features, IR estimation,
    EIS features, energy features, temperature features, and cycle
    metadata (cell_id, cycle_number, SOH).

    Args:
        processed_cells: Dictionary from the preprocessing pipeline
            (output of run_pipeline, loaded from processed_cells.pkl).
        soh_df: SOH labels DataFrame with columns cell_id, cycle_number, soh.
        min_peak_prominence: Minimum prominence for ICA peak detection.

    Returns:
        DataFrame with one row per (cell, discharge_cycle) and all features.
    """
    records = []

    for cell_id, cell_data in processed_cells.items():
        discharge_cycles = [c for c in cell_data["cycles"] if c["type"] == "discharge"]
        charge_cycles = [c for c in cell_data["cycles"] if c["type"] == "charge"]
        impedance_cycles = [c for c in cell_data["cycles"] if c["type"] == "impedance"]

        charge_caps = {}
        for cc in charge_cycles:
            charge_caps[cc["cycle_number"]] = _compute_charge_capacity(cc)

        soh_cell = soh_df[soh_df["cell_id"] == cell_id].set_index("cycle_number")

        soh_values = []
        for dc in discharge_cycles:
            cn = dc["cycle_number"]
            if cn in soh_cell.index:
                soh_values.append(soh_cell.loc[cn, "soh"])
            else:
                soh_values.append(np.nan)
        soh_arr = np.array(soh_values, dtype=np.float64)
        fade_rates = compute_capacity_fade_rate(soh_arr, window=10)

        for i, dc in enumerate(discharge_cycles):
            cn = dc["cycle_number"]
            record: dict[str, Any] = {
                "cell_id": cell_id,
                "dataset": cell_data.get("dataset", "unknown"),
                "cycle_number": cn,
                "soh": float(soh_arr[i]) if not np.isnan(soh_arr[i]) else np.nan,
                "rated_capacity": cell_data.get("rated_capacity", 2.0),
                "cutoff_voltage": cell_data.get("cutoff_voltage", 2.5),
                "ambient_temperature": dc.get("ambient_temperature", np.nan),
            }

            ica_feats = extract_ica_features(
                dc["voltage_resampled"],
                dc["capacity_grid"],
                min_peak_prominence=min_peak_prominence,
            )
            record.update(ica_feats)

            ir = estimate_ir_from_discharge(dc["voltage"], dc["current"], dc["time"])
            record["internal_resistance"] = ir

            nearest_eis = None
            if impedance_cycles:
                min_dist = float("inf")
                for ic in impedance_cycles:
                    dist = abs(ic["cycle_number"] - cn)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_eis = ic.get("eis")
            eis_feats = extract_eis_features(nearest_eis)
            record.update(eis_feats)

            energy = compute_discharge_energy(dc["voltage"], dc["current"], dc["time"])
            record["discharge_energy"] = energy

            v_mean = compute_mean_discharge_voltage(dc["voltage"], dc["current"], dc["time"])
            record["mean_discharge_voltage"] = v_mean

            q_discharge = dc["capacity"] if dc["capacity"] is not None else np.nan
            following_charge_cn = cn + 1
            q_charge = charge_caps.get(following_charge_cn, np.nan)
            ce = compute_coulombic_efficiency(q_discharge, q_charge)
            record["coulombic_efficiency"] = ce

            temp_feats = compute_temperature_features(dc["temperature"])
            record.update(temp_feats)

            record["capacity_fade_rate"] = float(fade_rates[i])
            record["discharge_capacity"] = q_discharge

            records.append(record)

    df = pd.DataFrame(records)
    logger.info("Feature matrix assembled: %s", df.shape)
    return df


def select_features(
    feature_df: pd.DataFrame,
    correlation_threshold: float = 0.95,
    top_k: int = 20,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply feature selection: correlation filter + RF importance ranking.

    Steps:
        1. Identify numeric feature columns (exclude metadata).
        2. Compute pairwise Pearson correlation matrix.
        3. Remove one feature from each pair with |r| > threshold.
        4. Train a Random Forest on remaining features, rank by importance.
        5. Keep the top_k features.

    Args:
        feature_df: Full feature matrix from build_feature_matrix.
        correlation_threshold: Maximum allowed pairwise correlation.
        top_k: Number of top features to keep.

    Returns:
        Tuple of (selected_df, feature_names) where selected_df contains
        only the selected feature columns plus metadata columns.
    """
    metadata_cols = [
        "cell_id",
        "dataset",
        "cycle_number",
        "soh",
        "rated_capacity",
        "cutoff_voltage",
        "ambient_temperature",
        "discharge_capacity",
    ]
    feature_cols = [c for c in feature_df.columns if c not in metadata_cols]

    numeric_df = feature_df[feature_cols].select_dtypes(include=[np.number])
    feature_cols = list(numeric_df.columns)

    nan_fractions = numeric_df.isna().mean()
    valid_cols = [c for c in feature_cols if nan_fractions[c] < 0.30]

    variances = numeric_df[valid_cols].var()
    valid_cols = [c for c in valid_cols if variances[c] > 1e-10]

    if len(valid_cols) < 2:
        logger.warning("Fewer than 2 valid features, returning all")
        return feature_df, feature_cols

    corr_matrix = numeric_df[valid_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated = [
            c for c in upper.index[upper[col] > correlation_threshold] if c not in to_drop
        ]
        if correlated:
            to_drop.add(col)

    remaining_cols = [c for c in valid_cols if c not in to_drop]
    logger.info(
        "Correlation filter: %d -> %d features (dropped %d)",
        len(valid_cols),
        len(remaining_cols),
        len(to_drop),
    )

    complete_mask = feature_df[remaining_cols].notna().all(axis=1)
    complete_df = feature_df.loc[complete_mask]

    if len(complete_df) < 10:
        logger.warning("Too few complete rows for RF importance, skipping RF selection")
        selected_cols = remaining_cols[:top_k]
        return feature_df, selected_cols

    X = complete_df[remaining_cols].values
    y = complete_df["soh"].values

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=remaining_cols)
    importances = importances.sort_values(ascending=False)

    selected_cols = list(importances.index[:top_k])
    logger.info("RF importance: kept top %d features", len(selected_cols))
    logger.info("Top 5 features: %s", list(importances.index[:5]))

    return feature_df, selected_cols


def save_feature_matrix(
    feature_df: pd.DataFrame,
    selected_cols: list[str],
    output_dir: str = "data/features",
    dataset: str = "all",
) -> Path:
    """Save the feature matrix with selected features to parquet.

    Args:
        feature_df: Full feature matrix.
        selected_cols: Names of selected feature columns.
        output_dir: Output directory path.
        dataset: Dataset name for filename suffix ('nasa', 'calce', or 'all').

    Returns:
        Path to the saved parquet file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata_cols = ["cell_id", "dataset", "cycle_number", "soh"]
    save_cols = metadata_cols + [c for c in selected_cols if c not in metadata_cols]
    save_df = feature_df[save_cols].copy()

    suffix = f"_{dataset}" if dataset != "all" else ""
    filepath = out_path / f"feature_matrix{suffix}.parquet"
    save_df.to_parquet(filepath, index=False)
    logger.info("Saved feature matrix: %s (%s)", filepath, save_df.shape)

    return filepath
