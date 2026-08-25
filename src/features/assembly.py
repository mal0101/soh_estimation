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


_DATASET_DEFAULTS = {
    "nasa_pcoe": {"rated_capacity": 2.0, "cutoff_voltage": 2.5},
    "calce": {"rated_capacity": 1.1, "cutoff_voltage": 2.7},
}


def _dataset_default(cell_data: dict[str, Any], key: str) -> float:
    """Return a dataset-appropriate default for missing cell metadata."""
    dataset = cell_data.get("dataset", "unknown")
    return _DATASET_DEFAULTS.get(dataset, {}).get(key, 2.0 if key == "rated_capacity" else 2.5)


def _build_preceding_charge_map(ordered_cycles: list[dict[str, Any]]) -> dict[int, float]:
    """Map each discharge cycle_number to its nearest PRECEDING charge capacity.

    Coulombic efficiency pairs a discharge with the charge that directly
    precedes it in time. Numbering conventions differ across datasets
    (NASA numbers all cycle types globally; CALCE numbers the charge
    before discharge k as k-1), so pairing is done by temporal order in
    the cycle list rather than by cycle-number arithmetic.

    Args:
        ordered_cycles: Cell cycles sorted by cycle_number.

    Returns:
        Dict: discharge cycle_number -> preceding charge capacity (Ah),
        NaN when no charge cycle precedes the discharge.
    """
    charge_caps: dict[int, float] = {}
    for pos, cyc in enumerate(ordered_cycles):
        if cyc["type"] != "charge":
            continue
        try:
            charge_caps[pos] = _compute_charge_capacity(cyc)
        except (ValueError, KeyError):
            charge_caps[pos] = np.nan

    mapping: dict[int, float] = {}
    last_charge_cap = np.nan
    for pos, cyc in enumerate(ordered_cycles):
        if cyc["type"] == "charge":
            if pos in charge_caps and np.isfinite(charge_caps[pos]):
                last_charge_cap = charge_caps[pos]
        elif cyc["type"] == "discharge":
            mapping[cyc["cycle_number"]] = last_charge_cap
    return mapping


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
        min_peak_prominence: Minimum prominence fraction for ICA peak detection.

    Returns:
        DataFrame with one row per (cell, discharge_cycle) and all features.
    """
    records = []

    for cell_id, cell_data in processed_cells.items():
        discharge_cycles = [c for c in cell_data["cycles"] if c["type"] == "discharge"]
        impedance_cycles = [c for c in cell_data["cycles"] if c["type"] == "impedance"]

        ordered_cycles = sorted(cell_data["cycles"], key=lambda c: c["cycle_number"])
        preceding_charge_cap = _build_preceding_charge_map(ordered_cycles)

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
                "rated_capacity": cell_data.get(
                    "rated_capacity", _dataset_default(cell_data, "rated_capacity")
                ),
                "cutoff_voltage": cell_data.get(
                    "cutoff_voltage", _dataset_default(cell_data, "cutoff_voltage")
                ),
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

            # EIS reference: only PAST impedance cycles (a future measurement
            # would not exist at prediction time).
            nearest_eis = None
            past_eis = [ic for ic in impedance_cycles if ic["cycle_number"] <= cn]
            if past_eis:
                nearest = min(past_eis, key=lambda ic: cn - ic["cycle_number"])
                nearest_eis = nearest.get("eis")
            eis_feats = extract_eis_features(nearest_eis)
            record.update(eis_feats)

            energy = compute_discharge_energy(dc["voltage"], dc["current"], dc["time"])
            record["discharge_energy"] = energy

            v_mean = compute_mean_discharge_voltage(dc["voltage"], dc["current"], dc["time"])
            record["mean_discharge_voltage"] = v_mean

            q_discharge = dc["capacity"] if dc["capacity"] is not None else np.nan
            q_charge = preceding_charge_cap.get(cn, np.nan)
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


_METADATA_COLS = [
    "cell_id",
    "dataset",
    "cycle_number",
    "soh",
    "rated_capacity",
    "cutoff_voltage",
    "ambient_temperature",
    "discharge_capacity",
]


def fit_feature_selection(
    train_df: pd.DataFrame,
    correlation_threshold: float = 0.95,
    top_k: int = 20,
) -> list[str]:
    """Fit the feature-selection pipeline on TRAINING data only.

    Steps (all computed exclusively on ``train_df``):
        1. Identify numeric candidate feature columns (exclude metadata).
        2. Drop columns with >=30% NaN or near-zero variance.
        3. Correlation filter: remove one feature from each pair with
           |r| > threshold.
        4. RandomForest importance ranking against SOH; keep top_k.

    Calling this per LOOCV fold (on the training cells only) keeps
    feature selection inside the cross-validation loop; fitting it once
    on the full dataset would leak test-cell label information into the
    chosen feature set and inflate reported metrics.

    Args:
        train_df: Feature matrix restricted to training cells.
        correlation_threshold: Maximum allowed pairwise correlation.
        top_k: Number of top features to keep.

    Returns:
        Ordered list of selected feature column names.
    """
    feature_cols = [c for c in train_df.columns if c not in _METADATA_COLS]

    numeric_df = train_df[feature_cols].select_dtypes(include=[np.number])
    feature_cols = list(numeric_df.columns)

    nan_fractions = numeric_df.isna().mean()
    valid_cols = [c for c in feature_cols if nan_fractions[c] < 0.30]

    variances = numeric_df[valid_cols].var()
    valid_cols = [c for c in valid_cols if variances[c] > 1e-10]

    if len(valid_cols) < 2:
        logger.warning("Fewer than 2 valid features after screening")
        return valid_cols

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

    complete_mask = train_df[remaining_cols].notna().all(axis=1)
    complete_df = train_df.loc[complete_mask]

    if len(complete_df) < 10 or complete_df["soh"].nunique() < 2:
        logger.warning("Too few complete rows for RF importance, keeping correlation-filtered set")
        return remaining_cols[:top_k]

    X = complete_df[remaining_cols].values
    y = complete_df["soh"].values

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=remaining_cols)
    importances = importances.sort_values(ascending=False)

    selected_cols = list(importances.index[:top_k])
    logger.info("RF importance: kept top %d features", len(selected_cols))
    logger.info("Top 5 features: %s", list(importances.index[:5]))

    return selected_cols


def select_features(
    feature_df: pd.DataFrame,
    correlation_threshold: float = 0.95,
    top_k: int = 20,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply feature selection over a full feature matrix.

    NOTE: this convenience wrapper fits selection on every row of
    ``feature_df`` and is appropriate only for exploratory analysis.
    Model training must call :func:`fit_feature_selection` inside each
    cross-validation fold instead.

    Args:
        feature_df: Full feature matrix from build_feature_matrix.
        correlation_threshold: Maximum allowed pairwise correlation.
        top_k: Number of top features to keep.

    Returns:
        Tuple of (selected_df, feature_names) where selected_df contains
        only the selected feature columns plus metadata columns.
    """
    selected_cols = fit_feature_selection(
        feature_df, correlation_threshold=correlation_threshold, top_k=top_k
    )
    return feature_df, selected_cols


def save_feature_matrix(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str = "data/features",
    dataset: str = "all",
) -> Path:
    """Save the feature matrix (metadata + given feature columns) to parquet.

    The builder passes ALL candidate features here; supervised selection
    happens per LOOCV fold during training.

    Args:
        feature_df: Full feature matrix.
        feature_cols: Feature columns to store alongside metadata.
        output_dir: Output directory path.
        dataset: Dataset name for filename suffix ('nasa', 'calce', or 'all').

    Returns:
        Path to the saved parquet file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata_cols = ["cell_id", "dataset", "cycle_number", "soh"]
    save_cols = metadata_cols + [c for c in feature_cols if c not in metadata_cols]
    save_df = feature_df[save_cols].copy()

    # Every dataset gets an explicit suffix so files can never silently
    # overwrite each other across runs.
    filepath = out_path / f"feature_matrix_{dataset}.parquet"
    save_df.to_parquet(filepath, index=False)
    logger.info("Saved feature matrix: %s (%s)", filepath, save_df.shape)

    return filepath
