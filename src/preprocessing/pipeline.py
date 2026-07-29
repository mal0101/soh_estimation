"""Preprocessing pipeline orchestrator.

Chains the full preprocessing pipeline: load -> validate -> compute
capacity axis -> filter -> resample -> compute SOH. Produces processed
cell data ready for feature engineering.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

from src.preprocessing.capacity import compute_cumulative_capacity
from src.preprocessing.data_loader import load_all_nasa_cells
from src.preprocessing.filtering import savgol_filter_voltage
from src.preprocessing.resampling import resample_to_uniform_grid
from src.preprocessing.segmentation import validate_cycles
from src.preprocessing.soh import compute_q_initial, compute_soh_for_all_cells

logger = logging.getLogger(__name__)


def preprocess_cycle(
    cycle: dict[str, Any],
    window_length: int = 51,
    polyorder: int = 3,
    n_points: int = 1000,
) -> dict[str, Any]:
    """Apply filtering and resampling to a single cycle.

    For charge and discharge cycles: computes cumulative capacity,
    applies Savitzky-Golay filter to voltage, then resamples onto
    a uniform capacity grid. Impedance cycles are passed through
    unchanged.

    Args:
        cycle: Cycle dictionary from load_nasa_cell.
        window_length: Savitzky-Golay window length.
        polyorder: Savitzky-Golay polynomial order.
        n_points: Number of points in the resampled grid.

    Returns:
        Copy of cycle dict with added fields: capacity_grid,
        voltage_filtered, voltage_resampled.
    """
    result = dict(cycle)

    if cycle["type"] not in ("charge", "discharge"):
        return result

    voltage = cycle["voltage"]
    current = cycle["current"]
    time_arr = cycle["time"]

    capacity = compute_cumulative_capacity(current, time_arr)

    voltage_filtered = savgol_filter_voltage(
        voltage, window_length=window_length, polyorder=polyorder
    )

    cap_grid, voltage_resampled = resample_to_uniform_grid(
        voltage_filtered, capacity, n_points=n_points
    )

    result["capacity_grid"] = cap_grid
    result["voltage_filtered"] = voltage_filtered
    result["voltage_resampled"] = voltage_resampled
    result["cumulative_capacity"] = capacity

    return result


def preprocess_cell(
    cell_data: dict[str, Any],
    q_initial: float,
    window_length: int = 51,
    polyorder: int = 3,
    n_points: int = 1000,
    min_discharge_fraction: float = 0.90,
) -> dict[str, Any]:
    """Apply the full preprocessing pipeline to a single cell.

    Pipeline: validate cycles -> preprocess each cycle.

    Args:
        cell_data: Raw cell dictionary from load_nasa_cell.
        q_initial: Reference capacity for partial cycle filtering.
        window_length: Savitzky-Golay window length.
        polyorder: Savitzky-Golay polynomial order.
        n_points: Number of points in the resampled grid.
        min_discharge_fraction: Minimum fraction of q_initial for
            early discharge cycles to be considered complete.

    Returns:
        Cell dict with preprocessed cycles (added: capacity_grid,
        voltage_filtered, voltage_resampled, cumulative_capacity).
    """
    filtered = validate_cycles(
        cell_data, q_initial, min_discharge_fraction=min_discharge_fraction
    )

    preprocessed_cycles = []
    for cycle in filtered["cycles"]:
        processed = preprocess_cycle(
            cycle,
            window_length=window_length,
            polyorder=polyorder,
            n_points=n_points,
        )
        preprocessed_cycles.append(processed)

    result = dict(filtered)
    result["cycles"] = preprocessed_cycles
    return result


def run_pipeline(config_path: str = "config/default.yaml") -> None:
    """Run the full preprocessing pipeline from configuration.

    Loads all cells, preprocesses them, computes SOH labels, and saves
    the processed data and labels to disk.
    """
    from src.utils.config import Config

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

    config = Config.from_yaml(config_path)
    raw_dir = config.get("data.raw_dir", "data/raw")
    processed_dir = Path(config.get("data.processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    nasa_pcoe_dir = Path(raw_dir) / "nasa_pcoe"
    window_length = config.get("preprocessing.savgol.window_length", 51)
    polyorder = config.get("preprocessing.savgol.polynomial_order", 3)
    n_points = config.get("preprocessing.resampling.n_points", 1000)
    min_frac = config.get("preprocessing.cycle_segmentation.min_discharge_fraction", 0.90)

    logger.info("Loading NASA PCoE cells from %s", nasa_pcoe_dir)
    all_cells = load_all_nasa_cells(nasa_pcoe_dir)

    q_initial_cycles = tuple(config.get("preprocessing.soh.q_initial_cycles", [3, 10]))

    processed_cells = {}
    for cell_id, cell_data in all_cells.items():
        q_initial = compute_q_initial(cell_data, q_initial_cycles)
        processed = preprocess_cell(
            cell_data,
            q_initial,
            window_length=window_length,
            polyorder=polyorder,
            n_points=n_points,
            min_discharge_fraction=min_frac,
        )
        processed_cells[cell_id] = processed

        n_discharge = sum(1 for c in processed["cycles"] if c["type"] == "discharge")
        logger.info("  %s: %d discharge cycles preprocessed", cell_id, n_discharge)

    pkl_path = processed_dir / "processed_cells.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(processed_cells, f)
    logger.info("Saved processed cells to %s", pkl_path)

    soh_df = compute_soh_for_all_cells(all_cells, q_initial_cycles)
    soh_path = processed_dir / "soh_labels.parquet"
    soh_df.to_parquet(soh_path, index=False)
    logger.info("Saved SOH labels to %s (%d rows)", soh_path, len(soh_df))

    _log_pipeline_summary(processed_cells)


def _log_pipeline_summary(processed_cells: dict[str, dict]) -> None:
    """Log a summary table of the preprocessing results."""
    logger.info("=" * 70)
    logger.info("Preprocessing Summary")
    logger.info("=" * 70)
    for cell_id, cell_data in processed_cells.items():
        charge = sum(1 for c in cell_data["cycles"] if c["type"] == "charge")
        discharge = sum(1 for c in cell_data["cycles"] if c["type"] == "discharge")
        impedance = sum(1 for c in cell_data["cycles"] if c["type"] == "impedance")
        has_resampled = any(
            "voltage_resampled" in c for c in cell_data["cycles"]
        )
        logger.info(
            "  %s: %d charge, %d discharge, %d impedance (resampled=%s)",
            cell_id, charge, discharge, impedance, has_resampled,
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    run_pipeline(args.config)
