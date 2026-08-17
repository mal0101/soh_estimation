"""Cycle validation and filtering for battery cycling data.

For NASA PCoE data, cycles are already pre-segmented by the data loader.
This module validates cycle completeness and filters partial cycles.
For future datasets (e.g., CALCE), it can detect cycle boundaries from
the current signal.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def validate_cycles(
    cell_data: dict[str, Any],
    q_initial: float,
    min_discharge_fraction: float = 0.90,
    early_cycle_window: int = 20,
) -> dict[str, Any]:
    """Filter out incomplete discharge cycles from loaded cell data.

    The capacity threshold (min_discharge_fraction * q_initial) is only
    applied to early cycles (cycle_number <= early_cycle_window) where
    capacity should be near rated. Later cycles are naturally degraded
    and are always kept as long as they have valid capacity data.

    Args:
        cell_data: Dictionary returned by load_nasa_cell.
        q_initial: Reference capacity (average discharge capacity over
            cycles 3-10) used to determine the minimum acceptable capacity.
        min_discharge_fraction: Minimum fraction of q_initial required
            for an early discharge cycle to be considered complete.
            Default 0.90.
        early_cycle_window: Only apply the capacity threshold to cycles
            with cycle_number <= this value. Default 20.

    Returns:
        Copy of cell_data with partial discharge cycles removed.
    """
    cell_id = cell_data.get("cell_id", "unknown")
    min_capacity = min_discharge_fraction * q_initial
    original_count = len(cell_data["cycles"])

    filtered_cycles = []
    n_discarded = 0

    for cycle in cell_data["cycles"]:
        if cycle["type"] != "discharge":
            filtered_cycles.append(cycle)
            continue

        if cycle["capacity"] is None:
            filtered_cycles.append(cycle)
            continue

        is_early = cycle["cycle_number"] <= early_cycle_window
        if is_early and cycle["capacity"] < min_capacity:
            n_discarded += 1
            logger.info(
                "%s: discarding early discharge cycle %d (capacity=%.4f < %.4f)",
                cell_id,
                cycle["cycle_number"],
                cycle["capacity"],
                min_capacity,
            )
        else:
            filtered_cycles.append(cycle)

    result = dict(cell_data)
    result["cycles"] = filtered_cycles

    logger.info(
        "%s: kept %d/%d cycles (discarded %d partial discharge cycles)",
        cell_id,
        len(filtered_cycles),
        original_count,
        n_discarded,
    )

    return result


def segment_from_current(
    current: np.ndarray,
    time: np.ndarray,
    rest_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Detect charge/discharge boundaries from a raw current signal.

    Segments a continuous current time-series into charge, discharge,
    and rest periods based on current sign and magnitude. This is used
    for datasets where cycle boundaries are not provided (e.g., CALCE).

    Convention: positive current = charge, negative current = discharge.
    Periods where |current| < rest_threshold are classified as rest.

    Args:
        current: Raw current array in Amps.
        time: Time array in seconds, same length as current.
        rest_threshold: Current magnitude below which the cell is
            considered at rest. Default 0.05 A.

    Returns:
        List of dicts with keys: start_idx, end_idx, cycle_type
        ("charge", "discharge", or "rest"), start_time, end_time.
    """
    if len(current) != len(time):
        raise ValueError("current and time must have the same length")

    segments: list[dict[str, Any]] = []
    if len(current) < 2:
        return segments

    def classify(val: float) -> str:
        if abs(val) < rest_threshold:
            return "rest"
        return "charge" if val > 0 else "discharge"

    current_type = classify(current[0])
    start_idx = 0

    for i in range(1, len(current)):
        seg_type = classify(current[i])
        if seg_type != current_type:
            segments.append(
                {
                    "start_idx": start_idx,
                    "end_idx": i - 1,
                    "cycle_type": current_type,
                    "start_time": float(time[start_idx]),
                    "end_time": float(time[i - 1]),
                }
            )
            start_idx = i
            current_type = seg_type

    segments.append(
        {
            "start_idx": start_idx,
            "end_idx": len(current) - 1,
            "cycle_type": current_type,
            "start_time": float(time[start_idx]),
            "end_time": float(time[-1]),
        }
    )

    return segments
