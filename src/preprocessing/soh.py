"""State of Health (SOH) label computation.

Computes per-cell Q_initial and SOH curves from raw discharge capacity
measurements. SOH is defined as:

    SOH(n) = Q_discharge(n) / Q_initial

where Q_initial is the average measured discharge capacity over cycles 3-10
of each specific cell.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_q_initial(
    cell_data: dict,
    q_initial_cycles: tuple[int, int] = (3, 10),
) -> float:
    """Compute the initial reference capacity for a cell.

    Q_initial is the mean discharge capacity over cycles in the range
    [q_initial_cycles[0], q_initial_cycles[1]]. Using measured capacity
    (not manufacturer-rated) avoids systematic offsets from manufacturing
    tolerance.

    Args:
        cell_data: Dictionary returned by load_nasa_cell.
        q_initial_cycles: Tuple of (start, end) cycle numbers (inclusive)
            over which to average. Defaults to (3, 10) to skip formation
            cycles while using early-life data.

    Returns:
        Mean discharge capacity in Ah.

    Raises:
        ValueError: If fewer than 2 valid discharge capacities are found
            in the specified cycle range.
    """
    cell_id = cell_data["cell_id"]
    start, end = q_initial_cycles

    capacities = []
    for cycle in cell_data["cycles"]:
        if cycle["type"] != "discharge":
            continue
        if cycle["capacity"] is None:
            continue
        if start <= cycle["cycle_number"] <= end:
            capacities.append(cycle["capacity"])

    if len(capacities) < 2:
        raise ValueError(
            f"{cell_id}: found only {len(capacities)} valid discharge capacities "
            f"in cycles {start}-{end}. Need at least 2."
        )

    q_initial = float(np.mean(capacities))
    logger.info(
        "%s: Q_initial = %.4f Ah (averaged over %d cycles in range %d-%d)",
        cell_id,
        q_initial,
        len(capacities),
        start,
        end,
    )
    return q_initial


def compute_soh_curve(
    cell_data: dict,
    q_initial: float,
    cap: float = 1.0,
) -> list[dict]:
    """Compute SOH for every discharge cycle in a cell.

    Args:
        cell_data: Dictionary returned by load_nasa_cell.
        q_initial: Reference capacity from compute_q_initial.
        cap: Maximum SOH value. Cycles where capacity exceeds q_initial
            (common in first few cycles due to formation) are capped at
            this value. Defaults to 1.0.

    Returns:
        List of dicts with keys: cycle_number, soh, q_discharge, q_initial.
    """
    cell_id = cell_data["cell_id"]
    soh_records = []

    for cycle in cell_data["cycles"]:
        if cycle["type"] != "discharge":
            continue
        if cycle["capacity"] is None:
            continue

        soh = cycle["capacity"] / q_initial
        soh = min(soh, cap)

        soh_records.append(
            {
                "cycle_number": cycle["cycle_number"],
                "soh": soh,
                "q_discharge": cycle["capacity"],
                "q_initial": q_initial,
            }
        )

    logger.info(
        "%s: computed SOH for %d discharge cycles (range: %.4f - %.4f)",
        cell_id,
        len(soh_records),
        min(r["soh"] for r in soh_records) if soh_records else 0,
        max(r["soh"] for r in soh_records) if soh_records else 0,
    )
    return soh_records


def compute_soh_for_all_cells(
    all_cells: dict[str, dict],
    q_initial_cycles: tuple[int, int] = (3, 10),
) -> pd.DataFrame:
    """Compute SOH labels for all loaded cells and return a unified DataFrame.

    Args:
        all_cells: Dictionary mapping cell_id to loaded cell data
            (output of load_all_nasa_cells).
        q_initial_cycles: Cycle range for Q_initial computation.

    Returns:
        DataFrame with columns: cell_id, dataset, cycle_number, soh,
        q_discharge, q_initial. One row per (cell, discharge_cycle) pair.
    """
    all_records = []

    for cell_id, cell_data in all_cells.items():
        try:
            q_initial = compute_q_initial(cell_data, q_initial_cycles)
        except ValueError as e:
            logger.error("Skipping %s: %s", cell_id, e)
            continue

        soh_records = compute_soh_curve(cell_data, q_initial)
        for record in soh_records:
            record["cell_id"] = cell_id
            record["dataset"] = cell_data["dataset"]
        all_records.extend(soh_records)

    df = pd.DataFrame(all_records)
    df = df[["cell_id", "dataset", "cycle_number", "soh", "q_discharge", "q_initial"]]
    logger.info("SOH labels computed: %d total records across %d cells", len(df), len(all_cells))
    return df
