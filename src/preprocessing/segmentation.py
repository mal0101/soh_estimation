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
    interruption_fraction: float = 0.75,
    interruption_window: int = 5,
    run_drop_fraction: float = 0.70,
    run_entry_fraction: float = 0.75,
) -> dict[str, Any]:
    """Filter out incomplete discharge cycles from loaded cell data.

    Two complementary filters are applied to discharge cycles:

    1. Early-cycle rule (as before): cycles with
       ``cycle_number <= early_cycle_window`` must reach at least
       ``min_discharge_fraction * q_initial`` — formation/incomplete
       early discharges are removed.
    2. Interruption rule (ALL cycles): a discharge whose capacity falls
       below ``interruption_fraction`` times the median capacity of its
       neighbouring discharge cycles (± ``interruption_window``
       neighbours in cycle order, self excluded) is an instrumentation
       interruption/pause and is removed regardless of cell age. A
       local-median reference is used so genuine gradual degradation
       late in life is never discarded (the local median tracks the
       fade), while isolated abrupt drops are caught anywhere in the
       series.
    3. Anomalous-run rule: contiguous blocks of discharges whose
       capacity stays below ``interruption_fraction * q_initial`` are
       removed only if they later recover above that threshold AND
       their mean depth is below ``run_drop_fraction * q_initial``.
       Real end-of-life fade never recovers (kept), genuine reversible
       transients recover but stay shallow near EOL (kept), while
       storage/test pauses crash capacity far below EOL and recover
       (dropped) — e.g., the CALCE CS2 storage gaps spanning ~250
       consecutive depressed cycles that local-median logic cannot see
       because neighbours inside the block are equally depressed.

    Args:
        cell_data: Dictionary returned by load_nasa_cell / load_calce_cell.
        q_initial: Reference capacity (average discharge capacity over
            the initial cycles) used for the thresholds.
        min_discharge_fraction: Minimum fraction of q_initial required
            for an early discharge cycle to be considered complete.
            Default 0.90.
        early_cycle_window: Only apply rule 1 to cycles with
            cycle_number <= this value. Default 20.
        interruption_fraction: Fraction of the reference/local-median
            capacity below which a discharge is flagged as interrupted.
            Default 0.75.
        interruption_window: Number of neighbouring discharge cycles on
            each side used for the local median. Default 5.
        run_drop_fraction: Mean-depth threshold for dropping a recovered
            low-capacity run as an anomaly. Default 0.70.
        run_entry_fraction: Fraction of q_initial below which a cycle
            joins a candidate anomalous run. Default 0.75.

    Returns:
        Copy of cell_data with partial/interrupted discharge cycles removed.
    """
    cell_id = cell_data.get("cell_id", "unknown")
    min_capacity = min_discharge_fraction * q_initial

    discharge_indices = [
        i for i, c in enumerate(cell_data["cycles"]) if c["type"] == "discharge"
    ]
    capacities = np.array(
        [
            (
                cell_data["cycles"][i]["capacity"]
                if cell_data["cycles"][i]["capacity"] is not None
                else np.nan
            )
            for i in discharge_indices
        ],
        dtype=np.float64,
    )

    # Rule 3: recovery-guarded contiguous-run detection. A run is an
    # anomaly only if it is DEEP on average (< run_drop_fraction x
    # q_initial): storage/test pauses crash capacity far below the EOL
    # line, whereas genuine reversible-fade transients (e.g., NASA
    # B0006 rest-recovery around SOH~0.72) stay shallow.
    run_drop = np.zeros(len(discharge_indices), dtype=bool)
    finite = np.isfinite(capacities)
    # NOTE: run membership uses its own entry fraction so that tightening
    # the per-cycle interruption_fraction does not change which cycles
    # form candidate runs.
    abs_threshold = run_entry_fraction * q_initial
    run_threshold = run_drop_fraction * q_initial
    below = finite & (capacities < abs_threshold)
    run_start = None
    for j in range(len(discharge_indices) + 1):
        if j < len(discharge_indices) and below[j]:
            if run_start is None:
                run_start = j
        elif run_start is not None:
            segment = capacities[run_start:j]
            recovered = bool(np.any(finite[j:] & (capacities[j:] >= abs_threshold)))
            if recovered and float(np.mean(segment)) < run_threshold:
                run_drop[run_start:j] = True
                logger.info(
                    "%s: flagged anomalous discharge run (cycles %d-%d, "
                    "n=%d, mean=%.4f < %.2f x q_initial %.4f, later recovers)",
                    cell_id,
                    cell_data["cycles"][discharge_indices[run_start]]["cycle_number"],
                    cell_data["cycles"][discharge_indices[j - 1]]["cycle_number"],
                    j - run_start,
                    float(np.mean(segment)),
                    run_drop_fraction,
                    q_initial,
                )
            run_start = None

    # Local-median reference per discharge cycle (self excluded).
    local_median = np.full(len(discharge_indices), np.nan)
    for j in range(len(discharge_indices)):
        lo = max(0, j - interruption_window)
        hi = min(len(discharge_indices), j + interruption_window + 1)
        neighbours = np.concatenate([capacities[lo:j], capacities[j + 1 : hi]])
        neighbours = neighbours[np.isfinite(neighbours)]
        if len(neighbours) > 0:
            local_median[j] = float(np.median(neighbours))

    filtered_cycles = []
    n_discarded_early = 0
    n_discarded_interrupt = 0
    n_discarded_run = 0
    discharge_pos = {i: j for j, i in enumerate(discharge_indices)}

    for idx, cycle in enumerate(cell_data["cycles"]):
        if cycle["type"] != "discharge":
            filtered_cycles.append(cycle)
            continue

        if cycle["capacity"] is None:
            filtered_cycles.append(cycle)
            continue

        if cycle["cycle_number"] <= early_cycle_window and cycle["capacity"] < min_capacity:
            n_discarded_early += 1
            logger.info(
                "%s: discarding incomplete early discharge cycle %d "
                "(capacity=%.4f < %.4f)",
                cell_id,
                cycle["cycle_number"],
                cycle["capacity"],
                min_capacity,
            )
            continue

        j = discharge_pos[idx]

        ref = local_median[j]
        if np.isfinite(ref) and cycle["capacity"] < interruption_fraction * ref:
            n_discarded_interrupt += 1
            logger.info(
                "%s: discarding interrupted discharge cycle %d "
                "(capacity=%.4f < %.2f x local median %.4f)",
                cell_id,
                cycle["cycle_number"],
                cycle["capacity"],
                interruption_fraction,
                ref,
            )
            continue

        if run_drop[j]:
            n_discarded_run += 1
            logger.info(
                "%s: discarding anomalous-run discharge cycle %d "
                "(capacity=%.4f < %.2f x q_initial %.4f; capacity later recovers)",
                cell_id,
                cycle["cycle_number"],
                cycle["capacity"],
                interruption_fraction,
                q_initial,
            )
            continue

        filtered_cycles.append(cycle)

    result = dict(cell_data)
    result["cycles"] = filtered_cycles
    result["n_cycles_discarded"] = (
        n_discarded_early + n_discarded_interrupt + n_discarded_run
    )

    logger.info(
        "%s: kept %d/%d cycles (discarded %d early partial + %d interrupted "
        "+ %d anomalous-run discharges)",
        cell_id,
        len(filtered_cycles),
        len(cell_data["cycles"]),
        n_discarded_early,
        n_discarded_interrupt,
        n_discarded_run,
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
