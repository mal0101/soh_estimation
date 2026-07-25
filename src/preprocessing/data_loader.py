"""Data loading module for NASA PCoE and CALCE battery datasets.

Reads raw dataset files into a standardized dictionary structure that
downstream preprocessing and feature engineering modules consume.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

logger = logging.getLogger(__name__)

NASA_CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]
NASA_RATED_CAPACITY = 2.0
NASA_CUTOFF_VOLTAGES = {"B0005": 2.7, "B0006": 2.5, "B0007": 2.2, "B0018": 2.5}


def load_nasa_cell(filepath: str | Path) -> dict[str, Any]:
    """Load a single NASA PCoE battery cell from a .mat file.

    Parses the MATLAB struct hierarchy and extracts charge, discharge,
    and impedance cycles into a flat, standardized dictionary.

    Args:
        filepath: Path to the .mat file (e.g., 'data/raw/nasa_pcoe/B0005.mat').

    Returns:
        Dictionary with keys: cell_id, dataset, rated_capacity, cutoff_voltage,
        cycles (list of cycle dicts). Each cycle dict contains: cycle_number,
        type, ambient_temperature, voltage, current, temperature, time,
        capacity, eis.

    Raises:
        FileNotFoundError: If the .mat file does not exist.
        ValueError: If the file cannot be parsed or contains no cycles.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MAT file not found: {filepath}")

    cell_id = filepath.stem
    mat = scipy.io.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)

    if cell_id not in mat:
        raise ValueError(
            f"Expected key '{cell_id}' in {filepath.name}, found: {[k for k in mat if not k.startswith('__')]}"
        )

    raw_cycles = mat[cell_id].cycle
    logger.info("Loaded %s: %d raw cycles", cell_id, len(raw_cycles))

    cycles = []
    for i, raw_cycle in enumerate(raw_cycles):
        cycle = _parse_nasa_cycle(raw_cycle, cycle_number=i + 1)
        if cycle is not None:
            cycles.append(cycle)

    n_charge = sum(1 for c in cycles if c["type"] == "charge")
    n_discharge = sum(1 for c in cycles if c["type"] == "discharge")
    n_impedance = sum(1 for c in cycles if c["type"] == "impedance")
    logger.info(
        "  Parsed: %d charge, %d discharge, %d impedance",
        n_charge,
        n_discharge,
        n_impedance,
    )

    return {
        "cell_id": cell_id,
        "dataset": "nasa_pcoe",
        "rated_capacity": NASA_RATED_CAPACITY,
        "cutoff_voltage": NASA_CUTOFF_VOLTAGES.get(cell_id, 2.5),
        "cycles": cycles,
    }


def _parse_nasa_cycle(raw_cycle: Any, cycle_number: int) -> dict[str, Any] | None:
    """Parse a single NASA cycle struct into a standardized dictionary.

    Args:
        raw_cycle: MATLAB struct with fields type, ambient_temperature, time, data.
        cycle_number: Sequential 1-based cycle index.

    Returns:
        Parsed cycle dictionary, or None if the cycle has invalid/empty data.
    """
    cycle_type = str(raw_cycle.type)
    data = raw_cycle.data

    voltage = None
    current = None
    temperature = None
    time = None
    capacity = None
    eis = None

    if cycle_type in ("charge", "discharge"):
        if not hasattr(data, "Voltage_measured"):
            logger.warning("Cycle %d (%s): missing Voltage_measured, skipping", cycle_number, cycle_type)
            return None

        voltage = np.asarray(data.Voltage_measured, dtype=np.float64)
        current = np.asarray(data.Current_measured, dtype=np.float64)
        temperature = np.asarray(data.Temperature_measured, dtype=np.float64)
        time = np.asarray(data.Time, dtype=np.float64)

        if voltage.size == 0 or np.all(np.isnan(voltage)):
            logger.warning("Cycle %d (%s): empty or all-NaN voltage, skipping", cycle_number, cycle_type)
            return None

        if cycle_type == "discharge" and hasattr(data, "Capacity"):
            capacity = float(data.Capacity)
            if np.isnan(capacity) or capacity <= 0:
                logger.warning("Cycle %d: invalid discharge capacity %.4f, skipping", cycle_number, capacity)
                return None

    if cycle_type == "impedance":
        eis = {}
        for field in ("Battery_impedance", "Rectified_Impedance", "Re", "Rct"):
            if hasattr(data, field):
                eis[field.lower()] = np.asarray(getattr(data, field))
        if not eis:
            logger.warning("Cycle %d: impedance cycle has no recognized EIS fields", cycle_number)

    return {
        "cycle_number": cycle_number,
        "type": cycle_type,
        "ambient_temperature": float(raw_cycle.ambient_temperature),
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "time": time,
        "capacity": capacity,
        "eis": eis,
    }


def load_calce_cell(filepath: str | Path) -> dict[str, Any]:
    """Load a single CALCE battery cell. Not yet implemented.

    Args:
        filepath: Path to the CALCE data file.

    Raises:
        NotImplementedError: Always, until CALCE data is obtained.
    """
    raise NotImplementedError(
        "CALCE data loading is not yet implemented. "
        "Download CS2 cells from https://calce.umd.edu/battery-data and "
        "request access if required."
    )


def load_all_nasa_cells(data_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load all four standard NASA PCoE battery cells.

    Args:
        data_dir: Directory containing B0005.mat, B0006.mat, B0007.mat, B0018.mat.

    Returns:
        Dictionary mapping cell_id to loaded cell data.
    """
    data_dir = Path(data_dir)
    cells = {}
    for cell_id in NASA_CELL_IDS:
        filepath = data_dir / f"{cell_id}.mat"
        try:
            cells[cell_id] = load_nasa_cell(filepath)
        except FileNotFoundError:
            logger.warning("File not found for %s at %s, skipping", cell_id, filepath)
        except ValueError as e:
            logger.error("Failed to load %s: %s", cell_id, e)
    return cells


def verify_data_integrity(cell_data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Verify the integrity of a loaded cell's data.

    Checks for: at least one discharge cycle, valid capacity values,
    non-empty voltage arrays, and absence of all-NaN arrays.

    Args:
        cell_data: Dictionary returned by load_nasa_cell or load_calce_cell.

    Returns:
        Tuple of (is_valid, list_of_warnings). is_valid is True if all
        critical checks pass.
    """
    warnings = []
    cell_id = cell_data.get("cell_id", "unknown")

    discharge_cycles = [c for c in cell_data["cycles"] if c["type"] == "discharge"]
    if len(discharge_cycles) == 0:
        warnings.append(f"{cell_id}: no discharge cycles found")
        return False, warnings

    for cycle in discharge_cycles:
        cn = cycle["cycle_number"]
        if cycle["voltage"].size < 10:
            warnings.append(f"{cell_id} cycle {cn}: voltage array has fewer than 10 points")
        if np.all(np.isnan(cycle["voltage"])):
            warnings.append(f"{cell_id} cycle {cn}: voltage is all NaN")
        if cycle["capacity"] is not None and cycle["capacity"] <= 0:
            warnings.append(f"{cell_id} cycle {cn}: non-positive capacity {cycle['capacity']:.4f}")

    capacities = [c["capacity"] for c in discharge_cycles if c["capacity"] is not None]
    if len(capacities) < 2:
        warnings.append(f"{cell_id}: fewer than 2 valid discharge capacities")

    return len(warnings) == 0, warnings
