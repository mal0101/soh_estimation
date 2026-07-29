"""Per-cycle temperature statistics."""

import numpy as np


def compute_temperature_features(temperature: np.ndarray) -> dict[str, float]:
    """Compute temperature statistics for a single cycle.

    Features:
        temp_mean: Mean temperature in °C.
        temp_max: Maximum temperature in °C.
        temp_min: Minimum temperature in °C.
        temp_range: Temperature range (max - min) in °C.

    Args:
        temperature: Temperature array in °C for one cycle.

    Returns:
        Dictionary of temperature features.
    """
    if len(temperature) == 0:
        return {
            "temp_mean": np.nan,
            "temp_max": np.nan,
            "temp_min": np.nan,
            "temp_range": np.nan,
        }

    return {
        "temp_mean": float(np.mean(temperature)),
        "temp_max": float(np.max(temperature)),
        "temp_min": float(np.min(temperature)),
        "temp_range": float(np.max(temperature) - np.min(temperature)),
    }
