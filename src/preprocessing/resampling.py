"""Uniform capacity grid resampling for voltage-capacity curves."""

import numpy as np
from scipy.interpolate import interp1d


def resample_to_uniform_grid(
    voltage: np.ndarray,
    capacity: np.ndarray,
    n_points: int = 1000,
    kind: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate voltage onto a uniform capacity grid.

    Maps a voltage-capacity curve (with non-uniformly spaced capacity
    values from cumulative integration) onto a uniform grid of n_points
    evenly spaced capacity values.

    Args:
        voltage: Voltage values to interpolate.
        capacity: Cumulative capacity array (from compute_cumulative_capacity).
            Must be monotonically increasing and have the same length as voltage.
        n_points: Number of points in the output uniform grid.
        kind: Interpolation kind passed to scipy.interpolate.interp1d.
            Default is "linear" to avoid overshoot on sharp transitions.

    Returns:
        Tuple of (capacity_grid, voltage_resampled), both arrays of
        length n_points. capacity_grid ranges from 0 to capacity[-1].

    Raises:
        ValueError: If voltage and capacity have different lengths, or
            if n_points < 2.
    """
    if len(voltage) != len(capacity):
        raise ValueError(
            f"voltage and capacity must have the same length, "
            f"got {len(voltage)} and {len(capacity)}"
        )
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    if len(voltage) < 2:
        cap_grid = np.linspace(0, 1, n_points)
        return cap_grid, np.full(n_points, voltage[0] if len(voltage) == 1 else 0.0)

    cap_max = capacity[-1]
    if cap_max <= 0:
        cap_grid = np.linspace(0, 1, n_points)
        return cap_grid, np.full(n_points, np.mean(voltage))

    cap_grid = np.linspace(0, cap_max, n_points)
    interpolator = interp1d(
        capacity, voltage, kind=kind, fill_value="extrapolate"
    )
    voltage_resampled = interpolator(cap_grid)

    return cap_grid, voltage_resampled
