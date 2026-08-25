"""Cumulative capacity computation from current and time arrays.

Computes the capacity axis for voltage-capacity curves by integrating
the absolute current over time using the trapezoidal rule.
"""

import numpy as np


def compute_cumulative_capacity(current: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Compute cumulative capacity (Ah) from current and time arrays.

    Uses trapezoidal integration: Q(t) = integral of |I(t)| dt / 3600.
    The result starts at 0 and increases monotonically to the total
    measured capacity of the cycle.

    Args:
        current: Current array in Amps. Sign convention does not matter;
            absolute values are used internally.
        time: Time array in seconds. Must be monotonically increasing
            and have the same length as current.

    Returns:
        Cumulative capacity array in Ah, same length as input.

    Raises:
        ValueError: If current and time have different lengths, or if
            time is not monotonically increasing.
    """
    if len(current) != len(time):
        raise ValueError(
            f"current and time must have the same length, got {len(current)} and {len(time)}"
        )
    if len(current) < 2:
        return np.zeros_like(current, dtype=np.float64)

    dt = np.diff(time)
    if not np.all(np.isfinite(dt)):
        raise ValueError(
            "time array contains non-finite values; NaN timestamps would "
            "silently corrupt the capacity axis"
        )
    if np.any(dt < 0):
        raise ValueError("time array must be monotonically increasing")

    avg_current = (np.abs(current[:-1]) + np.abs(current[1:])) / 2.0
    cumulative = np.asarray(
        np.concatenate([[0.0], np.cumsum(avg_current * dt) / 3600.0]),
        dtype=np.float64,
    )

    return cumulative
