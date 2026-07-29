"""Capacity fade rate trend feature."""

import numpy as np


def compute_capacity_fade_rate(
    soh_curve: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Compute the local SOH slope over a sliding window.

    SOH_slope(n) = (SOH(n) - SOH(n - window)) / window

    For the first `window` cycles, uses all available prior cycles.
    The result has the same length as the input; the first value is 0.

    Args:
        soh_curve: Array of SOH values in cycle order.
        window: Number of cycles over which to compute the slope.

    Returns:
        Array of SOH slope values, same length as soh_curve.
    """
    n = len(soh_curve)
    slopes = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        lookback = min(i, window)
        slopes[i] = (soh_curve[i] - soh_curve[i - lookback]) / lookback

    return slopes
