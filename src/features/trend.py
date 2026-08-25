"""Capacity fade rate trend feature."""

import numpy as np


def compute_capacity_fade_rate(
    soh_curve: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Compute the local SOH slope over a trailing window of PAST labels.

    fade_rate(n) = (SOH(n-1) - SOH(n-1-window)) / window

    Only strictly-past SOH labels are used (indices <= n-1), which keeps
    the feature valid for deployment: at prediction time the current
    cycle's SOH is unknown (it is the target), while previously measured
    capacities are available. Including SOH(n) itself would leak the
    target into the feature.

    The first ``window + 1`` values are NaN because insufficient past
    history exists; they are intentionally NOT filled with 0 so that no
    synthetic "no fade" signal is invented downstream (NaN rows are
    dropped per-fold during training).

    Args:
        soh_curve: Array of SOH values in cycle order.
        window: Number of cycles over which to compute the slope.

    Returns:
        Array of SOH slope values, same length as soh_curve. Entries
        with fewer than two prior labels are NaN.
    """
    n = len(soh_curve)
    slopes = np.full(n, np.nan, dtype=np.float64)

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    for i in range(n):
        end = i - 1
        if end < 1:
            continue
        lookback = min(end, window)
        prev = soh_curve[end - lookback]
        cur = soh_curve[end]
        if np.isfinite(cur) and np.isfinite(prev):
            slopes[i] = (cur - prev) / lookback

    return slopes
