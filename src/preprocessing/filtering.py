"""Savitzky-Golay noise filtering for voltage signals."""

import numpy as np
from scipy.signal import savgol_filter


def savgol_filter_voltage(
    voltage: np.ndarray,
    window_length: int = 51,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay filter to smooth a voltage signal.

    The filter fits a local polynomial of the given order within a sliding
    window and evaluates it at the center point, producing a smoothed signal
    that preserves peak shapes better than a moving average.

    Args:
        voltage: Raw voltage array to smooth.
        window_length: Filter window length in samples. Must be odd and
            >= polyorder + 2. If window_length exceeds len(voltage), it
            is reduced to the largest odd number <= len(voltage).
        polyorder: Polynomial order for the local fit. Must be < window_length.

    Returns:
        Smoothed voltage array of the same length as the input.
    """
    if len(voltage) < 3:
        return voltage.copy()

    effective_window = window_length
    if effective_window % 2 == 0:
        effective_window -= 1

    max_window = len(voltage)
    if max_window % 2 == 0:
        max_window -= 1

    effective_window = min(effective_window, max_window)
    effective_window = max(effective_window, polyorder + 2)

    if effective_window % 2 == 0:
        effective_window -= 1

    effective_window = min(effective_window, max_window)

    if effective_window < polyorder + 2:
        return voltage.copy()

    return savgol_filter(voltage, window_length=effective_window, polyorder=polyorder)
