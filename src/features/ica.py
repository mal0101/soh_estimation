"""Incremental Capacity Analysis (ICA) feature extraction.

Computes dQ/dV curves from voltage-capacity data and extracts
physically meaningful features: peak height, position, width,
area, and primary/secondary peak ratio.
"""

import logging
from typing import Any

import numpy as np
from scipy.signal import find_peaks

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

logger = logging.getLogger(__name__)


def compute_dQdV(
    voltage: np.ndarray,
    capacity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the incremental capacity analysis curve (dQ/dV).

    Uses numpy.gradient for numerical differentiation, which handles
    non-uniform spacing in the capacity grid.

    Args:
        voltage: Voltage values (resampled onto uniform capacity grid).
        capacity: Corresponding capacity values.

    Returns:
        Tuple of (voltage_grid, dQdV) arrays. The voltage_grid excludes
        the endpoints where the gradient is less reliable, and dQdV is
        computed on the interior points.
    """
    if len(voltage) < 3:
        return voltage.copy(), np.zeros_like(voltage)

    dQdV = np.gradient(capacity, voltage)

    return voltage.copy(), dQdV


def extract_ica_features(
    voltage: np.ndarray,
    capacity: np.ndarray,
    min_peak_prominence: float = 0.01,
) -> dict[str, Any]:
    """Extract ICA features from a discharge voltage-capacity curve.

    Features extracted:
        ica_peak_height: Height of the primary dQ/dV peak.
        ica_peak_voltage: Voltage position of the primary peak.
        ica_peak_fwhm: Full width at half maximum of primary peak.
        ica_peak_area: Area under the primary peak region.
        ica_secondary_ratio: Ratio of primary to secondary peak heights
            (NaN if no secondary peak is detected).

    Args:
        voltage: Resampled voltage array.
        capacity: Corresponding capacity array.
        min_peak_prominence: Minimum prominence for peak detection.
            Higher values reduce false positives from noise.

    Returns:
        Dictionary of ICA features.
    """
    voltage_grid, dQdV = compute_dQdV(voltage, capacity)

    features = {
        "ica_peak_height": np.nan,
        "ica_peak_voltage": np.nan,
        "ica_peak_fwhm": np.nan,
        "ica_peak_area": np.nan,
        "ica_secondary_ratio": np.nan,
    }

    if len(dQdV) < 5:
        return features

    valid_mask = np.isfinite(dQdV)
    if not np.any(valid_mask):
        return features

    dQdV_clean = np.where(valid_mask, dQdV, 0.0)

    try:
        peaks, properties = find_peaks(
            dQdV_clean,
            prominence=min_peak_prominence,
        )
    except Exception:
        return features

    if len(peaks) == 0:
        return features

    prominences = properties.get("prominences", np.zeros(len(peaks)))
    primary_idx = np.argmax(prominences)
    primary_peak = peaks[primary_idx]

    features["ica_peak_height"] = float(dQdV_clean[primary_peak])
    features["ica_peak_voltage"] = float(voltage_grid[primary_peak])

    half_max = features["ica_peak_height"] / 2.0

    if half_max > 0:
        left_mask = dQdV_clean[:primary_peak] >= half_max
        right_mask = dQdV_clean[primary_peak + 1:] >= half_max

        left_idx = np.where(left_mask)[0]
        right_idx = np.where(right_mask)[0]

        if len(left_idx) > 0 and len(right_idx) > 0:
            fwhm_start = left_idx[0]
            fwhm_end = primary_peak + 1 + right_idx[-1]
            fwhm_capacity = abs(capacity[fwhm_end] - capacity[fwhm_start])
            features["ica_peak_fwhm"] = float(fwhm_capacity)

    peak_region_mask = dQdV_clean > max(0, features["ica_peak_height"] * 0.1)
    peak_voltages = voltage_grid[peak_region_mask]
    peak_dqdV = dQdV_clean[peak_region_mask]
    if len(peak_voltages) > 1:
        positive_mask = peak_dqdV > 0
        if np.any(positive_mask):
            features["ica_peak_area"] = float(abs(_trapz(
                peak_dqdV[positive_mask], peak_voltages[positive_mask]
            )))

    if len(peaks) >= 2:
        secondary_idx = np.argsort(prominences)[-2]
        secondary_peak = peaks[secondary_idx]
        secondary_height = float(dQdV_clean[secondary_peak])
        if secondary_height > 0:
            features["ica_secondary_ratio"] = (
                features["ica_peak_height"] / secondary_height
            )

    return features
