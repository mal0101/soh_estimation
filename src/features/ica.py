"""Incremental Capacity Analysis (ICA) feature extraction.

Computes dQ/dV curves from voltage-capacity data and extracts
physically meaningful features: peak height, position, width,
area, and primary/secondary peak ratio.

Sign convention: for discharge data the capacity axis increases while
voltage decreases, so ``np.gradient(capacity, voltage)`` is negative.
We negate it so that dQ/dV is positive and genuine ICA peaks appear
as local maxima (standard convention in the literature).
"""

import logging
from typing import Any

import numpy as np
from scipy.signal import find_peaks

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Physically plausible voltage window for Li-ion ICA primary peaks.
ICA_VOLTAGE_RANGE = (3.0, 4.35)

logger = logging.getLogger(__name__)


def compute_dQdV(
    voltage: np.ndarray,
    capacity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the incremental capacity analysis curve (dQ/dV).

    Uses numpy.gradient for numerical differentiation. Near-duplicate
    consecutive voltage samples are dropped first: dividing by a ~zero
    voltage gap produces gradient explosions (~1e5) that would otherwise
    poison peak detection. The raw gradient is negated so discharge
    curves (capacity increasing while voltage decreases) yield a
    positive dQ/dV with peaks as local maxima.

    Args:
        voltage: Voltage values (resampled onto uniform capacity grid).
        capacity: Corresponding capacity values.

    Returns:
        Tuple of (voltage_grid, dQdV) arrays of equal length (shorter
        than the input when duplicates were removed).
    """
    voltage = np.asarray(voltage, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)

    if len(voltage) < 3:
        return voltage.copy(), np.zeros_like(voltage)

    # Drop consecutive near-duplicate voltages (numerical zero-gap guard).
    keep = np.ones(len(voltage), dtype=bool)
    eps = 1e-6 * max(float(np.ptp(voltage)), 1e-9)
    keep[1:] = np.abs(np.diff(voltage)) > eps
    voltage = voltage[keep]
    capacity = capacity[keep]

    if len(voltage) < 3:
        return voltage.copy(), np.zeros_like(voltage)

    dQdV = -np.gradient(capacity, voltage)

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
        ica_peak_fwhm: Full width at half maximum of primary peak,
            expressed in Ah along the capacity axis.
        ica_peak_area: Area under the primary peak region (dQ/dV vs V).
        ica_secondary_ratio: Ratio of secondary to primary peak heights
            (NaN if no secondary peak is detected).

    Args:
        voltage: Resampled voltage array.
        capacity: Corresponding capacity array.
        min_peak_prominence: Minimum peak prominence as a FRACTION of
            the maximum dQ/dV value (relative threshold so the noise
            gate scales with signal magnitude). Default 0.01 (1%).

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

    # Physical constraint: capacity cannot decrease during a discharge,
    # so genuine dQ/dV is non-negative. Negative excursions are
    # measurement noise (e.g., end-of-discharge voltage plateaus) and
    # are clipped before peak search.
    dQdV_clean = np.where(valid_mask, dQdV, 0.0)
    dQdV_clean = np.clip(dQdV_clean, 0.0, None)
    if np.all(dQdV_clean <= 0):
        return features

    # Relative prominence threshold: scale with signal magnitude.
    prominence = min_peak_prominence * float(np.max(dQdV_clean))

    try:
        peaks, properties = find_peaks(
            dQdV_clean,
            prominence=prominence,
        )
    except Exception:
        return features

    if len(peaks) == 0:
        return features

    prominences = properties.get("prominences", np.zeros(len(peaks)))
    primary_idx = int(np.argmax(prominences))
    primary_peak = peaks[primary_idx]

    peak_voltage = float(voltage_grid[primary_peak])
    peak_height = float(dQdV_clean[primary_peak])

    # Physical plausibility gate: the primary ICA peak of a Li-ion
    # discharge must lie inside a physically meaningful voltage window.
    if not (ICA_VOLTAGE_RANGE[0] <= peak_voltage <= ICA_VOLTAGE_RANGE[1]):
        logger.debug("ICA peak at %.3f V outside plausible range, rejecting", peak_voltage)
        return features

    features["ica_peak_height"] = peak_height
    features["ica_peak_voltage"] = peak_voltage

    half_max = features["ica_peak_height"] / 2.0

    left_mask = dQdV_clean[:primary_peak] >= half_max
    right_mask = dQdV_clean[primary_peak + 1 :] >= half_max

    left_idx = np.where(left_mask)[0]
    right_idx = np.where(right_mask)[0]

    if len(left_idx) > 0 and len(right_idx) > 0:
        fwhm_start = int(left_idx[0])
        fwhm_end = primary_peak + 1 + int(right_idx[-1])
        fwhm_capacity = abs(capacity[fwhm_end] - capacity[fwhm_start])
        features["ica_peak_fwhm"] = float(fwhm_capacity)

    peak_region_mask = dQdV_clean > features["ica_peak_height"] * 0.1
    peak_voltages = voltage_grid[peak_region_mask]
    peak_dqdV = dQdV_clean[peak_region_mask]
    if len(peak_voltages) > 1 and _trapz is not None:
        features["ica_peak_area"] = float(abs(_trapz(peak_dqdV, peak_voltages)))

    if len(peaks) >= 2:
        secondary_idx = int(np.argsort(prominences)[-2])
        secondary_peak = peaks[secondary_idx]
        secondary_height = float(dQdV_clean[secondary_peak])
        if secondary_height > 0:
            features["ica_secondary_ratio"] = secondary_height / features["ica_peak_height"]

    return features
