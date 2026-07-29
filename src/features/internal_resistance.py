"""Internal resistance estimation from discharge voltage response and EIS."""

import numpy as np


def estimate_ir_from_discharge(
    voltage: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
    initial_samples: int = 5,
) -> float:
    """Estimate internal resistance from the start of a discharge pulse.

    IR = delta_V / delta_I, where delta_V is the instantaneous voltage
    drop at the onset of discharge and delta_I is the applied current step.

    Args:
        voltage: Voltage array for the discharge cycle.
        current: Current array for the discharge cycle.
        time: Time array in seconds.
        initial_samples: Number of samples at the start to use for
            estimating the voltage drop and current step.

    Returns:
        Estimated internal resistance in Ohms. Returns NaN if the
        estimation fails (e.g., zero current step).
    """
    n = min(initial_samples + 1, len(voltage), len(current))
    if n < 2:
        return np.nan

    v_before = float(np.mean(voltage[: min(2, n - 1)]))
    v_after = float(np.mean(voltage[2:n]))
    i_before = float(np.mean(current[: min(2, n - 1)]))
    i_after = float(np.mean(current[2:n]))

    delta_v = abs(v_before - v_after)
    delta_i = abs(i_after - i_before)

    if delta_i < 1e-6:
        return np.nan

    return delta_v / delta_i


def extract_eis_features(eis_data: dict | None) -> dict[str, float]:
    """Extract internal resistance features from EIS measurements.

    Uses Re (electrolyte resistance) and Rct (charge transfer resistance)
    from impedance cycles if available.

    Args:
        eis_data: EIS dictionary from a cycle's data field. Contains keys
            like 're', 'rct', 'battery_impedance'. None if not available.

    Returns:
        Dictionary with eis_re and eis_rct features. Values are NaN
        if not available.
    """
    features = {
        "eis_re": np.nan,
        "eis_rct": np.nan,
    }

    if eis_data is None:
        return features

    if "re" in eis_data and eis_data["re"] is not None:
        re_val = eis_data["re"]
        if hasattr(re_val, "item"):
            re_val = re_val.item()
        features["eis_re"] = float(re_val)

    if "rct" in eis_data and eis_data["rct"] is not None:
        rct_val = eis_data["rct"]
        if hasattr(rct_val, "item"):
            rct_val = rct_val.item()
        features["eis_rct"] = float(rct_val)

    return features
