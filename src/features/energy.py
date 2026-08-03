"""Discharge energy, mean voltage, and coulombic efficiency features."""

import numpy as np

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def compute_discharge_energy(
    voltage: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
) -> float:
    """Compute total discharge energy in Wh.

    E = integral of V(t) * |I(t)| dt / 3600.

    Args:
        voltage: Voltage array.
        current: Current array (negative during discharge).
        time: Time array in seconds.

    Returns:
        Discharge energy in Wh.
    """
    if len(voltage) < 2 or len(current) < 2 or len(time) < 2:
        return np.nan

    power = voltage * np.abs(current)
    energy_joules = _trapz(power, time)
    return energy_joules / 3600.0


def compute_mean_discharge_voltage(
    voltage: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
) -> float:
    """Compute the mean voltage during discharge.

    V_mean = E / Q, where E is discharge energy and Q is discharge capacity.

    Args:
        voltage: Voltage array.
        current: Current array.
        time: Time array in seconds.

    Returns:
        Mean discharge voltage in Volts.
    """
    energy = compute_discharge_energy(voltage, current, time)
    if np.isnan(energy) or energy <= 0:
        return np.nan

    capacity_ah = _trapz(np.abs(current), time) / 3600.0
    if capacity_ah <= 0:
        return np.nan

    return energy / capacity_ah


def compute_coulombic_efficiency(
    q_discharge: float,
    q_charge: float,
) -> float:
    """Compute coulombic efficiency.

    CE = Q_discharge / Q_charge. Should be close to 1.0 for healthy cells
    and deviate with aging and temperature.

    Args:
        q_discharge: Discharge capacity in Ah.
        q_charge: Charge capacity in Ah.

    Returns:
        Coulombic efficiency (dimensionless). Returns NaN if q_charge
        is zero or negative.
    """
    if q_charge <= 0:
        return np.nan
    return q_discharge / q_charge
