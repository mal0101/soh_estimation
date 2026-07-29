"""Tests for src/preprocessing/ — filtering, resampling, capacity, segmentation."""

import numpy as np

from src.preprocessing.capacity import compute_cumulative_capacity
from src.preprocessing.filtering import savgol_filter_voltage
from src.preprocessing.resampling import resample_to_uniform_grid
from src.preprocessing.segmentation import validate_cycles


class TestSavgolFilter:
    """Tests for Savitzky-Golay voltage filtering."""

    def test_reduces_noise(self):
        rng = np.random.RandomState(42)
        x = np.linspace(0, 10, 200)
        signal = np.sin(x) + rng.randn(200) * 0.5
        filtered = savgol_filter_voltage(signal, window_length=21, polyorder=3)
        noise_original = np.std(signal - np.sin(x))
        noise_filtered = np.std(filtered - np.sin(x))
        assert noise_filtered < noise_original

    def test_preserves_length(self):
        signal = np.random.randn(200)
        filtered = savgol_filter_voltage(signal, window_length=21, polyorder=3)
        assert len(filtered) == len(signal)

    def test_output_not_nan(self):
        signal = np.random.randn(100)
        filtered = savgol_filter_voltage(signal, window_length=21, polyorder=3)
        assert not np.isnan(filtered).any()


class TestResampling:
    """Tests for uniform capacity grid resampling."""

    def test_output_length(self):
        voltage = np.random.randn(300)
        capacity = np.sort(np.random.randn(300) * 10)
        cap_grid, volt_resampled = resample_to_uniform_grid(voltage, capacity, n_points=500)
        assert len(cap_grid) == 500
        assert len(volt_resampled) == 500

    def test_uniform_spacing(self):
        voltage = np.random.randn(300)
        capacity = np.sort(np.random.randn(300) * 10)
        cap_grid, _ = resample_to_uniform_grid(voltage, capacity, n_points=500)
        diffs = np.diff(cap_grid)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)


class TestCumulativeCapacity:
    """Tests for cumulative capacity computation."""

    def test_monotonic_for_discharge(self):
        n = 200
        current = np.full(n, -2.0)
        time_arr = np.linspace(0, 3600, n)
        cap = compute_cumulative_capacity(current, time_arr)
        assert np.all(np.diff(cap) >= 0)

    def test_positive_output(self):
        current = np.full(100, -1.5)
        time_arr = np.linspace(0, 1800, 100)
        cap = compute_cumulative_capacity(current, time_arr)
        assert (cap >= 0).all()


class TestValidateCycles:
    """Tests for early-cycle validation/filtering."""

    def test_filters_early_partial_cycles(self):
        cell_data = {
            "q_initial": 2.0,
            "rated_capacity": 2.0,
            "cycles": [
                {"cycle_number": 1, "capacity": 0.5, "type": "discharge"},
                {"cycle_number": 2, "capacity": 1.2, "type": "discharge"},
                {"cycle_number": 5, "capacity": 2.0, "type": "discharge"},
                {"cycle_number": 10, "capacity": 1.9, "type": "discharge"},
                {"cycle_number": 20, "capacity": 1.8, "type": "discharge"},
            ],
        }
        result = validate_cycles(cell_data, q_initial=2.0, early_cycle_window=5)
        remaining_cns = [c["cycle_number"] for c in result["cycles"]]
        assert 1 not in remaining_cns
        assert 2 not in remaining_cns
        assert 5 in remaining_cns
        assert 10 in remaining_cns
        assert 20 in remaining_cns

    def test_keeps_late_low_soh(self):
        cell_data = {
            "q_initial": 2.0,
            "rated_capacity": 2.0,
            "cycles": [
                {"cycle_number": 10, "capacity": 2.0, "type": "discharge"},
                {"cycle_number": 50, "capacity": 1.5, "type": "discharge"},
                {"cycle_number": 100, "capacity": 1.2, "type": "discharge"},
            ],
        }
        result = validate_cycles(cell_data, q_initial=2.0, early_cycle_window=5)
        assert len(result["cycles"]) == 3
