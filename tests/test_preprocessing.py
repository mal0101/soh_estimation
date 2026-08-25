"""Tests for src/preprocessing/ — filtering, resampling, capacity, segmentation."""

import numpy as np
import pytest

from src.preprocessing.capacity import compute_cumulative_capacity
from src.preprocessing.data_loader import (
    CALCE_CELLS,
    CALCE_CUTOFF_VOLTAGE,
    CALCE_RATED_CAPACITY,
    load_all_calce_cells,
    load_calce_cell,
)
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
        """Genuine gradual degradation must never be filtered out."""
        n_flat, n_fade = 25, 35
        caps = [2.0] * n_flat + list(np.linspace(2.0, 1.2, n_fade)[1:])
        cell_data = {
            "q_initial": 2.0,
            "rated_capacity": 2.0,
            "cycles": [
                {"cycle_number": int(i + 1), "capacity": float(c), "type": "discharge"}
                for i, c in enumerate(caps)
            ],
        }
        result = validate_cycles(cell_data, q_initial=2.0)
        assert len(result["cycles"]) == len(caps)

    def test_discards_mid_life_interruption(self):
        """An abrupt isolated drop anywhere in life is an interruption."""
        n = 60
        capacities = np.linspace(2.0, 1.85, n)
        caps = [float(c) for c in capacities]
        caps[30] = 0.3  # instrumentation pause mid-life
        cell_data = {
            "q_initial": 2.0,
            "rated_capacity": 2.0,
            "cycles": [
                {"cycle_number": i + 1, "capacity": caps[i], "type": "discharge"}
                for i in range(n)
            ],
        }
        result = validate_cycles(cell_data, q_initial=2.0)
        remaining_cns = [c["cycle_number"] for c in result["cycles"]]
        assert 31 not in remaining_cns
        assert len(remaining_cns) == n - 1

    def test_discards_early_partial_beyond_window(self):
        """A partial discharge just after the early window is still caught."""
        n = 40
        caps = [2.0 - 0.001 * i for i in range(n)]
        caps[25] = 1.0  # 50% of neighbours -> interruption
        cell_data = {
            "q_initial": 2.0,
            "rated_capacity": 2.0,
            "cycles": [
                {"cycle_number": i + 1, "capacity": float(caps[i]), "type": "discharge"}
                for i in range(n)
            ],
        }
        result = validate_cycles(cell_data, q_initial=2.0, early_cycle_window=20)
        remaining_cns = [c["cycle_number"] for c in result["cycles"]]
        assert 26 not in remaining_cns

    def test_consecutive_interruptions_removed(self):
        """Two adjacent interruption cycles are both flagged."""
        n = 40
        caps = [2.0 - 0.001 * i for i in range(n)]
        caps[25] = 0.4
        caps[26] = 0.5
        cell_data = {
            "q_initial": 2.0,
            "rated_capacity": 2.0,
            "cycles": [
                {"cycle_number": i + 1, "capacity": float(caps[i]), "type": "discharge"}
                for i in range(n)
            ],
        }
        result = validate_cycles(cell_data, q_initial=2.0)
        remaining_cns = [c["cycle_number"] for c in result["cycles"]]
        assert 26 not in remaining_cns and 27 not in remaining_cns

    def test_discard_count_reported(self):
        n = 30
        caps = [2.0 - 0.001 * i for i in range(n)]
        caps[15] = 0.2
        cell_data = {
            "cell_id": "X",
            "rated_capacity": 2.0,
            "cycles": [
                {"cycle_number": i + 1, "capacity": float(caps[i]), "type": "discharge"}
                for i in range(n)
            ],
        }
        result = validate_cycles(cell_data, q_initial=2.0)
        assert result["n_cycles_discarded"] == 1


class TestCalceLoader:
    """Tests for CALCE data loading."""

    @pytest.fixture(scope="class")
    def calce_cell(self):
        """Load a single CALCE cell for testing (once per class)."""
        return load_calce_cell("data/raw/calce/CS2_33")

    def test_load_calce_cell_returns_dict(self, calce_cell):
        assert isinstance(calce_cell, dict)
        assert calce_cell["cell_id"] == "CS2_33"
        assert calce_cell["dataset"] == "calce"

    def test_calce_rated_capacity(self, calce_cell):
        assert calce_cell["rated_capacity"] == CALCE_RATED_CAPACITY

    def test_calce_cutoff_voltage(self, calce_cell):
        assert calce_cell["cutoff_voltage"] == CALCE_CUTOFF_VOLTAGE

    def test_calce_has_discharge_cycles(self, calce_cell):
        discharge = [c for c in calce_cell["cycles"] if c["type"] == "discharge"]
        assert len(discharge) > 0

    def test_calce_discharge_capacity_positive(self, calce_cell):
        discharge = [c for c in calce_cell["cycles"] if c["type"] == "discharge"]
        capacities = [c["capacity"] for c in discharge if c["capacity"] is not None]
        assert all(c > 0 for c in capacities)

    def test_calce_voltage_array_not_empty(self, calce_cell):
        discharge = [c for c in calce_cell["cycles"] if c["type"] == "discharge"]
        for cycle in discharge:
            assert cycle["voltage"].size > 0

    def test_calce_time_array_not_empty(self, calce_cell):
        discharge = [c for c in calce_cell["cycles"] if c["type"] == "discharge"]
        for cycle in discharge:
            assert cycle["time"].size > 0

    def test_calce_no_eis(self, calce_cell):
        discharge = [c for c in calce_cell["cycles"] if c["type"] == "discharge"]
        for cycle in discharge:
            assert cycle["eis"] is None

    def test_calce_temperature_placeholder(self, calce_cell):
        discharge = [c for c in calce_cell["cycles"] if c["type"] == "discharge"]
        for cycle in discharge:
            assert cycle["temperature"] is not None
            assert np.all(cycle["temperature"] == 25.0)

    def test_load_all_calce_cells(self):
        cells = load_all_calce_cells("data/raw/calce")
        assert len(cells) == 4
        for cell_id in CALCE_CELLS:
            assert cell_id in cells
            assert cells[cell_id]["dataset"] == "calce"

    def test_calce_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_calce_cell("data/raw/calce/NONEXISTENT")
