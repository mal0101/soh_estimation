"""Tests for src/features/ — ICA, energy, temperature, IR, trend, assembly."""

import numpy as np
import pandas as pd
import pytest

from src.features.assembly import build_feature_matrix, select_features
from src.features.energy import compute_discharge_energy, compute_mean_discharge_voltage
from src.features.ica import compute_dQdV, extract_ica_features
from src.features.internal_resistance import estimate_ir_from_discharge
from src.features.temperature import compute_temperature_features
from src.features.trend import compute_capacity_fade_rate


def _make_discharge_cycle(n=300):
    """Create a synthetic discharge cycle dict for testing."""
    rng = np.random.RandomState(42)
    time_arr = np.linspace(0, 3600, n)
    voltage = 4.2 - 0.8 * np.linspace(0, 1, n) + rng.randn(n) * 0.01
    current = np.full(n, -2.0)
    temperature = 25 + 5 * np.linspace(0, 1, n) + rng.randn(n) * 0.5
    capacity = np.abs(np.cumsum(current * np.diff(time_arr, prepend=0) / 3600))
    return {
        "type": "discharge",
        "cycle_number": 10,
        "voltage": voltage,
        "current": current,
        "time": time_arr,
        "temperature": temperature,
        "capacity": capacity,
        "voltage_resampled": voltage,
        "capacity_grid": capacity,
    }


class TestICA:
    """Tests for incremental capacity analysis."""

    def test_compute_dQdV_output_shape(self):
        cycle = _make_discharge_cycle()
        V_grid, dQdV = compute_dQdV(cycle["voltage_resampled"], cycle["capacity_grid"])
        assert len(V_grid) == len(dQdV)

    def test_compute_dQdV_same_length_as_input(self):
        cycle = _make_discharge_cycle(500)
        V_grid, dQdV = compute_dQdV(cycle["voltage_resampled"], cycle["capacity_grid"])
        assert len(V_grid) == 500

    def test_extract_ica_features_keys(self):
        cycle = _make_discharge_cycle()
        feats = extract_ica_features(cycle["voltage_resampled"], cycle["capacity_grid"])
        expected_keys = {
            "ica_peak_height", "ica_peak_voltage", "ica_peak_area",
            "ica_peak_fwhm", "ica_secondary_ratio",
        }
        assert expected_keys == set(feats.keys())

    def test_peak_voltage_in_valid_range(self):
        cycle = _make_discharge_cycle()
        feats = extract_ica_features(cycle["voltage_resampled"], cycle["capacity_grid"])
        if not np.isnan(feats["ica_peak_voltage"]):
            assert 2.5 <= feats["ica_peak_voltage"] <= 4.5


class TestEnergy:
    """Tests for energy features."""

    def test_discharge_energy_positive(self):
        n = 200
        voltage = np.linspace(4.0, 3.0, n)
        current = np.full(n, -2.0)
        time_arr = np.linspace(0, 3600, n)
        energy = compute_discharge_energy(voltage, current, time_arr)
        assert energy > 0

    def test_mean_voltage_in_range(self):
        n = 200
        voltage = np.linspace(4.0, 3.0, n)
        current = np.full(n, -2.0)
        time_arr = np.linspace(0, 3600, n)
        mean_v = compute_mean_discharge_voltage(voltage, current, time_arr)
        assert 2.5 <= mean_v <= 4.5


class TestTemperature:
    """Tests for temperature features."""

    def test_returns_four_keys(self):
        temp = np.array([25.0, 30.0, 28.0, 35.0, 20.0])
        feats = compute_temperature_features(temp)
        assert set(feats.keys()) == {"temp_mean", "temp_max", "temp_min", "temp_range"}

    def test_range_equals_max_minus_min(self):
        temp = np.array([20.0, 25.0, 30.0])
        feats = compute_temperature_features(temp)
        assert feats["temp_range"] == pytest.approx(feats["temp_max"] - feats["temp_min"])

    def test_mean_in_range(self):
        temp = np.array([20.0, 25.0, 30.0])
        feats = compute_temperature_features(temp)
        assert feats["temp_min"] <= feats["temp_mean"] <= feats["temp_max"]


class TestInternalResistance:
    """Tests for internal resistance estimation."""

    def test_ir_non_negative_or_nan(self):
        """IR can be NaN for degenerate synthetic data, but should not be negative."""
        n = 200
        voltage = np.linspace(4.0, 3.0, n)
        current = np.full(n, -2.0)
        time_arr = np.linspace(0, 3600, n)
        ir = estimate_ir_from_discharge(voltage, current, time_arr)
        assert np.isnan(ir) or ir >= 0


class TestTrend:
    """Tests for capacity fade rate."""

    def test_output_length(self):
        soh = np.linspace(1.0, 0.8, 50)
        rates = compute_capacity_fade_rate(soh, window=10)
        assert len(rates) == len(soh)

    def test_rates_negative_for_degradation(self):
        soh = np.linspace(1.0, 0.8, 50)
        rates = compute_capacity_fade_rate(soh, window=10)
        valid = rates[~np.isnan(rates)]
        assert (valid <= 0).all()


class TestAssembly:
    """Tests for feature matrix assembly and selection."""

    def _make_processed_cells(self):
        """Create minimal processed_cells dict for assembly."""
        cells = {}
        for cell_id in ["cell_A", "cell_B"]:
            cycles = []
            for cn in range(5, 55, 2):
                np.random.RandomState(cn)
                n = 300
                voltage = np.linspace(4.0, 3.0, n)
                current = np.full(n, -2.0)
                time_arr = np.linspace(0, 3600, n)
                temp = np.full(n, 25.0)
                capacity = np.abs(np.cumsum(current * np.diff(time_arr, prepend=0) / 3600))
                cycles.append({
                    "type": "discharge",
                    "cycle_number": cn,
                    "voltage": voltage,
                    "current": current,
                    "time": time_arr,
                    "temperature": temp,
                    "capacity": capacity,
                    "voltage_resampled": voltage,
                    "capacity_grid": capacity,
                    "eis": None,
                })
            cells[cell_id] = {
                "q_initial": 2.0,
                "rated_capacity": 2.0,
                "cutoff_voltage": 2.7,
                "ambient_temperature": 25,
                "cycles": cycles,
            }
        return cells

    def test_build_feature_matrix_shape(self):
        cells = self._make_processed_cells()
        soh_records = []
        for cell_id, cell in cells.items():
            for c in cell["cycles"]:
                soh_records.append({
                    "cell_id": cell_id,
                    "dataset": "test",
                    "cycle_number": c["cycle_number"],
                    "soh": 0.95,
                    "rated_capacity": 2.0,
                })
        soh_df = pd.DataFrame(soh_records)
        df = build_feature_matrix(cells, soh_df)
        assert len(df) > 0
        assert "cell_id" in df.columns
        assert "soh" in df.columns

    def test_select_features_removes_constants(self):
        rng = np.random.RandomState(42)
        n = 50
        x = rng.randn(n)
        df = pd.DataFrame({
            "cell_id": ["A"] * n,
            "dataset": ["test"] * n,
            "cycle_number": range(n),
            "soh": 0.9 + 0.1 * x,
            "feat_a": x,
            "feat_b": x * 0.5 + rng.randn(n) * 0.1,
            "feat_c": rng.randn(n),
            "feat_d": rng.randn(n),
            "feat_e": rng.randn(n),
            "feat_constant": np.full(n, 5.0),
        })
        _, selected = select_features(df, top_k=5)
        assert "feat_constant" not in selected

    def test_correlation_filter_one_of_pair(self):
        rng = np.random.RandomState(42)
        x = rng.randn(50)
        df = pd.DataFrame({
            "cell_id": ["A"] * 50,
            "dataset": ["test"] * 50,
            "cycle_number": range(50),
            "soh": rng.randn(50),
            "a": x,
            "b": x + rng.randn(50) * 0.001,
            "c": rng.randn(50),
        })
        _, selected = select_features(df, correlation_threshold=0.99, top_k=5)
        a_in = "a" in selected
        b_in = "b" in selected
        assert not (a_in and b_in), "Both correlated features should not both be selected"
