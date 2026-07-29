"""Tests for src/evaluation/ — error_analysis, comparison, visualizations."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.comparison import build_comparison_table, rank_models
from src.evaluation.deployability import deployability_report, model_size_mb
from src.evaluation.error_analysis import phase_rmse, phase_summary_table


class TestPhaseRMSE:
    """Tests for phase-based error breakdown."""

    def test_all_phases_present(self):
        y_true = np.array([0.95, 0.85, 0.75, 0.92, 0.88, 0.70])
        y_pred = np.array([0.94, 0.83, 0.78, 0.91, 0.86, 0.72])
        soh = y_true.copy()
        results = phase_rmse(y_true, y_pred, soh)
        assert set(results.keys()) == {"early", "mid", "end"}

    def test_n_samples_correct(self):
        soh = np.array([0.95, 0.85, 0.75, 0.92, 0.88, 0.70])
        y_true = soh.copy()
        y_pred = soh.copy()
        results = phase_rmse(y_true, y_pred, soh)
        assert results["early"]["n_samples"] == 2
        assert results["mid"]["n_samples"] == 2
        assert results["end"]["n_samples"] == 2

    def test_perfect_predictions_zero_rmse(self):
        soh = np.array([0.95, 0.85, 0.75])
        results = phase_rmse(soh, soh, soh)
        for phase in results:
            if results[phase]["n_samples"] > 0:
                assert results[phase]["rmse"] == pytest.approx(0.0)

    def test_empty_phase(self):
        soh = np.array([0.95, 0.92])
        y_true = soh.copy()
        y_pred = soh + 0.01
        results = phase_rmse(y_true, y_pred, soh)
        assert results["end"]["n_samples"] == 0
        assert np.isnan(results["end"]["rmse"])


class TestPhaseSummaryTable:
    """Tests for summary table formatting."""

    def test_returns_string(self):
        results = {"early": {"rmse": 0.01, "mae": 0.008, "n_samples": 100}}
        table = phase_summary_table(results)
        assert isinstance(table, str)
        assert "early" in table


class TestBuildComparisonTable:
    """Tests for comparison table construction."""

    def test_returns_dataframe(self):
        results = {
            "rf": {"rmse_mean": 0.015, "rmse_std": 0.002, "mae_mean": 0.012, "mae_std": 0.001},
            "gpr": {"rmse_mean": 0.012, "rmse_std": 0.003, "mae_mean": 0.010, "mae_std": 0.002},
        }
        df = build_comparison_table(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


class TestRankModels:
    """Tests for model ranking."""

    def test_best_first(self):
        results = {
            "rf": {"rmse_mean": 0.020},
            "gpr": {"rmse_mean": 0.010},
            "svr": {"rmse_mean": 0.015},
        }
        ranked = rank_models(results)
        assert ranked.iloc[0]["Model"] == "gpr"
        assert ranked.iloc[0]["Rank"] == 1


class TestModelSizeMB:
    """Tests for model file size measurement."""

    def test_returns_positive_float(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_bytes(b"\x00" * 1024)
        size = model_size_mb(f)
        assert size > 0
        assert abs(size - 1024 / (1024 * 1024)) < 1e-6


class TestDeployabilityReport:
    """Tests for deployability report generation."""

    def test_report_keys(self):
        inf_times = {"rf": {"mean_inference_ms": 1.0}}
        sizes = {"rf": 0.5}
        report = deployability_report(inf_times, sizes)
        assert "rf" in report
        assert report["rf"]["meets_inference_target"] is True
