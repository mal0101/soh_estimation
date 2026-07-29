"""Tests for src/evaluation/metrics.py."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_all_metrics, mae, maxae, r2, rmse


class TestRMSE:
    """Tests for root mean squared error."""

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 3.0, 3.0])
        expected = np.sqrt(np.mean([0.0, 1.0, 0.0]))
        assert abs(rmse(y_true, y_pred) - expected) < 1e-10

    def test_returns_float(self):
        assert isinstance(rmse(np.array([1.0]), np.array([2.0])), float)


class TestMAE:
    """Tests for mean absolute error."""

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_symmetric(self):
        y_true = np.array([1.0, 2.0, 3.0])
        assert mae(y_true, y_true + 0.5) == mae(y_true, y_true - 0.5)

    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, -1.0])
        assert mae(y_true, y_pred) == 1.0


class TestMaxAE:
    """Tests for maximum absolute error."""

    def test_picks_largest(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.5, 3.0])
        assert maxae(y_true, y_pred) == pytest.approx(0.5)

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0])
        assert maxae(y, y) == 0.0


class TestR2:
    """Tests for coefficient of determination."""

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2(y, y) == pytest.approx(1.0)

    def test_baseline_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert r2(y_true, y_pred) == pytest.approx(0.0)

    def test_worse_than_mean(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 1.0, 2.0])
        assert r2(y_true, y_pred) < 0.0


class TestComputeAllMetrics:
    """Tests for the combined metrics function."""

    def test_returns_correct_keys(self):
        y = np.array([1.0, 2.0])
        metrics = compute_all_metrics(y, y)
        assert set(metrics.keys()) == {"rmse", "mae", "maxae", "r2"}

    def test_consistent_with_individual(self):
        y_true = np.array([0.9, 0.85, 0.8])
        y_pred = np.array([0.91, 0.83, 0.82])
        metrics = compute_all_metrics(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(rmse(y_true, y_pred))
        assert metrics["mae"] == pytest.approx(mae(y_true, y_pred))
        assert metrics["maxae"] == pytest.approx(maxae(y_true, y_pred))
        assert metrics["r2"] == pytest.approx(r2(y_true, y_pred))

    def test_all_values_are_float(self):
        metrics = compute_all_metrics(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
        for v in metrics.values():
            assert isinstance(v, float)
