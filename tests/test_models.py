"""Tests for src/models/ — baseline, RF, SVR, GPR."""

import numpy as np
import pytest

from src.models.baseline import NaiveBaseline
from src.models.gpr_model import train_gpr
from src.models.rf_model import train_rf
from src.models.svr_model import train_svr


@pytest.fixture
def train_test_data():
    """Small synthetic dataset for model tests."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = 0.8 + 0.1 * X[:, 0] - 0.05 * X[:, 1] + rng.randn(100) * 0.02
    return X[:80], y[:80], X[80:], y[80:]


class TestNaiveBaseline:
    """Tests for mean-prediction baseline."""

    def test_predicts_mean(self, train_test_data):
        X_tr, y_tr, X_te, _ = train_test_data
        model = NaiveBaseline().fit(X_tr, y_tr)
        preds = model.predict(X_te)
        np.testing.assert_allclose(preds, y_tr.mean())

    def test_fit_returns_self(self, train_test_data):
        X_tr, y_tr, _, _ = train_test_data
        model = NaiveBaseline()
        result = model.fit(X_tr, y_tr)
        assert result is model

    def test_output_length(self, train_test_data):
        X_tr, y_tr, X_te, _ = train_test_data
        model = NaiveBaseline().fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(X_te)


class TestRF:
    """Tests for Random Forest with Optuna."""

    def test_returns_correct_shape(self, train_test_data):
        X_tr, y_tr, X_te, y_te = train_test_data
        model, params, metrics = train_rf(
            X_tr,
            y_tr,
            X_te,
            y_te,
            n_trials=3,
            param_space={
                "n_estimators": (50, 100),
                "max_depth": [5, 10],
                "min_samples_leaf": (1, 3),
                "max_features": ["sqrt"],
            },
        )
        assert hasattr(model, "predict")
        assert isinstance(params, dict)
        assert isinstance(metrics, dict)

    def test_metrics_keys(self, train_test_data):
        X_tr, y_tr, X_te, y_te = train_test_data
        _, _, metrics = train_rf(
            X_tr,
            y_tr,
            X_te,
            y_te,
            n_trials=3,
            param_space={
                "n_estimators": (50, 100),
                "max_depth": [5],
                "min_samples_leaf": (1, 3),
                "max_features": ["sqrt"],
            },
        )
        assert set(metrics.keys()) == {"rmse", "mae", "maxae", "r2"}

    def test_rmse_reasonable(self, train_test_data):
        X_tr, y_tr, X_te, y_te = train_test_data
        _, _, metrics = train_rf(
            X_tr,
            y_tr,
            X_te,
            y_te,
            n_trials=5,
            param_space={
                "n_estimators": (50, 100),
                "max_depth": [10, None],
                "min_samples_leaf": (1, 3),
                "max_features": ["sqrt", 0.5],
            },
        )
        assert metrics["rmse"] < 0.1

    def test_default_params_works(self, train_test_data):
        X_tr, y_tr, X_te, y_te = train_test_data
        model, params, metrics = train_rf(X_tr, y_tr, X_te, y_te, n_trials=2)
        assert hasattr(model, "predict")


class TestSVR:
    """Tests for Support Vector Regression."""

    def test_returns_correct_shape(self, train_test_data):
        X_tr, y_tr, X_te, y_te = train_test_data
        model, params, metrics = train_svr(
            X_tr,
            y_tr,
            X_te,
            y_te,
            n_trials=3,
            param_space={
                "C": (0.1, 10),
                "epsilon": (0.001, 0.05),
                "gamma": ["scale", "auto"],
            },
        )
        assert hasattr(model, "predict")
        assert isinstance(params, dict)

    def test_rmse_reasonable(self, train_test_data):
        X_tr, y_tr, X_te, y_te = train_test_data
        _, _, metrics = train_svr(
            X_tr,
            y_tr,
            X_te,
            y_te,
            n_trials=5,
            param_space={
                "C": (0.1, 100),
                "epsilon": (0.001, 0.1),
                "gamma": ["scale", "auto"],
            },
        )
        assert metrics["rmse"] < 0.15

    def test_two_element_c_tuple(self):
        """After fix, param_space C should be 2-element tuple."""
        param_space = {"C": (0.01, 100), "epsilon": (0.001, 0.1), "gamma": ["scale"]}
        assert len(param_space["C"]) == 2


class TestGPR:
    """Tests for Gaussian Process Regression."""

    def test_returns_correct_shape(self, train_test_data):
        X_tr, y_tr, X_te, y_te = train_test_data
        model, metrics = train_gpr(
            X_tr,
            y_tr,
            X_te,
            y_te,
            max_train_samples=1000,
            n_restarts=1,
        )
        assert hasattr(model, "predict")
        assert isinstance(metrics, dict)

    def test_subsampling(self, train_test_data):
        X_tr, y_tr, X_te, y_te = train_test_data
        model, metrics = train_gpr(
            X_tr,
            y_tr,
            X_te,
            y_te,
            max_train_samples=30,
            n_restarts=1,
        )
        assert hasattr(model, "predict")

    def test_small_dataset(self, train_test_data):
        X_tr, y_tr, X_te, y_te = train_test_data
        model, metrics = train_gpr(
            X_tr,
            y_tr,
            X_te,
            y_te,
            max_train_samples=1000,
            n_restarts=1,
        )
        assert metrics["rmse"] < 0.2
