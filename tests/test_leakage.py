"""Leakage and data-integrity regression tests.

These tests pin the methodological guarantees of the remediated pipeline:

    1. capacity_fade_rate never contains the current row's label.
    2. Feature selection is fitted on training cells only.
    3. Inner hyperparameter tuning excludes no outer-test information.
    4. validate_cycles removes interruptions/storage runs but keeps
       genuine gradual degradation.
    5. Q_initial (median) is robust to outlier cycles.
    6. Physically impossible CE values are rejected.
    7. Phase analysis includes SOH-capped samples.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.error_analysis import phase_rmse
from src.features.assembly import fit_feature_selection
from src.features.energy import compute_coulombic_efficiency
from src.features.trend import compute_capacity_fade_rate
from src.models.rf_model import optimize_rf
from src.preprocessing.segmentation import validate_cycles
from src.preprocessing.soh import compute_q_initial


class TestFadeRateNoTargetLeakage:
    """The fade-rate feature must depend only on PAST labels."""

    def test_current_label_does_not_affect_own_feature(self):
        rng = np.random.RandomState(0)
        soh = np.linspace(1.0, 0.8, 30) + rng.randn(30) * 0.001
        base = compute_capacity_fade_rate(soh, window=10)
        perturbed = soh.copy()
        perturbed[20] += 0.05  # change ONLY row 20's label
        new = compute_capacity_fade_rate(perturbed, window=10)
        # Row 20's own feature must be identical; later rows may change.
        assert base[20] == pytest.approx(new[20])

    def test_warmup_is_nan_not_zero(self):
        soh = np.linspace(1.0, 0.9, 15)
        fade = compute_capacity_fade_rate(soh, window=10)
        # Only the first two rows lack two past labels.
        assert np.isnan(fade[:2]).all()
        assert np.isfinite(fade[2:]).all()
        assert (fade[2:] != 0).any()

    def test_trailing_window_only(self):
        soh = np.linspace(1.0, 0.9, 15)
        fade = compute_capacity_fade_rate(soh, window=3)
        for i in range(4, len(soh)):
            expected = (soh[i - 1] - soh[i - 4]) / 3
            assert fade[i] == pytest.approx(expected)


class TestSelectionFittedOnTrainOnly:
    """fit_feature_selection must be callable on any subset of cells."""

    def _make_df(self):
        rng = np.random.RandomState(42)
        records = []
        for cid, offset in [("c1", 0.0), ("c2", 0.01), ("c3", 0.02), ("c4", 0.03)]:
            for i in range(40):
                soh = 1.0 - 0.004 * i - offset
                records.append(
                    {
                        "cell_id": cid,
                        "cycle_number": i,
                        "soh": soh,
                        "f_energy": 50 * soh + rng.randn() * 0.5,
                        "f_energy2": 20 * soh + rng.randn() * 0.3,
                        "f_noise": rng.randn(),
                    }
                )
        return pd.DataFrame(records)

    def test_signal_feature_ranked_top_on_train_cells(self):
        df = self._make_df()
        train = df[df["cell_id"].isin(["c1", "c2", "c3"])]
        cols = fit_feature_selection(train, top_k=2)
        assert cols[0] == "f_energy"

    def test_selection_is_sensitive_to_train_rows(self):
        """Selection output must actually depend on the rows it is given:
        destroying a signal column's information must demote it."""
        df = self._make_df()
        train = df[df["cell_id"].isin(["c1", "c2", "c3"])].copy()
        cols_a = fit_feature_selection(train, top_k=2)
        assert cols_a[0] == "f_energy"

        corrupted = train.copy()
        # Replace the primary signal with pure noise across ALL training rows;
        # the secondary signal must take over the top spot.
        corrupted["f_energy"] = np.random.RandomState(3).rand(len(corrupted))
        cols_b = fit_feature_selection(corrupted, top_k=2)

        assert cols_b[0] == "f_energy2", (
            f"expected surviving signal to lead, got {cols_b} -> selection ignores its inputs"
        )

    def test_rf_objective_scores_against_validation_split(self):
        """optimize_rf's objective must evaluate candidate models on X_val
        (n_val samples), never on the tuning split. Verified with a spy on
        predict(): every scored prediction must have validation size."""
        import src.models.rf_model as rf_mod

        rng = np.random.RandomState(7)
        X = rng.randn(120, 4)
        y = X[:, 0] * 0.5 + rng.randn(120) * 0.01
        X_tr, y_tr = X[:80], y[:80]
        X_val, y_val = X[80:], y[80:]

        observed_sizes: list[int] = []
        original_predict = rf_mod.RandomForestRegressor.predict

        class SpyRF(rf_mod.RandomForestRegressor):
            def predict(self, X):  # noqa: N802 (sklearn API name)
                observed_sizes.append(len(X))
                return original_predict(self, X)

        original_cls = rf_mod.RandomForestRegressor
        rf_mod.RandomForestRegressor = SpyRF
        try:
            optimize_rf(X_tr, y_tr, X_val, y_val, n_trials=4)
        finally:
            rf_mod.RandomForestRegressor = original_cls

        # n_trials fits + n_trials objective predicts + final build_rf fit
        # (build_rf is constructed inside objective too). Every predict call
        # must be exactly the validation-set size.
        assert observed_sizes, "objective never evaluated a model"
        assert all(n == len(y_val) for n in observed_sizes), (
            f"predict() called with non-validation sizes: {sorted(set(observed_sizes))}"
        )


class TestInnerCellSplit:
    """The inner tuning split must hold out an entire cell."""

    def test_val_cell_excluded_from_tune(self):
        from src.models.train_classical import _inner_cell_split

        fold = {
            "train_df": pd.DataFrame(
                {
                    "cell_id": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
                    "feat": np.arange(30, dtype=float),
                }
            )
        }
        tr_idx, val_idx = _inner_cell_split(fold)
        assert set(fold["train_df"].iloc[val_idx]["cell_id"]) == {"c"}
        assert set(fold["train_df"].iloc[tr_idx]["cell_id"]) == {"a", "b"}


class TestCycleIntegrityFilters:
    """Interruptions removed everywhere; genuine degradation kept."""

    @staticmethod
    def _cell(caps):
        return {
            "cell_id": "T",
            "rated_capacity": 2.0,
            "cycles": [
                {"cycle_number": i + 1, "capacity": float(c), "type": "discharge"}
                for i, c in enumerate(caps)
            ],
        }

    def test_single_interruption_removed(self):
        caps = [2.0 - 0.001 * i for i in range(40)]
        caps[25] = 0.3
        out = validate_cycles(self._cell(caps), q_initial=2.0)
        cns = [c["cycle_number"] for c in out["cycles"]]
        assert 26 not in cns and len(cns) == 39

    def test_storage_gap_with_recovery_removed(self):
        caps = [2.0 - 0.0005 * i for i in range(200)]
        caps[100:150] = [0.2] * 50  # depressed block at indices 100-149
        out = validate_cycles(self._cell(caps), q_initial=2.0)
        cns = [c["cycle_number"] for c in out["cycles"]]
        assert all(not (101 <= cn <= 150) for cn in cns)

    def test_unrecovered_eol_fade_kept(self):
        caps = list(np.linspace(2.0, 1.0, 120))
        out = validate_cycles(self._cell(caps), q_initial=2.0)
        assert len(out["cycles"]) == 120


class TestQInitialRobustness:
    def test_median_ignores_window_outlier(self):
        caps_clean = [1.98, 1.99, 2.00, 2.01]
        cycles = [
            {"cycle_number": i + 3, "capacity": c, "type": "discharge"}
            for i, c in enumerate(caps_clean)
        ]
        poisoned = dict(cell_id="T", cycles=cycles + [{"cycle_number": 8, "capacity": 0.1, "type": "discharge"}])
        clean_cell = dict(cell_id="T", cycles=[c for c in poisoned["cycles"] if c["capacity"] > 1])
        q_poisoned = compute_q_initial(poisoned, (3, 10))
        q_clean = compute_q_initial(clean_cell, (3, 10))
        mean_poisoned = float(np.mean([c["capacity"] for c in poisoned["cycles"]]))
        # Median is essentially unaffected; the mean would shift by ~19%.
        assert q_poisoned == pytest.approx(q_clean, rel=0.01)
        assert abs(mean_poisoned - q_clean) / q_clean > 0.15


class TestCERejection:
    def test_impossible_ratio_nan(self):
        assert np.isnan(compute_coulombic_efficiency(1.0, 0.05))
        assert np.isnan(compute_coulombic_efficiency(2.0, 1.0))

    def test_plausible_ratio_kept(self):
        assert compute_coulombic_efficiency(0.97, 1.0) == pytest.approx(0.97)


class TestPhaseAnalysisIncludesCap:
    def test_soh_one_counted(self):
        y = np.array([1.0, 1.0, 0.85, 0.5])
        p = np.array([0.9, 1.0, 0.9, 0.6])
        res = phase_rmse(y, p, y)
        assert res["early"]["n_samples"] == 2
        assert res["mid"]["n_samples"] == 1
        assert res["end"]["n_samples"] == 1
