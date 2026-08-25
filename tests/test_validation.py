"""Tests for src/evaluation/validation.py."""

import json

import numpy as np

from src.evaluation.validation import (
    cell_based_loocv,
    cell_fold_splits,
    materialize_fold,
    save_fold_indices,
    scale_features,
)


class TestCellBasedLOOCV:
    """Tests for cell-based leave-one-cell-out cross-validation."""

    def test_fold_count_matches_cells(self, feature_df_small):
        folds = cell_based_loocv(feature_df_small, ["feat_1", "feat_2"])
        assert len(folds) == feature_df_small["cell_id"].nunique()

    def test_no_overlap(self, feature_df_small):
        folds = cell_based_loocv(feature_df_small, ["feat_1", "feat_2"])
        for f in folds:
            train_set = set(f["train_indices"])
            test_set = set(f["test_indices"])
            assert train_set.isdisjoint(test_set)

    def test_all_data_covered(self, feature_df_small):
        folds = cell_based_loocv(feature_df_small, ["feat_1", "feat_2"])
        all_indices = set(range(len(feature_df_small)))
        covered = set()
        for f in folds:
            covered.update(f["train_indices"])
            covered.update(f["test_indices"])
        assert covered == all_indices

    def test_holdout_cell(self, feature_df_small):
        folds = cell_based_loocv(feature_df_small, ["feat_1", "feat_2"])
        for f in folds:
            test_df = feature_df_small.loc[f["test_indices"]]
            assert (test_df["cell_id"] == f["test_cell"]).all()

    def test_with_nans(self, feature_df_small):
        df = feature_df_small.copy()
        rng = np.random.RandomState(99)
        nan_mask = rng.rand(len(df)) < 0.1
        df.loc[nan_mask, "feat_3"] = np.nan
        folds = cell_based_loocv(df, ["feat_1", "feat_2", "feat_3"])
        for f in folds:
            assert len(f["X_train"]) > 0
            assert len(f["X_test"]) > 0
            assert not np.isnan(f["X_train"]).any()
            assert not np.isnan(f["X_test"]).any()

    def test_correct_shapes(self, feature_df_small):
        feature_cols = ["feat_1", "feat_2", "feat_3"]
        folds = cell_based_loocv(feature_df_small, feature_cols)
        for f in folds:
            assert f["X_train"].shape[1] == len(feature_cols)
            assert f["X_test"].shape[1] == len(feature_cols)
            assert len(f["X_train"]) == len(f["y_train"])
            assert len(f["X_test"]) == len(f["y_test"])


class TestScaleFeatures:
    """Tests for per-fold feature scaling."""

    def test_train_zero_mean(self, numpy_data):
        X, _ = numpy_data
        X_train, X_test = X[:15], X[15:]
        X_tr_s, _, _ = scale_features(X_train, X_test)
        means = X_tr_s.mean(axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_train_unit_variance(self, numpy_data):
        X, _ = numpy_data
        X_train, X_test = X[:15], X[15:]
        X_tr_s, _, _ = scale_features(X_train, X_test)
        stds = X_tr_s.std(axis=0, ddof=0)
        np.testing.assert_allclose(stds, 1.0, atol=1e-10)

    def test_no_leakage(self, numpy_data):
        X, _ = numpy_data
        X_train, X_test = X[:15], X[15:]
        _, X_te_s, scaler = scale_features(X_train, X_test)
        assert not np.allclose(X_te_s.mean(axis=0), 0.0, atol=0.1)

    def test_returns_scaler(self, numpy_data):
        X, _ = numpy_data
        _, _, scaler = scale_features(X[:15], X[15:])
        assert hasattr(scaler, "transform")


class TestSaveFoldIndices:
    """Tests for fold index JSON serialization."""

    def test_roundtrip(self, feature_df_small, tmp_path):
        folds = cell_based_loocv(feature_df_small, ["feat_1", "feat_2"])
        filepath = save_fold_indices(folds, output_dir=tmp_path)
        assert filepath.exists()
        assert filepath.name == "fold_indices_all.json"

        with open(filepath) as f:
            loaded = json.load(f)

        assert loaded["dataset"] == "all"
        fold_list = loaded["folds"]
        assert len(fold_list) == len(folds)
        assert fold_list[0]["test_cell"] == folds[0]["test_cell"]
        assert fold_list[0]["train_indices"] == folds[0]["train_indices"].tolist()

    def test_suffix_per_dataset(self, feature_df_small, tmp_path):
        folds = cell_fold_splits(feature_df_small)
        p_all = save_fold_indices(folds, output_dir=tmp_path, dataset="all")
        p_nasa = save_fold_indices(folds, output_dir=tmp_path, dataset="nasa")
        assert p_all.name == "fold_indices_all.json"
        assert p_nasa.name == "fold_indices_nasa.json"


class TestCellFoldSplits:
    """Tests for feature-agnostic LOOCV split construction."""

    def test_disjoint_and_complete(self, feature_df_small):
        splits = cell_fold_splits(feature_df_small)
        assert len(splits) == feature_df_small["cell_id"].nunique()
        all_idx = set(feature_df_small.index)
        for s in splits:
            assert not set(s["train_indices"]) & set(s["test_indices"])
            assert set(s["train_indices"]) | set(s["test_indices"]) == all_idx

    def test_materialize_fold_dropna(self, feature_df_small):
        df = feature_df_small.copy()
        df.loc[df.index[:3], "feat_1"] = np.nan
        splits = cell_fold_splits(df)
        # Choose the split where the NaN rows are in train.
        s = splits[1]
        fold = materialize_fold(df, s, ["feat_1", "feat_2"])
        assert len(fold["X_train"]) == len(fold["train_df"])
        assert not np.isnan(fold["X_train"]).any()
        assert not np.isnan(fold["X_test"]).any()
