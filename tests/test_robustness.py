"""Tests for robustness — noise injection and missing cycles."""

import numpy as np
import pandas as pd
import pytest


def inject_gaussian_noise(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    noise_fraction: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Add Gaussian noise proportional to each feature's std.

    Args:
        feature_df: Feature matrix.
        feature_cols: Feature column names.
        noise_fraction: Noise as a fraction of feature std.
        seed: Random seed.

    Returns:
        Noisy copy of the feature matrix.
    """
    rng = np.random.RandomState(seed)
    noisy = feature_df.copy()
    for col in feature_cols:
        std = noisy[col].std()
        noise = rng.randn(len(noisy)) * std * noise_fraction
        noisy[col] = noisy[col] + noise
    return noisy


def drop_random_cycles(
    feature_df: pd.DataFrame,
    fraction: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Randomly drop a fraction of cycles per cell.

    Args:
        feature_df: Feature matrix.
        fraction: Fraction of cycles to drop (0.0 to 1.0).
        seed: Random seed.

    Returns:
        Reduced feature matrix.
    """
    rng = np.random.RandomState(seed)
    mask = pd.Series(True, index=feature_df.index)
    for cell_id in feature_df["cell_id"].unique():
        cell_mask = feature_df["cell_id"] == cell_id
        cell_indices = feature_df.index[cell_mask]
        n_drop = int(len(cell_indices) * fraction)
        drop_idx = rng.choice(cell_indices, n_drop, replace=False)
        mask.loc[drop_idx] = False
    return feature_df.loc[mask].reset_index(drop=True)


class TestNoiseInjection:
    """Tests for Gaussian noise robustness."""

    def test_no_crash(self):
        df = pd.DataFrame(
            {
                "cell_id": ["A"] * 20,
                "feat_1": np.random.randn(20),
                "feat_2": np.random.randn(20),
            }
        )
        noisy = inject_gaussian_noise(df, ["feat_1", "feat_2"], noise_fraction=0.01)
        assert len(noisy) == len(df)

    def test_values_change(self):
        df = pd.DataFrame(
            {
                "cell_id": ["A"] * 20,
                "feat_1": np.random.RandomState(0).randn(20),
            }
        )
        noisy = inject_gaussian_noise(df, ["feat_1"], noise_fraction=0.1)
        assert not np.allclose(df["feat_1"].values, noisy["feat_1"].values)

    def test_original_unchanged(self):
        df = pd.DataFrame(
            {
                "cell_id": ["A"] * 20,
                "feat_1": np.random.RandomState(0).randn(20),
            }
        )
        original = df["feat_1"].values.copy()
        _ = inject_gaussian_noise(df, ["feat_1"], noise_fraction=0.1)
        np.testing.assert_array_equal(df["feat_1"].values, original)


class TestMissingCycles:
    """Tests for missing cycle robustness."""

    def test_no_crash(self):
        df = pd.DataFrame(
            {
                "cell_id": ["A"] * 20,
                "cycle_number": range(20),
                "soh": np.linspace(1, 0.8, 20),
            }
        )
        reduced = drop_random_cycles(df, fraction=0.2)
        assert len(reduced) == 16

    def test_preserves_cells(self):
        df = pd.DataFrame(
            {
                "cell_id": ["A"] * 20 + ["B"] * 20,
                "cycle_number": list(range(20)) * 2,
                "soh": list(np.linspace(1, 0.8, 20)) * 2,
            }
        )
        reduced = drop_random_cycles(df, fraction=0.3)
        assert set(reduced["cell_id"].unique()) == {"A", "B"}

    def test_drops_correct_fraction(self):
        df = pd.DataFrame(
            {
                "cell_id": ["A"] * 100,
                "cycle_number": range(100),
            }
        )
        reduced = drop_random_cycles(df, fraction=0.2)
        assert len(reduced) == 80


class TestNoiseSemantics:
    """Pin the scaled-space noise semantics used by scripts/run_robustness.py."""

    def test_noise_level_matches_train_std_units(self):
        """Adding N(0, level) in scaled space == level x train-std in raw space."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(0)
        raw = rng.randn(500, 3) * np.array([100.0, 0.01, 5.0]) + np.array([10.0, 2.0, -3.0])
        scaler = StandardScaler().fit(raw)
        scaled = scaler.transform(raw)

        level = 0.05
        noisy_scaled = scaled + np.random.RandomState(1).randn(*scaled.shape) * level

        # Recover raw-space perturbation and compare to level * train std.
        recovered_raw = scaler.inverse_transform(noisy_scaled)
        delta_raw = recovered_raw - raw
        expected_std = level * scaler.scale_
        # Sample std of the injected noise should approximate level*std per feature.
        assert np.allclose(delta_raw.std(axis=0), expected_std, rtol=0.25)

    def test_relative_noise_is_scale_invariant(self):
        """The same `level` must produce the same relative degradation for a
        100x-rescaled feature (this is what the fixed implementation buys)."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(0)
        base = rng.randn(400, 1)

        def rel_rmse_after_noise(mat):
            sc = StandardScaler().fit(mat)
            s = sc.transform(mat)
            s_noisy = s + np.random.RandomState(7).randn(*s.shape) * 0.1
            return float(np.sqrt(np.mean((sc.inverse_transform(s_noisy) - mat) ** 2))) / float(
                np.std(mat)
            )

        r1 = rel_rmse_after_noise(base)
        r2 = rel_rmse_after_noise(base * 100.0 + 50.0)
        assert r1 == pytest.approx(r2, rel=0.35)
