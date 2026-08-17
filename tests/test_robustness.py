"""Tests for robustness — noise injection and missing cycles."""

import numpy as np
import pandas as pd


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
