"""Feature matrix builder CLI.

Loads preprocessed cells and SOH labels, computes all candidate
features, and saves the FULL candidate feature matrix (no supervised
selection applied). Feature selection is performed per cross-validation
fold inside the training scripts via ``fit_feature_selection`` — doing
it here on the full dataset would leak test-cell label information into
the chosen feature set.

Usage:
    python -m src.features.build_features --config config/default.yaml --dataset nasa
"""

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.features.assembly import _METADATA_COLS, build_feature_matrix, save_feature_matrix

logger = logging.getLogger(__name__)

# Helper columns produced by build_feature_matrix that must NEVER be
# treated as model features. discharge_capacity in particular is the
# raw label before SOH normalisation; rated_capacity/cutoff_voltage/
# ambient_temperature are constants or protocol metadata.
_EXCLUDED_COLS = [
    "rated_capacity",
    "cutoff_voltage",
    "ambient_temperature",
    "discharge_capacity",
]


def run_build_features(config_path: str = "config/default.yaml", dataset: str = "all") -> None:
    """Build and save the full candidate feature matrix for a dataset.

    Args:
        config_path: Path to the YAML configuration file (anchored to the
            project root when relative).
        dataset: Which dataset to process: 'nasa', 'calce', or 'all'.
    """
    from src.utils.config import Config
    from src.utils.paths import project_root

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = project_root() / config_path
    config = Config.from_yaml(str(cfg_path))
    processed_dir = Path(config.get("data.processed_dir", "data/processed"))
    features_dir = config.get("data.features_dir", "data/features")
    min_peak_prominence = config.get("features.ica.min_peak_prominence", 0.01)

    suffix = f"_{dataset}"

    pkl_path = processed_dir / f"processed_cells{suffix}.pkl"
    labels_path = processed_dir / f"soh_labels{suffix}.parquet"

    logger.info("Loading processed cells from %s", pkl_path)
    with open(pkl_path, "rb") as f:
        cells = pickle.load(f)
    soh_df = pd.read_parquet(labels_path)
    logger.info("Loaded %d cells, %d SOH labels", len(cells), len(soh_df))

    feature_df = build_feature_matrix(
        cells, soh_df, min_peak_prominence=min_peak_prominence
    )

    drop_cols = [c for c in _EXCLUDED_COLS if c in feature_df.columns]
    feature_df = feature_df.drop(columns=drop_cols)

    feature_cols = [c for c in feature_df.columns if c not in _METADATA_COLS]
    n_nan = int(feature_df[feature_cols].isna().any(axis=1).sum())
    logger.info(
        "Candidate matrix: %d rows x %d features (%d rows contain NaN, "
        "dropped per-fold during training)",
        len(feature_df),
        len(feature_cols),
        n_nan,
    )

    save_feature_matrix(feature_df, feature_cols, output_dir=features_dir, dataset=dataset)


def main() -> None:
    """CLI entry point for feature matrix building."""
    parser = argparse.ArgumentParser(description="Build the full feature matrix")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nasa", "calce", "all"],
        default="all",
        help="Dataset to process: nasa, calce, or all (default: all)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    run_build_features(args.config, dataset=args.dataset)


if __name__ == "__main__":
    main()
