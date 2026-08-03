"""Deep learning training orchestrator.

Runs LSTM, 1D CNN, and Transformer through cell-based LOOCV with Optuna
hyperparameter search and multiple seeds, logging results to MLflow.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.validation import cell_based_loocv
from src.models.cnn import train_cnn
from src.models.dl_base import create_sequences
from src.models.lstm import train_lstm
from src.models.transformer import train_transformer
from src.utils.config import Config
from src.utils.seeding import get_device as device_manager
from src.utils.seeding import set_seed
from src.utils.tracking import init_tracking, log_metrics, log_params, start_run

logger = logging.getLogger(__name__)


def run_dl_pipeline(config_path: str = "config/default.yaml") -> dict:
    """Run all DL models through LOOCV.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of aggregated results per model.
    """
    config = Config.from_yaml(config_path)
    init_tracking(
        tracking_uri=config["tracking"]["tracking_uri"],
        experiment_name=config["tracking"]["experiment_name"],
    )

    feature_df = pd.read_parquet(Path(config["data"]["features_dir"]) / "feature_matrix.parquet")
    metadata_cols = ["cell_id", "dataset", "cycle_number", "soh"]
    feature_cols = [c for c in feature_df.columns if c not in metadata_cols]

    dl_cfg = config["models"]["dl"]
    window_size = config["models"]["sequence_window"]
    n_seeds = dl_cfg.get("n_seeds", 5)
    device = device_manager()

    folds = cell_based_loocv(feature_df, feature_cols)
    results = {
        "lstm": {"fold_metrics": [], "fold_times": []},
        "cnn": {"fold_metrics": [], "fold_times": []},
        "transformer": {"fold_metrics": [], "fold_times": []},
    }

    for fold in folds:
        fold_idx = fold["fold"]
        logger.info("=" * 60)
        logger.info("Fold %d (test cell: %s)", fold_idx, fold["test_cell"])
        logger.info("=" * 60)

        train_df = feature_df.loc[fold["train_indices"]]
        test_df = feature_df.loc[fold["test_indices"]]

        train_df_scaled, test_df_scaled, _ = _scale_feature_dfs(train_df, test_df, feature_cols)

        model_trainers = {
            "lstm": train_lstm,
            "cnn": train_cnn,
            "transformer": train_transformer,
        }

        for model_name, trainer in model_trainers.items():
            seed_metrics = []
            seed_times = []

            for seed_idx in range(n_seeds):
                seed = 42 + seed_idx

                set_seed(seed)
                X_train, y_train = create_sequences(train_df_scaled, feature_cols, window_size)
                X_test, y_test = create_sequences(test_df_scaled, feature_cols, window_size)

                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning("Fold %d seed %d: empty sequences, skipping", fold_idx, seed)
                    continue

                with start_run(
                    run_name=f"fold_{fold_idx}_{model_name}_seed{seed}",
                    tags={"fold": str(fold_idx), "model": model_name, "seed": str(seed)},
                ):
                    t0 = time.perf_counter()
                    model_cfg = dl_cfg.get(model_name, {})
                    n_trials = model_cfg.get("n_trials", 10)
                    best_params, metrics = trainer(
                        X_train, y_train, X_test, y_test,
                        config=dl_cfg,
                        device=device,
                        seed=seed,
                        n_trials=n_trials,
                    )
                    elapsed = time.perf_counter() - t0

                    log_params({"model": model_name, "seed": seed, **best_params})
                    log_metrics(metrics)

                    seed_metrics.append(metrics)
                    seed_times.append(elapsed)
                    logger.info(
                        "%s seed=%d RMSE=%.6f R2=%.4f (%.1fs)",
                        model_name, seed, metrics["rmse"], metrics["r2"], elapsed,
                    )

            if seed_metrics:
                agg = {}
                for metric in ["rmse", "mae", "maxae", "r2"]:
                    vals = [m[metric] for m in seed_metrics]
                    agg[f"{metric}_mean"] = float(np.mean(vals))
                    agg[f"{metric}_std"] = float(np.std(vals))
                results[model_name]["fold_metrics"].append(agg)
                results[model_name]["fold_times"].append(float(np.mean(seed_times)))

    summary = {}
    for model_name, data in results.items():
        if not data["fold_metrics"]:
            continue
        fold_metrics = data["fold_metrics"]
        agg = {}
        for metric in ["rmse_mean", "mae_mean", "maxae_mean", "r2_mean"]:
            base = metric.replace("_mean", "")
            vals = [m[metric] for m in fold_metrics]
            agg[metric] = float(np.mean(vals))
            agg[f"{base}_std"] = float(np.std(vals))
        agg["inference_time_mean_s"] = float(np.mean(data["fold_times"]))
        summary[model_name] = agg
        logger.info(
            "%s: RMSE=%.6f±%.6f R2=%.4f±%.4f",
            model_name, agg["rmse_mean"], agg.get("rmse_std", 0),
            agg["r2_mean"], agg.get("r2_std", 0),
        )

    results_path = Path("experiments") / "dl_results.yaml"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    logger.info("Results saved to %s", results_path)

    return summary


def _scale_feature_dfs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Scale feature columns in train/test DataFrames.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        feature_cols: Feature column names to scale.

    Returns:
        Tuple of (scaled_train_df, scaled_test_df, fitted_scaler).
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled, scaler


def main() -> None:
    """CLI entry point for DL training."""
    parser = argparse.ArgumentParser(description="Deep learning training orchestrator")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    summary = run_dl_pipeline(args.config)

    print("\n" + "=" * 60)
    print("DEEP LEARNING RESULTS SUMMARY")
    print("=" * 60)
    for model_name, metrics in summary.items():
        print(f"\n{model_name.upper()}:")
        print(f"  RMSE:  {metrics['rmse_mean']:.6f} ± {metrics.get('rmse_std', 0):.6f}")
        print(f"  MAE:   {metrics['mae_mean']:.6f} ± {metrics.get('mae_std', 0):.6f}")
        print(f"  MaxAE: {metrics['maxae_mean']:.6f} ± {metrics.get('maxae_std', 0):.6f}")
        print(f"  R²:    {metrics['r2_mean']:.4f} ± {metrics.get('r2_std', 0):.4f}")


if __name__ == "__main__":
    main()
