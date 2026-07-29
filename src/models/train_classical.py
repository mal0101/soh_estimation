"""Classical ML training orchestrator.

Runs all three classical models (RF, SVR, GPR) plus naive baseline
through cell-based LOOCV, logging results to MLflow.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.validation import cell_based_loocv, save_fold_indices, scale_features
from src.models.baseline import NaiveBaseline
from src.models.gpr_model import train_gpr
from src.models.rf_model import train_rf
from src.models.svr_model import train_svr
from src.utils.config import Config
from src.utils.tracking import init_tracking, log_metrics, log_params, start_run

logger = logging.getLogger(__name__)


def run_classical_pipeline(config_path: str = "config/default.yaml") -> dict:
    """Run all classical ML models through LOOCV.

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

    folds = cell_based_loocv(feature_df, feature_cols)
    save_fold_indices(folds, output_dir="experiments")

    results = {
        "naive": {"fold_metrics": [], "fold_times": []},
        "rf": {"fold_metrics": [], "fold_times": []},
        "svr": {"fold_metrics": [], "fold_times": []},
        "gpr": {"fold_metrics": [], "fold_times": []},
    }

    for fold in folds:
        fold_idx = fold["fold"]
        X_train, y_train = fold["X_train"], fold["y_train"]
        X_test, y_test = fold["X_test"], fold["y_test"]

        X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

        logger.info("=" * 60)
        logger.info("Fold %d (test cell: %s)", fold_idx, fold["test_cell"])
        logger.info("=" * 60)

        with start_run(run_name=f"fold_{fold_idx}_naive", tags={"fold": str(fold_idx), "model": "naive"}):
            t0 = time.perf_counter()
            baseline = NaiveBaseline().fit(X_train_s, y_train)
            y_pred = baseline.predict(X_test_s)
            elapsed = time.perf_counter() - t0
            metrics = compute_all_metrics(y_test, y_pred)
            log_params({"model": "naive"})
            log_metrics(metrics)
            results["naive"]["fold_metrics"].append(metrics)
            results["naive"]["fold_times"].append(elapsed)
            logger.info("Naive RMSE=%.6f R2=%.4f (%.1fs)", metrics["rmse"], metrics["r2"], elapsed)

        with start_run(run_name=f"fold_{fold_idx}_rf", tags={"fold": str(fold_idx), "model": "rf"}):
            t0 = time.perf_counter()
            rf_cfg = config["models"]["classical"]["rf"]
            model, best_params, metrics = train_rf(
                X_train_s, y_train, X_test_s, y_test,
                n_trials=rf_cfg["n_trials"],
                param_space={
                    "n_estimators": tuple(rf_cfg["param_space"]["n_estimators"]),
                    "max_depth": rf_cfg["param_space"]["max_depth"],
                    "min_samples_leaf": tuple(rf_cfg["param_space"]["min_samples_leaf"]),
                    "max_features": rf_cfg["param_space"]["max_features"],
                },
            )
            elapsed = time.perf_counter() - t0
            log_params({"model": "rf", **best_params})
            log_metrics(metrics)
            results["rf"]["fold_metrics"].append(metrics)
            results["rf"]["fold_times"].append(elapsed)
            logger.info("RF RMSE=%.6f R2=%.4f (%.1fs)", metrics["rmse"], metrics["r2"], elapsed)

        with start_run(run_name=f"fold_{fold_idx}_svr", tags={"fold": str(fold_idx), "model": "svr"}):
            t0 = time.perf_counter()
            svr_cfg = config["models"]["classical"]["svr"]
            model, best_params, metrics = train_svr(
                X_train_s, y_train, X_test_s, y_test,
                n_trials=svr_cfg["n_trials"],
                param_space={
                    "C": tuple(svr_cfg["param_space"]["C"]),
                    "epsilon": tuple(svr_cfg["param_space"]["epsilon"]),
                    "gamma": svr_cfg["param_space"]["gamma"],
                },
            )
            elapsed = time.perf_counter() - t0
            log_params({"model": "svr", **best_params})
            log_metrics(metrics)
            results["svr"]["fold_metrics"].append(metrics)
            results["svr"]["fold_times"].append(elapsed)
            logger.info("SVR RMSE=%.6f R2=%.4f (%.1fs)", metrics["rmse"], metrics["r2"], elapsed)

        with start_run(run_name=f"fold_{fold_idx}_gpr", tags={"fold": str(fold_idx), "model": "gpr"}):
            t0 = time.perf_counter()
            gpr_cfg = config["models"]["classical"]["gpr"]
            model, metrics = train_gpr(
                X_train_s, y_train, X_test_s, y_test,
                max_train_samples=gpr_cfg["max_train_samples"],
                n_restarts=gpr_cfg["n_restarts_optimizer"],
            )
            elapsed = time.perf_counter() - t0
            log_params({"model": "gpr", "n_train": gpr_cfg["max_train_samples"]})
            log_metrics(metrics)
            results["gpr"]["fold_metrics"].append(metrics)
            results["gpr"]["fold_times"].append(elapsed)
            logger.info("GPR RMSE=%.6f R2=%.4f (%.1fs)", metrics["rmse"], metrics["r2"], elapsed)

    summary = {}
    for model_name, data in results.items():
        fold_metrics = data["fold_metrics"]
        agg = {}
        for metric in ["rmse", "mae", "maxae", "r2"]:
            vals = [m[metric] for m in fold_metrics]
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals))
        agg["inference_time_mean_s"] = float(np.mean(data["fold_times"]))
        summary[model_name] = agg
        logger.info("%s: RMSE=%.6f±%.6f R2=%.4f±%.4f",
                     model_name, agg["rmse_mean"], agg["rmse_std"], agg["r2_mean"], agg["r2_std"])

    results_path = Path("experiments") / "classical_results.yaml"
    with open(results_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    logger.info("Results saved to %s", results_path)

    return summary


def main() -> None:
    """CLI entry point for classical ML training."""
    parser = argparse.ArgumentParser(description="Classical ML training orchestrator")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    summary = run_classical_pipeline(args.config)

    print("\n" + "=" * 60)
    print("CLASSICAL ML RESULTS SUMMARY")
    print("=" * 60)
    for model_name, metrics in summary.items():
        print(f"\n{model_name.upper()}:")
        print(f"  RMSE:  {metrics['rmse_mean']:.6f} ± {metrics['rmse_std']:.6f}")
        print(f"  MAE:   {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
        print(f"  MaxAE: {metrics['maxae_mean']:.6f} ± {metrics['maxae_std']:.6f}")
        print(f"  R²:    {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")


if __name__ == "__main__":
    main()
