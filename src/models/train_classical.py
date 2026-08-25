"""Classical ML training orchestrator.

Runs naive baseline, Random Forest, SVR and GPR through cell-based
Leave-One-Cell-Out cross-validation with a leakage-safe protocol:

    1. Feature selection (correlation filter + RF importance) is fitted
       PER FOLD on the training cells only.
    2. Hyperparameter selection (Optuna) uses an INNER cell split of the
       training cells — the outer test cell never influences tuning.
    3. The final model per fold is refit on ALL training rows with the
       chosen parameters, then evaluated once on the held-out cell.
    4. Inference latency is benchmarked on the fitted model with
       single-sample predict() calls; fold wall time is reported
       separately as training time.

Every run is logged to MLflow with a dataset tag; models are persisted
to experiments/models/{dataset}/.
"""

import argparse
import logging
import time

import joblib
import numpy as np
import pandas as pd
import yaml

from src.evaluation.deployability import benchmark_inference_time
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.validation import (
    cell_fold_splits,
    materialize_fold,
    save_fold_indices,
    scale_features,
)
from src.features.assembly import fit_feature_selection
from src.models.baseline import NaiveBaseline
from src.models.gpr_model import build_gpr
from src.models.rf_model import build_rf, optimize_rf
from src.models.svr_model import build_svr, optimize_svr
from src.utils.config import Config
from src.utils.paths import project_root
from src.utils.tracking import init_tracking, log_metrics, log_params, start_run

logger = logging.getLogger(__name__)


def _inner_cell_split(fold: dict) -> tuple[np.ndarray, np.ndarray]:
    """Split a fold's training rows into inner-tune/inner-val BY CELL.

    The alphabetically-last training cell becomes the inner validation
    cell (deterministic choice); the remainder drives Optuna fitting.
    With fewer than two training cells the split degenerates to using
    all training rows for both (no valid split exists).

    Args:
        fold: Fold dict from materialize_fold (needs train_df).

    Returns:
        Tuple of (inner_train_indices, inner_val_indices) into the
        fold's train_df.
    """
    train_df: pd.DataFrame = fold["train_df"]
    cells = sorted(train_df["cell_id"].unique())
    if len(cells) < 2:
        idx = np.arange(len(train_df))
        return idx, idx

    val_cell = cells[-1]
    val_mask = train_df["cell_id"] == val_cell
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def run_classical_pipeline(config_path: str = "config/default.yaml", dataset: str = "all") -> dict:
    """Run all classical ML models through leakage-safe LOOCV.

    Args:
        config_path: Path to the YAML configuration file.
        dataset: Dataset to use: 'nasa', 'calce', or 'all'.

    Returns:
        Dictionary of aggregated results per model.
    """
    root = project_root()
    config = Config.from_yaml(str(root / config_path))
    init_tracking(
        tracking_uri=config["tracking"]["tracking_uri"],
        experiment_name=config["tracking"]["experiment_name"],
    )

    # Every dataset gets an explicit suffix (nasa/calce/all) so result
    # files can never silently overwrite each other.
    suffix = f"_{dataset}"
    feature_path = root / config["data"]["features_dir"] / f"feature_matrix{suffix}.parquet"
    feature_df = pd.read_parquet(feature_path)
    logger.info("Loaded candidate matrix %s: %s", feature_path, feature_df.shape)

    experiments_dir = root / "experiments"
    models_dir = experiments_dir / "models" / dataset
    models_dir.mkdir(parents=True, exist_ok=True)

    rf_cfg = config["models"]["classical"]["rf"]
    svr_cfg = config["models"]["classical"]["svr"]
    gpr_cfg = config["models"]["classical"]["gpr"]
    n_inf_repeats = config.get("evaluation.deployability.n_inference_repeats", 200)
    base_seed = int(config.get("seeding.base_seed", 42))

    folds_splits = cell_fold_splits(feature_df)
    save_fold_indices(folds_splits, output_dir=experiments_dir, dataset=dataset)

    results: dict[str, dict[str, list]] = {
        name: {"fold_metrics": [], "fold_val_metrics": [], "fold_train_times": [],
               "fold_inference": [], "fold_features": []}
        for name in ["naive", "rf", "svr", "gpr"]
    }

    for split in folds_splits:
        # ---- Per-fold feature selection (training rows only) ----------
        sel_train_df = feature_df.loc[split["train_indices"]]
        selected_cols = fit_feature_selection(
            sel_train_df,
            correlation_threshold=config.get("features.feature_selection.correlation_threshold", 0.95),
            top_k=config.get("features.feature_selection.top_k_features", 20),
        )
        fold = materialize_fold(feature_df, split, selected_cols)

        logger.info("=" * 60)
        logger.info("Fold %d (test cell: %s) | features=%d", fold["fold"], fold["test_cell"], len(selected_cols))
        logger.info("=" * 60)

        X_train, y_train = fold["X_train"], fold["y_train"]
        X_test, y_test = fold["X_test"], fold["y_test"]
        X_train_s, X_test_s, _scaler = scale_features(X_train, X_test)

        # Inner split (on scaled full-train arrays) for tuning.
        inner_tr_idx, inner_val_idx = _inner_cell_split(fold)
        Xin_tr, Yin_tr = X_train_s[inner_tr_idx], y_train[inner_tr_idx]
        Xin_val, Yin_val = X_train_s[inner_val_idx], y_train[inner_val_idx]
        if len(inner_val_idx) < len(y_train):
            logger.info(
                "Inner split: tune=%d rows / val=%d rows (val cell excluded from test)",
                len(Xin_tr), len(Xin_val),
            )

        common_tags = {"dataset": dataset}

        # ---------------- Naive baseline --------------------------------
        with start_run(run_name=f"fold_{fold['fold']}_naive",
                       tags={**common_tags, "fold": str(fold["fold"]), "model": "naive"}):
            t0 = time.perf_counter()
            baseline = NaiveBaseline().fit(X_train_s, y_train)
            train_time = time.perf_counter() - t0
            inf_stats = benchmark_inference_time(baseline, X_test_s, n_repeats=n_inf_repeats)
            y_pred = baseline.predict(X_test_s)
            metrics = compute_all_metrics(y_test, y_pred)
            log_params({"model": "naive"})
            log_metrics(metrics)
            _record(results["naive"], metrics, train_time, inf_stats, selected_cols, None)
            logger.info("Naive RMSE=%.6f R2=%.4f", metrics["rmse"], metrics["r2"])

        # ---------------- Random Forest ---------------------------------
        with start_run(run_name=f"fold_{fold['fold']}_rf",
                       tags={**common_tags, "fold": str(fold["fold"]), "model": "rf"}):
            t0 = time.perf_counter()
            best = optimize_rf(
                Xin_tr, Yin_tr, Xin_val, Yin_val,
                n_trials=rf_cfg["n_trials"],
                param_space={
                    "n_estimators": tuple(rf_cfg["param_space"]["n_estimators"]),
                    "max_depth": rf_cfg["param_space"]["max_depth"],
                    "min_samples_leaf": tuple(rf_cfg["param_space"]["min_samples_leaf"]),
                    "max_features": rf_cfg["param_space"]["max_features"],
                },
                seed=base_seed,
            )
            model = build_rf(best, seed=base_seed).fit(X_train_s, y_train)
            train_time = time.perf_counter() - t0
            inf_stats = benchmark_inference_time(model, X_test_s, n_repeats=n_inf_repeats)
            y_pred = model.predict(X_test_s)
            metrics = compute_all_metrics(y_test, y_pred)
            log_params({"model": "rf", **best})
            log_metrics(metrics)
            model_path = models_dir / f"rf_fold{fold['fold']}.joblib"
            joblib.dump({"model": model, "feature_cols": selected_cols}, model_path)
            _record(results["rf"], metrics, train_time, inf_stats, selected_cols, best)
            logger.info("RF RMSE=%.6f R2=%.4f", metrics["rmse"], metrics["r2"])

        # ---------------- SVR --------------------------------------------
        with start_run(run_name=f"fold_{fold['fold']}_svr",
                       tags={**common_tags, "fold": str(fold["fold"]), "model": "svr"}):
            t0 = time.perf_counter()
            best = optimize_svr(
                Xin_tr, Yin_tr, Xin_val, Yin_val,
                n_trials=svr_cfg["n_trials"],
                param_space={
                    "C": tuple(svr_cfg["param_space"]["C"]),
                    "epsilon": tuple(svr_cfg["param_space"]["epsilon"]),
                    "gamma": svr_cfg["param_space"]["gamma"],
                },
                seed=base_seed,
            )
            model = build_svr(best).fit(X_train_s, y_train)
            train_time = time.perf_counter() - t0
            inf_stats = benchmark_inference_time(model, X_test_s, n_repeats=n_inf_repeats)
            y_pred = model.predict(X_test_s)
            metrics = compute_all_metrics(y_test, y_pred)
            log_params({"model": "svr", **best})
            log_metrics(metrics)
            model_path = models_dir / f"svr_fold{fold['fold']}.joblib"
            joblib.dump({"model": model, "feature_cols": selected_cols}, model_path)
            _record(results["svr"], metrics, train_time, inf_stats, selected_cols, best)
            logger.info("SVR RMSE=%.6f R2=%.4f", metrics["rmse"], metrics["r2"])

        # ---------------- GPR (no hyperparameter search) ------------------
        with start_run(run_name=f"fold_{fold['fold']}_gpr",
                       tags={**common_tags, "fold": str(fold["fold"]), "model": "gpr"}):
            t0 = time.perf_counter()
            model = build_gpr(n_restarts=gpr_cfg["n_restarts_optimizer"], seed=base_seed)
            model.fit(X_train_s, y_train)
            train_time = time.perf_counter() - t0
            inf_stats = benchmark_inference_time(model, X_test_s, n_repeats=min(n_inf_repeats, 100))
            y_pred = model.predict(X_test_s)
            metrics = compute_all_metrics(y_test, y_pred)
            log_params({"model": "gpr", "n_train": len(X_train_s)})
            log_metrics(metrics)
            model_path = models_dir / f"gpr_fold{fold['fold']}.joblib"
            joblib.dump({"model": model, "feature_cols": selected_cols}, model_path)
            _record(results["gpr"], metrics, train_time, inf_stats, selected_cols, None)
            logger.info("GPR RMSE=%.6f R2=%.4f", metrics["rmse"], metrics["r2"])

    summary = _aggregate(results)
    summary["_meta"] = {
        "dataset": dataset,
        "n_folds": len(folds_splits),
        "protocol": "cell-based LOOCV; per-fold feature selection; "
        "inner-cell-split hyperparameter tuning; refit on full training fold",
        "inference_benchmark": f"single-sample predict() x{n_inf_repeats}",
    }

    results_path = experiments_dir / f"classical_results{suffix}.yaml"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    logger.info("Results saved to %s", results_path)

    return summary


def _record(entry: dict, metrics: dict, train_time: float, inf_stats: dict,
            selected_cols: list, best_params: dict | None) -> None:
    """Append one fold's outcomes to a model's result accumulator."""
    entry["fold_metrics"].append(metrics)
    entry["fold_train_times"].append(train_time)
    entry["fold_inference"].append(inf_stats)
    entry["fold_features"].append(list(selected_cols))


def _aggregate(results: dict) -> dict:
    """Aggregate per-fold outcomes into summary statistics."""
    summary = {}
    for model_name, data in results.items():
        agg: dict = {}
        for metric in ["rmse", "mae", "maxae", "r2"]:
            vals = [m[metric] for m in data["fold_metrics"]]
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals))
        agg["train_time_mean_s"] = float(np.mean(data["fold_train_times"]))
        agg["inference_time_ms_mean"] = float(
            np.mean([s["mean_inference_ms"] for s in data["fold_inference"]]
                    ) if data["fold_inference"] else float("nan")
        )
        agg["inference_time_ms_p95"] = float(
            np.mean([s["p95_ms"] for s in data["fold_inference"]]
                    ) if data["fold_inference"] else float("nan")
        )
        # Frequency of each feature across folds' selections (traceability).
        feat_counts: dict[str, int] = {}
        for cols in data["fold_features"]:
            for c in cols:
                feat_counts[c] = feat_counts.get(c, 0) + 1
        agg["selected_features_by_fold_count"] = dict(
            sorted(feat_counts.items(), key=lambda kv: -kv[1])
        )
        agg["per_fold_rmse"] = [float(m["rmse"]) for m in data["fold_metrics"]]
        agg["per_fold_r2"] = [float(m["r2"]) for m in data["fold_metrics"]]
        summary[model_name] = agg
    return summary


def main() -> None:
    """CLI entry point for classical ML training."""
    parser = argparse.ArgumentParser(description="Classical ML training orchestrator")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    parser.add_argument(
        "--dataset",
        choices=["nasa", "calce", "all"],
        default="all",
        help="Dataset to train on: nasa, calce, or all (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    summary = run_classical_pipeline(args.config, dataset=args.dataset)

    print("\n" + "=" * 60)
    print("CLASSICAL ML RESULTS SUMMARY")
    print("=" * 60)
    for model_name, metrics in summary.items():
        if model_name == "_meta":
            continue
        print(f"\n{model_name.upper()}:")
        print(f"  RMSE:  {metrics['rmse_mean']:.6f} ± {metrics['rmse_std']:.6f}")
        print(f"  MAE:   {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
        print(f"  MaxAE: {metrics['maxae_mean']:.6f} ± {metrics['maxae_std']:.6f}")
        print(f"  R²:    {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
        print(f"  Inference: {metrics['inference_time_ms_mean']:.3f} ms")


if __name__ == "__main__":
    main()
