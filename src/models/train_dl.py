"""Deep learning training orchestrator.

Runs LSTM, 1D CNN and Transformer through cell-based Leave-One-Cell-Out
cross-validation with a leakage-safe protocol:

    1. Feature selection is fitted PER FOLD on training cells only.
    2. Hyperparameter selection (Optuna) and early stopping use an INNER
       cell split of the training cells; the outer test cell never
       influences selection.
    3. The final model per (fold, seed) is refit on ALL training rows
       for exactly the epoch count chosen during selection, then
       evaluated once on the held-out cell.
    4. Sequence datasets are built directly from cell-grouped frames, so
       windows never cross cell boundaries.
    5. Inference latency is benchmarked on the fitted model; fold wall
       time is reported separately as training time.

Every run is logged to MLflow with dataset/fold/model/seed tags; model
checkpoints are persisted under experiments/models/{dataset}/.
"""

import argparse
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.deployability import benchmark_inference_time
from src.evaluation.validation import cell_fold_splits, materialize_fold, save_fold_indices
from src.features.assembly import fit_feature_selection
from src.models.cnn import evaluate_cnn, optimize_cnn, train_cnn_final
from src.models.lstm import evaluate_lstm, optimize_lstm, train_lstm_final
from src.models.transformer import (
    evaluate_transformer,
    optimize_transformer,
    train_transformer_final,
)
from src.utils.config import Config
from src.utils.paths import project_root
from src.utils.seeding import get_device as device_manager
from src.utils.seeding import set_seed
from src.utils.tracking import init_tracking, log_metrics, log_params, start_run

logger = logging.getLogger(__name__)


def _scale_dfs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale features fitting the scaler on train only.

    Args:
        train_df: Fold-training DataFrame.
        test_df: Held-out DataFrame.
        feature_cols: Feature columns to scale.

    Returns:
        Tuple of (scaled_train_df, scaled_test_df).
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled


def _inner_split_by_cell(train_scaled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the alphabetically-last training cell for tuning.

    Args:
        train_scaled: Scaled fold-training DataFrame.

    Returns:
        Tuple of (inner_train_df, inner_val_df). If fewer than two
        cells exist, the val frame equals train (degenerate; selection
        then monitors training data — logged loudly).
    """
    cells = sorted(train_scaled["cell_id"].unique())
    if len(cells) < 2:
        logger.warning("Fewer than 2 training cells; inner split degenerates")
        return train_scaled, train_scaled
    val_cell = cells[-1]
    mask = train_scaled["cell_id"] == val_cell
    return train_scaled.loc[~mask], train_scaled.loc[mask]


def run_dl_pipeline(config_path: str = "config/default.yaml", dataset: str = "all") -> dict:
    """Run all DL models through leakage-safe LOOCV.

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

    suffix = f"_{dataset}"
    feature_path = root / config["data"]["features_dir"] / f"feature_matrix{suffix}.parquet"
    feature_df = pd.read_parquet(feature_path)
    logger.info("Loaded candidate matrix %s: %s", feature_path, feature_df.shape)

    experiments_dir = root / "experiments"
    models_dir = experiments_dir / "models" / dataset
    models_dir.mkdir(parents=True, exist_ok=True)

    dl_cfg = dict(config["models"]["dl"])
    window_size = config["models"]["sequence_window"]
    n_seeds = int(dl_cfg.get("n_seeds", 3))
    base_seed = int(config.get("seeding.base_seed", 42))
    n_inf_repeats = int(config.get("evaluation.deployability.n_inference_repeats", 200))
    device = device_manager()

    folds_splits = cell_fold_splits(feature_df)
    save_fold_indices(folds_splits, output_dir=experiments_dir, dataset=dataset)

    model_names = ["lstm", "cnn", "transformer"]
    skipped: list[str] = []
    results: dict[str, dict[str, list]] = {
        name: {
            "fold_metrics": [],
            "fold_times": [],
            "fold_inference": [],
            "fold_features": [],
            "fold_val_rmse": [],
        }
        for name in model_names
    }

    optimizers: dict[str, Any] = {
        "lstm": optimize_lstm,
        "cnn": optimize_cnn,
        "transformer": optimize_transformer,
    }
    final_trainers: dict[str, Any] = {
        "lstm": train_lstm_final,
        "cnn": train_cnn_final,
        "transformer": train_transformer_final,
    }
    evaluators: dict[str, Any] = {
        "lstm": evaluate_lstm,
        "cnn": evaluate_cnn,
        "transformer": evaluate_transformer,
    }

    def _coerce(val: Any) -> Any:
        """YAML 1.1 parses bare ``1e-4`` as a string; coerce numerics."""
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return val
        return val

    def _space(model_name: str) -> dict | None:
        cfg_model = config["models"]["dl"].get(model_name, {})
        ps = cfg_model.get("param_space", {})
        out: dict = {}
        for key, raw in ps.items():
            if isinstance(raw, list):
                vals = [_coerce(v) for v in raw]
                out[key] = (
                    tuple(vals) if key in ("dropout", "learning_rate") else vals
                )
            else:
                out[key] = _coerce(raw)
        return out or None

    def _dense_units(model_name: str) -> int | None:
        cfg_model = config["models"]["dl"].get(model_name, {})
        defaults: dict = cfg_model.get("default", {}) or {}
        value = defaults.get("dense_units")
        return int(value) if value is not None else None

    def _fallback_params(model_name: str) -> dict[str, Any]:
        """Sane middle-of-space defaults when Optuna selection fails."""
        default_lr = float(dl_cfg.get("learning_rate", 0.001))
        fallbacks = {
            "lstm": {
                "lstm_1_units": 64, "lstm_2_units": 32,
                "dropout": 0.25, "learning_rate": default_lr,
            },
            "cnn": {
                "filters_1": 32, "filters_2": 64, "filters_3": 128,
                "kernel_size": 3, "dropout": 0.25, "learning_rate": default_lr,
            },
            "transformer": {
                "d_model": 64, "n_heads": 4, "n_encoder_blocks": 2,
                "dropout": 0.2, "learning_rate": default_lr,
            },
        }
        params = dict(fallbacks[model_name])
        params["best_epoch"] = max(int(dl_cfg.get("max_epochs", 80)) // 4, 5)
        return params

    for split in folds_splits:
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

        train_scaled, test_scaled = _scale_dfs(fold["train_df"], fold["test_df"], selected_cols)
        inner_train_df, inner_val_df = _inner_split_by_cell(train_scaled)

        for model_name in model_names:
            cfg_model = config["models"]["dl"].get(model_name, {})
            n_trials = int(cfg_model.get("n_trials", 10))

            seed_metrics, seed_times, seed_inf, seed_val_rmse = [], [], [], []

            for seed_idx in range(n_seeds):
                seed = base_seed + seed_idx
                set_seed(seed)

                with start_run(
                    run_name=f"fold_{fold['fold']}_{model_name}_seed{seed}",
                    tags={
                        "dataset": dataset,
                        "fold": str(fold["fold"]),
                        "model": model_name,
                        "seed": str(seed),
                    },
                ):
                    t0 = time.perf_counter()
                    try:
                        kwargs = {}
                        dense_units = _dense_units(model_name)
                        if model_name in ("lstm", "cnn") and dense_units is not None:
                            kwargs["dense_units"] = int(dense_units)
                        params = optimizers[model_name](
                            inner_train_df,
                            inner_val_df,
                            selected_cols,
                            window_size,
                            dl_cfg,
                            device,
                            seed=seed,
                            n_trials=n_trials,
                            param_space=_space(model_name),
                            **kwargs,
                        )
                    except (ValueError, RuntimeError) as e:
                        logger.warning(
                            "%s fold %d seed %d: optimization failed (%s); using defaults",
                            model_name, fold["fold"], seed, e,
                        )
                        params = _fallback_params(model_name)

                    try:
                        final_kwargs = {}
                        if model_name in ("lstm", "cnn") and dense_units is not None:
                            final_kwargs["dense_units"] = int(dense_units)
                        model, _history = final_trainers[model_name](
                            train_scaled,
                            selected_cols,
                            window_size,
                            dl_cfg,
                            device,
                            seed,
                            params,
                            **final_kwargs,
                        )
                        y_true, y_pred, metrics = evaluators[model_name](
                            model, test_scaled, selected_cols, window_size,
                            int(dl_cfg.get("batch_size", 64)), device,
                        )
                    except ValueError as e:
                        logger.warning(
                            "%s fold %d seed %d: skipped (%s)", model_name, fold["fold"], seed, e
                        )
                        skipped.append(f"{model_name}/fold{fold['fold']}/seed{seed}")
                        continue

                    elapsed = time.perf_counter() - t0

                    # Real single-sample inference benchmark.
                    from src.models.dl_base import SOHDataset

                    test_ds = SOHDataset(test_scaled, selected_cols, window_size)
                    sample = test_ds[0][0].unsqueeze(0).numpy()
                    inf_stats = benchmark_inference_time(
                        model, sample, n_repeats=n_inf_repeats, is_torch=True, device=device
                    )

                    log_params({"model": model_name, "seed": seed, **{
                        k: v for k, v in params.items() if k not in ("best_epoch", "val_rmse")
                    }})
                    log_metrics(metrics)

                    ckpt_path = models_dir / f"{model_name}_fold{fold['fold']}_seed{seed}.pt"
                    import torch

                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "params": dict(params),
                            "feature_cols": list(selected_cols),
                            "window_size": window_size,
                            "input_dim": len(selected_cols),
                        },
                        ckpt_path,
                    )

                    seed_metrics.append(metrics)
                    seed_times.append(elapsed)
                    seed_inf.append(inf_stats)
                    seed_val_rmse.append(float(params.get("val_rmse", float("nan"))))
                    logger.info(
                        "%s seed=%d RMSE=%.6f R2=%.4f (%.1fs)",
                        model_name, seed, metrics["rmse"], metrics["r2"], elapsed,
                    )

            if seed_metrics:
                fold_agg = {
                    m: {
                        "mean": float(np.mean([x[m] for x in seed_metrics])),
                        "std": float(np.std([x[m] for x in seed_metrics])),
                    }
                    for m in ["rmse", "mae", "maxae", "r2"]
                }
                entry = results[model_name]
                entry["fold_metrics"].append(fold_agg)
                entry["fold_times"].append(float(np.mean(seed_times)))
                entry["fold_inference"].append(seed_inf)
                entry["fold_features"].append(list(selected_cols))
                entry["fold_val_rmse"].append(float(np.nanmean(seed_val_rmse)))

    summary = _aggregate(results, model_names)

    summary["_meta"] = {
        "dataset": dataset,
        "n_folds": len(folds_splits),
        "n_seeds": n_seeds,
        "window_size": window_size,
        "protocol": (
            "cell-based LOOCV; per-fold feature selection; inner-cell-split "
            "Optuna selection with early stopping; final refit for the selected "
            "epoch count on the full training fold with min-train-loss weight "
            "selection (train-only information)"
        ),
        "skipped_runs": skipped,
    }

    results_path = experiments_dir / f"dl_results{suffix}.yaml"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    logger.info("Results saved to %s", results_path)

    return summary


def _aggregate(results: dict, model_names: list[str]) -> dict:
    """Aggregate per-fold outcomes (each already seed-averaged) into summary."""
    summary: dict[str, dict] = {}
    for model_name in model_names:
        data = results[model_name]
        if not data["fold_metrics"]:
            continue

        agg: dict = {}
        fold_means: dict[str, list[float]] = {m: [] for m in ["rmse", "mae", "maxae", "r2"]}
        seed_stds: dict[str, list[float]] = {m: [] for m in ["rmse", "mae", "maxae", "r2"]}
        for fold_agg in data["fold_metrics"]:
            for m in fold_means:
                fold_means[m].append(fold_agg[m]["mean"])
                seed_stds[m].append(fold_agg[m]["std"])
        for m in fold_means:
            agg[f"{m}_mean"] = float(np.mean(fold_means[m]))
            agg[f"{m}_std"] = float(np.std(fold_means[m]))
            # Average within-fold across-seed variability, reported
            # separately so the fold-level std stays comparable to the
            # classical models' fold std.
            agg[f"{m}_seed_std_mean"] = float(np.mean(seed_stds[m]))

        agg["train_time_mean_s"] = float(np.mean(data["fold_times"]))
        agg["inference_time_ms_mean"] = float(
            np.mean([np.mean([s["mean_inference_ms"] for s in row]) for row in data["fold_inference"]])
        )
        agg["inference_time_ms_p95"] = float(
            np.mean([np.mean([s["p95_ms"] for s in row]) for row in data["fold_inference"]])
        )
        feat_counts: dict[str, int] = {}
        for cols in data.get("fold_features", []):
            for c in cols:
                feat_counts[c] = feat_counts.get(c, 0) + 1
        agg["selected_features_by_fold_count"] = dict(
            sorted(feat_counts.items(), key=lambda kv: -kv[1])
        )
        agg["per_fold_rmse"] = [float(v) for v in fold_means["rmse"]]
        agg["per_fold_r2"] = [float(v) for v in fold_means["r2"]]
        agg["per_fold_val_rmse"] = [float(v) for v in data.get("fold_val_rmse", [])]
        summary[model_name] = agg
    return summary


def main() -> None:
    """CLI entry point for DL training."""
    parser = argparse.ArgumentParser(description="Deep learning training orchestrator")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    parser.add_argument(
        "--dataset",
        choices=["nasa", "calce", "all"],
        default="all",
        help="Dataset to train on: nasa, calce, or all (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    summary = run_dl_pipeline(args.config, dataset=args.dataset)

    print("\n" + "=" * 60)
    print("DEEP LEARNING RESULTS SUMMARY")
    print("=" * 60)
    for model_name, metrics in summary.items():
        if model_name == "_meta":
            continue
        print(f"\n{model_name.upper()}:")
        print(f"  RMSE:  {metrics['rmse_mean']:.6f} ± {metrics.get('rmse_std', 0):.6f}")
        print(f"  R²:    {metrics['r2_mean']:.4f} ± {metrics.get('r2_std', 0):.4f}")
        print(f"  Inference: {metrics['inference_time_ms_mean']:.3f} ms")


if __name__ == "__main__":
    main()
