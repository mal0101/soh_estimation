"""Robustness analysis: sensor noise and missing-cycle degradation.

Uses the persisted per-fold models (experiments/models/{dataset}/*.joblib)
and the recorded fold splits to evaluate classical models under:

    1. Additive Gaussian noise on test features at fractions of each
       feature's training-set std (0.5%, 1%, 2%).
    2. Random missing cycles: evaluating only on a random subset of the
       test cell's cycles (10/20/30% removed), reflecting gappy telemetry.

Outputs experiments/robustness_{dataset}.yaml.

Usage:
    python scripts/run_robustness.py --dataset nasa
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_all_metrics  # noqa: E402

NOISE_LEVELS = [0.005, 0.01, 0.02]
MISSING_FRACTIONS = [0.1, 0.2, 0.3]


def load_fold_data(dataset: str) -> tuple[pd.DataFrame, list[dict]]:
    suffix = f"_{dataset}"
    fm_path = ROOT / "data" / "features" / f"feature_matrix{suffix}.parquet"
    folds_path = ROOT / "experiments" / f"fold_indices{suffix}.json"
    df = pd.read_parquet(fm_path)
    payload = json.loads(folds_path.read_text())
    return df, payload["folds"]


def run_noise(df: pd.DataFrame, folds: list[dict], dataset: str, rng_seed: int = 42) -> dict:
    results: dict[str, dict[str, dict]] = {m: {} for m in ["rf", "svr", "gpr"]}
    for fold in folds:
        fold_idx = fold["fold"]
        train_df = df.loc[fold["train_indices"]].dropna(subset=["soh"])
        # Feature columns for this fold come from the saved artifact.
        for model_name in ["rf", "svr", "gpr"]:
            path = ROOT / "experiments" / "models" / dataset / f"{model_name}_fold{fold_idx}.joblib"
            if not path.exists():
                continue
            art = joblib.load(path)
            model = art["model"]
            cols = art["feature_cols"]

            tr = train_df.dropna(subset=cols)
            te = df.loc[fold["test_indices"]].dropna(subset=cols + ["soh"])
            X_te_raw = te[cols].values
            y_te = te["soh"].values

            # Scale exactly as the pipeline did: fit scaler on train.
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(tr[cols].values)
            X_te_s = scaler.transform(X_te_raw)

            for level in NOISE_LEVELS:
                key = str(level)
                rng = np.random.RandomState(rng_seed + fold_idx)
                sigma = level * scaler.scale_  # fraction of train std (scaled space ~1)
                Xn = X_te_s + rng.randn(*X_te_s.shape) * sigma
                pred = model.predict(Xn)
                m = compute_all_metrics(y_te, pred)
                if key not in results[model_name]:
                    results[model_name][key] = {"rmse": [], "mae": []}
                results[model_name][key]["rmse"].append(m["rmse"])
                results[model_name][key]["mae"].append(m["mae"])

    out = {}
    for model_name, levels in results.items():
        out[model_name] = {}
        for level, acc in levels.items():
            out[model_name][level] = {
                k: float(np.mean(v)) for k, v in acc.items() if v
            }
    return out


def run_missing_cycles(
    df: pd.DataFrame, folds: list[dict], dataset: str, rng_seed: int = 42
) -> dict:
    results: dict[str, dict[str, dict]] = {m: {} for m in ["rf", "svr", "gpr"]}
    for fold in folds:
        fold_idx = fold["fold"]
        train_df = df.loc[fold["train_indices"]].dropna(subset=["soh"])
        for model_name in ["rf", "svr", "gpr"]:
            path = ROOT / "experiments" / "models" / dataset / f"{model_name}_fold{fold_idx}.joblib"
            if not path.exists():
                continue
            art = joblib.load(path)
            model, cols = art["model"], art["feature_cols"]

            tr = train_df.dropna(subset=cols)
            te = df.loc[fold["test_indices"]].dropna(subset=cols + ["soh"])

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(tr[cols].values)

            for frac in MISSING_FRACTIONS:
                key = str(frac)
                rng = np.random.RandomState(rng_seed + fold_idx)
                keep_mask = rng.rand(len(te)) >= frac
                if keep_mask.sum() < 5:
                    continue
                sub = te.iloc[np.where(keep_mask)[0]]
                Xs = scaler.transform(sub[cols].values)
                pred = model.predict(Xs)
                m = compute_all_metrics(sub["soh"].values, pred)
                if key not in results[model_name]:
                    results[model_name][key] = {"rmse": [], "mae": []}
                results[model_name][key]["rmse"].append(m["rmse"])
                results[model_name][key]["mae"].append(m["mae"])

    out = {}
    for model_name, fracs in results.items():
        out[model_name] = {}
        for frac, acc in fracs.items():
            out[model_name][frac] = {
                k: float(np.mean(v)) for k, v in acc.items() if v
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness analysis")
    parser.add_argument("--dataset", choices=["nasa", "calce", "all"], default="all")
    args = parser.parse_args()

    df, folds = load_fold_data(args.dataset)
    print(f"Loaded {len(df)} rows, {len(folds)} folds")

    noise = run_noise(df, folds, args.dataset)
    missing = run_missing_cycles(df, folds, args.dataset)

    out_path = ROOT / "experiments" / f"robustness_{args.dataset}.yaml"
    with open(out_path, "w") as fh:
        yaml.dump({"noise": noise, "missing_cycles": missing}, fh, default_flow_style=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
