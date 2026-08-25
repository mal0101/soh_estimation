"""Analysis extras: SHAP importance, GPR uncertainty calibration, deployability.

Produces, per dataset (where artifacts exist):
    - experiments/analysis/shap_importance_{dataset}.yaml  (global RF SHAP)
    - experiments/analysis/gpr_calibration_{dataset}.yaml  (coverage stats)
    - experiments/deployability_{dataset}.yaml             (latency + size vs targets)

Usage:
    python scripts/run_analysis_extras.py --dataset nasa
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

from src.evaluation.deployability import model_size_mb  # noqa: E402


def load_fold_data(dataset: str) -> tuple[pd.DataFrame, list[dict]]:
    suffix = f"_{dataset}"
    df = pd.read_parquet(ROOT / "data" / "features" / f"feature_matrix{suffix}.parquet")
    folds = json.loads(
        (ROOT / "experiments" / f"fold_indices{suffix}.json").read_text()
    )["folds"]
    return df, folds


def shap_analysis(df: pd.DataFrame, folds: list[dict], dataset: str) -> dict | None:
    """Global RF feature importance via SHAP on the first fold's model."""
    import shap

    fold = folds[0]
    path = ROOT / "experiments" / "models" / dataset / "rf_fold0.joblib"
    if not path.exists():
        return None
    art = joblib.load(path)
    model, cols = art["model"], art["feature_cols"]

    train_df = df.loc[fold["train_indices"]].dropna(subset=cols + ["soh"])
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_df[cols].values)
    X = scaler.transform(train_df[cols].values)[:500]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)[::-1]
    return {
        "model": "rf_fold0",
        "n_background": int(len(X)),
        "importance": {cols[i]: float(importance[i]) for i in order},
    }


def gpr_calibration(df: pd.DataFrame, folds: list[dict], dataset: str) -> dict | None:
    """Empirical coverage of GPR's predictive intervals."""
    out: dict[str, list] = {}
    for fold in folds:
        path = ROOT / "experiments" / "models" / dataset / f"gpr_fold{fold['fold']}.joblib"
        if not path.exists():
            continue
        art = joblib.load(path)
        model, cols = art["model"], art["feature_cols"]

        train_df = df.loc[fold["train_indices"]].dropna(subset=cols + ["soh"])
        te = df.loc[fold["test_indices"]].dropna(subset=cols + ["soh"])

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(train_df[cols].values)
        X_te = scaler.transform(te[cols].values)
        y_te = te["soh"].values

        mean, std = model.predict(X_te, return_std=True)
        for level_name, z in [("68", 1.0), ("95", 1.96)]:
            lo, hi = mean - z * std, mean + z * std
            cov = float(np.mean((y_te >= lo) & (y_te <= hi)))
            out.setdefault(level_name, []).append(cov)

    if not out:
        return None
    return {
        lvl: {
            "empirical_coverage_mean": float(np.mean(v)),
            "nominal": 0.68 if lvl == "68" else 0.95,
            "per_fold": [float(x) for x in v],
        }
        for lvl, v in out.items()
    }


def deployability(dataset: str, targets: dict | None = None) -> dict:
    targets = targets or {"target_inference_ms": 200.0, "target_size_mb": 4.0}
    models_dir = ROOT / "experiments" / "models" / dataset
    report = {}
    for model_name in ["naive", "rf", "svr", "gpr"]:
        sizes = []
        inf_ms = []
        for p in sorted(models_dir.glob(f"{model_name}_fold*.joblib")):
            sizes.append(model_size_mb(p))
        # Inference times come from the results YAML (real benchmarks).
        res_path = ROOT / "experiments" / f"classical_results_{dataset}.yaml"
        if res_path.exists():
            data = yaml.safe_load(res_path.read_text()).get(model_name, {})
            inf_ms = data.get("inference_time_ms_mean", np.nan)
        size_mean = float(np.mean(sizes)) if sizes else float("nan")
        report[model_name] = {
            "model_size_mb_mean": size_mean,
            "inference_time_ms_mean": inf_ms,
            "meets_size_target": (
                bool(size_mean <= targets["target_size_mb"]) if np.isfinite(size_mean) else None
            ),
            "meets_inference_target": (
                bool(inf_ms <= targets["target_inference_ms"]) if np.isfinite(inf_ms) else None
            ),
        }
    return {"targets": targets, "models": report}


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP + calibration + deployability")
    parser.add_argument("--dataset", choices=["nasa", "calce", "all"], default="all")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    analysis_dir = ROOT / "experiments" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df, folds = load_fold_data(args.dataset)

    shap_out = shap_analysis(df, folds, args.dataset)
    if shap_out:
        p = analysis_dir / f"shap_importance_{args.dataset}.yaml"
        with open(p, "w") as fh:
            yaml.safe_dump(shap_out, fh, sort_keys=False)
        print("top-5 SHAP:", list(shap_out["importance"].items())[:5])
        print(f"Wrote {p}")

    cal = gpr_calibration(df, folds, args.dataset)
    if cal:
        p = analysis_dir / f"gpr_calibration_{args.dataset}.yaml"
        with open(p, "w") as fh:
            yaml.safe_dump(cal, fh, sort_keys=False)
        print(f"Wrote {p}")

    cfg_path = ROOT / args.config
    dep_targets = {}
    if cfg_path.exists():
        _cfg = yaml.safe_load(cfg_path.read_text())
        dep_targets = {
            "target_inference_ms": float(_cfg["evaluation"]["deployability"]["target_inference_ms"]),
            "target_size_mb": float(_cfg["evaluation"]["deployability"]["target_size_mb"]),
        }
    dep = deployability(args.dataset, targets=dep_targets)
    p = ROOT / "experiments" / f"deployability_{args.dataset}.yaml"
    with open(p, "w") as fh:
        yaml.safe_dump(dep, fh, sort_keys=False)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
