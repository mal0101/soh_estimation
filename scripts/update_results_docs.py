"""Generate README results section + machine-readable numbers manifest.

Reads the final experiments/*.yaml files and:
    1. Rewrites the README.md block between RESULTS markers with
       markdown tables regenerated from the YAMLs (no hand-copied
       numbers anywhere).
    2. Writes report/numbers_manifest.json containing every headline
       metric for the LaTeX report/presentation handoff.

Usage:
    python scripts/update_results_docs.py
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS_CLASSICAL = ["naive", "rf", "svr", "gpr"]
MODELS_DL = ["lstm", "cnn", "transformer"]
MODEL_LABELS = {
    "naive": "Naive (mean)",
    "rf": "RF",
    "svr": "SVR",
    "gpr": "GPR",
    "lstm": "LSTM",
    "cnn": "CNN",
    "transformer": "Transformer",
}
DATASETS = {"nasa": "NASA PCoE", "calce": "CALCE CS2", "all": "Combined"}


def load(dataset: str) -> dict:
    out = {}
    for kind, models in [("classical", MODELS_CLASSICAL), ("dl", MODELS_DL)]:
        p = ROOT / "experiments" / f"{kind}_results_{dataset}.yaml"
        if not p.exists():
            continue
        data = yaml.safe_load(p.read_text())
        for m in models:
            if m in data and isinstance(data[m], dict) and "rmse_mean" in data[m]:
                data[m]["_kind"] = kind
                out[m] = data[m]
    return out


def fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def table_for(dataset: str) -> str:
    results = load(dataset)
    lines = [
        "| Model | RMSE | MAE | MaxAE | R² | Inference (ms) |",
        "|-------|------|-----|-------|----|----------------|",
    ]
    rows = sorted(results.items(), key=lambda kv: kv[1]["rmse_mean"])
    for name, m in rows:
        inf_ms = m.get("inference_time_ms_mean", float("nan"))
        lines.append(
            f"| {MODEL_LABELS[name]} | {fmt(m['rmse_mean'])} ± {fmt(m['rmse_std'])} "
            f"| {fmt(m['mae_mean'])} | {fmt(m['maxae_mean'])} "
            f"| {fmt(m['r2_mean'], 2)} ± {fmt(m['r2_std'], 2)} "
            f"| {inf_ms:.3f} |"
        )
    return "\n".join(lines)


def build_readme_block() -> str:
    parts = []
    meta = {}
    for ds in DATASETS:
        p = ROOT / "experiments" / f"classical_results_{ds}.yaml"
        if p.exists():
            meta[ds] = yaml.safe_load(p.read_text()).get("_meta", {})
    for ds, label in DATASETS.items():
        parts.append(f"### {label}\n\n{table_for(ds)}\n")
    return "\n".join(parts)


def manifest() -> dict:
    out = {"generated_by": "scripts/update_results_docs.py", "protocol_note": (
        "Cell-based LOOCV; per-fold feature selection fitted on training cells only; "
        "hyperparameters selected via inner cell split; final refit on full training fold; "
        "inference latency measured as single-sample predict() latency."
    )}
    for ds in DATASETS:
        out[ds] = load(ds)
    return out


def main() -> None:
    block = build_readme_block()
    readme_path = ROOT / "README.md"
    text = readme_path.read_text()

    start_marker = "<!-- RESULTS:AUTO -->"
    end_marker = "<!-- /RESULTS:AUTO -->"
    if start_marker in text and end_marker in text:
        pre = text.split(start_marker)[0]
        post = text.split(end_marker)[1]
        new_text = f"{pre}{start_marker}\n{block}\n{end_marker}{post}"
        readme_path.write_text(new_text)
        print(f"README updated ({len(block)} chars of tables)")
    else:
        print("Markers not found in README; printing block instead:\n")
        print(block)

    mp = ROOT / "report" / "numbers_manifest.json"
    mp.write_text(json.dumps(manifest(), indent=2))
    print(f"Manifest written to {mp}")


if __name__ == "__main__":
    main()
