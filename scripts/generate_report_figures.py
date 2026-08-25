"""Generate report-ready figures from canonical experiment artifacts.

Outputs PNGs into report/figures/ (the LaTeX sources reference these in a
later deliverable). All data comes from experiments/*.yaml + persisted
models — nothing recomputed ad hoc.

Usage:
    python scripts/generate_report_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
DATASETS = ["nasa", "calce", "all"]
LABELS = {"nasa": "NASA PCoE", "calce": "CALCE CS2", "all": "Combined"}
DL = {"lstm", "cnn", "transformer"}


def load_results(ds: str) -> dict:
    out = {}
    for kind in ["classical", "dl"]:
        p = ROOT / "experiments" / f"{kind}_results_{ds}.yaml"
        if p.exists():
            data = yaml.safe_load(p.read_text())
            for m, v in data.items():
                if isinstance(v, dict) and "rmse_mean" in v:
                    v["_kind"] = kind
                    out[m] = v
    return out


def fig_comparison(ds: str, results: dict) -> None:
    rows = sorted(results.items(), key=lambda kv: kv[1]["rmse_mean"])
    names = [k.upper() for k, _ in rows]
    rmse = [v["rmse_mean"] for _, v in rows]
    stds = [v["rmse_std"] for _, v in rows]
    colors = ["#2196F3" if v["_kind"] == "classical" else "#FF5722" for _, v in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(names, rmse, yerr=stds, capsize=4, color=colors)
    axes[0].set_title(f"{LABELS[ds]} — RMSE (lower is better)")
    axes[0].set_ylabel("RMSE (SOH units)")

    order = sorted(results.items(), key=lambda kv: -kv[1]["r2_mean"])
    names2 = [k.upper() for k, _ in order]
    r2 = [v["r2_mean"] for _, v in order]
    colors2 = ["#2196F3" if v["_kind"] == "classical" else "#FF5722" for _, v in order]
    axes[1].bar(names2, r2, color=colors2)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_title(f"{LABELS[ds]} — R²")

    handles = [plt.Rectangle((0, 0), 1, 1, color="#2196F3"),
               plt.Rectangle((0, 0), 1, 1, color="#FF5722")]
    fig.legend(handles, ["Classical", "Deep learning"], loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"comparison_{ds}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_shap(ds: str) -> None:
    p = ROOT / "experiments" / "analysis" / f"shap_importance_{ds}.yaml"
    if not p.exists():
        return
    imp = yaml.safe_load(p.read_text())["importance"]
    s = pd.Series(imp).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    s.plot.barh(ax=ax, color="#2196F3")
    ax.set_title(f"{LABELS.get(ds, ds)} — RF SHAP feature importance")
    ax.set_xlabel("mean |SHAP value|")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"shap_{ds}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_robustness(ds: str) -> None:
    p = ROOT / "experiments" / f"robustness_{ds}.yaml"
    if not p.exists():
        return
    data = yaml.safe_load(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model_name in ["rf", "svr", "gpr"]:
        noise = data["noise"].get(model_name, {})
        levels = sorted(noise.keys(), key=float)
        axes[0].plot([float(x) * 100 for x in levels],
                     [noise[x]["rmse"] for x in levels], marker="o", label=model_name.upper())
        missing = data["missing_cycles"].get(model_name, {})
        fracs = sorted(missing.keys(), key=float)
        axes[1].plot([float(x) * 100 for x in fracs],
                     [missing[x]["rmse"] for x in fracs], marker="o", label=model_name.upper())
    axes[0].set_title(f"{LABELS[ds]} — RMSE vs sensor noise")
    axes[0].set_xlabel("noise level (% of train std)")
    axes[1].set_title(f"{LABELS[ds]} — RMSE vs missing cycles")
    axes[1].set_xlabel("removed test cycles (%)")
    for ax in axes:
        ax.set_ylabel("RMSE")
        ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"robustness_{ds}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_calibration() -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    width = 0.35
    x = np.arange(len(DATASETS))
    for off, lvl, color in [(-width / 2, "68", "#2196F3"), (width / 2, "95", "#FF9800")]:
        vals = []
        for ds in DATASETS:
            p = ROOT / "experiments" / "analysis" / f"gpr_calibration_{ds}.yaml"
            vals.append(yaml.safe_load(p.read_text())[lvl]["empirical_coverage_mean"]
                        if p.exists() else np.nan)
        ax.bar(x + off, vals, width, label=f"nominal {lvl}%", color=color)
    ax.axhline(0.68, ls="--", color="#2196F3", lw=0.8)
    ax.axhline(0.95, ls="--", color="#FF9800", lw=0.8)
    ax.set_xticks(x, [LABELS[d] for d in DATASETS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("empirical coverage")
    ax.set_title("GPR predictive-interval coverage across held-out cells")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "gpr_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        results = load_results(ds)
        if not results:
            continue
        fig_comparison(ds, results)
        fig_shap(ds)
        fig_robustness(ds)
        print(f"figures done: {ds}")
    fig_calibration()
    print(f"All figures written to {FIG_DIR}")


if __name__ == "__main__":
    sys.exit(main())
