# SOH Estimation — Inspection Guide

This guide reflects the **remediated** pipeline (leakage-safe protocol,
cleaned labels, explicit per-dataset artifact naming). Every number shown
anywhere in the repo is generated from `experiments/*.yaml` by
`scripts/update_results_docs.py` — nothing is hand-copied.

## 1. Environment Setup

```bash
# Activate the correct environment (MANDATORY)
conda activate soh

# Verify Python version — must be 3.11.x
python --version   # expected: Python 3.11.x

# Verify key packages exist
python -c "import numpy, pandas, sklearn, torch, mlflow, optuna, shap, captum, h5py; print('OK')"
```

## 2. Project Layout

```
soh_estimation/
├── config/default.yaml          ← Central configuration (ALL hyperparams)
├── src/                         ← Core library (editable install)
│   ├── preprocessing/           ← load → filter → capacity axis → resample → SOH
│   ├── features/                ← ICA, IR, energy, temperature, trend (+ build_features CLI)
│   ├── models/                  ← RF, SVR, GPR, LSTM, CNN, Transformer + orchestrators
│   ├── evaluation/              ← metrics, folds, comparison, SHAP, deployability
│   └── utils/                   ← config loader, MLflow tracking, seeding, paths
├── data/
│   ├── raw/nasa_pcoe/*.mat      ← NASA PCoE B0005/B0006/B0007/B0018
│   ├── raw/calce/CS2_*/         ← CALCE CS2-33..36 (.xlsx sessions)
│   ├── processed/               ← processed_cells_{nasa,calce,all}.pkl
│   │                              soh_labels_{nasa,calce,all}.parquet
│   └── features/                ← feature_matrix_{nasa,calce,all}.parquet
│                                  (FULL candidate matrices — selection happens
│                                   per LOOCV fold inside training)
├── experiments/                 ← training outputs (committed)
│   ├── classical_results_{nasa,calce,all}.yaml
│   ├── dl_results_{nasa,calce,all}.yaml
│   ├── fold_indices_{nasa,calce,all}.json
│   ├── robustness_{nasa,calce,all}.yaml
│   ├── deployability_{nasa,calce,all}.yaml
│   ├── models/{dataset}/        ← persisted fold models (.joblib / .pt)
│   └── analysis/                ← SHAP importances, GPR calibration
├── notebooks/                   ← 01..06 (consume canonical artifacts)
├── scripts/                     ← update_results_docs.py, run_robustness.py,
│                                  run_analysis_extras.py
├── report/                      ← main.tex, presentation.tex,
│                                  numbers_manifest.json
└── mlflow.db                    ← all runs tagged with dataset/fold/model/seed
```

**Naming convention:** every dataset gets an explicit suffix
(`_nasa`, `_calce`, `_all`) on every artifact. Nothing is ever overwritten
across datasets.

## 3. Pipeline Steps (end-to-end order)

### Step 0 — Verify raw data

```bash
ls data/raw/nasa_pcoe/   # B0005.mat B0006.mat B0007.mat B0018.mat
ls data/raw/calce/       # CS2_33 CS2_34 CS2_35 CS2_36 (+ zips)
```

Expected discharge-cycle counts after loading (before filtering):
B0005/B0006/B0007: 168 each; B0018: 132; CALCE 866/775/932/973.

### Step 1 — Preprocessing

```bash
python -m src.preprocessing.pipeline --config config/default.yaml --dataset nasa
python -m src.preprocessing.pipeline --config config/default.yaml --dataset calce
python -m src.preprocessing.pipeline --config config/default.yaml --dataset all
```

What it does:
1. Loads cells (NASA .mat structs; CALCE .xlsx session sheets).
2. Computes Q_initial per cell as the **median** measured discharge
   capacity over cycles [3,10] (robust to outlier cycles).
3. Filters discharge cycles with three rules:
   - early partials: cycles ≤ 20 below 90% of Q_initial;
   - isolated interruptions: >7% below the ±5-neighbour local median
     (catches CALCE reference-performance-test dips anywhere in life);
   - anomalous runs: contiguous depressed blocks (<75% of Q_initial)
     that later RECOVER and average <70% of Q_initial (storage/test
     pauses). Genuine unrecovered end-of-life fade and shallow
     reversible transients (e.g., NASA B0006) are kept.
4. Savitzky-Golay voltage filter (window 51, order 3), uniform
   1000-point capacity grid.
5. SOH = Q(n)/Q_initial capped at `preprocessing.soh.soh_cap`.

Expected label counts: nasa=636, calce=2559, all=3195 rows.

### Step 2 — Candidate feature matrix

```bash
python -m src.features.build_features --config config/default.yaml --dataset all   # or nasa/calce
```

Produces the FULL 16-candidate-feature matrix (metadata + features).
Supervised feature selection is NOT applied here — it runs per LOOCV
fold during training (`fit_feature_selection` on training cells only).

### Step 3 — Classical ML

```bash
python -m src.models.train_classical --config config/default.yaml --dataset nasa    # then calce, all
```

Protocol per fold: select features (train cells only) → scale (train fit) →
tune RF/SVR via Optuna against an inner held-out CELL → refit best params
on the full training fold → evaluate once on the test cell → benchmark
single-sample inference latency → persist model to
`experiments/models/{dataset}/`.

### Step 4 — Deep learning

```bash
python -m src.models.train_dl --config config/default.yaml --dataset nasa           # then calce, all
```

Same protocol plus: sequence windows built directly from cell-grouped
frames (windows never cross cells); Optuna trials early-stop on the inner
val cell and record `best_epoch`; final models refit for exactly
`best_epoch` epochs; train_loop then keeps the epoch with the lowest
training loss (train-only information — no test involvement).

Duration: hours (dominated by CALCE/all). Run overnight.

### Step 5 — Analysis extras & robustness

```bash
python scripts/run_robustness.py --dataset all          # repeat per dataset if needed
python scripts/run_analysis_extras.py --dataset all
```

Outputs: `experiments/robustness_*.yaml`, `experiments/deployability_*.yaml`,
`experiments/analysis/shap_importance_*.yaml`,
`experiments/analysis/gpr_calibration_*.yaml`.

### Step 6 — Regenerate documentation numbers

```bash
python scripts/update_results_docs.py
```

Rewrites the README results block between the `RESULTS:AUTO` markers and
writes `report/numbers_manifest.json`.

### Step 7 — Tests, lint, types

```bash
python -m pytest tests/ -q            # 134 tests expected green
ruff check src/ tests/                # clean
mypy src/                             # clean
```

## 4. Notebooks

| Notebook | Purpose |
|---|---|
| 01_eda_nasa | Raw NASA signals, fade trajectories, SOH distribution |
| 01_eda_calce | CALCE sessions; loads pipeline-produced labels |
| 02_preprocessing_validation | Filter/resample checks + cycle accounting |
| 03_feature_engineering | ICA/IR/energy/temperature validation (NASA) |
| 04_classical_ml | Combined-dataset results from canonical YAMLs + persisted models |
| 05_dl_training_analysis | NASA DL results + MLflow run inspection |
| 06_final_comparison | NASA + CALCE comparison, phases, SHAP, deployability |

All notebooks use the `soh` kernel and locate the project root by walking
up until `config/default.yaml` is found, so they run from any CWD.

## 5. MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Runs are tagged `dataset`, `fold`, `model`, `seed`. Legacy pre-remediation
runs have no dataset tag; notebook 05 filters them out.

## 6. Key Files to Inspect

| File | What to check |
|------|--------------|
| `config/default.yaml` | All hyperparams; preprocessing filter thresholds |
| `src/features/trend.py` | Fade-rate uses strictly PAST labels (no target leak) |
| `src/features/assembly.py` | `fit_feature_selection` (fold-safe) + CE/EIS pairing |
| `src/preprocessing/segmentation.py` | Three-rule cycle integrity filter |
| `src/models/train_classical.py` | Inner-cell-split tuning orchestration |
| `src/models/train_dl.py` | Fixed-epoch refit + best-weight restoration flow |
| `tests/test_leakage.py` | Regression pins for every anti-leakage guarantee |

## 7. Troubleshooting

**ModuleNotFoundError: src** → `pip install -e .`

**numpy trapezoid errors** → handled by compat wrapper in
`src/features/{energy,ica}.py`.

**MLflow URI issues** → `.env`: `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`
(relative sqlite paths are anchored to the project root automatically).

**Very long DL training** → shrink in a scratch copy of the config:
`n_seeds: 1`, `max_epochs: 5`, `n_trials: 1–2`.

**Jupyter kernel** → always select the `soh` kernel.

## 8. Data Provenance

- NASA PCoE: 18650 LCO, 2.0 Ah rated; charge 1.5 A CC-CV to 4.2 V;
  discharge 2 A CC; cutoffs B0005 2.7 V / B0006 2.5 V / B0007 2.2 V /
  B0018 2.5 V. Q_initial (median over cycles 3–10): B0005 1.8353,
  B0006 2.0133, B0007 1.8806, B0018 1.8491 Ah.
- CALCE CS2: pouch cells, 1.1 Ah rated, 2.7 V cutoff. Q_initial:
  CS2_33 1.0771, CS2_34 1.0832, CS2_35 1.0476, CS2_36 1.0409 Ah
  (CS2_36 was previously poisoned by a 0.147 Ah interruption inside its
  averaging window — fixed by the median reduction).
- EOL definition: SOH ≤ 0.80 (informational; used in reporting/phases).

## 9. Reproducibility Contract

What is tracked in git vs. what must be regenerated locally:

| Artifact | Tracked | Regenerate with |
|---|---|---|
| `experiments/classical_results_{ds}.yaml` | yes | `soh-train-classical --dataset {nasa\|calce\|all}` (~30-60 min total) |
| `experiments/dl_results_{ds}.yaml` | yes | `soh-train-dl --dataset {nasa\|calce\|all}` (~hours; run overnight) |
| `experiments/fold_indices_{ds}.json` | yes | produced by either training command |
| `experiments/robustness_{ds}.yaml` | yes | `python scripts/run_robustness.py --dataset {ds}` (needs local models) |
| `experiments/deployability_{ds}.yaml` | yes | `python scripts/run_analysis_extras.py --dataset {ds}` (needs local models) |
| `experiments/analysis/*.yaml` | yes | same as above |
| `report/figures/*.png`, `numbers_manifest.json` | yes | `generate_report_figures.py`, `update_results_docs.py` |
| `mlflow.db` | yes | grows on every training run |
| Raw + processed data products (`data/**`) | no | `soh-preprocess --dataset all`, `soh-build-features --dataset all` |
| Persisted fold models (`experiments/models/`, ~1 GB) | **no** | training commands above |

Consequence: robustness / deployability / SHAP analyses and the model-based
sections of notebooks 04 and 06 require locally trained models. Re-running the
classical suite takes ~30-60 min; DL is the long pole. All *reported metrics*
live in the tracked YAMLs and never depend on local binaries.

Gate check after any change:

```bash
python scripts/verify_consistency.py   # README block, notebook outputs, stale refs
```
