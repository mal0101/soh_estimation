# SOH Estimation — Detailed Execution Plan

## Environment Decision

**Recommendation: Conda (Miniconda).** Rationale:

1. `project_plan.md` lists "conda or venv" as options.
2. We need Python 3.11 specifically — conda handles this without touching system pyenv (which has 3.13.5).
3. PyTorch with MPS support installs more reliably via conda (`pytorch::pytorch` channel).
4. Non-Python dependencies (MKL, BLAS, HDF5 for h5py) resolve automatically.
5. Jupyter notebook integration is trivial.
6. Deployability phase requires MKL-optimized CPU inference measurements, which conda provides.

---

## Phase 0: Setup and Environment ✅

### Step 0.1: Conda Environment ✅

```bash
conda create -n soh python=3.11 -y
conda activate soh
conda install -c conda-forge numpy=1.24 pandas=2.1 scipy=1.11 h5py=3.10 matplotlib=3.8 seaborn=0.13 -y
conda install -c conda-forge scikit-learn=1.4 optuna=3.5 shap=0.44 -y
conda install -c pytorch pytorch=2.2 torchvision=0.17 -c defaults -y
pip install captum==0.7.0 mlflow==2.10 jupyterlab==4.1 openpyxl xlrd
pip freeze > requirements.txt
conda env export --from-history > environment.yml
```

**Actual environment:** Conda `soh`, Python 3.11.14, PyTorch 2.5.1 (MPS: True), numpy 1.26.4, MLflow 3.14, editable package.

### Step 0.2: Configuration Files ✅

- `pyproject.toml` — Project metadata, ruff/black config
- `setup.py` — Editable install of `src` package
- `src/__init__.py` + all subpackage `__init__.py` files
- `config/default.yaml` — Centralized hyperparameters, paths, dataset configs

### Step 0.3: Fix `.gitignore` ✅

Removed `docs/` from `.gitignore`. Added data and experiment ignores including `mlflow.db`.

### Step 0.4: Documentation Scaffolding ✅

- `README.md` — Setup, usage, directory structure, license
- `CHANGELOG.md` — Updated at each phase boundary
- `docs/data_dictionary.md` — Describes every column in feature_matrix
- `docs/decisions_log.md` — Records all design decisions with rationale
- `docs/mlops.md` — Reproducibility, retraining, evaluation instructions

### Step 0.5: Utility Modules ✅

- `src/utils/config.py` — YAML config loader
- `src/utils/tracking.py` — MLflow helper functions
- `src/utils/seeding.py` — Seed all RNGs for reproducibility

### Step 0.6: Experiment Tracking ✅

MLflow initialized with local SQLite backend (`sqlite:///mlflow.db`). `src/utils/tracking.py` provides helpers for logging params, metrics, artifacts.

---

## Phase 1: Data Acquisition and EDA ✅

### Step 1.1: Download Data ✅

- NASA PCoE: `.mat` files in `data/raw/nasa_pcoe/` — cells B0005, B0006, B0007, B0018

### Step 1.2: Data Loading Module ✅

`src/preprocessing/data_loader.py`:
- `load_nasa_cell(filepath) -> dict`
- `load_calce_cell(filepath) -> dict`
- `verify_data_integrity(cell_data) -> bool`

### Step 1.3: EDA Notebooks ✅

- `notebooks/01_eda_nasa.ipynb` — Capacity fade, voltage evolution, temperature, data inventory

### Step 1.4: SOH Label Definition ✅

`src/preprocessing/soh.py`:
- `compute_q_initial(cell_data, cycles=(3, 10)) -> float`
- `compute_soh_curve(cell_data, q_initial) -> np.ndarray`

**Cell statistics:**
| Cell | Cycles | Charge | Discharge | Impedance | Q_initial (Ah) |
|------|--------|--------|-----------|-----------|-----------------|
| B0005 | 616 | 170 | 168 | 278 | 1.8379 |
| B0006 | 616 | 170 | 168 | 278 | 2.0131 |
| B0007 | 616 | 170 | 168 | 278 | 1.8804 |
| B0018 | 319 | 134 | 132 | 53 | 1.8491 |

**SOH labels:** 636 rows total (4 cells × ~159 discharge cycles each after filtering).

---

## Phase 2: Data Preprocessing ✅

### Data Observations (from Phase 1)

| Characteristic | Value |
|---|---|
| Data segmentation | Pre-segmented by cycle type (charge/discharge/impedance) |
| Discharge points/cycle | 179–371 (variable-length) |
| Current convention | Negative = discharge (~-2A), positive = charge (CC at 1.5A) |
| Capacity axis | Computed via cumulative trapezoidal integration of \|current\| |
| Measured capacity | Only available for discharge cycles (field `capacity`) |
| Partial cycles | None detected in B0005 (all 179+ points) |

### Step 2.1: Capacity Axis Computation ✅

`src/preprocessing/capacity.py`:
- `compute_cumulative_capacity(current, time) -> np.ndarray`
  - Trapezoidal integration: Q(t) = integral of |I(t)| dt / 3600

### Step 2.2: Noise Filtering ✅

`src/preprocessing/filtering.py`:
- `savgol_filter_voltage(voltage, window_length=51, polyorder=3) -> np.ndarray`

### Step 2.3: Resampling to Uniform Capacity Grid ✅

`src/preprocessing/resampling.py`:
- `resample_to_uniform_grid(voltage, capacity, n_points=1000, kind="linear") -> tuple`

### Step 2.4: Cycle Validation and Filtering ✅

`src/preprocessing/segmentation.py`:
- `validate_cycle(cell_data, min_discharge_fraction=0.90) -> dict`
  - Note: `validate_cycles` only filters early cycles (`early_cycle_window=20`). Late-life low capacity is natural degradation.

### Step 2.5: Pipeline Orchestrator ✅

`src/preprocessing/pipeline.py`:
- `preprocess_cell(cell_data, config) -> dict`
- `run_pipeline(config_path="config/default.yaml") -> None`
- Output: `data/processed/processed_cells.pkl`

### Step 2.6: Validation Notebook ✅

`notebooks/02_preprocessing_validation.ipynb`

---

## Phase 3: Feature Engineering ✅

### Step 3.1: Incremental Capacity Analysis (ICA) ✅

`src/features/ica.py`:
- `compute_dQdV(voltage, capacity) -> tuple[np.ndarray, np.ndarray]`
- `extract_ica_features(voltage, capacity, min_peak_prominence=0.01) -> dict`
  - Features: ica_peak_height, ica_peak_voltage, ica_peak_area
  - Note: `ica_peak_fwhm` and `ica_secondary_ratio` are always NaN (dQ/dV peaks are narrow single-point spikes on resampled grid)

### Step 3.2: Internal Resistance Estimation ✅

`src/features/internal_resistance.py`:
- `estimate_ir_from_discharge(voltage, current, time) -> float`
- `extract_eis_features(eis_data) -> dict[str, float]` → produces `eis_re`

### Step 3.3: Energy and Efficiency ✅

`src/features/energy.py`:
- `compute_discharge_energy(voltage, current, time) -> float`
- `compute_mean_discharge_voltage(voltage, capacity) -> float`
- `compute_coulombic_efficiency(q_discharge, q_charge) -> float`
  - Note: `coulombic_efficiency` dropped from final feature set (low variance across cycles)

### Step 3.4: Temperature Features ✅

`src/features/temperature.py`:
- Features: temp_mean, temp_max, temp_min, temp_range

### Step 3.5: Capacity Fade Rate (Trend Feature) ✅

`src/features/trend.py`:
- `compute_capacity_fade_rate(soh_curve, window=10) -> np.ndarray`

### Step 3.6: Feature Matrix Assembly ✅

`src/features/assembly.py`:
- `build_feature_matrix(processed_cells, config) -> pd.DataFrame`
- `select_features(feature_df, correlation_threshold=0.95, top_k=20) -> pd.DataFrame`
  - Step 1: Drop zero-variance features
  - Step 2: Remove one feature from each pair with |r| > 0.95
  - Step 3: Train RF, rank by importance, keep top_k
- Output: `data/features/feature_matrix.parquet` (636 rows × 16 columns, 12 selected features)

### Final Selected Features (12)

| Feature | Description |
|---|---|
| mean_discharge_voltage | Average voltage during discharge (V) |
| internal_resistance | Delta_V / delta_I at discharge onset (Ohm) |
| discharge_energy | Integral of V*I*dt over discharge (Wh) |
| eis_re | Real impedance at 1 kHz from EIS (Ohm) |
| capacity_fade_rate | Local SOH slope over last 10 cycles |
| temp_mean | Mean temperature during discharge (deg C) |
| temp_max | Maximum temperature during discharge (deg C) |
| temp_range | Temperature range: max - min (deg C) |
| ica_peak_voltage | Voltage position of primary ICA peak (V) |
| temp_min | Minimum temperature during discharge (deg C) |
| ica_peak_height | Height of tallest peak in dQ/dV curve |
| ica_peak_area | Area under primary ICA peak (Ah/V) |

### Step 3.7: Feature Engineering Notebook ✅

`notebooks/03_feature_engineering.ipynb`

---

## Phase 4: Validation Strategy ✅

### Step 4.1: Cross-Validation ✅

`src/evaluation/validation.py`:
- `cell_based_loocv(feature_matrix, dataset) -> List[Tuple[train, test]]`
- Sequence windowing for DL: `create_sequences(features, soh, window=20) -> (X, y)`
- Fold indices saved to `experiments/fold_indices.json`

### Step 4.2: Feature Scaling ✅

StandardScaler per fold.

---

## Phase 5: Classical ML Training ✅

### Step 5.0: Baseline ✅

`src/models/baseline.py` — naive mean prediction.

### Step 5.1: Random Forest ✅

`src/models/rf_model.py` — Optuna 100 trials, log to MLflow.

### Step 5.2: SVR ✅

`src/models/svr_model.py` — Optuna 50 trials, RBF kernel.

### Step 5.3: GPR ✅

`src/models/gpr_model.py` — Matern(1.5) + WhiteKernel, subsample to 5000.

### Step 5.4: Classical ML Orchestrator ✅

`src/models/train_classical.py` — CLI entry point.

### Results

| Model | RMSE (mean±std) | MAE (mean±std) | MaxAE (mean±std) | R² (mean±std) |
|---|---|---|---|---|
| SVR | 0.015 ± 0.006 | 0.011 ± 0.004 | 0.054 ± 0.017 | 0.973 ± 0.020 |
| RF | 0.025 ± 0.017 | 0.020 ± 0.014 | 0.084 ± 0.036 | 0.927 ± 0.073 |
| GPR | 0.030 ± 0.019 | 0.027 ± 0.019 | 0.098 ± 0.020 | 0.885 ± 0.097 |
| Naive | 0.112 ± 0.027 | 0.096 ± 0.025 | 0.203 ± 0.047 | -0.285 ± 0.250 |

---

## Phase 6: Deep Learning Training ✅

### Step 6.0: Shared DL Infrastructure ✅

`src/models/dl_base.py`:
- SOHDataset, train_loop, evaluate, device detection, checkpointing

### Step 6.1: LSTM ✅

`src/models/lstm.py` — 2-layer LSTM, Optuna 10 trials per model.

### Step 6.2: 1D CNN ✅

`src/models/cnn.py` — 3-layer Conv1D + GAP, Optuna 10 trials.

### Step 6.3: Transformer ✅

`src/models/transformer.py` — 2-block encoder, 4 heads, Optuna 10 trials.

### Step 6.4: DL Orchestrator ✅

`src/models/train_dl.py` — CLI, 3 seeds per config (reduced from 5 for feasibility).

### Results (3 seeds, 10 Optuna trials per model, 80 max epochs)

| Model | RMSE (mean±std) | MAE (mean±std) | MaxAE (mean±std) | R² (mean±std) |
|---|---|---|---|---|
| LSTM | 0.048 ± 0.020 | 0.043 ± 0.019 | 0.098 ± 0.034 | -0.135 ± 1.366 |
| Transformer | 0.067 ± 0.022 | 0.059 ± 0.021 | 0.139 ± 0.040 | -0.142 ± 0.513 |
| CNN | 0.104 ± 0.059 | 0.085 ± 0.044 | 0.227 ± 0.144 | -2.504 ± 3.071 |

**Note:** All DL models have negative R², indicating they do not generalize well on the small dataset (636 samples, 4 cells). Classical ML dominates.

---

## Phase 7: Evaluation and Comparison ✅

### Step 7.1: Metrics ✅

`src/evaluation/metrics.py` — RMSE, MAE, MaxAE, R2.

### Step 7.2: Error Analysis ✅

`src/evaluation/error_analysis.py` — Phase-based RMSE breakdown (early/mid/late life).

### Step 7.3: Visualization ✅

`src/evaluation/visualizations.py` — Trajectory, scatter, comparison plots.

### Step 7.4: SHAP and Attribution ✅

`src/evaluation/interpretability.py` — SHAP TreeExplainer + Captum Integrated Gradients.

### Step 7.5: GPR Calibration ✅

`compute_coverage_probability(y_true, y_mean, y_std, confidence=0.95) -> float`

### Step 7.6: Comparison Table ✅

`src/evaluation/comparison.py` — Full model × dataset × metric table.

---

## Phase 8: Robustness Assessment ✅

### Step 8.1: Inference Benchmarking ✅

`src/evaluation/deployability.py`:
- Sequential CPU inference time (1000 repeats)
- Model file size (MB)

### Step 8.2: Noise Robustness ✅

- Add Gaussian noise (0.5%, 1%, 2%) → recompute features → re-evaluate
- Report RMSE degradation

### Step 8.3: Missing Cycle Robustness ✅

- Drop 10%, 20%, 30% cycles → forward-fill → re-evaluate

---

## Phase 9: Report and Presentation (Next)

### Incremental Writing Schedule

| Section | Write During |
|---|---|
| Introduction | Phase 0 |
| Related Work | Phase 0 |
| Datasets | Phase 1 |
| Methodology | Phase 2–4 |
| Models | Phase 5–6 |
| Results | Phase 7 |
| Discussion | Phase 7–8 |
| Conclusion | Phase 9 |

### Figures to Produce

| Figure | Phase |
|---|---|
| Capacity fade curves | Phase 1 |
| ICA dQ/dV evolution | Phase 3 |
| Feature correlation heatmap | Phase 3 |
| Feature importance bar chart | Phase 3 |
| DL training loss curves | Phase 6 |
| Model comparison table | Phase 7 |
| Predicted vs. true scatter | Phase 7 |
| SOH trajectory plots | Phase 7 |
| SHAP beeswarm + summary | Phase 7 |
| GPR confidence bands | Phase 7 |
| Robustness bar charts | Phase 8 |

---

## Documentation Strategy

### Code-Level
- Every file: module-level docstring
- Every function: Google-style docstring with Args/Returns/Raises
- Type hints on all signatures

### Notebook-Level
- Markdown header before every code cell
- Summary cell at end of each notebook
- Naming: `{phase}_{descriptive_name}.ipynb`

### Project-Level
- `README.md`: Setup, usage, structure
- `docs/`: plan, data dictionary, decisions log, MLOps guide
- `CHANGELOG.md`: Updated at phase boundaries
- MLflow: Full run provenance

### MLOps
- `docs/mlops.md`: Reproducibility instructions
- Dual-track: `environment.yml` + `requirements.txt`
- Seed all RNGs, config-driven hyperparameters
- Model registry via MLflow

---

## Target File Structure

```
soh_estimation/
├── .gitignore
├── LICENSE
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── setup.py
├── config/
│   └── default.yaml
├── data/
│   ├── raw/
│   │   └── nasa_pcoe/
│   ├── processed/
│   │   ├── soh_labels.parquet
│   │   └── processed_cells.pkl
│   └── features/
│       └── feature_matrix.parquet
├── docs/
│   ├── project_plan.md
│   ├── execution_plan.md
│   ├── data_dictionary.md
│   ├── decisions_log.md
│   └── mlops.md
├── notebooks/
│   ├── 01_eda_nasa.ipynb
│   ├── 01_eda_calce.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_classical_ml.ipynb
│   ├── 05_dl_training_analysis.ipynb
│   └── 06_final_comparison.ipynb
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── tracking.py
│   │   ├── config.py
│   │   └── seeding.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── filtering.py
│   │   ├── resampling.py
│   │   ├── segmentation.py
│   │   ├── soh.py
│   │   └── pipeline.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── ica.py
│   │   ├── internal_resistance.py
│   │   ├── energy.py
│   │   ├── temperature.py
│   │   ├── trend.py
│   │   └── assembly.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── rf_model.py
│   │   ├── svr_model.py
│   │   ├── gpr_model.py
│   │   ├── dl_base.py
│   │   ├── lstm.py
│   │   ├── cnn.py
│   │   ├── transformer.py
│   │   ├── train_classical.py
│   │   └── train_dl.py
│   └── evaluation/
│       ├── __init__.py
│       ├── validation.py
│       ├── metrics.py
│       ├── error_analysis.py
│       ├── interpretability.py
│       ├── deployability.py
│       ├── comparison.py
│       └── visualizations.py
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_metrics.py
│   ├── test_validation.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_dl.py
│   ├── test_evaluation.py
│   └── test_robustness.py
├── experiments/
│   ├── classical_results.yaml
│   ├── dl_results.yaml
│   ├── fold_indices.json
│   └── figures/
│       ├── phase1/
│       ├── phase2/
│       ├── phase3/
│       ├── phase6/
│       ├── phase7/
│       └── phase8/
└── report/
    ├── main.tex
    ├── references.bib
    └── presentation/
```

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| CALCE data access delay | Resolved: full CALCE CS2 integration completed (loader, preprocessing, results) |
| GPR O(n^3) memory blowup | Subsample to 5000 hard cap |
| Transformer overfits on small data | Reported honestly; negative R² documented |
| DL training too slow on CPU | Used MPS backend (Apple Silicon); reduced trials/seeds for feasibility |
| Scope creep | Strict phase gates; signed scope doc |
| Writing panic at end | Write incrementally per phase |
| MLflow file store deprecation | Migrated to SQLite backend (`sqlite:///mlflow.db`) |


---

## Post-Plan Remediation (2026-08-25)

A full audit identified target leakage (capacity_fade_rate), selection-on-test
(feature selection + Optuna on the LOOCV test cell), corrupted CALCE labels
(storage gaps, RPT dips, poisoned Q_initial), degenerate ICA features, and
mislabelled inference times. All were fixed; every experiment was re-run under
a leakage-safe protocol. See docs/decisions_log.md D-009..D-016 and
CHANGELOG.md. Numbers quoted earlier in this plan are superseded by
experiments/*.yaml.
