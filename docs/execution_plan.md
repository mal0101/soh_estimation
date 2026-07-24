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

## Phase 0: Setup and Environment (Day 1–2)

### Step 0.1: Conda Environment

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

### Step 0.2: Configuration Files

- `pyproject.toml` — Project metadata, ruff/black config
- `setup.py` — Editable install of `src` package
- `src/__init__.py` + all subpackage `__init__.py` files
- `config/default.yaml` — Centralized hyperparameters, paths, dataset configs
- `.env.example` — Template for secrets (W&B API key if used)

### Step 0.3: Fix `.gitignore`

Current `.gitignore` has `docs/` on line 8, which excludes the project plan. Remove that line. Add data and experiment ignores.

### Step 0.4: Documentation Scaffolding

- `README.md` — Setup, usage, directory structure, license
- `CHANGELOG.md` — Updated at each phase boundary
- `docs/data_dictionary.md` — Describes every column in feature_matrix
- `docs/decisions_log.md` — Records all design decisions with rationale
- `docs/meeting_notes.md` — Supervisor meeting summaries
- `docs/mlops.md` — Reproducibility, retraining, evaluation instructions

### Step 0.5: Utility Modules

- `src/utils/config.py` — YAML config loader
- `src/utils/tracking.py` — MLflow helper functions
- `src/utils/seeding.py` — Seed all RNGs for reproducibility

### Step 0.6: Experiment Tracking

Initialize MLflow with local backend. Create `src/utils/tracking.py` with helpers for logging params, metrics, artifacts.

---

## Phase 1: Data Acquisition and EDA (Day 3–5)

### Step 1.1: Download Data

- NASA PCoE: `.mat` files to `data/raw/nasa_pcoe/` — cells B0005, B0006, B0007, B0018
- CALCE: CSV/Excel to `data/raw/calce/` — cells CS2-33, CS2-34, CS2-35, CS2-36

### Step 1.2: Data Loading Module

`src/preprocessing/data_loader.py`:
- `load_nasa_cell(filepath) -> dict`
- `load_calce_cell(filepath) -> dict`
- `verify_data_integrity(cell_data) -> bool`

### Step 1.3: EDA Notebooks

- `notebooks/01_eda_nasa.ipynb` — Capacity fade, voltage evolution, temperature, data inventory
- `notebooks/01_eda_calce.ipynb` — Same for CALCE

### Step 1.4: SOH Label Definition

`src/preprocessing/soh.py`:
- `compute_q_initial(cell_data, cycles=(3, 10)) -> float`
- `compute_soh_curve(cell_data, q_initial) -> np.ndarray`

---

## Phase 2: Data Preprocessing (Day 6–13)

### Step 2.1: Noise Filtering

`src/preprocessing/filtering.py`:
- `savgol_filter_voltage(voltage, capacity, window=51, polyorder=3) -> np.ndarray`

### Step 2.2: Resampling

`src/preprocessing/resampling.py`:
- `resample_to_uniform_grid(voltage, capacity, n_points=1000) -> (capacity_grid, voltage_resampled)`

### Step 2.3: Cycle Segmentation

`src/preprocessing/segmentation.py`:
- `segment_cycles(current, time) -> List[Tuple[start, end, type]]`

### Step 2.4: Pipeline Orchestrator

`src/preprocessing/pipeline.py`:
- Chains: load → segment → filter → resample → compute SOH
- CLI: `python -m src.preprocessing.pipeline --config config/default.yaml`

### Step 2.5: Validation Notebook

`notebooks/02_preprocessing_validation.ipynb`

---

## Phase 3: Feature Engineering (Day 14–20)

### Step 3.1: ICA Features

`src/features/ica.py`:
- `compute_dQdV(voltage, capacity) -> (voltage_grid, dQdV)`
- `extract_ica_features(dQdV, voltage_grid) -> dict`
- Features: primary peak height/voltage/FWHM/area, secondary peak ratio

### Step 3.2: Internal Resistance

`src/features/internal_resistance.py`:
- `estimate_ir(voltage, current, time) -> float`

### Step 3.3: Energy and Efficiency

`src/features/energy.py`:
- `compute_discharge_energy(voltage, current, time) -> float`
- `compute_mean_discharge_voltage(...) -> float`
- `compute_coulombic_efficiency(q_discharge, q_charge) -> float`

### Step 3.4: Temperature Features

`src/features/temperature.py`:
- `compute_temperature_features(temperature) -> dict`

### Step 3.5: Trend Feature

`src/features/trend.py`:
- `compute_capacity_fade_rate(soh_curve, window=10) -> np.ndarray`

### Step 3.6: Feature Matrix Assembly

`src/features/assembly.py`:
- Combines all features, removes correlated pairs (|r| > 0.95), keeps top 15-20 via RF importance
- Saves feature_matrix.parquet

### Step 3.7: Feature Engineering Notebook

`notebooks/03_feature_engineering.ipynb`

---

## Phase 4: Validation Strategy (Day 21)

### Step 4.1: Cross-Validation

`src/evaluation/validation.py`:
- `cell_based_loocv(feature_matrix, dataset) -> List[Tuple[train, test]]`
- Sequence windowing: `create_sequences(features, soh, window=20) -> (X, y)`
- Save fold indices to `experiments/fold_indices.json`

### Step 4.2: Feature Scaling

StandardScaler per fold, saved to `experiments/scalers/`

---

## Phase 5: Classical ML Training (Day 22–27)

### Step 5.0: Baseline

`src/models/baseline.py` — naive mean prediction

### Step 5.1: Random Forest

`src/models/rf.py` — Optuna 100 trials, log to MLflow

### Step 5.2: SVR

`src/models/svr.py` — Optuna 50 trials, RBF kernel

### Step 5.3: GPR

`src/models/gpr.py` — Matern(1.5) + WhiteKernel, subsample to 5000

### Step 5.4: Classical ML Orchestrator

`src/models/train_classical.py` — CLI entry point

---

## Phase 6: Deep Learning Training (Day 28–37)

### Step 6.0: Shared DL Infrastructure

`src/models/dl_base.py`:
- SOHDataset, train_loop, evaluate, device detection, checkpointing

### Step 6.1: LSTM

`src/models/lstm.py` — 2-layer LSTM, Optuna 30 trials

### Step 6.2: 1D CNN

`src/models/cnn.py` — 3-layer Conv1D + GAP, Optuna 30 trials

### Step 6.3: Transformer

`src/models/transformer.py` — 2-block encoder, 4 heads, Optuna 30 trials

### Step 6.4: DL Orchestrator

`src/models/train_dl.py` — CLI, 5 seeds per config

---

## Phase 7: Evaluation and Comparison (Day 38–41)

### Step 7.1: Metrics

`src/evaluation/metrics.py` — RMSE, MAE, MaxAE, R2

### Step 7.2: Error Analysis

`src/evaluation/error_analysis.py` — Phase-based RMSE breakdown

### Step 7.3: Visualization

`src/evaluation/visualizations.py` — Trajectory, scatter, comparison plots

### Step 7.4: SHAP and Attribution

`src/evaluation/interpretability.py` — SHAP TreeExplainer + Captum Integrated Gradients

### Step 7.5: GPR Calibration

`compute_coverage_probability(y_true, y_mean, y_std, confidence=0.95) -> float`

### Step 7.6: Comparison Table

`src/evaluation/comparison.py` — Full model × dataset × metric table

---

## Phase 8: Deployability Assessment (Day 42–44)

### Step 8.1: Inference Benchmarking

`src/evaluation/deployability.py`:
- Sequential CPU inference time (1000 repeats)
- Model file size (MB)

### Step 8.2: Noise Robustness

- Add Gaussian noise (0.5%, 1%, 2%) → recompute features → re-evaluate
- Report RMSE degradation; <20% increase under 1% noise = robust

### Step 8.3: Missing Cycle Robustness

- Drop 10%, 20%, 30% cycles → forward-fill → re-evaluate

---

## Phase 9: Report and Presentation (Day 45–50)

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
- `docs/`: plan, literature notes, data dictionary, decisions log, meeting notes
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
soh-estimation/
├── .gitignore
├── LICENSE
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── setup.py
├── requirements.txt
├── environment.yml
├── config/
│   └── default.yaml
├── data/
│   ├── raw/
│   │   ├── nasa_pcoe/
│   │   └── calce/
│   ├── processed/
│   └── features/
├── docs/
│   ├── project_plan.md
│   ├── execution_plan.md
│   ├── literature_notes.md
│   ├── data_dictionary.md
│   ├── decisions_log.md
│   ├── meeting_notes.md
│   └── mlops.md
├── notebooks/
│   ├── 01_eda_nasa.ipynb
│   ├── 01_eda_calce.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_classical_ml_analysis.ipynb
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
│   │   ├── rf.py
│   │   ├── svr.py
│   │   ├── gpr.py
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
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_evaluation.py
├── experiments/
│   ├── mlruns/
│   ├── artifacts/
│   ├── fold_indices.json
│   ├── scalers/
│   ├── figures/
│   │   ├── phase1/
│   │   ├── phase2/
│   │   ├── phase3/
│   │   ├── phase6/
│   │   ├── phase7/
│   │   └── phase8/
│   ├── classical/
│   ├── dl/
│   └── results/
└── report/
    ├── main.tex
    ├── references.bib
    └── presentation/
```

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| CALCE data access delay | Start with NASA PCoE only; add CALCE when available |
| GPR O(n^3) memory blowup | Subsample to 5000 hard cap |
| Transformer overfits on small data | Report honestly; do not force competitiveness |
| DL training too slow on CPU | Use Colab/Kaggle free GPU |
| Scope creep | Strict phase gates; signed scope doc |
| Writing panic at end | Write incrementally per phase |
