# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- LaTeX report writing (`report/main.tex`, `report/references.bib`)
- Presentation slides (`report/presentation/`)

## [0.4.0] - 2025-07-27

### Added
- Deep learning training orchestrator (`src/models/train_dl.py`)
- LSTM model with Optuna hyperparameter search (`src/models/lstm.py`)
- 1D CNN model with Optuna hyperparameter search (`src/models/cnn.py`)
- Transformer encoder model with Optuna hyperparameter search (`src/models/transformer.py`)
- Shared DL infrastructure: SOHDataset, train_loop, evaluate, checkpointing (`src/models/dl_base.py`)
- DL results saved to `experiments/dl_results.yaml`
- DL training analysis notebook (`notebooks/05_dl_training_analysis.ipynb`)
- Final comparison notebook (`notebooks/06_final_comparison.ipynb`)

### Changed
- DL config tuned for feasibility: n_seeds 5→3, n_trials 30→10, max_epochs 150→80
- SVR gamma bounds changed from string `1e-4` to float `0.0001` in config
- MLflow tracking migrated from file store to SQLite backend (`sqlite:///mlflow.db`)

### Fixed
- DL `_arrays_to_df` column naming mismatch (string keys vs integer keys) in lstm.py, cnn.py, transformer.py
- SVR gamma type conversion (string from Optuna categorical) in svr_model.py
- SVR Optuna objective crash on failed trials (added try/except)
- `train_dl.py` sequence_window config path and n_trials config read

## [0.3.0] - 2025-07-27

### Added
- Error analysis module: phase-based RMSE breakdown (`src/evaluation/error_analysis.py`)
- Interpretability module: SHAP TreeExplainer + Captum Integrated Gradients (`src/evaluation/interpretability.py`)
- Deployability module: inference time benchmark, model size (`src/evaluation/deployability.py`)
- Comparison module: model × metric table, ranking (`src/evaluation/comparison.py`)
- Visualization module: trajectory, scatter, comparison bar plots (`src/evaluation/visualizations.py`)
- Robustness utilities: noise injection, missing cycle simulation (`tests/test_robustness.py`)
- Comprehensive unit tests: 102 tests all passing

### Fixed
- Correlation filter exclusion in feature selection (`src/features/assembly.py`)
- Raise from exception handling in config loader (`src/utils/config.py`)

## [0.2.0] - 2025-07-27

### Added
- Classical ML training orchestrator (`src/models/train_classical.py`)
- Random Forest model with Optuna hyperparameter search (`src/models/rf_model.py`)
- SVR model with Optuna hyperparameter search (`src/models/svr_model.py`)
- GPR model with Matérn(1.5) kernel (`src/models/gpr_model.py`)
- Baseline naive mean prediction (`src/models/baseline.py`)
- Classical ML training notebook (`notebooks/04_classical_ml.ipynb`)
- Classical results saved to `experiments/classical_results.yaml`

### Results
- SVR: RMSE 0.015±0.006, R² 0.973±0.020
- RF: RMSE 0.025±0.017, R² 0.927±0.073
- GPR: RMSE 0.030±0.019, R² 0.885±0.097
- Naive: RMSE 0.112±0.027, R² -0.285±0.250

## [0.1.0] - 2025-07-23

### Added
- Initial repository with `docs/project_plan.md`
- MIT License
- Base `.gitignore` template
- Empty directory scaffolding: `src/`, `notebooks/`, `data/`, `experiments/`, `report/`

### Added (Phase 0)
- Project scaffolding and directory structure
- Conda environment setup with Python 3.11 (actual: 3.11.14, PyTorch 2.5.1 with MPS)
- Configuration system (`config/default.yaml`, `src/utils/config.py`)
- MLflow experiment tracking integration (`src/utils/tracking.py`)
- Reproducibility utilities (`src/utils/seeding.py`)
- `pyproject.toml` and `setup.py` for project packaging
- Unit test structure (`tests/`)

### Added (Phase 1)
- NASA PCoE data loader (`src/preprocessing/data_loader.py`)
- SOH label computation (`src/preprocessing/soh.py`)
- EDA notebook (`notebooks/01_eda_nasa.ipynb`)
- SOH labels saved to `data/processed/soh_labels.parquet` (636 rows)

### Added (Phase 2)
- Capacity axis computation (`src/preprocessing/capacity.py`)
- Savitzky-Golay voltage filtering (`src/preprocessing/filtering.py`)
- Uniform grid resampling (`src/preprocessing/resampling.py`)
- Cycle validation and filtering (`src/preprocessing/segmentation.py`)
- Preprocessing pipeline orchestrator (`src/preprocessing/pipeline.py`)
- Preprocessing validation notebook (`notebooks/02_preprocessing_validation.ipynb`)
- Preprocessed data saved to `data/processed/processed_cells.pkl`

### Added (Phase 3)
- ICA feature extraction (`src/features/ica.py`)
- Internal resistance estimation (`src/features/internal_resistance.py`)
- Energy and efficiency features (`src/features/energy.py`)
- Temperature features (`src/features/temperature.py`)
- Capacity fade rate feature (`src/features/trend.py`)
- Feature matrix assembly and selection (`src/features/assembly.py`)
- Feature engineering notebook (`notebooks/03_feature_engineering.ipynb`)
- Feature matrix saved to `data/features/feature_matrix.parquet` (636×16, 12 selected features)

### Added (Phase 4)
- Evaluation metrics: RMSE, MAE, MaxAE, R² (`src/evaluation/metrics.py`)
- Cell-based LOOCV, feature scaling, fold indices (`src/evaluation/validation.py`)
- Fold indices saved to `experiments/fold_indices.json`

### Changed
- Fixed `.gitignore` to exclude data and experiment artifacts while tracking docs
