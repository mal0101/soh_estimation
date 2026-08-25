# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] - 2026-08-25 — Methodology Remediation

Full audit found methodological defects that inflated all previously reported
metrics, plus stale/mixed-provenance artifacts. Every defect was fixed and the
entire benchmark re-run under a leakage-safe protocol. All numbers in the
README/notebooks/report are regenerated from `experiments/*.yaml`.

### Fixed (methodology)
- `capacity_fade_rate` no longer contains the current row's SOH target
  (strictly past labels; NaN warm-up instead of synthetic zeros)
- Feature selection (correlation filter + RF importance) fitted per LOOCV
  fold on training cells only (`fit_feature_selection`); saved matrices are
  FULL candidate sets
- Optuna tuning and DL early stopping use an inner cell split of the training
  cells; the outer test cell is touched once for final reporting
- DL early stopping restores best weights; final models refit for exactly the
  selected epoch count
- DL double-windowing removed: sequence datasets built directly from
  cell-grouped frames so windows never cross cell boundaries
- `inference_time_mean_s` stored fold TRAINING time: renamed
  `train_time_mean_s`; real single-sample latency benchmarked into
  `inference_time_ms_mean/p95`

### Fixed (data integrity)
- CALCE SOH labels were produced by an EDA notebook on RAW unfiltered cells:
  labels are now pipeline-only output
- CS2_36 Q_initial poisoning (+12% label bias) fixed via median reduction over
  cycles [3,10] (`preprocessing.soh.q_reduction: median`)
- Cycle-integrity filters remove storage/test-pause runs (~250-cycle depressed
  blocks that later recover) and isolated RPT dips anywhere in life; genuine
  EOL fade and NASA B0006 reversible transients retained (CALCE labels
  3,546 → 2,559 rows; zero sub-0.6 SOH artifacts remain)
- dQ/dV sign inversion fixed (discharge ICA peaks were artifacts); near-zero
  voltage gaps deduplicated; peaks gated to [3.0, 4.35] V
- Coulombic efficiency pairing fixed (off-by-one on CALCE, 84% NaN on NASA);
  impossible ratios rejected
- Internal-resistance estimator requires a genuine current step (relative ΔI guard)
- Phase analysis includes SOH-capped samples (top phase closed interval)

### Added
- `src/features/build_features.py` CLI (`soh-build-features`)
- `scripts/update_results_docs.py`: regenerates README tables +
  report/numbers_manifest.json from YAMLs
- `scripts/run_robustness.py`: sensor-noise and missing-cycle robustness
- `scripts/run_analysis_extras.py`: SHAP importances, GPR calibration,
  deployability vs targets
- Persisted per-fold models (`experiments/models/{dataset}/`) as joblib/torch
- Per-dataset MLflow run tags; `tests/test_leakage.py` (regression pins for
  every anti-leakage guarantee; suite now 134 tests)
- Fold-level diagnostics in results YAMLs (per-fold metrics, selected
  features, val RMSE, seed variability for DL)

### Changed
- All artifacts use explicit suffixes `_nasa` / `_calce` / `_all`; stale
  unsuffixed/orphan results deleted
- Config: dead keys removed (`dl.optimizer/loss/max_epochs_cpu`, `ffn_dim`,
  `bms_flash_*`, unread tracking flags), search spaces now actually consumed
  by trainers, `seeding.base_seed` wired through
- Packaging: portable requirements.txt, environment.yml synced with
  pyproject.toml, setup.py reduced to a shim, python pinned >=3.11,<3.12

### Removed
- Pre-remediation result YAMLs, stale unsuffixed `fold_indices.json`,
  notebook-based label writing

## [Unreleased]

### Added
- CALCE CS2 data loader (`load_calce_cell`, `load_all_calce_cells`) in `src/preprocessing/data_loader.py`
- Extracted CALCE CS2 cells (CS2_33, CS2_34, CS2_35, CS2_36) to `data/raw/calce/`
- `--dataset` CLI flag to preprocessing pipeline (`nasa`, `calce`, `all`)
- `--dataset` CLI flag to classical ML and DL training scripts
- CALCE preprocessing pipeline output (`data/processed/processed_cells_calce.pkl`, `soh_labels_calce.parquet`)
- CALCE feature matrix (`data/features/feature_matrix_calce.parquet`) and combined matrix
- Classical ML results on CALCE and combined datasets (`experiments/*_calce.yaml`, `*_all.yaml`)
- DL results on CALCE (LSTM RMSE=0.061, CNN RMSE=0.144, Transformer RMSE=0.042)
- CALCE EDA notebook (`notebooks/01_eda_calce.ipynb`)
- Updated final comparison notebook with multi-dataset results
- LaTeX technical report (`report/main.tex`) — 8-page benchmark study
- Beamer presentation (`report/presentation.tex`) — 11 slides
- `console_scripts` entry points: `soh-preprocess`, `soh-train-classical`, `soh-train-dl`
- Input shape validation in `captum_integrated_gradients`
- `target_size_mb` parameter in `deployability_report`

### Changed
- `assembly.py` uses `cell_data["dataset"]` instead of hardcoded `"nasa_pcoe"`
- `save_feature_matrix` accepts `dataset` parameter for filename suffix
- Feature matrix saved with per-dataset suffixes (e.g., `feature_matrix_nasa.parquet`)
- `deployability.py` imports `torch` lazily (no longer blocks classical-only usage)
- `rank_models` uses stable sort (`kind="mergesort"`) for reproducible tie-breaking
- `logging.basicConfig` moved from `run_pipeline()` to CLI entry points only
- `train_classical.py` creates `experiments/` directory before writing results

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

Note: entries below [Unreleased] describe the pre-remediation codebase; the
0.2.0 "Results" block reflects the stale numbers superseded by 0.5.0.
