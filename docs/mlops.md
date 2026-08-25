# MLOps Guide

Reproducibility, experiment tracking, and deployment assessment instructions.

## Reproducing Results

### Environment Reproduction

```bash
# Option A: From conda environment.yml (exact conda packages)
conda env create -f environment.yml
conda activate soh

# Option B: From pip requirements.txt (pip-installed packages only)
conda create -n soh python=3.11 -y
conda activate soh
pip install -r requirements.txt
```

**Actual environment:** Conda `soh`, Python 3.11.14, PyTorch 2.5.1 (MPS: True), numpy 1.26.4, MLflow 3.14.

### Data Reproduction

1. Download NASA PCoE `.mat` files to `data/raw/nasa_pcoe/` (cells B0005, B0006, B0007, B0018).
2. Run preprocessing: `python -m src.preprocessing.pipeline --config config/default.yaml`
3. Run feature engineering: `python -m src.features.assembly --config config/default.yaml`

### Training Reproduction

All training runs are logged to MLflow SQLite backend (`sqlite:///mlflow.db`):

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Retrain classical ML models
python -m src.models.train_classical --config config/default.yaml --model rf
python -m src.models.train_classical --config config/default.yaml --model svr
python -m src.models.train_classical --config config/default.yaml --model gpr

# Retrain DL models (models trained together; ~hours)
python -m src.models.train_dl --config config/default.yaml --dataset nasa   # or calce / all

# Analysis extras + robustness
python scripts/run_robustness.py --dataset nasa
python scripts/run_analysis_extras.py --dataset nasa
```

### Evaluation Reproduction

Model comparison tables are produced by the training orchestrators and
regenerated into docs via `scripts/update_results_docs.py`.

## Experiment Tracking

### MLflow Organization

```
Experiment: soh_benchmark  (tracking URI sqlite:///mlflow.db, anchored to project root)
├── Run: fold_0_naive            tags: {dataset, fold=0, model=naive}
├── Run: fold_0_rf               params: Optuna best; metrics incl. per-fold rmse/r2
├── Run: fold_0_svr / fold_0_gpr
├── Run: fold_0_lstm_seed42      tags: {dataset, fold, model=lstm, seed=42}
├── ... (3 models x n_folds x n_seeds per dataset)
└── Legacy pre-remediation runs have NO dataset tag (kept for history;
    notebooks filter them out via `tags.dataset`).

Authoritative metric values live in experiments/*_results_{dataset}.yaml —
this file is not updated with numbers to avoid a second source of truth.
```

### Log Standards

- Every run must log: hyperparameters, all evaluation metrics, model artifact
- Use descriptive run names: `{model}_{dataset}_{fold}_{seed}`
- Tag runs with phase number and any special conditions

## Model Registry

Models are saved in:
- `experiments/classical/{model}/fold_{i}/` — sklearn pickle files
- `experiments/dl/{model}/seed_{i}/` — PyTorch state dicts

For the deployability assessment, the best model per type is registered in MLflow.

## Deployment Assessment

We do NOT deploy to production. We assess deployability by measuring:

1. **Inference latency**: 1000 sequential CPU forward passes, averaged
2. **Model file size**: On-disk size in MB
3. **Noise robustness**: RMSE degradation under 0.5%, 1%, 2% Gaussian noise
4. **Missing cycle robustness**: RMSE degradation under 10%, 20%, 30% cycle dropout

Target thresholds (from BMS hardware constraints):
- RMSE < 2% on held-out cells
- MaxAE < 5%
- Inference time < 200 ms on CPU
- Model size < 4 MB (BMS flash limit)

## CI/CD (Future)

Not implemented in this internship scope. Potential additions:
- GitHub Actions for linting (ruff) and testing (pytest)
- Automated model evaluation on push
- DVC for data versioning


## Post-Remediation Notes (2026-08-25)

- Run naming: `fold_{i}_{model}` (classical), `fold_{i}_{model}_seed{s}` (DL).
- Every run carries `dataset`, `fold`, `model`(, `seed`) tags.
- Persisted models live in `experiments/models/{dataset}/` (joblib for
  classical, torch state dicts for DL); sizes feed deployability reports.
- Inference latency in results YAMLs is REAL single-sample predict() latency
  (`inference_time_ms_mean/p95`); `train_time_mean_s` is the fold wall time.
