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

### Data Reproduction

1. Download datasets to `data/raw/` as documented in `docs/project_plan.md` Section 4.
2. Run preprocessing: `python -m src.preprocessing.pipeline --config config/default.yaml`
3. Run feature engineering: `python -m src.features.assembly --config config/default.yaml`

### Training Reproduction

All training runs are seeded (seed=42 by default) and logged to MLflow:

```bash
# View all runs
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db

# Retrain a specific model
python -m src.models.train_classical --config config/default.yaml --model rf
python -m src.models.train_dl --config config/default.yaml --model lstm
```

### Evaluation Reproduction

```bash
python -m src.evaluation.comparison --config config/default.yaml
```

## Experiment Tracking

### MLflow Organization

```
Experiment: soh_benchmark
├── Run: rf_nasa_fold_0
│   ├── Params: {n_estimators: 300, max_depth: 20, ...}
│   ├── Metrics: {rmse: 0.018, mae: 0.012, maxae: 0.041, r2: 0.987}
│   └── Artifacts: model.pkl, feature_importance.png
├── Run: lstm_nasa_fold_0_seed_42
│   ├── Params: {lstm_1_units: 64, lr: 0.001, ...}
│   ├── Metrics: {rmse: 0.015, mae: 0.010, ...}
│   └── Artifacts: model.pt, training_loss.csv
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
