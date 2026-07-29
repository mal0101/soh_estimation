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

# Retrain DL models
python -m src.models.train_dl --config config/default.yaml --model lstm
python -m src.models.train_dl --config config/default.yaml --model cnn
python -m src.models.train_dl --config config/default.yaml --model transformer
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
│   ├── Metrics: {rmse: 0.025, mae: 0.020, maxae: 0.084, r2: 0.927}
│   └── Artifacts: model.pkl
├── Run: svr_nasa_fold_0
│   ├── Params: {C: 10.0, epsilon: 0.01, gamma: 0.01, ...}
│   ├── Metrics: {rmse: 0.015, mae: 0.011, maxae: 0.054, r2: 0.973}
│   └── Artifacts: model.pkl
├── Run: gpr_nasa_fold_0
│   ├── Params: {kernel: Matern(1.5), ...}
│   ├── Metrics: {rmse: 0.030, mae: 0.027, maxae: 0.098, r2: 0.885}
│   └── Artifacts: model.pkl
├── Run: lstm_nasa_fold_0_seed_42
│   ├── Params: {lstm_1_units: 64, lr: 0.001, ...}
│   ├── Metrics: {rmse: 0.048, mae: 0.043, maxae: 0.098, r2: -0.135}
│   └── Artifacts: model.pt
├── Run: cnn_nasa_fold_0_seed_42
│   ├── Params: {conv1_channels: 32, lr: 0.001, ...}
│   ├── Metrics: {rmse: 0.104, mae: 0.085, maxae: 0.227, r2: -2.504}
│   └── Artifacts: model.pt
└── Run: transformer_nasa_fold_0_seed_42
    ├── Params: {d_model: 64, nhead: 4, lr: 0.001, ...}
    ├── Metrics: {rmse: 0.067, mae: 0.059, maxae: 0.139, r2: -0.142}
    └── Artifacts: model.pt
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
