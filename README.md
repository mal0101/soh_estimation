# State of Health Estimation of Lithium-Ion EV Batteries

Benchmark study comparing classical machine learning models (Random Forest, Support Vector Regression, Gaussian Process Regression) against deep learning architectures (LSTM, 1D CNN, Transformer) for estimating the State of Health of lithium-ion batteries.

SOH is defined as the ratio of a battery's current maximum capacity to its original rated capacity: `SOH = (Current Capacity / Rated Capacity) × 100`. It is the primary indicator of battery aging and remaining usable life. Since SOH cannot be measured directly in real operating conditions — Battery Management Systems only have access to voltage, current, and temperature signals — data-driven estimation from cycling data is a critical research problem.

This project develops a robust feature engineering pipeline extracting physically meaningful features from raw cycling data (voltage, current, temperature), validates all models under a cell-based leave-one-cell-out cross-validation scheme with **leakage-safe model selection** (feature selection, hyperparameter tuning, and early stopping all fitted inside the training folds), and produces a deployability assessment covering single-sample inference latency, model size, and robustness to sensor noise.

## Datasets

- **NASA PCoE**: 18650 LCO cells (B0005, B0006, B0007, B0018) — 636 discharge cycles
- **CALCE**: CS2 pouch cells (CS2-33, CS2-34, CS2-35, CS2-36) — 3,546 raw discharge cycles (2,559 after storage-gap/RPT cleaning)

Data integrity notes: CALCE cells contain multi-week test-pause periods (contiguous depressed-capacity runs that later recover) and periodic reference-performance-test dips; both are removed by the preprocessing filters. NASA B0006's reversible-fade transients are genuine electrochemistry and are retained.

## Key Results

All numbers below are regenerated programmatically from `experiments/*_results_*.yaml` — never hand-edited. RMSE/MAE/MaxAE are in SOH units; inference time is measured single-sample `predict()` latency on CPU.

<!-- RESULTS:AUTO -->
### NASA PCoE

| Model | RMSE | MAE | MaxAE | R² | Inference (ms) |
|-------|------|-----|-------|----|----------------|
| SVR | 0.025 ± 0.019 | 0.022 | 0.051 | 0.90 ± 0.11 | 0.041 |
| GPR | 0.028 ± 0.017 | 0.026 | 0.056 | 0.89 ± 0.10 | 0.099 |
| RF | 0.029 ± 0.019 | 0.024 | 0.082 | 0.88 ± 0.13 | 19.795 |
| LSTM | 0.040 ± 0.013 | 0.034 | 0.086 | 0.63 ± 0.24 | 0.941 |
| Transformer | 0.048 ± 0.014 | 0.042 | 0.092 | 0.36 ± 0.51 | 4.581 |
| CNN | 0.059 ± 0.019 | 0.048 | 0.132 | 0.07 ± 0.75 | 0.786 |
| Naive (mean) | 0.108 ± 0.025 | 0.090 | 0.210 | -0.41 ± 0.37 | 0.001 |

### CALCE CS2

| Model | RMSE | MAE | MaxAE | R² | Inference (ms) |
|-------|------|-----|-------|----|----------------|
| GPR | 0.019 ± 0.010 | 0.015 | 0.067 | 0.92 ± 0.06 | 1.067 |
| RF | 0.021 ± 0.010 | 0.016 | 0.059 | 0.91 ± 0.07 | 18.389 |
| SVR | 0.021 ± 0.010 | 0.016 | 0.104 | 0.90 ± 0.09 | 0.079 |
| Transformer | 0.021 ± 0.004 | 0.017 | 0.102 | 0.91 ± 0.03 | 2.205 |
| LSTM | 0.025 ± 0.004 | 0.020 | 0.129 | 0.89 ± 0.02 | 0.858 |
| CNN | 0.040 ± 0.013 | 0.032 | 0.192 | 0.62 ± 0.27 | 0.656 |
| Naive (mean) | 0.076 ± 0.011 | 0.064 | 0.192 | -0.01 ± 0.01 | 0.001 |

### Combined

| Model | RMSE | MAE | MaxAE | R² | Inference (ms) |
|-------|------|-----|-------|----|----------------|
| GPR | 0.025 ± 0.012 | 0.020 | 0.084 | 0.89 ± 0.09 | 2.255 |
| RF | 0.025 ± 0.017 | 0.020 | 0.073 | 0.89 ± 0.10 | 25.730 |
| SVR | 0.034 ± 0.018 | 0.026 | 0.127 | 0.79 ± 0.22 | 0.083 |
| Transformer | 0.034 ± 0.012 | 0.028 | 0.112 | 0.79 ± 0.14 | 2.078 |
| LSTM | 0.041 ± 0.024 | 0.035 | 0.120 | 0.71 ± 0.25 | 0.769 |
| CNN | 0.049 ± 0.027 | 0.041 | 0.147 | 0.46 ± 0.64 | 0.624 |
| Naive (mean) | 0.103 ± 0.038 | 0.091 | 0.187 | -0.42 ± 0.48 | 0.001 |

<!-- /RESULTS:AUTO -->

**Key findings (leakage-free protocol):**
- Classical ML remains the strongest choice at this scale; SVR offers the best accuracy-per-byte for embedded deployment.
- On NASA, SVR leads classical models while the LSTM is the strongest DL model (R² ≈ 0.6).
- On CALCE (~4× more cycles), the Transformer reaches near-parity with classical models (R² ≈ 0.91), confirming that DL needs substantially more data to compete.
- GPR's predictive intervals are poorly calibrated across held-out cells (empirical coverage far below nominal); see `experiments/analysis/gpr_calibration_*.yaml`.

For full methodology, pipeline instructions, and verification steps, see [GUIDE.md](GUIDE.md).

## Reproducing

```bash
conda env create -f environment.yml && conda activate soh
pip install -e .

soh-preprocess --dataset all          # processed pkls + SOH labels
soh-build-features --dataset all      # candidate feature matrices
soh-train-classical --dataset nasa    # repeat for calce, all
soh-train-dl --dataset nasa           # repeat for calce, all (~hours)
python scripts/run_robustness.py --dataset all
python scripts/run_analysis_extras.py --dataset all
python scripts/update_results_docs.py # regenerate README tables + manifest
```

## License

MIT License
