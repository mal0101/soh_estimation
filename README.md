# State of Health Estimation of Lithium-Ion EV Batteries

Benchmark study comparing classical machine learning models (Random Forest, Support Vector Regression, Gaussian Process Regression) against deep learning architectures (LSTM, 1D CNN, Transformer) for estimating the State of Health of lithium-ion batteries.

SOH is defined as the ratio of a battery's current maximum capacity to its original rated capacity: `SOH = (Current Capacity / Rated Capacity) × 100`. It is the primary indicator of battery aging and remaining usable life. Since SOH cannot be measured directly in real operating conditions — Battery Management Systems only have access to voltage, current, and temperature signals — data-driven estimation from cycling data is a critical research problem.

This project develops a robust feature engineering pipeline extracting physically meaningful features from raw cycling data (voltage, current, temperature), validates all models under a cell-based leave-one-cell-out cross-validation scheme, and produces a deployability assessment covering inference latency, model size, and robustness to sensor noise.

## Datasets

- **NASA PCoE**: 18650 LCO cells (B0005, B0006, B0007, B0018)
- **CALCE**: CS2 pouch cells (CS2-33, CS2-34, CS2-35, CS2-36)

## Key Results

### NASA PCoE Dataset

| Model | RMSE | R² |
|-------|------|----|
| **SVR** | **0.015 ± 0.006** | **0.973 ± 0.020** |
| RF | 0.026 ± 0.018 | 0.918 ± 0.081 |
| GPR | 0.037 ± 0.020 | 0.835 ± 0.117 |
| LSTM | 0.048 ± 0.020 | -0.135 ± 1.366 |
| Transformer | 0.067 ± 0.022 | -0.142 ± 0.513 |
| CNN | 0.104 ± 0.059 | -2.504 ± 3.071 |

### CALCE Dataset

| Model | RMSE | R² |
|-------|------|----|
| RF | 0.023 ± 0.014 | 0.969 ± 0.034 |
| **Transformer** | **0.042 ± 0.010** | **0.914 ± 0.060** |
| SVR | 0.059 ± 0.043 | 0.870 ± 0.096 |
| GPR | 0.037 ± 0.025 | 0.844 ± 0.235 |
| LSTM | 0.061 ± 0.016 | 0.781 ± 0.227 |
| CNN | 0.144 ± 0.106 | -0.062 ± 1.015 |

### Combined Dataset (8 cells)

| Model | RMSE | R² |
|-------|------|----|
| **RF** | **0.024 ± 0.013** | **0.941 ± 0.050** |
| SVR | 0.055 ± 0.033 | 0.711 ± 0.354 |
| GPR | 0.123 ± 0.127 | -2.807 ± 7.435 |

**Key findings:**
- Classical ML (SVR, RF) dramatically outperforms all deep learning models on small datasets.
- On CALCE alone, the Transformer achieves competitive performance (R² = 0.914), suggesting DL can work with the right data conditions.
- On the combined dataset, RF is the most robust cross-chemistry model (RMSE = 0.024, R² = 0.941).
- All DL models produce negative R² on the combined dataset, indicating they fail to generalize across chemistries with this sample size.

For full methodology, pipeline instructions, and verification steps, see [GUIDE.md](GUIDE.md).

## License

MIT License
