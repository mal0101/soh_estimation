# State of Health Estimation of Lithium-Ion EV Batteries

Benchmark study comparing classical machine learning models (Random Forest, Support Vector Regression, Gaussian Process Regression) against deep learning architectures (LSTM, 1D CNN, Transformer) for estimating the State of Health of lithium-ion batteries.

SOH is defined as the ratio of a battery's current maximum capacity to its original rated capacity: `SOH = (Current Capacity / Rated Capacity) × 100`. It is the primary indicator of battery aging and remaining usable life. Since SOH cannot be measured directly in real operating conditions — Battery Management Systems only have access to voltage, current, and temperature signals — data-driven estimation from cycling data is a critical research problem.

This project develops a robust feature engineering pipeline extracting physically meaningful features from raw cycling data (voltage, current, temperature), validates all models under a cell-based leave-one-cell-out cross-validation scheme, and produces a deployability assessment covering inference latency, model size, and robustness to sensor noise.

## Datasets

- **NASA PCoE**: 18650 LCO cells (B0005, B0006, B0007, B0018)
- **CALCE**: CS2 pouch cells (CS2-33, CS2-34, CS2-35, CS2-36)

## License

MIT License
