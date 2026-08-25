# State of Health (SOH) Estimation of Lithium-Ion EV Batteries
## Complete Project Plan — ML & DL Benchmark Study
### Capgemini Internship — Revised Scope (July–August 2025)

---

## Table of Contents

1. Project Overview and Objectives
2. Scope Definition
3. Prerequisites and Background Knowledge
4. Data Sources
5. Phase-by-Phase Project Plan
6. Time Estimates
7. Tools and Environment Setup
8. Evaluation Metrics
9. Pitfalls and Common Mistakes to Avoid
10. Deliverables Checklist
11. References and Learning Resources

---

## 1. Project Overview and Objectives

### What This Project Is About

State of Health (SOH) is defined as the ratio of a battery's current maximum capacity to its original rated capacity, expressed as a percentage: SOH = (Current Capacity / Rated Capacity) x 100. It is the primary indicator of battery aging and remaining usable life. As lithium-ion batteries degrade over charge-discharge cycles, temperature exposure, and varying usage conditions, their SOH declines — affecting EV range, safety, warranty calculations, and second-life reuse decisions.

The central challenge is that SOH cannot be measured directly in real operating conditions. Battery Management Systems (BMS) only have access to voltage, current, and temperature signals. True capacity is only known through a full discharge test, which makes ground-truth labels sparse and expensive to obtain. Physics-based models can estimate SOH but require internal electrochemical parameters that are not accessible in deployed systems and are computationally heavy. Classical machine learning methods require manual feature engineering and generalize poorly across different cell chemistries and operating conditions. Pure deep learning approaches achieve high accuracy but demand large labeled datasets.

The gap this project addresses is the lack of hybrid ML/DL models that are simultaneously accurate, generalizable across chemistries and conditions, interpretable, and lightweight enough for deployment inside production BMS hardware.

### Core Objectives

- Build and rigorously benchmark three classical ML models (Random Forest, Support Vector Regression, Gaussian Process Regression) against three deep learning architectures (LSTM, 1D CNN, Transformer) for SOH estimation — six models in total.
- Develop a robust feature engineering pipeline that extracts physically meaningful features from raw cycling data (voltage, current, temperature, time).
- Validate all models under a cell-based leave-one-cell-out cross-validation scheme to ensure results are not inflated by data leakage.
- Produce a deployability assessment covering inference latency, model size, and robustness to sensor noise.
- Deliver a technical report suitable for submission as a workshop paper or preprint.

### Why This Matters

- Safety: early detection of abnormal degradation before it leads to thermal events.
- Economic efficiency: fairer warranty structuring and resale pricing.
- Sustainability: enables confident second-life battery reuse and informed recycling decisions.
- Industry adoption: a lightweight, deployable estimator removes the main barrier to real-world BMS integration.
- Driver trust: transparent, real-time health and range estimates.

---

## 2. Scope Definition

This section defines exactly what is in and out of scope. Agree on this with your Capgemini supervisor before starting Phase 1. Scope creep is the primary risk on a 5-6 week timeline.

### In Scope

- Two datasets: NASA PCoE and CALCE. These cover LCO chemistry under controlled lab conditions and provide enough cells for robust cross-validation.
- SOH definition: capacity-based, SOH(n) = Q_discharge(n) / Q_initial, where Q_initial is the average of cycles 3-10 for each individual cell.
- End-of-life threshold: SOH = 0.80 (80% of initial capacity), the industry standard.
- Six models benchmarked under identical experimental conditions: RF, SVR, GPR (classical ML) and LSTM, 1D CNN, Transformer (deep learning).
- Feature engineering: ICA-derived features, internal resistance, discharge energy, coulombic efficiency, temperature statistics per cycle.
- Validation strategy: cell-based leave-one-cell-out cross-validation within each dataset.
- Interpretability: SHAP analysis on RF, gradient attribution on the best-performing DL model.
- Deployability assessment: CPU inference time and model size benchmarking, plus a robustness test under simulated sensor noise.
- Technical report (6-10 pages) covering the full methodology and results.

### Out of Scope

- Cross-dataset training (training on NASA, testing on Oxford or CALCE). This would require harmonizing two preprocessing pipelines and is not feasible in 5-6 weeks without cutting something more important.
- Additional datasets beyond NASA PCoE and CALCE. If preprocessing finishes ahead of schedule, adding the Severson/MATR dataset is the first and only recommended extension.
- Calendar aging (degradation due to storage rather than cycling).
- Model quantization, pruning, or ONNX export. Deployability is assessed analytically, not through actual embedded deployment.
- Electrochemical Impedance Spectroscopy (EIS) features, even where the dataset provides them. They add complexity to preprocessing without being necessary for competitive SOH estimation.

### Stretch Goals (only if core scope is complete before mid-August)

In order of priority:
1. Add the Severson/MATR dataset (LFP chemistry, 124 cells) as a third dataset and repeat the benchmark — this significantly strengthens the generalization claim in the paper.
2. Add a fourth DL model: GRU, as a lightweight alternative to LSTM, to complete the recurrent model comparison.
3. Attempt ONNX export of the best-performing DL model and benchmark on CPU with the ONNX runtime.

---

## 3. Prerequisites and Background Knowledge

Before starting the project work, you should be comfortable with or must quickly acquire the following.

### Electrochemistry and Battery Fundamentals

You do not need to be an electrochemist, but you need enough domain knowledge to make sense of the data and interpret results. Specifically, understand:

- The charge-discharge cycle and what capacity fade means physically.
- The difference between State of Charge (SOC) and State of Health (SOH). SOC is the current energy level (like a fuel gauge); SOH is the long-term health of the battery itself. They are completely different quantities and are frequently confused.
- Why degradation is non-linear: it depends on depth of discharge, C-rate (how fast the battery is charged or discharged), temperature history, and cell chemistry.
- What Incremental Capacity Analysis (ICA) is — the most important physics-informed feature extraction technique in this project. ICA computes dQ/dV, whose peaks correspond to phase transitions in the cathode material and shift measurably with aging.
- What internal resistance (IR) is and why it increases with aging.
- The two lithium-ion chemistries in your datasets: LCO (Lithium Cobalt Oxide) in NASA PCoE, and LCO/LFP in CALCE. Their degradation curves look different and this will show up in your cross-cell variance.

### Signal Processing

- Noise filtering: raw voltage and current signals contain sensor noise. You need Savitzky-Golay filtering before computing derivatives for ICA.
- Resampling: datasets record signals at different frequencies. Uniform resampling onto a capacity grid is required before feature extraction.
- Cycle segmentation: identifying the start and end of each charge or discharge cycle from the current signal is a non-trivial but essential preprocessing step.

### Machine Learning

You likely know most of this already, but confirm comfort with:

- The full supervised learning pipeline: train/validation/test split, cross-validation, overfitting detection, hyperparameter tuning.
- Random Forest: ensemble of decision trees, aggregated by averaging. Inherently interpretable via feature importance. No feature scaling required.
- Support Vector Regression: margin-based regression using kernel trick. Requires feature scaling. Does not scale well beyond ~50,000 samples.
- Gaussian Process Regression: non-parametric Bayesian model that outputs a mean prediction and a calibrated uncertainty estimate (confidence interval) for every prediction. This is what makes it uniquely valuable for safety-critical applications. Computationally O(n^3) in training — only feasible on subsampled data or with sparse approximations.
- SHAP (SHapley Additive exPlanations): a game-theory-based method for attributing a model's prediction to individual input features. Works best with tree models; also available for DL via DeepSHAP.

### Deep Learning

- LSTM (Long Short-Term Memory): the foundational recurrent architecture for sequence modeling. Uses gating mechanisms to selectively remember or forget information across timesteps. This is your anchor DL model — the one with the most literature support for battery SOH.
- 1D CNN (Convolutional Neural Network applied to sequences): applies learnable filters along the time dimension to extract local patterns. Faster to train than LSTM, parallelizable, and often competitive in accuracy. Teaches you a fundamentally different inductive bias from recurrence.
- Transformer: uses self-attention to model dependencies between all positions in a sequence simultaneously, without recurrence. Currently the dominant architecture in sequence modeling broadly. For small datasets it requires more careful regularization than LSTM. Understanding attention is essential regardless of this project's outcome because it underpins almost all modern foundation models.
- Gradient-based attribution (Integrated Gradients): the DL equivalent of SHAP for attribution analysis. Quantifies how much each input feature contributed to a specific prediction.

### New Topic to Learn: Gaussian Process Regression

Since you have not worked with GPR before, budget dedicated study time before implementing it. The key concepts to understand are:

- A Gaussian Process is a prior over functions, not over parameters. Instead of learning fixed weights, it maintains a distribution over all possible functions that are consistent with the training data.
- The kernel (covariance) function defines what "similar" means between two inputs. The RBF (squared exponential) kernel assumes smooth, infinitely differentiable functions. The Matern 3/2 kernel assumes once-differentiable functions and is more appropriate for physical processes with sharper transitions. For SOH estimation, Matern 3/2 is the better default choice.
- The output of GPR is a Gaussian distribution at each test point: a mean (your point prediction) and a variance (your uncertainty). The uncertainty is wide where training data is sparse and narrow where it is dense. This is physically meaningful for SOH: early in battery life, many degradation trajectories are plausible, so uncertainty should be high. Near end of life, trajectories converge, so uncertainty should decrease.
- GPR does not scale to large datasets without approximation. With more than roughly 5,000 training samples, use scikit-learn's GaussianProcessRegressor with a subsample, or switch to GPyTorch with inducing points (sparse GP). For NASA PCoE and CALCE combined, you are likely within the direct computation range, but check the sample count before deciding.

Recommended resource for GPR: Chapter 2 of "Gaussian Processes for Machine Learning" by Rasmussen and Williams (freely available at gaussianprocess.org). Read sections 2.1 and 2.2 — that is sufficient for implementation.

---

## 4. Data Sources

### Primary Datasets (In Scope)

**NASA Prognostics Center of Excellence (PCoE) Battery Dataset**
- URL: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
- What it contains: 18650 lithium-ion cells (LCO chemistry) cycled at room temperature under various C-rates until end of life. Channels: voltage, current, temperature, impedance, and measured discharge capacity per cycle. Data is provided as MATLAB .mat files.
- Why it is the right starting point: the most widely cited benchmark in SOH estimation literature. Large enough to train deep learning models. Has clear end-of-life markers. Directly comparable to most published results, which helps you contextualize your own numbers.
- Key limitation: single chemistry (LCO), controlled room temperature, constant C-rates. Models trained here alone will not generalize to real-world conditions.
- Cells to use: B0005, B0006, B0007, B0018 are the standard four cells used in most papers. Use all four, with one held out per cross-validation fold.

**CALCE (Center for Advanced Life Cycle Engineering) Battery Dataset**
- URL: https://calce.umd.edu/battery-data
- What it contains: Multiple cell types including CS2 (LCO pouch, 1.1 Ah) and CX2 cells, cycled under different temperature and C-rate conditions. Includes both charge and discharge data per cycle, stored as CSV or Excel files.
- Why it complements NASA PCoE: different cell form factor (pouch vs. cylindrical), different rated capacity, different cycling conditions. Using both datasets in your cross-validation pool forces the model to generalize across cell types, which strengthens any generalization claim.
- Key limitation: dataset organization can be inconsistent across cell batches. Some files require registration. Send a short request email to CALCE stating the internship context — they typically respond within a few days.
- Cells to use: the CS2 series (CS2-33, CS2-34, CS2-35, CS2-36 are commonly used). Verify availability at the time of access.

### Additional Recommended Dataset (Stretch Goal Only)

**Stanford Battery Aging Dataset (Severson et al., 2019)**
- URL: https://data.matr.io/1/
- What it contains: 124 LFP/graphite cells cycled under 72 different fast-charging protocols. The largest publicly available aging dataset.
- Why it is valuable: LFP chemistry has a notoriously flat voltage curve, which makes ICA features behave differently than LCO. Including it would add a third chemistry to the benchmark and significantly strengthen the paper's generalization claims.
- Only attempt this if core scope is complete by early-to-mid August.

### Data Licensing

Document the license for each dataset you use in your report. NASA PCoE and CALCE are available for non-commercial research use, which covers an internship context. Severson/MATR data is released under a Creative Commons license. Confirm terms at the time of access and cite the original dataset papers, not just the download URLs.

---

## 5. Phase-by-Phase Project Plan

### Phase 0: Setup and Literature Review (Days 1-7)

This phase establishes your understanding of the problem space and prevents you from duplicating known work. One week maximum — do not let it expand.

**Step 0.1: Read the essential papers**

Read the following in this order. Each is chosen because it directly informs a decision you will make in this project:

- Severson et al. (2019), Nature Energy: establishes the benchmark for data-driven SOH from features extracted early in battery life. Read the Methods section carefully — their feature engineering is directly reusable.
- Shen et al. (2019), Journal of Energy Storage: demonstrates CNN for battery SOH estimation. Gives you the baseline CNN architecture to adapt.
- Zhang et al. (2020), Nature Communications: demonstrates how ICA and DVA features change with aging. Essential for understanding what your features are measuring physically.
- Hu et al. (2020), Joule: a review paper. Skim it for the model comparison tables and the discussion of failure modes. You do not need to read it cover to cover.
- One recent (2022-2024) paper using Transformers for battery SOH: search Google Scholar for "Transformer state of health lithium-ion" filtered to the last two years. Pick the most-cited result. Read the architecture section and the experimental setup section.

Do not aim for exhaustive literature coverage. Five papers read carefully is worth more than twenty skimmed.

**Step 0.2: Write and share your scope document**

Write a one-page scope document based on Section 2 of this plan. Share it with your Capgemini supervisor and get explicit written confirmation before starting data work. This protects you if scope creep becomes a conversation later.

**Step 0.3: Set up your environment and project structure**

Set up a Git repository immediately. Use this directory structure from day one:

```
soh-estimation/
├── data/
│   ├── raw/           # untouched downloaded files
│   ├── processed/     # output of preprocessing scripts
│   └── features/      # output of feature engineering scripts
├── notebooks/         # exploratory analysis only; no final code lives here
├── src/
│   ├── preprocessing/ # cycle segmentation, filtering, SOH labeling
│   ├── features/      # ICA, IR, energy, temperature features
│   ├── models/        # one file per model: rf.py, svr.py, gpr.py, lstm.py, cnn.py, transformer.py
│   └── evaluation/    # metrics, plots, SHAP
├── experiments/       # saved results, model artifacts, logs
├── report/            # LaTeX or Markdown source for the paper
└── README.md
```

Install and configure your experiment tracking tool (MLflow or Weights & Biases) before writing any model code. Every training run from Phase 5 onward must be logged automatically.

---

### Phase 1: Data Acquisition and Inventory (Days 7-10)

**Step 1.1: Download and verify all files**

Download NASA PCoE and CALCE datasets. For NASA PCoE, the standard cells are B0005, B0006, B0007, and B0018. Verify file integrity. Load each file in Python and confirm you can read the voltage, current, temperature, time, and capacity channels without errors. For .mat files, use scipy.io.loadmat or h5py depending on the MATLAB version used to create them (h5py is required for MATLAB v7.3+ files).

**Step 1.2: Exploratory data analysis (EDA)**

For each cell in each dataset, independently:

- Plot capacity vs. cycle number. Confirm the generally declining trend. Flag any cell where capacity jumps upward mid-life or drops to zero suddenly — these may be corrupted and should be excluded with documentation.
- Plot the raw voltage vs. time (or vs. capacity) for cycles 1, 50, 100, and the final cycle of the same cell on one chart. You should see the curve shift downward and narrow as the cell ages. If you do not see this, something is wrong with your data loading.
- Record: number of cycles, sampling frequency, C-rate, temperature, capacity range, rated capacity, and file format. Compile this into a data inventory table.

**Step 1.3: Define your SOH label**

Use capacity-based SOH throughout: SOH(n) = Q_discharge(n) / Q_initial, where Q_initial is the average measured discharge capacity over cycles 3-10 of that specific cell. Do not use the manufacturer's rated capacity as the denominator — individual cells vary by up to 5% from nominal due to manufacturing tolerance, and this introduces a systematic offset that the model will partially learn rather than the true degradation signal.

Cap SOH at 1.0 for the first few cycles where capacity may slightly exceed Q_initial due to formation effects.

---

### Phase 2: Data Preprocessing (Days 10-18)

This is the most time-consuming phase and the one where the most irreversible mistakes happen. Budget 8 days. Do not rush it.

**Step 2.1: Noise filtering**

Apply a Savitzky-Golay filter to the voltage vs. capacity curve of each cycle before computing any derivative. Standard parameters: window length 51, polynomial order 3. Adjust window length per dataset if the signal is noisier or smoother than expected. Verify the filter does not significantly distort the voltage plateau regions by overlaying the raw and filtered curves on the same plot for at least three cycles per cell.

Do not use a simple moving average for this — it blurs the ICA peaks too aggressively.

**Step 2.2: Resampling to a uniform capacity grid**

Interpolate each voltage-capacity curve onto a uniform capacity grid of 1000 points from 0 to Q_initial. This makes all cycles from all cells directly comparable for feature extraction. Use linear interpolation (scipy.interpolate.interp1d) for this step.

**Step 2.3: Cycle segmentation**

For each cell, identify the start and end of each charge cycle and each discharge cycle. Some datasets provide a cycle number column; others require you to infer cycle boundaries from the sign of the current signal (positive = charge, negative = discharge, near-zero = rest). Implement a segmentation function and manually verify it on at least three cells per dataset by plotting the segmented cycles.

Handle partial cycles explicitly. Define a threshold (e.g., cycles where less than 90% of Q_initial was discharged are discarded) and apply it consistently across all cells and datasets. Document how many cycles were discarded per cell.

**Step 2.4: Compute SOH labels**

For each complete discharge cycle of each cell, compute and store the SOH value. At the end of this step, you should have a table with one row per (cell, cycle) pair and columns for: cell ID, dataset, cycle number, SOH, and the filtered/resampled voltage-capacity curve for that cycle.

---

### Phase 3: Feature Engineering (Days 18-25)

This phase translates domain knowledge into the numerical inputs your models will consume. Do not skip or abbreviate it — the quality of your features is the single largest determinant of model performance on this problem.

**Step 3.1: Incremental Capacity Analysis (ICA)**

Compute dQ/dV as a function of voltage for each cycle's charge phase. From the dQ/dV curve, extract the following features per cycle:

- Height of the primary peak (the tallest peak in the dQ/dV curve).
- Voltage position of the primary peak.
- Width of the primary peak at half its maximum height (FWHM).
- Area under the primary peak (integral of dQ/dV over the peak region).
- If a secondary peak is visible (chemistry-dependent): the ratio of primary to secondary peak heights.

ICA is only reliable at low charge rates (below C/5). For datasets with higher C-rates, the peaks will be smeared and the features will be noisy. Check this for each dataset and document it. If the C-rate is too high for reliable ICA, include the raw dQ/dV curve features but flag them as lower-confidence in the report.

Plot the dQ/dV curves at early, mid, and late life for one cell in each dataset before computing features. You should see the peaks shift and shrink with aging — this visual confirmation is essential before you trust the automated feature extraction.

**Step 3.2: Internal Resistance Estimation**

Estimate internal resistance per cycle from the voltage response at the start of the discharge pulse: IR = delta_V / delta_I, where delta_V is the instantaneous voltage drop and delta_I is the applied current step. This feature typically increases monotonically with aging and is one of the most reliable single predictors of SOH.

If the dataset includes Electrochemical Impedance Spectroscopy (EIS) data (NASA PCoE does), extract the real part of impedance at 1 kHz as a supplementary internal resistance estimate.

**Step 3.3: Cycle-Level Energy and Efficiency Features**

For each discharge cycle:

- Total discharge energy: integral of voltage times current over the cycle (V * I * dt, summed). This captures both how much charge was delivered and at what voltage, making it more informative than capacity alone.
- Mean discharge voltage: a proxy for the overall state of the voltage curve.
- Discharge duration at constant current: longer duration indicates higher capacity.
- Coulombic efficiency: Q_discharge(n) / Q_charge(n). This ratio should be close to 1.0 for healthy cells and deviates with aging and temperature. Compute it for every cycle.

**Step 3.4: Temperature Features per Cycle**

For each cycle, compute: mean temperature, maximum temperature, minimum temperature, and temperature range (max - min). Elevated mean temperature accelerates degradation. High temperature range within a single cycle may indicate thermal management issues.

**Step 3.5: Capacity Fade Rate (Trend Feature)**

Compute the local slope of the SOH curve over the last 10 cycles as a feature: SOH_slope(n) = (SOH(n) - SOH(n-10)) / 10. This tells the model how fast the battery is currently degrading, which is predictive of near-future SOH independently of the absolute SOH value.

Only compute this feature for cycles n > 10. For the first 10 cycles, use the slope computed from available cycles (e.g., for cycle 5, use cycles 1-5).

**Step 3.6: Feature Selection**

After computing all features, perform the following cleanup:

- Compute a pairwise Pearson correlation matrix for all features. Remove one feature from each pair with correlation above 0.95 — keeping both adds no information and can destabilize SVR and GPR.
- Train a quick Random Forest on the full feature set and rank features by importance. Keep the top 15-20 features. Record which features were kept and which were dropped — this is a result worth reporting.
- Scale the feature matrix using StandardScaler (fit on training cells only, applied to all cells) before feeding to SVR and GPR. RF does not require scaling. The DL models use the same scaled features.

---

### Phase 4: Validation Strategy (Day 25)

This is a half-day planning step, but it governs the validity of everything that follows. Get it right.

**Cell-based leave-one-cell-out cross-validation (LOOCV)**

For each dataset independently, perform leave-one-cell-out cross-validation: train on all cells except one, test on the left-out cell, rotate through all cells. This produces one test score per cell. Report the mean and standard deviation of the metric across all folds.

This is the correct validation strategy for this problem. It ensures the model is evaluated on cells it has never seen during training, which is the actual deployment scenario. Any other split strategy (random cycle split, random sample split) leaks information and inflates performance.

For the DL models, within each fold, use 80% of the training cycles for model training and 20% as a validation set for early stopping. The test cell's cycles are never used for any training or tuning decision.

**Sequence window definition for DL models**

DL models receive a sequence of W consecutive cycles as input and predict SOH at cycle W+1. Set W = 20 cycles. This window size is large enough to capture degradation trends while keeping sequence length tractable. Slide the window across each cell's history to generate training samples. Never slide a window across cell boundaries — a window must belong entirely to one cell.

---

### Phase 5: Classical ML Training (Days 26-32)

Once the feature matrix and validation folds are prepared, classical ML training is fast. The bottleneck here is hyperparameter search, not computation.

**Step 5.0: Establish a baseline**

Before training any model, compute the naive baseline: predict the mean SOH of the training set for all test samples. Record RMSE and MAE. Every model must beat this baseline. If a model does not, something is wrong with either the model or the features.

**Step 5.1: Random Forest (RF)**

Hyperparameters to tune using Optuna (Bayesian search, 50-100 trials):
- Number of trees: search range 100 to 500.
- Max depth: search range None (unlimited) and 10 to 40.
- Min samples per leaf: search range 1 to 10.
- Max features per split: search over "sqrt", "log2", and 0.5.

RF does not require feature scaling. After tuning, extract and record feature importances — this is your first interpretability result and will directly inform the SHAP analysis.

**Step 5.2: Support Vector Regression (SVR)**

Always scale the feature matrix before fitting SVR. Use the StandardScaler fitted on training cells only.

Hyperparameters to tune:
- Kernel: fix to RBF (no need to search others for this problem).
- C (regularization strength): log-uniform search over 0.01 to 1000.
- Epsilon (tube width): search over 0.001 to 0.1.
- Gamma: search over "scale", "auto", and log-uniform 1e-4 to 1.

If the training set exceeds 50,000 samples after windowing, switch to sklearn's LinearSVR or use a random subsample of 30,000 samples for SVR only. Document this if it occurs.

**Step 5.3: Gaussian Process Regression (GPR)**

GPR is the most computationally expensive classical model and requires careful handling.

First, check the size of your training set per fold. If it exceeds 5,000 samples, subsample to 5,000 randomly selected cycles for GPR training only. Apply the same scaling as SVR.

Kernel choice: use Matern(nu=1.5) + WhiteKernel(). The Matern 1.5 kernel assumes once-differentiable functions, which is appropriate for the smooth-but-not-infinitely-smooth degradation curves you will see. The WhiteKernel accounts for observation noise.

When fitting, set n_restarts_optimizer=5 to avoid local optima in the kernel hyperparameter optimization.

GPR outputs: for each test point, you get a predicted mean (your SOH estimate) and a predicted standard deviation (your uncertainty). Plot the predicted SOH trajectory with the 95% confidence band (mean +/- 1.96 * std) against the true SOH for at least one cell per dataset. Compute coverage probability: the fraction of true SOH values falling within the 95% band should be approximately 0.95 for a well-calibrated model.

If you want to scale GPR to the full dataset without subsampling, use GPyTorch with inducing points (sparse GP approximation). This is a stretch goal — only pursue it if the subsampled GPR already works correctly.

---

### Phase 6: Deep Learning Training (Days 32-42)

**Step 6.0: Shared setup for all DL models**

All three DL models use the same:
- Input: sequence of W=20 consecutive cycles, each represented by the feature vector from Phase 3.
- Output: scalar SOH value at cycle W+1.
- Loss function: Mean Squared Error (MSE).
- Optimizer: Adam with initial learning rate 1e-3.
- Learning rate schedule: ReduceLROnPlateau on validation loss, factor 0.5, patience 5.
- Early stopping: patience 10 on validation loss.
- Batch size: 64.
- Maximum epochs: 150.
- Random seeds: train each model 5 times with different seeds. Report mean and standard deviation of test metrics.

Use a GPU if available (Google Colab or Kaggle Notebooks provide free T4 GPUs). If training on CPU, reduce max epochs to 50 and batch size to 32 — CPU training for 150 epochs on LSTM will take several hours per fold.

Track every run in MLflow or Weights & Biases: log architecture, hyperparameters, training loss curve, validation loss curve, and all test metrics.

**Step 6.1: LSTM**

Architecture:
- Input layer: shape (20, num_features).
- LSTM layer 1: 64 units, return_sequences=True, dropout=0.2.
- LSTM layer 2: 32 units, return_sequences=False, dropout=0.2.
- Dense layer: 16 units, ReLU activation.
- Output layer: 1 unit, no activation (regression).

Hyperparameters to search (Optuna, 30 trials per fold):
- Hidden units for layer 1: 32, 64, 128.
- Hidden units for layer 2: 16, 32, 64.
- Dropout rate: 0.1 to 0.4.
- Learning rate: log-uniform 1e-4 to 1e-2.

LSTM is your anchor model and the one with the most literature support. If only one DL model finishes training in time, it should be LSTM.

**Step 6.2: 1D CNN**

Architecture:
- Input layer: shape (20, num_features).
- Conv1D layer 1: 32 filters, kernel size 3, ReLU, same padding.
- Conv1D layer 2: 64 filters, kernel size 3, ReLU, same padding.
- Conv1D layer 3: 128 filters, kernel size 3, ReLU, same padding.
- GlobalAveragePooling1D.
- Dense layer: 64 units, ReLU, dropout 0.2.
- Output layer: 1 unit, no activation.

Key insight to understand: unlike LSTM, the CNN processes the entire input sequence in parallel, with no sequential dependency. This makes it faster to train and faster at inference. The trade-off is that it cannot capture long-range sequential dependencies as naturally as LSTM, though for a window of 20 cycles this is rarely a practical limitation.

Hyperparameters to search: number of filters per layer (16/32/64 or 32/64/128 or 64/128/256), kernel size (3, 5, or 7), dropout rate, learning rate.

**Step 6.3: Transformer**

The Transformer for sequence regression is architecturally different from the recurrent models. Understanding the architecture is part of the learning objective here.

Architecture:
- Input projection: a Dense layer mapping each cycle's feature vector to a model dimension d_model = 64.
- Positional encoding: sinusoidal positional encoding added to the projected input (this tells the model the order of cycles, since attention is permutation-invariant by itself).
- Transformer encoder block x2: each block contains multi-head self-attention (4 heads, key/query/value dimension 16 each) followed by a Feed-Forward sublayer (dimension 128), with LayerNorm and residual connections around both.
- Global average pooling across the sequence dimension.
- Dense layer: 32 units, ReLU.
- Output layer: 1 unit, no activation.

Regularization is critical for Transformers on small datasets: use dropout 0.1-0.3 inside the attention layers and the Feed-Forward sublayers. Without regularization, Transformers overfit rapidly on datasets the size of NASA PCoE.

Hyperparameters to search: d_model (32, 64), number of heads (2, 4), number of encoder blocks (1, 2, 3), dropout rate, learning rate.

If the Transformer underperforms LSTM or CNN, that is a valid and publishable result. Do not modify the architecture after seeing the test results to make it competitive — that is a form of post-hoc tuning on the test set. Report the result honestly and discuss why it may have underperformed (likely: insufficient data for the attention mechanism to be advantageous at this sequence length).

---

### Phase 7: Evaluation and Comparison (Days 42-46)

**Step 7.1: Metrics to report**

Compute and report the following for every model, in every cross-validation fold, and then aggregate as mean ± standard deviation across folds:

- Root Mean Square Error (RMSE): the primary metric.
- Mean Absolute Error (MAE): more interpretable than RMSE.
- Maximum Absolute Error (MaxAE): worst-case prediction error. Critical for safety-critical applications.
- R-squared (R2): fraction of variance explained.
- Inference time (milliseconds per prediction, measured on CPU): run 1,000 predictions and average.
- Model size (MB): size of the saved model file on disk.

**Step 7.2: Error analysis by degradation phase**

Break down RMSE separately for:
- Early life: SOH > 0.90.
- Mid life: 0.80 < SOH <= 0.90.
- End of life: SOH <= 0.80.

This is important because the practical consequence of prediction error varies by phase. An error of 3% at SOH = 0.95 is negligible; the same error at SOH = 0.81 could trigger or miss an end-of-life decision.

**Step 7.3: Predicted SOH trajectory plots**

For each model, plot the predicted SOH vs. true SOH over cycles for at least two held-out cells (one from each dataset). This is the most intuitive visualization of model performance and is mandatory in any paper on this topic. Include GPR's confidence interval bands on its trajectory plot.

**Step 7.4: SHAP analysis**

Run SHAP TreeExplainer on the trained RF model. Produce:
- A summary plot showing the mean absolute SHAP value per feature (global importance ranking).
- A beeswarm plot showing the direction of each feature's effect (positive or negative correlation with SOH).

Verify that the top features align with known electrochemical behavior. ICA peak height should rank highly (it is the most physics-grounded feature). If cycle number ranks above ICA features, the model may be memorizing temporal patterns rather than learning degradation physics — investigate and discuss.

For the best-performing DL model, compute Integrated Gradients (using the Captum library for PyTorch, or tf-explain for TensorFlow) to produce an equivalent feature attribution analysis.

**Step 7.5: GPR calibration check**

For GPR, compute the coverage probability at 95% confidence intervals: count the fraction of test SOH values that fall within the GPR's predicted 95% interval. A well-calibrated model should produce coverage close to 0.95. Under-coverage (e.g., 0.70) means the model is overconfident. Over-coverage (e.g., 0.99) means the model is too conservative and its intervals are uninformative.

---

### Phase 8: Deployability Assessment (Days 46-49)

This phase does not require you to build a deployed system. It requires you to rigorously characterize the models on dimensions that matter for deployment.

**Step 8.1: Inference time and model size**

For every model, measure CPU inference time as follows: run 1,000 forward passes sequentially (not in batch) and record the mean time per pass. This simulates a BMS that processes one cycle's data at a time in real time. Report in milliseconds.

Record model file size: for RF, the size of the pickled model file; for SVR and GPR, same; for DL models, the size of the saved weights file.

For context, a production BMS typically has 256 KB to 4 MB of available flash storage for an inference model, and a cycle evaluation budget of several hundred milliseconds.

**Step 8.2: Robustness to sensor noise**

Simulate realistic BMS sensor noise by adding Gaussian noise to the voltage and current signals at the preprocessing stage, then recomputing features. Test noise levels of 0.5%, 1%, and 2% (as a fraction of the signal range). Report how the RMSE of each model degrades as noise increases.

A model whose RMSE increases by less than 20% relative under 1% noise is considered robust. Models that degrade dramatically under small noise are poor deployment candidates regardless of their clean-data performance.

**Step 8.3: Robustness to missing cycles**

In real operation, cycles may be missing (the vehicle sat parked for weeks, or data logging failed). Simulate this by randomly dropping 10%, 20%, and 30% of cycles from the test cell's history before making predictions. For DL models, fill dropped cycles with the last observed feature vector (forward fill). Report RMSE degradation.

---

### Phase 9: Report and Presentation (Days 49-end of August)

**Step 9.1: Report structure**

Write a technical report of 6-10 pages (excluding references and appendices) in the following structure:

1. Abstract (150-200 words): problem, approach, key results, conclusion.
2. Introduction: why SOH estimation matters, what the current approaches miss, what this paper contributes.
3. Related work: 5-8 cited papers organized by approach (physics-based, classical ML, deep learning, hybrid). Do not summarize papers one by one — organize thematically and compare approaches.
4. Datasets: describe NASA PCoE and CALCE, including cell type, chemistry, cycling conditions, number of cells used, and total cycles.
5. Methodology:
   - Preprocessing pipeline (filtering, resampling, cycle segmentation, SOH labeling).
   - Feature engineering (ICA, IR, energy, efficiency, temperature, trend features).
   - Model architectures (one subsection per model, precise enough to reproduce).
   - Validation strategy (cell-based LOOCV, sequence window definition).
6. Results:
   - Full model comparison table (RMSE, MAE, MaxAE, R2, inference time, model size).
   - Error analysis by degradation phase.
   - Predicted SOH trajectory plots.
   - Feature importance and SHAP analysis.
   - GPR uncertainty calibration.
   - Robustness results (noise and missing cycles).
7. Discussion: what the results mean, where models succeed and fail, physical interpretation of feature importances, trade-offs between accuracy and deployability.
8. Conclusion and future work: summarize findings, list concrete next steps (cross-dataset validation, calendar aging, hardware deployment).
9. References.

**Step 9.2: Figures to produce**

Every figure takes time. Plan which figures you need and produce them as you go through the project rather than all at once at the end:

- Capacity fade curves (cycle vs. capacity) for representative cells from each dataset. Produce this in Phase 1.
- ICA dQ/dV curves at early, mid, and late life for one cell. Produce this in Phase 3.
- Feature correlation heatmap and feature importance bar chart. Produce this in Phase 3.
- Training and validation loss curves for each DL model. Produce these in Phase 6.
- Model comparison table (can be a formatted table in LaTeX or Markdown, not a figure).
- Predicted vs. true SOH scatter plot for each model. Produce in Phase 7.
- Predicted SOH trajectory over cycles for representative cells, for each model. Produce in Phase 7.
- SHAP beeswarm plot and summary plot. Produce in Phase 7.
- GPR confidence interval trajectory plot. Produce in Phase 7.
- Robustness bar charts (RMSE vs. noise level, RMSE vs. missing cycle fraction). Produce in Phase 8.

**Step 9.3: Presentation**

Prepare 10-12 slides covering: problem and motivation (1 slide), related work (1 slide), datasets and preprocessing (1 slide), feature engineering (1 slide with visual), model architectures overview (1-2 slides), main results table (1 slide), SOH trajectory plots (1 slide), interpretability findings (1 slide), deployability assessment (1 slide), conclusion and next steps (1 slide).

---

## 6. Time Estimates

The following schedule is calibrated for a 5-6 week full-time internship ending at the end of August 2025, starting around July 23.

| Phase | Duration | Approximate Dates |
|---|---|---|
| Phase 0: Setup and Literature Review | 7 days | Jul 23 - Jul 29 |
| Phase 1: Data Acquisition and EDA | 3 days | Jul 30 - Aug 1 |
| Phase 2: Preprocessing | 8 days | Aug 2 - Aug 9 |
| Phase 3: Feature Engineering | 7 days | Aug 10 - Aug 16 |
| Phase 4: Validation Strategy | 1 day | Aug 17 |
| Phase 5: Classical ML Training | 6 days | Aug 18 - Aug 23 |
| Phase 6: DL Training | 10 days | Aug 24 - Sep 2 |
| Phase 7: Evaluation and Comparison | 4 days | Sep 3 - Sep 6 |
| Phase 8: Deployability Assessment | 3 days | Sep 7 - Sep 9 |
| Phase 9: Report and Presentation | remaining time | Sep 10 - end |

Total estimated duration: approximately 9-10 weeks for the full scope with six models.

Given that your internship ends at the end of August, be aware that Phases 6-9 will likely overlap with your deadline. The recommended mitigation is to treat Phase 6 as a rolling process: start LSTM first (Day 32), then CNN (Day 35), then Transformer (Day 38), so you have at least two DL models trained and evaluated if the Transformer runs over time. Never let DL training push Phase 9 out entirely — a finished report with four models is worth more than an unfinished report with six.

---

## 7. Tools and Environment Setup

### Programming Language

Python 3.10 or 3.11. Do not use MATLAB for modeling, even where the datasets provide .mat files — read them in Python via scipy.io.loadmat or h5py.

### Core Libraries

- Data handling: pandas, numpy, h5py, scipy.io
- Signal processing: scipy.signal (Savitzky-Golay), scipy.interpolate (resampling)
- Classical ML: scikit-learn (RF, SVR, GPR), XGBoost (optional stretch)
- Deep learning: PyTorch (preferred) or TensorFlow/Keras
- Interpretability: shap, captum (PyTorch gradient attribution)
- Hyperparameter optimization: optuna
- Experiment tracking: mlflow or wandb (Weights & Biases free tier)
- Visualization: matplotlib, seaborn
- Environment: conda or venv with pinned requirements.txt

### Compute

- Classical ML: any modern laptop CPU is sufficient.
- DL models: use a GPU. Google Colab (free T4) or Kaggle Notebooks (free GPU sessions) are adequate. Train on GPU, evaluate on CPU.
- Pin all library versions in requirements.txt before starting model training. Version drift across a 6-week project is a real and avoidable problem.

---

## 8. Evaluation Metrics Summary

| Metric | Why It Matters |
|---|---|
| RMSE (primary) | Penalizes large errors more than small ones; appropriate for safety-critical regression |
| MAE | Directly interpretable in SOH percentage points; robust to outliers |
| MaxAE | Worst-case error; a 5% max error near end-of-life is a safety concern |
| R2 | Secondary; shows fraction of variance explained |
| Inference time (ms/cycle) | Deployment feasibility on BMS hardware |
| Model size (MB) | BMS memory constraints (typically 256 KB - 4 MB) |

Target for a credible research result: RMSE < 2% on held-out cells, MaxAE < 5%, inference time < 200 ms on CPU for all models.

---

## 9. Pitfalls and Common Mistakes to Avoid

### Data Pitfalls

**Data leakage through cycle-level splitting.** This is the single most common error in battery SOH papers and produces RMSE values that are 3-10x better than true generalization performance. Always split at the cell level: entire cells go to training or testing, never individual cycles of the same cell. If you do only one thing right in the experimental design, make it this.

**Using manufacturer-rated capacity instead of measured initial capacity.** Individual cells vary up to 5% from the rated nominal capacity. Use the measured capacity of each specific cell (averaged over cycles 3-10) as the denominator for SOH. Using the nominal rated capacity introduces a systematic offset that partially masks real degradation signals.

**Trusting the first few cycles.** Formation effects in the first 2-5 cycles can cause capacity to be irregular. Always compute Q_initial as the average of cycles 3-10, not cycle 1.

**Cross-dataset normalization errors.** Different datasets use different cutoff voltages (lower and upper voltage limits). A discharge capacity measured to 2.5V is larger than one measured to 2.7V. Before combining datasets, verify the voltage limits match and correct for them if they do not.

**Ignoring temperature as a feature.** Temperature is one of the strongest drivers of degradation rate. A model that ignores temperature will make systematic errors on any cell that operates at a different temperature than the training distribution. Always include temperature features.

### Modeling Pitfalls

**Tuning hyperparameters using test set performance.** Any hyperparameter decision informed by test set results constitutes data leakage. All tuning must be done on the validation set only. The test set (left-out cell in LOOCV) is used exactly once per fold, for final evaluation only.

**Reporting only the best random seed for DL models.** Deep learning training is stochastic. Report the mean and standard deviation of metrics over at least 5 runs with different random seeds. A model that achieves 1.0% RMSE once but averages 2.5% is not competitive — it got lucky.

**Treating the Transformer as the expected winner.** Transformers require substantially more data than LSTM or CNN to realize their advantages. On a dataset of 4-8 cells (NASA PCoE), the Transformer may underperform both LSTM and CNN. This is a legitimate result — discuss it honestly rather than over-engineering the Transformer to win.

**Using cycle number as a primary feature.** Cycle number is a proxy for time, not for physical battery state. A cell cycled aggressively at high C-rates is in much worse health at cycle 100 than a gently cycled cell. Models that rely heavily on cycle number generalize poorly to cells with different usage histories. Include it as a feature if you choose, but verify via SHAP that it is not dominating the predictions — if it is, your model has learned a shortcut rather than the true degradation physics.

**Underestimating GPR computation time.** GPR training is O(n^3) in the number of training samples. With 10,000 training samples, this may take several minutes per fit. With 50,000 samples, it will take hours and may run out of memory. Check your training set size before starting Phase 5 and decide on subsampling strategy in advance.

**Reporting average RMSE without per-cell breakdown.** A single cell with unusual degradation behavior can dominate the mean RMSE. Always report per-cell metrics alongside aggregate metrics. A model that fails catastrophically on one cell but performs well on others is not deployable, regardless of its average metric.

### Reporting Pitfalls

**Comparing your numbers to published results without verifying the experimental setup is identical.** Many published SOH papers use random cycle splits (data leakage). Their RMSE values are not directly comparable to your cell-based LOOCV results. When you compare to prior work, explicitly state what split strategy the prior work used and note that your numbers are more conservative by design.

**Omitting negative results.** If the Transformer underperforms LSTM, report it and discuss why. If GPR's uncertainty intervals are poorly calibrated, report it. A thorough analysis of failure modes is more valuable than a clean paper where everything works.

**Writing the report last.** Start the Introduction, Related Work, and Datasets sections during Phase 0 and Phase 1 respectively, while the material is fresh. Write the Methodology section during each phase. Leave only the Results and Discussion sections for the end. Writing incrementally prevents the panic of trying to write 8 pages in 3 days.

---

## 10. Deliverables Checklist

### Phase-End Check-ins (share with supervisor)

- [ ] Scope document with supervisor sign-off (end of Phase 0)
- [ ] Data inventory table with EDA plots (end of Phase 1)
- [ ] Preprocessing pipeline with before/after filtering plots (end of Phase 2)
- [ ] Feature matrix with correlation heatmap and importance ranking (end of Phase 3)
- [ ] First classical ML results and comparison to baseline (end of Phase 5)
- [ ] All model results with comparison table (end of Phase 7)

### Final Deliverables

- [ ] Git repository with clean, documented code and README
- [ ] Technical report (6-10 pages, submission-quality)
- [ ] Presentation slides (10-12 slides)
- [ ] Model comparison table (all 6 models, all metrics, both datasets)
- [ ] Saved model artifacts for all trained models
- [ ] SHAP analysis plots and DL gradient attribution plots
- [ ] GPR uncertainty calibration figure
- [ ] Robustness analysis results (noise and missing cycles)

---

## 11. References and Learning Resources

### Foundational Papers

- Severson, K.A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nature Energy, 4, 383-391.
- Zhang, Y. et al. (2020). Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using the distribution of relaxation times. Nature Communications, 11, 1706.
- Shen, S. et al. (2019). A deep learning method for online capacity estimation of lithium-ion batteries. Journal of Energy Storage, 25, 100817.
- Hu, C. et al. (2020). Battery lifetime prognostics. Joule, 4(2), 310-346.
- Richardson, R.R. et al. (2017). Gaussian process regression for forecasting battery state of health. Journal of Power Sources, 357, 209-219. (The primary reference for GPR in battery SOH — read this before implementing Phase 5.3.)

### For GPR Specifically

- Rasmussen, C.E. and Williams, C.K.I. (2006). Gaussian Processes for Machine Learning. MIT Press. Available free at gaussianprocess.org. Read Chapters 1 and 2 only.
- scikit-learn documentation on GaussianProcessRegressor: https://scikit-learn.org/stable/modules/gaussian_process.html. Read the regression examples in full.

### For Transformer Architecture

- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS. The original paper — read Section 3 (architecture) and Section 5 (results) for background.
- The Annotated Transformer by Harvard NLP: http://nlp.seas.harvard.edu/annotated-transformer. Line-by-line implementation walkthrough in PyTorch. Use this to build your Transformer model.

### For SHAP

- Lundberg, S.M. and Lee, S.I. (2017). A unified approach to interpreting model predictions. NeurIPS. The original SHAP paper.
- SHAP documentation with battery-relevant tabular examples: https://shap.readthedocs.io

### Community Resources

- Papers With Code (paperswithcode.com): search "state of health estimation" for recent papers with linked code. Valuable for understanding what the current state of the art achieves and how.
- NASA PCoE dataset page includes several papers that use the dataset directly — these are your most direct benchmarks.

---

*This plan is calibrated for a full-time internship ending at end of August, with a six-model benchmark scope. The non-negotiable core is: preprocessing done correctly (cell-based splits, per-cell SOH labeling) + feature engineering (ICA + IR + energy + temperature) + at least RF and LSTM trained and evaluated. Everything else is additive. If you hit a blocker in any phase, escalate to your supervisor immediately rather than losing days trying to resolve it alone — internship timelines have no slack for silent debugging.*

---

## Post-Plan Corrections (2026-08-25)

Statements above reflect initial planning and are superseded where they conflict
with the implemented system:

- Model count: **seven** models are benchmarked — naive mean baseline plus
  RF/SVR/GPR/LSTM/CNN/Transformer (plan said six).
- EIS features ARE in scope: `eis_re`/`eis_rct` are extracted for NASA cells
  (`src/features/internal_resistance.py`) when impedance data exists.
- Feature selection keeps up to `top_k_features: 20` per LOOCV fold
  (`fit_feature_selection`, train-cells-only); the plan's "top 15-20" was a
  design-stage estimate.
- All reported metrics come from the leakage-safe protocol described in
  docs/decisions_log.md D-009..D-016; earlier in-text metric examples are
  illustrative only.
