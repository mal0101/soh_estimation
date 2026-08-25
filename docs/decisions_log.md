# Decisions Log

Records all architectural and design decisions made during the project, with rationale and alternatives considered.

---

## D-001: Use Conda Instead of venv

**Date:** 2025-07-23
**Status:** Accepted
**Context:** Need to set up Python environment for the project.
**Decision:** Use Conda (Miniconda) with Python 3.11.
**Rationale:**
- System has Python 3.13.5 via pyenv; PyTorch and captum have limited support on 3.13.
- Conda handles non-Python deps (MKL, HDF5, BLAS) automatically.
- PyTorch MPS backend installs more reliably via conda channel.
- Jupyter integration is simpler with conda.
- Project plan explicitly lists conda as an option.
**Alternatives rejected:**
- venv + pyenv: Would require installing Python 3.11 via pyenv first; additional complexity.
- Docker: Overkill for a 6-week internship project.

---

## D-002: Python 3.11 (Not 3.10 or 3.12)

**Date:** 2025-07-23
**Status:** Accepted
**Context:** Project plan specifies "Python 3.10 or 3.11".
**Decision:** Use Python 3.11.
**Rationale:**
- 3.11 is newer with better performance (faster startup, better error messages) while maintaining full library compatibility.
- 3.10 is also fine but provides no advantage for this project.
- 3.12+ risks compatibility issues with some ML libraries at time of writing.

---

## D-003: MLflow Over Weights & Biases

**Date:** 2025-07-23
**Status:** Accepted
**Context:** Need experiment tracking for all training runs.
**Decision:** Use MLflow with local SQLite backend.
**Rationale:**
- No account/API key required (W&B free tier requires registration).
- Local tracking keeps data on-machine; no external dependencies.
- MLflow model registry is sufficient for our model versioning needs.
- Works offline; important for a internship laptop that may not always have internet.
**Migration:** Migrated from file store to SQLite backend (`sqlite:///mlflow.db`) on 2025-07-27 after MLflow 3.14 dropped file store support.
**Alternatives rejected:**
- W&B: Requires account, internet for syncing, free tier has run limits.
- TensorBoard: Less suited for sklearn models; primarily a DL tool.

---

## D-004: Centralized YAML Configuration

**Date:** 2025-07-23
**Status:** Accepted
**Context:** Need a single source of truth for all hyperparameters and paths.
**Decision:** Use `config/default.yaml` with a `Config` class providing dot-notation access.
**Rationale:**
- All hyperparameters in one file; no hardcoded values scattered across modules.
- YAML is human-readable and easy to edit.
- Config class enables `cfg.models.dl.learning_rate` style access.
- Supports CLI overrides via `--config` flag.
**Tuning:** DL models adjusted for feasibility: n_seeds 5→3, n_trials 30→10, max_epochs 150→80. SVR gamma bounds changed from `1e-4` string to `0.0001` float.
**Alternatives rejected:**
- argparse only: Becomes unwieldy with 50+ parameters.
- Hydra/OmegaConf: Overkill for this project's complexity level.

---

## D-005: Cell-Based Leave-One-Cell-Out Cross-Validation

**Date:** 2025-07-23
**Status:** Accepted
**Context:** Need a validation strategy that prevents data leakage.
**Decision:** Cell-based LOOCV within each dataset, independent of each other.
**Rationale:**
- Prevents data leakage (the #1 error in battery SOH papers).
- Models are evaluated on cells never seen during training.
- Produces per-cell metrics for honest performance assessment.
- Matches actual deployment scenario: predict SOH for a new cell.
**Alternatives rejected:**
- Random cycle split: Leaks information across cells; inflates performance.
- Random sample split: Same leakage problem.
- K-fold: Not applicable when cells are the natural grouping.

---

## D-006: Sequence Window of 20 Cycles for DL Models

**Date:** 2025-07-23
**Status:** Accepted
**Context:** DL models need sequential input; must define window size.
**Decision:** W = 20 consecutive cycles as input, predict SOH at cycle W+1.
**Rationale:**
- 20 cycles captures short-term degradation trends without excessive sequence length.
- Matches typical window sizes in published SOH literature.
- Keeps sequence length tractable for Transformer self-attention.
- Large enough to capture rate-of-change features.
**Alternatives rejected:**
- W = 10: Too short to capture meaningful trends.
- W = 50: Excessive for small datasets; risks overfitting.

---

## D-007: Matern 1.5 Kernel for GPR

**Date:** 2025-07-23
**Status:** Accepted
**Context:** GPR requires a kernel choice; RBF vs Matern.
**Decision:** Matern(nu=1.5) + WhiteKernel().
**Rationale:**
- Matern 1.5 assumes once-differentiable functions; appropriate for physical degradation.
- RBF assumes infinitely smooth functions; too strong an assumption for battery data.
- WhiteKernel accounts for observation noise in the measurements.
- Richardson et al. (2017) is the primary GPR reference for battery SOH.
**Alternatives rejected:**
- RBF: Too smooth; may miss sharp degradation transitions.
- Matern 2.5: Acceptable alternative; 1.5 is more conservative.

---

## D-008: 3 Random Seeds for DL Models

**Date:** 2025-07-27
**Status:** Accepted
**Context:** DL training is stochastic; must report mean ± std.
**Decision:** Train each DL model configuration 3 times with different seeds (reduced from initial plan of 5).
**Rationale:**
- DL training has high variance due to random initialization and data shuffling.
- 3 seeds is the minimum for meaningful statistics while keeping training feasible.
- Combined with reduced Optuna trials (10 per model) and max_epochs (80), keeps total training time manageable.
- A model achieving 1% RMSE once but averaging 2.5% is not competitive.
**Alternatives rejected:**
- 5 seeds: Computationally expensive with small dataset; 3 provides sufficient variance estimates.
- 1 seed: Insufficient for reporting statistics.

---

## D-009: Remediation of Target Leakage in capacity_fade_rate

**Date:** 2026-08-25
**Status:** Accepted
**Context:** Audit found the fade-rate feature for row *n* used SOH(n) itself
— an affine function of the prediction target — inflating every model.
**Decision:** Fade rate now uses strictly past labels:
(SOH(n−1) − SOH(n−1−window)) / window; warm-up rows are NaN, never 0.
**Consequence:** First two rows per cell carry no fade feature (dropped per fold).

---

## D-010: Feature Selection Inside the CV Loop

**Date:** 2026-08-25
**Status:** Accepted
**Context:** Correlation filtering + RF importance were fitted on ALL cells
(including future test cells) before LOOCV — selection-on-test inflation.
**Decision:** `fit_feature_selection(train_df)` runs per fold on training
cells only; the saved feature matrices are FULL candidate matrices.
**Alternatives rejected:** keeping global selection with a documented caveat
(rejected: results would remain inflated).

---

## D-011: Inner Cell Split for Hyperparameter Tuning

**Date:** 2026-08-25
**Status:** Accepted
**Context:** Optuna objectives and DL early stopping consumed the outer
LOOCV test cell; reported metrics came from that same cell.
**Decision:** Each fold splits its training cells by cell (alphabetically
last = inner validation). Classical tuning and DL selection/early-stopping
use only the inner split; final models refit on the full training fold;
the test cell is touched once, for reporting. DL refits run for exactly
`best_epoch` epochs chosen during selection, with best-weight restoration.
**Consequence:** Headline metrics are lower but honest (SVR NASA R² 0.97 → ~0.90).

---

## D-012: Robust Q_initial (Median) and Cycle Integrity Filters

**Date:** 2026-08-25
**Status:** Accepted
**Context:** CS2_36's Q_initial was poisoned by a 0.147 Ah interruption inside
cycles [3,10] (+12% label bias); CALCE cells contained storage/test pauses
(~250-cycle depressed blocks) and periodic RPT dips; the old filter only
checked cycles ≤ 20.
**Decision:** Q_initial = median over cycles [3,10]. Three integrity rules:
early partials (≤20, <90% Q_init); isolated interruptions (>7% below ±5-neighbour
local median); anomalous recovered runs (block <75% Q_init, later recovery,
mean <70% Q_init). Genuine unrecovered EOL fade and shallow reversible
transients (NASA B0006) are retained by design.
**Consequence:** CALCE labels 3,546 → 2,559 rows; zero sub-0.6 SOH artifacts
remain outside genuine B0006 deep discharge.

---

## D-013: ICA Sign Convention and Physical Gating

**Date:** 2026-08-25
**Status:** Accepted
**Context:** dQ/dV computed as gradient(capacity, voltage) is negative on
discharge curves; peak detection latched onto start-of-discharge artifacts and
FWHM was always NaN.
**Decision:** Negate the gradient; drop near-duplicate voltages before
differentiation; clip dQ/dV ≥ 0; reject peaks outside [3.0, 4.35] V;
prominence threshold relative to signal maximum.

---

## D-014: Explicit Dataset Suffixes on All Artifacts

**Date:** 2026-08-25
**Status:** Accepted
**Context:** `--dataset all` previously wrote unsuffixed files while some
results existed only as unproducible `_all` orphans; two contradictory
"combined" result sets coexisted.
**Decision:** Every artifact carries `_nasa` / `_calce` / `_all`; stale
unsuffixed files removed; MLflow runs tagged with `dataset`.

---

## D-015: Real Inference Latency Benchmarks

**Date:** 2026-08-25
**Status:** Accepted
**Context:** `inference_time_mean_s` stored full fold TRAINING wall time
(including Optuna search); report latency claims were therefore unsupported.
**Decision:** Fold wall time renamed `train_time_mean_s`; inference measured
via `benchmark_inference_time` (single-sample predict ×N) into
`inference_time_ms_mean/p95`; models persisted under
`experiments/models/{dataset}/` so sizes are measurable.

---

## D-016: Honest Reporting Over Legacy Numbers

**Date:** 2026-08-25
**Status:** Accepted
**Context:** Pre-remediation numbers were internally inconsistent (notebook 04
matched no YAML), partially fabricated-looking via staleness, and methodologically
inflated.
**Decision:** Full re-run of every experiment after fixes; all downstream
numbers regenerated programmatically from YAMLs (`scripts/update_results_docs.py`);
stale claims corrected everywhere (README/GUIDE/notebooks/docs).
