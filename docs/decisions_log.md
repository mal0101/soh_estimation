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

## D-008: 5 Random Seeds for DL Models

**Date:** 2025-07-23
**Status:** Accepted
**Context:** DL training is stochastic; must report mean ± std.
**Decision:** Train each DL model configuration 5 times with different seeds.
**Rationale:**
- DL training has high variance due to random initialization and data shuffling.
- 5 seeds is the minimum for meaningful statistics.
- A model achieving 1% RMSE once but averaging 2.5% is not competitive.
- Matches standard practice in ML papers.
**Alternatives rejected:**
- 3 seeds: Too few for reliable statistics.
- 10 seeds: Computationally expensive; diminishing returns.

---

