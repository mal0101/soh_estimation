# Data Dictionary

Describes every column in the feature matrices and intermediate data products.
All artifacts use explicit dataset suffixes (`_nasa`, `_calce`, `_all`); nothing
is ever overwritten across datasets.

## SOH Labels

Produced exclusively by `python -m src.preprocessing.pipeline` (never by notebooks).

### `data/processed/soh_labels_{nasa|calce|all}.parquet`

| Column | Type | Description |
|---|---|---|
| cell_id | str | Unique cell identifier (e.g., B0005, CS2_33) |
| dataset | str | Dataset name: 'nasa_pcoe' or 'calce' |
| cycle_number | int | Cycle number (matches raw data numbering) |
| soh | float | State of Health: Q_discharge(n) / Q_initial, capped at `preprocessing.soh.soh_cap` (=1.0) |
| q_discharge | float | Measured discharge capacity for this cycle (Ah) |
| q_initial | float | Reference capacity: **median** of discharge capacities over cycles [3,10] for this cell (Ah) |

**Shapes:** nasa = 636 rows · calce = 2,559 rows · all = 3,195 rows.

Q_initial uses a median (not mean) reduction so that outlier cycles inside the
reference window cannot poison every downstream label (this occurred on CS2_36
before the fix: a 0.147 Ah interruption shifted its reference by ~12%).

## Cycle Integrity Filtering

Applied inside `validate_cycles` before labels are computed:

1. **Early partials** — cycles ≤ `early_cycle_window` (20) below 90% of Q_initial.
2. **Isolated interruptions** — discharges >7% below the local median of ±5
   neighbouring discharge cycles (catches CALCE reference-performance-test dips).
3. **Anomalous runs** — contiguous blocks below 75% of Q_initial that later RECOVER
   and average below 70% of Q_initial (multi-week storage/test pauses). Genuine
   unrecovered end-of-life fade and shallow reversible transients are kept.

Per-cell discard counts are stored on each processed-cell dict as
`n_cycles_discarded`.

## Feature Matrix

### `data/features/feature_matrix_{nasa|calce|all}.parquet`

FULL **candidate** matrices produced by `python -m src.features.build_features`.
No supervised selection is applied at build time: correlation filtering and RF
importance ranking run **per LOOCV fold on training cells only**
(`src.features.assembly.fit_feature_selection`) during model training.

**Shape:** (n_labels, 4 metadata + 16 candidates):

- nasa: 636 × 20 · calce: 2,559 × 20 · all: 3,195 × 20.

### Metadata Columns

| Column | Type | Description |
|---|---|---|
| cell_id | str | Cell identifier |
| dataset | str | Dataset source |
| cycle_number | int | Cycle number |
| soh | float | Target (see labels above) |

### Candidate Features (16)

#### ICA (`src/features/ica.py`; dQ/dV negated so discharge peaks are positive)

| Column | Type | Description |
|---|---|---|
| ica_peak_voltage | float | Voltage of primary dQ/dV peak (V); rejected outside [3.0, 4.35] V |
| ica_peak_height | float | Height of primary peak (Ah/V) |
| ica_peak_area | float | Area under primary peak region (Ah) |
| ica_peak_fwhm | float | Full width at half maximum along capacity axis (Ah); often NaN (~70%) |
| ica_secondary_ratio | float | Secondary-to-primary peak height ratio; NaN when no secondary peak |

#### Internal resistance (`src/features/internal_resistance.py`)

| Column | Type | Description |
|---|---|---|
| internal_resistance | float | ΔV/ΔI over the first discharge samples (Ohm); NaN unless a genuine current step exists |
| eis_re | float | Re from nearest PAST impedance cycle (Ohm); NASA only (NaN for CALCE) |
| eis_rct | float | Rct from nearest PAST impedance cycle (Ohm); NASA only |

#### Energy (`src/features/energy.py`)

| Column | Type | Description |
|---|---|---|
| discharge_energy | float | ∫V·\|I\|dt over the discharge (Wh) |
| mean_discharge_voltage | float | Discharge energy / discharged capacity (V) |
| coulombic_efficiency | float | Q_dis / Q_charge of the immediately PRECEDING charge; NaN outside (0.8, 1.05] |

#### Temperature (`src/features/temperature.py`)

| Column | Type | Description |
|---|---|---|
| temp_mean / temp_max / temp_min / temp_range | float | Discharge temperature statistics (°C). CALCE logs no thermocouple data → constant placeholder 25.0 (variance-screened out on CALCE folds) |

#### Trend (`src/features/trend.py`)

| Column | Type | Description |
|---|---|---|
| capacity_fade_rate | float | Trailing slope over strictly PAST labels: (SOH(n-1) − SOH(n-1-window)) / window, window=10. The current row's own SOH is deliberately excluded (target leakage). First two rows per cell are NaN |

### Per-fold Selection

`fit_feature_selection` screens candidates per fold: drops columns with ≥30% NaN
or near-zero variance, removes pairs with \|r\| > 0.95, then keeps the top-k by
RF importance (k=20; effectively all survivors). Typical survivors:
`discharge_energy`, `internal_resistance`, `mean_discharge_voltage`,
`capacity_fade_rate`, `coulombic_efficiency`, ICA peaks, temperature stats.
Typical drops: `ica_peak_fwhm`, `ica_secondary_ratio` (>30% NaN), `eis_*`
(CALCE), temperature features (CALCE-only folds).

## Preprocessed Cells

### `data/processed/processed_cells_{nasa|calce|all}.pkl`

Dict keyed by cell_id. Each entry:

| Key | Type | Description |
|---|---|---|
| cell_id | str | Cell identifier |
| dataset | str | 'nasa_pcoe' or 'calce' |
| rated_capacity / cutoff_voltage | float | From loader (NASA 2.0 Ah; CALCE 1.1 Ah / 2.7 V) |
| q_initial | float | Robust reference used for filtering and labels (Ah) |
| n_cycles_discarded | int | Cycles removed by validate_cycles |
| cycles | list[dict] | Preprocessed cycles |

Each cycle dict contains: `cycle_number`, `type` ('charge'/'discharge'/'impedance'),
raw `voltage`, `current`, `temperature`, `time`, reported `capacity` (discharge),
plus for charge/discharge: `cumulative_capacity` (integrated Ah),
`voltage_filtered` (Savitzky-Golay), `capacity_grid` + `voltage_resampled`
(1000-point uniform grid).

## Removed Legacy Artifacts

The following pre-remediation artifacts were removed or replaced; do not
recreate them:

- `experiments/classical_results.yaml` / `dl_results.yaml` (no-suffix naming)
- `classical_results_all.yaml`-style files produced by an older code path
- `fold_indices.json` without dataset suffix
- SOH labels written from EDA notebooks (bypassed integrity filters)
