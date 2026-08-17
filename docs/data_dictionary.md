# Data Dictionary

Describes every column in the feature matrix and intermediate data products.

## SOH Labels

### Combined: `data/processed/soh_labels.parquet`

| Column | Type | Description |
|---|---|---|
| cell_id | str | Unique cell identifier (e.g., B0005) |
| dataset | str | Dataset name: 'nasa_pcoe' or 'calce' |
| cycle_number | int | Cycle number (not 1-indexed; matches raw data numbering) |
| soh | float | State of Health: Q_discharge(n) / Q_initial, capped at 1.0 |
| q_discharge | float | Measured discharge capacity for this cycle (Ah) |
| q_initial | float | Reference capacity: mean of cycles 3-10 for this cell (Ah) |

**Shape:** 636+ rows (NASA: 4 cells, CALCE: 4 cells)

### NASA-only: `data/processed/soh_labels_nasa.parquet`

Same schema as combined. **Shape:** 636 rows (B0005, B0006, B0007, B0018)

### CALCE-only: `data/processed/soh_labels_calce.parquet`

Same schema as combined. **Shape:** ~900 rows (CS2_33, CS2_34, CS2_35, CS2_36)

## Feature Matrix

### Combined: `data/features/feature_matrix.parquet`

Inherits all columns from SOH Labels plus 12 selected features. **Shape:** 636+ × 16.

### NASA-only: `data/features/feature_matrix_nasa.parquet`

Same schema. **Shape:** 636 × 16.

### CALCE-only: `data/features/feature_matrix_calce.parquet`

Same schema. **Shape:** ~900 × 16.

### Selected Features (12)

#### Energy Features (from `src/features/energy.py`)

| Column | Type | Description |
|---|---|---|
| mean_discharge_voltage | float | Average voltage during discharge (V) |
| discharge_energy | float | Integral of V*I*dt over discharge cycle (Wh) |

#### Internal Resistance (from `src/features/internal_resistance.py`)

| Column | Type | Description |
|---|---|---|
| internal_resistance | float | Estimated from discharge pulse onset: delta_V / delta_I (Ohm) |
| eis_re | float | Real impedance at 1 kHz from EIS data (Ohm, NaN if unavailable) |

#### ICA Features (from `src/features/ica.py`)

| Column | Type | Description |
|---|---|---|
| ica_peak_voltage | float | Voltage position of primary peak in dQ/dV curve (V) |
| ica_peak_height | float | Height of tallest peak in dQ/dV curve |
| ica_peak_area | float | Area under primary peak (Ah/V) |

Note: `ica_peak_fwhm` and `ica_secondary_ratio` were always NaN (dQ/dV peaks are narrow single-point spikes on the resampled grid) and were dropped from the final feature set.

#### Temperature Features (from `src/features/temperature.py`)

| Column | Type | Description |
|---|---|---|
| temp_mean | float | Mean temperature during discharge cycle (deg C) |
| temp_max | float | Maximum temperature during discharge cycle (deg C) |
| temp_min | float | Minimum temperature during discharge cycle (deg C) |
| temp_range | float | Temperature range: max - min (deg C) |

#### Trend Features (from `src/features/trend.py`)

| Column | Type | Description |
|---|---|---|
| capacity_fade_rate | float | Local slope of SOH over last 10 cycles: (SOH(n) - SOH(n-10)) / 10 |

### Metadata

| Column | Type | Description |
|---|---|---|
| cycle_number | int | Cycle number (inherited from SOH labels) |
| dataset | str | Dataset source (inherited) |
| cell_id | str | Cell identifier (inherited) |
| discharge_capacity | float | Total discharge capacity for this cycle (Ah) |

## Dropped Features

| Feature | Reason for Dropping |
|---|---|
| ica_peak_fwhm | Always NaN; dQ/dV peaks are too narrow on resampled grid |
| ica_secondary_ratio | Always NaN; no secondary peaks detected |
| coulombic_efficiency | Low variance across cycles; not discriminative |
| discharge_duration | Low discriminative power |
| secondary_ratio | Always NaN |

## Preprocessed Data

### Combined: `data/processed/processed_cells.pkl`

Pickle file containing a dict keyed by cell_id. Each entry is a dict with:

| Key | Type | Description |
|---|---|---|
| cell_id | str | Cell identifier |
| dataset | str | Dataset source |
| cycles | list[dict] | List of preprocessed cycle dicts |

Each cycle dict contains:

| Key | Type | Description |
|---|---|---|
| cycle_number | int | Cycle number |
| cycle_type | str | 'charge', 'discharge', or 'impedance' |
| voltage_filtered | array[float] | Savitzky-Golay filtered voltage (resampled to 1000 points) |
| current_filtered | array[float] | Filtered current (resampled) |
| temperature_filtered | array[float] | Filtered temperature (resampled) |
| capacity_grid | array[float] | Uniform capacity axis (1000 points) |
| soh | float | SOH label (discharge cycles only) |

### NASA-only: `data/processed/processed_cells_nasa.pkl`

Same schema. Contains only NASA cells (B0005, B0006, B0007, B0018).

### CALCE-only: `data/processed/processed_cells_calce.pkl`

Same schema. Contains only CALCE cells (CS2_33, CS2_34, CS2_35, CS2_36).
