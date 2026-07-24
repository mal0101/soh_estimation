# Data Dictionary

Describes every column in the feature matrix and intermediate data products.

## SOH Labels (`data/processed/soh_labels.parquet`)

| Column | Type | Description |
|---|---|---|
| cell_id | str | Unique cell identifier (e.g., B0005, CS2_33) |
| dataset | str | Dataset name: 'nasa_pcoe' or 'calce' |
| cycle_number | int | 1-indexed cycle number |
| soh | float | State of Health: Q_discharge(n) / Q_initial, capped at 1.0 |
| q_discharge | float | Measured discharge capacity for this cycle (Ah) |
| q_initial | float | Reference capacity: mean of cycles 3-10 for this cell (Ah) |

## Feature Matrix (`data/features/feature_matrix.parquet`)

Inherits all columns from SOH Labels plus:

### ICA Features (from `src/features/ica.py`)

| Column | Type | Description |
|---|---|---|
| ica_primary_peak_height | float | Height of tallest peak in dQ/dV curve |
| ica_primary_peak_voltage | float | Voltage position of primary peak (V) |
| ica_primary_peak_fwhm | float | Full width at half maximum of primary peak (V) |
| ica_primary_peak_area | float | Area under primary peak (Ah/V) |
| ica_secondary_peak_ratio | float | Ratio of secondary to primary peak height (if present) |

### Internal Resistance (from `src/features/internal_resistance.py`)

| Column | Type | Description |
|---|---|---|
| internal_resistance | float | Estimated from discharge pulse onset: delta_V / delta_I (Ohm) |
| eis_resistance_1khz | float | Real impedance at 1 kHz from EIS data (Ohm, NASA only) |

### Energy Features (from `src/features/energy.py`)

| Column | Type | Description |
|---|---|---|
| discharge_energy | float | Integral of V*I*dt over discharge cycle (Wh) |
| mean_discharge_voltage | float | Average voltage during discharge (V) |
| discharge_duration | float | Time from discharge start to end (s) |
| coulombic_efficiency | float | Q_discharge / Q_charge ratio (dimensionless) |

### Temperature Features (from `src/features/temperature.py`)

| Column | Type | Description |
|---|---|---|
| temp_mean | float | Mean temperature during discharge cycle (deg C) |
| temp_max | float | Maximum temperature during discharge cycle (deg C) |
| temp_min | float | Minimum temperature during discharge cycle (deg C) |
| temp_range | float | Temperature range: max - min (deg C) |

### Trend Features (from `src/features/trend.py`)

| Column | Type | Description |
|---|---|---|
| soh_slope | float | Local slope of SOH over last 10 cycles: (SOH(n) - SOH(n-10)) / 10 |

### Metadata

| Column | Type | Description |
|---|---|---|
| cycle_number | int | Cycle number (inherited from SOH labels) |
| dataset | str | Dataset source (inherited) |
| cell_id | str | Cell identifier (inherited) |

## Preprocessed Data (`data/processed/{dataset}_{cell}_preprocessed.parquet`)

| Column | Type | Description |
|---|---|---|
| cell_id | str | Cell identifier |
| dataset | str | Dataset source |
| cycle_number | int | Cycle number |
| cycle_type | str | 'charge', 'discharge', or 'rest' |
| soh | float | SOH label (discharge cycles only) |
| voltage_filtered | array[float] | Savitzky-Golay filtered voltage (resampled) |
| current_filtered | array[float] | Filtered current (resampled) |
| temperature_filtered | array[float] | Filtered temperature (resampled) |
| capacity_grid | array[float] | Uniform capacity axis (1000 points) |
