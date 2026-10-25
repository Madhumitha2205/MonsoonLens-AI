# MonsoonLens AI: Comprehensive Preprocessing Pipeline Report

**Project Title:** Historical Rainfall Pattern Analysis Using Machine Learning  
**Pipeline Run Timestamp:** 2026-10-25 05:04:24 UTC  
**Pipeline Script:** `src/preprocess.py`  
**Configuration:** `src/config.py`

---

## 1. Raw Dataset Overview

- **Source File:** `data/raw/rainfall_area-wt_sd_1901-2015.csv`
- **Original Dimensions:** 4,116 rows × 19 columns
- **Temporal Span:** 1901 – 2015 (115 years)
- **Geographic Coverage:** 36 meteorological subdivisions across India
- **Schema Received:**
  - Identifiers: `SUBDIVISION` (string), `YEAR` (integer)
  - Monthly Rainfall (12 columns): `JAN`, `FEB`, `MAR`, `APR`, `MAY`, `JUN`, `JUL`, `AUG`, `SEP`, `OCT`, `NOV`, `DEC` (mm)
  - Annual Rainfall: `ANNUAL` (mm)
  - Seasonal Rainfall (4 columns): `Jan-Feb`, `Mar-May`, `Jun-Sep`, `Oct-Dec` (mm)

---

## 2. Cleaning Operations Performed

1. **Text Standardization:**
   - Stripped leading/trailing whitespace and normalized multiple spaces in `SUBDIVISION` names.
   - Preserved original subdivision spelling in `SUBDIVISION_ORIGINAL` for full traceability.
2. **Type Casting and Coercion:**
   - Standardized `YEAR` to `int64`.
   - Standardized all 17 rainfall fields (`JAN`–`DEC`, `ANNUAL`, `Jan-Feb`, `Mar-May`, `Jun-Sep`, `Oct-Dec`) to `float64`.
   - Empty strings, whitespace, and source `NA` strings converted to standard IEEE floating-point `NaN`.
3. **Duplicate Detection:**
   - Audited for exact row duplicates: **0 detected**.
   - Audited for repeated `(SUBDIVISION, YEAR)` combinations: **0 detected**.
4. **Range and Validity Checks:**
   - Negative rainfall checks: **0 detected** (all values $\ge 0.0$ mm).
   - Infinite value checks: **0 detected**.
5. **Chronological Sorting:**
   - Observations sorted strictly by `SUBDIVISION` (alphabetical) and `YEAR` (ascending).

---

## 3. Missing Value Strategy and Findings

- **Total Rows with Missing Data:** 26 rows out of 4,116 (0.632% of total rows).
- **Total Missing Monthly Values:** 70 values out of 49,392 month-station observations (0.142%).
- **Distribution of Missing Rows Across Subdivisions:**
  - `LAKSHADWEEP`: 11 rows
  - `ANDAMAN & NICOBAR ISLANDS`: 6 rows
  - `ARUNACHAL PRADESH`: 6 rows
  - `JAMMU & KASHMIR`: 1 row (1936)
  - `WEST MADHYA PRADESH`: 1 row (1950)
  - `COASTAL KARNATAKA`: 1 row (1947)
- **Missing Value Handling Rules:**
  - Missing rainfall observations are **scientifically preserved as `NaN`**.
  - **No Zero-Filling:** Zero rainfall is a physical phenomenon (dry weather); missing rainfall is a recording absence. They are never conflated.
  - **No Global Mean Imputation:** Filling missing values with the national or subdivision mean would corrupt historical variance and bias extreme-value distributions.
  - **No Cross-Year Forward/Backward Fill:** Rainfall is non-stationary and exhibits high inter-annual variability; carrying observations forward across years would introduce severe artificial autocorrelation.

---

## 4. Seasonal and Annual Consistency Audit

We examined the relationship between reported monthly figures and reported aggregate totals:
- **Jan-Feb** vs. `JAN + FEB`: Max difference = 0.10 mm
- **Mar-May** vs. `MAR + APR + MAY`: Max difference = 0.10 mm
- **Jun-Sep** vs. `JUN + JUL + AUG + SEP`: Max difference = 0.20 mm
- **Oct-Dec** vs. `OCT + NOV + DEC`: Max difference = 0.10 mm
- **ANNUAL** vs. Sum of 12 Months: Max difference = 0.40 mm

### Key Mathematical Finding:
The maximum discrepancy of **0.40 mm** is strictly below the cumulative single-decimal rounding bound of $12 	imes 0.05 = 0.60$ mm.
Both `ANNUAL_SOURCE` and `ANNUAL_CALCULATED` are preserved, providing full transparency.
If any monthly value is missing, the corresponding calculated total is set to `NaN` (partial sums are strictly rejected).

---

## 5. Processed Datasets Generated

The pipeline created 5 clean, analysis-ready datasets in `data/processed/`:

| Dataset Filename | Format | Dimensions | Purpose & Analytical Suitability |
| :--- | :--- | :--- | :--- |
| `cleaned_wide.csv` | Wide tabular | 4,116 × 26 | Primary cleaned dataset containing original measurements, derived calculations, and consistency flags. |
| `monthly_rainfall_long.csv` | Tidy long | 49,392 × 8 | One record per subdivision-year-month with standard ISO date (`YYYY-MM-01`). Supports monthly climatology and time-series modeling. |
| `annual_rainfall.csv` | Annual series | 4,116 × 7 | Annual totals with source, calculated, validated measurements and missingness audit flags. |
| `seasonal_rainfall.csv` | Tidy seasonal | 16,464 × 9 | Four seasons per subdivision-year (Winter, Pre-Monsoon, Monsoon, Post-Monsoon) for inter-seasonal comparisons. |
| `quality_flags.csv` | Audit catalog | 453 × 7 | Granular log of every missing value, consistency discrepancy, and statistical extreme (>99.5th percentile). |
| `dataset_metadata.json` | JSON metadata | Structured | Complete machine-readable audit trail, software versions, schemas, and processing parameters. |

---

## 6. Time-Series Continuity and Feature Engineering Notes

- **Continuous Subdivisions (33):** Continuous 115-year record (1901–2015).
- **Discontinuous Subdivisions (3):**
  - `ANDAMAN & NICOBAR ISLANDS`: Missing 5 years (`[1909, 1943, 1944, 1945, 1948]`).
  - `ARUNACHAL PRADESH`: Missing 18 years (`1901–1915`, `1954–1956`).
  - `LAKSHADWEEP`: Missing 1 year (`1918`).
- **Feature Engineering Safeguard:** The feature engineering module (`src/feature_engineering.py`) explicitly checks year differences ($t - t_{prev} == 1$) before calculating lags. Lags across non-consecutive years are assigned `NaN` to prevent spurious autocorrelation.

---

## 7. Limitations and Next Steps

1. **Subdivision Boundary Evolution:** Meteorological subdivisions represent regional spatial aggregations defined by the India Meteorological Department. Internal rain gauge density evolved over 1901–2015.
2. **Missing Months in Island Territories:** Andaman & Nicobar and Lakshadweep have higher missingness rates due to island rain-gauge network logistics during the early 20th century.
3. **Next Stage:** The processed datasets are validated and ready for:
   - Exploratory data visualization
   - Trend and decadal shift analysis
   - Machine learning anomaly detection
   - Change-point detection (Pettitt test / CUSUM)
   - Interactive dashboard presentation
