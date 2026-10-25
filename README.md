# MonsoonLens AI: Historical Rainfall Pattern Analysis

**MonsoonLens AI** is an advanced machine learning and data science system designed to investigate historical rainfall patterns across India's 36 meteorological subdivisions (1901–2015). It provides rigorous, reproducible data preprocessing, statistical anomaly detection, non-parametric climate regime shift detection, leakage-free time-series forecasting, and an interactive 5-page Streamlit analytical dashboard.

---

## 1. Project Directory Structure

```
MonsoonLens-AI/
├── app.py                                         # Streamlit web application entry point & overview
├── pages/
│   ├── 1_Rainfall_Explorer.py                     # Deep-dive interactive slice & filter explorer
│   ├── 2_Regional_Analysis.py                     # Subdivisional rankings, CV%, & multi-region comparisons
│   ├── 3_Seasonal_Insights.py                     # 4-season dynamics, monsoon share, & correlation matrix
│   ├── 4_Data_Quality_and_Reports.py              # Quality audit logs, 9 research reports, & download center
│   ├── 5_Advanced_Analytics.py                    # Anomalies, Pettitt/CUSUM regime shifts, & ML forecasting
│   └── 6_Historical_vs_Recent_Monitor.py          # Real-time/cached OGD data.gov.in benchmark monitor
├── .streamlit/
│   └── config.toml                                # Theme configuration (dark navy, cyan accents)
├── data/
│   ├── raw/
│   │   └── rainfall_area-wt_sd_1901-2015.csv      # Original raw dataset (strictly read-only)
│   ├── processed/
│   │   ├── cleaned_wide.csv                       # Standardized wide dataset with quality flags
│   │   ├── monthly_rainfall_long.csv              # Tidy monthly time-series (49,392 records)
│   │   ├── annual_rainfall.csv                    # Annual series (source, calculated, validated)
│   │   ├── seasonal_rainfall.csv                  # Tidy seasonal series (16,464 records)
│   │   ├── quality_flags.csv                      # Granular audit log of missing & extreme values
│   │   └── dataset_metadata.json                  # Machine-readable metadata & provenance
│   └── external/
│       ├── data_gov_in_subdivision_recent.csv     # Local cached dataset - provenance not independently verified (2016-2023)
│       └── data_gov_in_metadata.json              # Provenance & candidate resource metadata
├── src/
│   ├── __init__.py                                # Package marker
│   ├── config.py                                  # Central configuration, schemas, thresholds & filepaths
│   ├── data_loader.py                             # Non-mutating cached data access & schema validator
│   ├── data_gov_in_client.py                      # Official data.gov.in REST client with offline fallback
│   ├── recent_rainfall_adapter.py                 # Schema harmonizer & subdivision normalizer
│   ├── historical_recent_comparator.py            # Baseline calculation, departure metrics & IMD categorization
│   ├── dashboard_utils.py                         # Loaders, caching, Plotly chart generators, & UI components
│   ├── inspect_data.py                            # Non-destructive exploratory dataset audit script
│   ├── preprocess.py                              # Core reproducible preprocessing pipeline
│   ├── validate_data.py                           # Post-processing validation & verification engine
│   ├── feature_engineering.py                     # Leakage-free lag, rolling, cyclical & trend features
│   ├── eda_analysis.py                            # Comprehensive EDA analysis & figure generation
│   ├── anomaly_detection.py                       # Multi-method anomaly detection (Z-score, IQR, Isolation Forest)
│   ├── change_point_analysis.py                   # Non-parametric regime shift tests (Pettitt, CUSUM, SNHT)
│   ├── rainfall_forecasting.py                    # Baselines & ML forecasting models with horizon projection
│   ├── model_evaluation.py                        # Standardized metrics (MAE, RMSE, MAPE, R², Skill Score)
│   ├── rainfall_insights.py                       # Natural language insight & climate narrative generation
│   └── backend_pipeline.py                        # End-to-end orchestration pipeline & offline report generator
├── reports/
│   ├── figures/                                   # 7 Publication-grade charts (trends, heatmaps, boxplots)
│   ├── initial_data_quality_report.md             # Baseline raw dataset audit report
│   ├── preprocessing_report.md                    # Comprehensive pipeline execution report
│   ├── validation_report.md                       # Systematic PASS/FAIL validation report
│   ├── data_quality_summary.csv                   # Tabular quality summary per column
│   ├── eda_report.md                              # Comprehensive Stage 2 EDA findings report
│   ├── anomaly_detection_report.md                # Research report on historical rainfall extremes
│   ├── change_point_analysis_report.md            # Structural break & regime shift research report
│   ├── forecasting_report.md                      # Out-of-time forecasting benchmark report
│   ├── model_evaluation_report.md                 # Metric comparison & model skill evaluation report
│   ├── backend_architecture.md                    # Engineering architecture & data flow documentation
│   └── historical_recent_rainfall_monitor_report.md # Methodology & findings report for Recent Monitor
├── notebooks/
│   └── 01_data_exploration.ipynb                 # Interactive visual exploration & diagnostic notebook
├── tests/
│   ├── conftest.py                                # Pytest root path resolution
│   ├── test_data_loader.py                        # Data loader immutability & schema tests
│   ├── test_preprocessing.py                      # Preprocessing test suite (18 unit/integration tests)
│   ├── test_eda.py                                # EDA calculation & figure test suite (7 tests)
│   ├── test_dashboard.py                          # Dashboard loaders, filtering & UI tests (20 tests)
│   ├── test_feature_engineering.py                # Temporal integrity & leakage prevention tests
│   ├── test_anomaly_detection.py                  # Statistical & ML anomaly detection tests
│   ├── test_change_point_analysis.py              # Pettitt, CUSUM, and SNHT change-point tests
│   ├── test_rainfall_forecasting.py               # Out-of-time forecasting pipeline tests
│   ├── test_model_evaluation.py                   # Metric & benchmark skill score tests
│   ├── test_rainfall_insights.py                  # Natural language narrative generation tests
│   ├── test_backend_pipeline.py                   # End-to-end backend orchestration tests
│   ├── test_data_gov_in_client.py                 # Offline-mocked data.gov.in client tests
│   └── test_historical_recent_comparator.py       # Baseline departure & IMD classification tests
├── requirements.txt                               # Project dependencies
├── .env.example                                   # Environment template with DATA_GOV_IN_API_KEY
├── README.md                                      # Project guide and execution documentation
└── .gitignore                                     # Version control exclusions
```

---

## 2. Environment Setup (Windows PowerShell)

Ensure Python 3.10+ is installed. Then open Windows PowerShell in the project directory:

```powershell
# 1. Navigate to the project root directory
cd "C:\Users\madhu\OneDrive\Desktop\MonsoonLens-AI"

# 2. (Optional) Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 3. Pipeline Execution Commands

Execute each stage of the pipeline using the exact commands below:

### Step 1: Inspect the Raw Dataset
Audits the raw CSV file without making any modifications, calculates missingness and geographic coverage, and produces the initial quality reports.
```powershell
python src/inspect_data.py
```

### Step 2: Run the Preprocessing Pipeline
Cleans text labels, standardizes numeric types, calculates seasonal and annual totals strictly from monthly constituent observations, and exports all processed datasets.
```powershell
python src/preprocess.py
```

### Step 3: Run the Validation Suite
Performs 19 automated integrity checks (non-negativity, missingness preservation, row counts, and rounding tolerances) and generates the validation reports.
```powershell
python src/validate_data.py
```

### Step 4: Run Stage 2 Exploratory Data Analysis (EDA)
Computes national and subdivisional descriptive statistics, generates 7 publication-grade figures in `reports/figures/`, and outputs the comprehensive report `reports/eda_report.md`.
```powershell
python src/eda_analysis.py
```

### Step 5: Run the Complete Backend Pipeline
Orchestrates anomaly detection across all 36 subdivisions, executes non-parametric change-point tests (Pettitt, CUSUM, SNHT), benchmarks baseline and ML forecasting models on an out-of-time test period (1996–2015), and generates 5 new research reports in `reports/`.
```powershell
python src/backend_pipeline.py
```

### Step 6: Run the Complete Automated Test Suite (77 Tests)
Executes all 77 unit and integration tests across data loading, preprocessing, EDA, dashboard UI, feature engineering, anomaly detection, regime shift detection, forecasting, and evaluation.
```powershell
python -m pytest -v
```

### Step 7: Launch Interactive Exploration Notebook
Explore the interactive visual workflow in Jupyter:
```powershell
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 4. Processed Output Datasets

All processed artifacts are written to `data/processed/`:

| File | Dimensions | Description |
| :--- | :--- | :--- |
| `cleaned_wide.csv` | 4,116 × 26 | Wide tabular format preserving original measurements alongside calculated totals, missing month counts, and consistency flags. |
| `monthly_rainfall_long.csv` | 49,392 × 8 | Tidy long format (1 row per subdivision-year-month) with standard ISO date (`YYYY-MM-01`), rainfall in mm, and missing flags. Ideal for monthly visualizations and ML modeling. |
| `annual_rainfall.csv` | 4,116 × 7 | Annual rainfall series comparing source-reported and monthly-calculated totals with validation flags. |
| `seasonal_rainfall.csv` | 16,464 × 9 | Tidy seasonal dataset across Winter (Jan-Feb), Pre-Monsoon (Mar-May), Monsoon (Jun-Sep), and Post-Monsoon (Oct-Dec). |
| `quality_flags.csv` | 453 × 7 | Granular audit trail tracking every missing observation, consistency check, and statistical extreme (>99.5th percentile). |
| `dataset_metadata.json` | JSON | Machine-readable metadata detailing processing timestamps, software versions, schema dictionaries, and scientific assumptions. |

---

## 5. Important Scientific Decisions & Data Quality Rules

1. **No Silent Imputation:**
   Missing rainfall values represent genuine gaps in historical gauge observations. They are preserved as `NaN`. Zero rainfall is a physical phenomenon (dry weather); missing rainfall is a recording absence. They are never conflated.
2. **Rejection of Partial Sums:**
   If any month is missing within a season or calendar year, the calculated seasonal/annual total is set to `NaN`. Partial sums are strictly rejected from serving as complete totals.
3. **Preservation of Meteorological Extremes:**
   Values exceeding the 99.5th percentile (e.g. Arunachal Pradesh in July 1948: 2,362.8 mm; Coastal Karnataka in Jun-Sep 1961: 4,536.9 mm) are verified historical weather phenomena, not data entry errors, and are fully preserved.
4. **Time-Series Continuity Safeguard:**
   33 subdivisions possess 115 continuous years (1901–2015), while 3 subdivisions contain historical observation gaps (Andaman & Nicobar Islands missing WWII years 1943–1945, 1909, 1948; Arunachal Pradesh starting in 1916; Lakshadweep missing 1918). Feature engineering strictly validates year differences ($YEAR_t - YEAR_{t-1} == 1$) to prevent computing false lag differences across non-consecutive years.
5. **Leakage-Free Feature Engineering:**
   Prior-year lag features and rolling window statistics are strictly shifted ($t-1$) so that target year $t$ rainfall is never exposed during feature creation.
6. **Honest Out-of-Time Forecasting & Benchmarks:**
   Forecasting models are evaluated against an out-of-time test set (1996–2015) using chronological temporal splits rather than randomized splits. All ML models are rigorously benchmarked against standard climatological baselines (Historical Mean, Naive Persistence, 5-Year Trailing Rolling Mean) using Forecast Skill Scores.
7. **Raw Data Immutability:**
   The raw dataset `data/raw/rainfall_area-wt_sd_1901-2015.csv` is accessed in read-only mode and is never overwritten or mutated.

---

## 6. Interactive Streamlit Web Dashboard

The **MonsoonLens AI** interactive frontend is a multi-page Streamlit analytical web platform designed for exploring 115 years of meteorological observations, detecting anomalies, analyzing regime shifts, and visualizing forecast scenarios.

### 6.1 Multi-Page Architecture

1. **Executive Overview (`app.py`)**:
   - High-level KPIs: temporal coverage (115 yrs), subdivision count (36), record volume (49,392), national mean annual precipitation (1,411.0 mm), and Southwest Monsoon share (74.6%).
   - All-India annual rainfall time series with 10-year rolling window and OLS trend fit.
   - Chronological monthly climatology distribution and percentage share.
   - Seasonal breakdown and Top-5 wettest vs. driest subdivision previews.
   - Direct navigation cards to all 5 analytical subpages.

2. **Interactive Rainfall Explorer (`pages/1_Rainfall_Explorer.py`)**:
   - Flexible multidimensional filtering: timeline slider (1901–2015), subdivision selector, month multiselect, and season selector.
   - Synchronized Plotly visualizations: annual time-series with configurable rolling averages, monthly climatology, Year × Month temporal heatmaps, and distribution boxplots.
   - Dynamic summary metrics (filtered row count, mean, max, and missingness percentage).
   - Filtered dataset table with direct CSV export.

3. **Regional & Subdivisional Analysis (`pages/2_Regional_Analysis.py`)**:
   - Ranked annual rainfall bar charts across all 36 subdivisions.
   - Coefficient of Variation (CV%) ranking highlighting the contrast between high-rainfall coastal/mountain zones and arid interior zones.
   - Side-by-side multi-subdivision comparisons for monthly profiles and annual trajectories.
   - Full 36-subdivision × 12-month climatology intensity heatmap.
   - Comprehensive regional statistical summary table with CSV export.

4. **Seasonal Insights & Dynamics (`pages/3_Seasonal_Insights.py`)**:
   - Investigation across IMD standard seasons: Winter (Jan–Feb), Pre-Monsoon (Mar–May), Southwest Monsoon (Jun–Sep), and Post-Monsoon (Oct–Dec).
   - Seasonal contribution donut chart and bar comparisons.
   - Multi-decadal annual trajectories for all four seasons plotted concurrently.
   - Inter-seasonal Pearson and Spearman correlation matrix heatmaps.
   - Seasonal summary tables with CSV download buttons.

5. **Data Quality, Audit Logs & Reports (`pages/4_Data_Quality_and_Reports.py`)**:
   - Complete inventory and file health check of all processed datasets and metadata.
   - Granular audit trail explorer for 453 recorded quality flags, filterable by flag type, severity, and subdivision.
   - Interactive report viewer for all 8 research and architecture reports.
   - Static publication figure gallery (`reports/figures/`).
   - Unified Research Artifact Download Center for all CSVs and Markdown reports.

6. **Advanced Analytics & ML Modeling (`pages/5_Advanced_Analytics.py`)**:
   - **Tab 1: Anomaly Detection**: Compare Z-Score, IQR, and Isolation Forest algorithms; interactive scatter plot highlighting extreme wet and drought years; natural language anomaly summary.
   - **Tab 2: Regime Shifts & Change Points**: Pettitt's test, CUSUM, and SNHT step-shift identification; pre- and post-shift regime lines; Cohen's $d$ effect size; complete 36-subdivision overview table.
   - **Tab 3: Forecasting & Horizon Projection**: Out-of-time evaluation (1996–2015) across baselines and Random Forest; Forecast Skill Score vs. Historical Mean; 1–10 year horizon projections with 95% empirical prediction intervals.

7. **Historical vs Recent Rainfall Monitor (`pages/6_Historical_vs_Recent_Monitor.py`)**:
   - **Open Government Data (data.gov.in) Integration**: Dual-mode architecture querying live REST endpoints when `DATA_GOV_IN_API_KEY` is present, or gracefully utilizing a local cached dataset (provenance not independently verified) for offline development.
   - **Climatological Baseline Benchmarking**: Real-time evaluation of recent observations against 115-year baseline distributions (1901-2015) across all 36 IMD subdivisions and the national aggregate.
   - **National Aggregate Baseline Limitation**: Explicitly documents that the ALL-INDIA historical baseline is calculated as an unweighted subdivision mean, distinct from official IMD area-weighted totals.
   - **Multi-Metric Departure Diagnostics**: Absolute deviation ($mm$), percentage departure ($\%$, with zero-division protection), standardized deviation ($z$-score), and historical percentile ranking.
   - **Strict 36-Subdivision Ranking**: Separates the 36 canonical meteorological subdivisions from the national aggregate, ensuring regional rankings and category percentages are not distorted.
   - **Standard IMD Classification**: Real-time categorical assignment into Large Excess ($\ge +60\%$), Excess ($+20\%$ to $+59\%$), Normal ($-19\%$ to $+19\%$), Deficient ($-20\%$ to $-59\%$), Large Deficient ($\le -60\%$), No Rain ($-100\%$), and Observation Missing (`NaN`).
   - **Interactive Visualizations & Export Center**: Plotly baseline distribution vs recent observation curves, ranked 36-subdivision departure bar charts, and direct downloads for CSV data tables and Markdown summary reports.

---

## 7. Dashboard Launch Instructions (Windows PowerShell)

Ensure Python 3.10+ is installed and your terminal is in the project root directory.

### Quick Launch Commands:

```powershell
# 1. Activate the project virtual environment (if using one)
.\venv\Scripts\Activate.ps1

# 2. Install / verify required dependencies
python -m pip install -r requirements.txt

# 3. Launch the Streamlit application
python -m streamlit run app.py
```

The application will start and automatically open in your default browser at:
`http://localhost:8501`

### PowerShell Execution Policy Workaround

If PowerShell displays an execution policy error when running `.\.venv\Scripts\Activate.ps1` (`File cannot be loaded because running scripts is disabled on this system`), use either of the following safe workarounds:

**Option A (Session-scoped bypass):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

**Option B (Direct execution via virtual environment Python):**
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

---

## 8. Running the Complete Automated Test Suite

MonsoonLens AI includes **77 automated unit and integration tests** covering data preprocessing, feature engineering continuity safeguards, EDA statistical calculations, figure integrity, dashboard loaders, schema enforcement, filtering logic, anomaly detection, change-point analysis, forecasting baselines, ML regressors, metric calculations, and natural language narrative generators.

```powershell
# Run all 77 automated tests via pytest
python -m pytest -v
```

All 77 tests execute and report `PASSED`:
- `tests/test_data_loader.py` (7 tests)
- `tests/test_preprocessing.py` (18 tests)
- `tests/test_eda.py` (7 tests)
- `tests/test_dashboard.py` (20 tests)
- `tests/test_feature_engineering.py` (3 tests)
- `tests/test_anomaly_detection.py` (5 tests)
- `tests/test_change_point_analysis.py` (5 tests)
- `tests/test_rainfall_forecasting.py` (5 tests)
- `tests/test_model_evaluation.py` (4 tests)
- `tests/test_rainfall_insights.py` (4 tests)
- `tests/test_backend_pipeline.py` (2 tests)

---

## 9. Scientific Limitations and Operational Notice

1. **Empirical Historical Data**: Current dashboard metrics reflect ground-truth historical rain-gauge observations (1901–2015).
2. **Missing Observations**: Historical observation gaps (e.g. WWII in Andaman & Nicobar) are strictly preserved as `NaN` rather than synthetic or zero-filled values.
3. **Forecasting Uncertainty**: Multi-year precipitation forecasting in tropical monsoon systems is subject to chaotic ocean-atmosphere teleconnections (ENSO, IOD, MJO). Projections generated in this system quantify statistical trends, baseline momentum, and planning scenarios with 95% empirical prediction intervals — **not deterministic severe storm or flash flood warnings**.
