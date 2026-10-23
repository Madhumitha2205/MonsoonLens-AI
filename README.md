# MonsoonLens AI: Historical Rainfall Pattern Analysis

**MonsoonLens AI** is a machine learning and data science system designed to investigate historical rainfall patterns across India's 36 meteorological subdivisions (1901–2015). It provides rigorous, reproducible data preprocessing, statistical anomaly detection, time-series feature engineering, and foundation datasets for climate trend analysis.

---

## 1. Project Directory Structure

```
MonsoonLens-AI/
├── data/
│   ├── raw/
│   │   └── rainfall_area-wt_sd_1901-2015.csv      # Original raw dataset (strictly read-only)
│   └── processed/
│       ├── cleaned_wide.csv                       # Standardized wide dataset with quality flags
│       ├── monthly_rainfall_long.csv              # Tidy monthly time-series (49,392 records)
│       ├── annual_rainfall.csv                    # Annual series (source, calculated, validated)
│       ├── seasonal_rainfall.csv                  # Tidy seasonal series (16,464 records)
│       ├── quality_flags.csv                      # Granular audit log of missing & extreme values
│       └── dataset_metadata.json                  # Machine-readable metadata & provenance
├── src/
│   ├── __init__.py                                # Package marker
│   ├── config.py                                  # Central configuration, schemas & thresholds
│   ├── inspect_data.py                            # Non-destructive exploratory dataset audit script
│   ├── preprocess.py                              # Core reproducible preprocessing pipeline
│   ├── validate_data.py                           # Post-processing validation & verification engine
│   ├── feature_engineering.py                     # Leakage-free time-series & anomaly features
│   └── eda_analysis.py                            # Comprehensive EDA analysis & figure generation
├── reports/
│   ├── figures/                                   # 7 Publication-grade charts (trends, heatmaps, boxplots)
│   ├── initial_data_quality_report.md             # Baseline raw dataset audit report
│   ├── preprocessing_report.md                    # Comprehensive pipeline execution report
│   ├── validation_report.md                       # Systematic PASS/FAIL validation report
│   ├── data_quality_summary.csv                   # Tabular quality summary per column
│   └── eda_report.md                              # Comprehensive Stage 2 EDA findings report
├── notebooks/
│   └── 01_data_exploration.ipynb                 # Interactive visual exploration & diagnostic notebook
├── tests/
│   ├── test_preprocessing.py                      # Preprocessing test suite (15 unit/integration tests)
│   └── test_eda.py                                # EDA calculation & figure test suite (7 tests)
├── logs/                                          # Execution logs
├── requirements.txt                               # Project dependencies
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

## 3. Execution Commands

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

### Step 4: Run the Preprocessing Unit Tests
Executes 15 comprehensive unit and integration tests covering edge cases, synthetic error detection, group isolation, and continuity handling.
```powershell
python -m unittest tests/test_preprocessing.py -v
```

### Step 5: Run Stage 2 Exploratory Data Analysis (EDA)
Computes national and subdivisional descriptive statistics, generates 7 publication-grade figures in `reports/figures/`, and outputs the comprehensive report `reports/eda_report.md`.
```powershell
python src/eda_analysis.py
```

### Step 6: Run All Automated Unit Tests (Preprocessing + EDA)
Executes all 22 project unit and integration tests verifying both preprocessing integrity and EDA statistical calculations.
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
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
5. **Raw Data Immutability:**
   The raw dataset `data/raw/rainfall_area-wt_sd_1901-2015.csv` is accessed in read-only mode and is never overwritten or mutated.
