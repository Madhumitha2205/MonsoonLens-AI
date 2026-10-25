"""
MonsoonLens AI - Central Configuration Module.

Provides paths, column schemas, seasonal mappings, quality thresholds,
and feature engineering parameters. Uses pathlib for robust cross-platform path handling.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# Base Directories
# -----------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
REPORTS_DIR: Path = BASE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
LOGS_DIR: Path = BASE_DIR / "logs"
TESTS_DIR: Path = BASE_DIR / "tests"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Dataset File Paths
# -----------------------------------------------------------------------------
RAW_DATA_PATH: Path = RAW_DATA_DIR / "rainfall_area-wt_sd_1901-2015.csv"

# Processed output files
CLEANED_WIDE_PATH: Path = PROCESSED_DATA_DIR / "cleaned_wide.csv"
MONTHLY_LONG_PATH: Path = PROCESSED_DATA_DIR / "monthly_rainfall_long.csv"
ANNUAL_PATH: Path = PROCESSED_DATA_DIR / "annual_rainfall.csv"
SEASONAL_PATH: Path = PROCESSED_DATA_DIR / "seasonal_rainfall.csv"
QUALITY_FLAGS_PATH: Path = PROCESSED_DATA_DIR / "quality_flags.csv"
METADATA_PATH: Path = PROCESSED_DATA_DIR / "dataset_metadata.json"

# External Official Datasets (data.gov.in)
RECENT_RAINFALL_PATH: Path = EXTERNAL_DATA_DIR / "data_gov_in_subdivision_recent.csv"
DATA_GOV_IN_METADATA_PATH: Path = EXTERNAL_DATA_DIR / "data_gov_in_metadata.json"

# Report paths
INITIAL_REPORT_PATH: Path = REPORTS_DIR / "initial_data_quality_report.md"
PREPROCESSING_REPORT_PATH: Path = REPORTS_DIR / "preprocessing_report.md"
VALIDATION_REPORT_PATH: Path = REPORTS_DIR / "validation_report.md"
DATA_QUALITY_SUMMARY_PATH: Path = REPORTS_DIR / "data_quality_summary.csv"
EDA_REPORT_PATH: Path = REPORTS_DIR / "eda_report.md"
ANOMALY_REPORT_PATH: Path = REPORTS_DIR / "anomaly_detection_report.md"
CHANGE_POINT_REPORT_PATH: Path = REPORTS_DIR / "change_point_analysis_report.md"
FORECASTING_REPORT_PATH: Path = REPORTS_DIR / "forecasting_report.md"
MODEL_EVALUATION_REPORT_PATH: Path = REPORTS_DIR / "model_evaluation_report.md"
BACKEND_ARCHITECTURE_REPORT_PATH: Path = REPORTS_DIR / "backend_architecture.md"
HISTORICAL_RECENT_REPORT_PATH: Path = REPORTS_DIR / "historical_recent_rainfall_monitor_report.md"

# -----------------------------------------------------------------------------
# Backend, Modeling & External API Parameters
# -----------------------------------------------------------------------------
DEFAULT_TRAIN_END_YEAR: int = 1990
DEFAULT_FORECAST_HORIZON: int = 5
ANOMALY_Z_SCORE_THRESHOLD: float = 2.0
ANOMALY_IQR_MULTIPLIER: float = 1.5
ANOMALY_IFOREST_CONTAMINATION: float = 0.05
CHANGE_POINT_P_VALUE_THRESHOLD: float = 0.05

# Official data.gov.in OGD Platform parameters
DATA_GOV_IN_BASE_URL: str = "https://api.data.gov.in/resource/"
DATA_GOV_IN_DEFAULT_RESOURCE_ID: str = "fb5e2a33-3e34-4087-83a1-f0e46f2748d0"
DATA_GOV_IN_ALL_INDIA_RESOURCE_ID: str = "7146eb85-c33f-4436-b75a-25dad6ea9d8a"
DATA_GOV_IN_REQUEST_TIMEOUT: int = 15

# Standard India Meteorological Department (IMD) Departure Categories
# Percentage departure bands:
# Large Excess (LE): >= +60%
# Excess (E): +20% to +59%
# Normal (N): -19% to +19%
# Deficient (D): -20% to -59%
# Large Deficient (LD): <= -60%
# No Rain (NR): -100%
IMD_CATEGORIES = {
    "LARGE_EXCESS": {"label": "Large Excess", "min": 60.0, "max": float("inf"), "color": "#1d4ed8"},
    "EXCESS": {"label": "Excess", "min": 20.0, "max": 59.99, "color": "#0284c7"},
    "NORMAL": {"label": "Normal", "min": -19.99, "max": 19.99, "color": "#10b981"},
    "DEFICIENT": {"label": "Deficient", "min": -59.99, "max": -20.0, "color": "#f59e0b"},
    "LARGE_DEFICIENT": {"label": "Large Deficient", "min": -99.99, "max": -60.0, "color": "#ef4444"},
    "NO_RAIN": {"label": "No Rain", "min": -100.0, "max": -100.0, "color": "#64748b"},
}

# -----------------------------------------------------------------------------
# Column Schemas and Mappings
# -----------------------------------------------------------------------------
MONTH_COLUMNS: List[str] = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]

MONTH_NUMBERS: Dict[str, int] = {
    month: idx + 1 for idx, month in enumerate(MONTH_COLUMNS)
}

SEASON_COLUMNS: List[str] = [
    "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"
]

SEASON_MAPPING: Dict[str, List[str]] = {
    "Jan-Feb": ["JAN", "FEB"],
    "Mar-May": ["MAR", "APR", "MAY"],
    "Jun-Sep": ["JUN", "JUL", "AUG", "SEP"],
    "Oct-Dec": ["OCT", "NOV", "DEC"]
}

SEASON_NUMBERS: Dict[str, int] = {
    "Jan-Feb": 1,
    "Mar-May": 2,
    "Jun-Sep": 3,
    "Oct-Dec": 4
}

SEASON_DESCRIPTIONS: Dict[str, str] = {
    "Jan-Feb": "Winter",
    "Mar-May": "Pre-Monsoon / Summer",
    "Jun-Sep": "Southwest Monsoon",
    "Oct-Dec": "Post-Monsoon / Northeast Monsoon"
}

RAW_EXPECTED_COLUMNS: List[str] = (
    ["SUBDIVISION", "YEAR"]
    + MONTH_COLUMNS
    + ["ANNUAL"]
    + SEASON_COLUMNS
)

# -----------------------------------------------------------------------------
# Validation and Quality Thresholds
# -----------------------------------------------------------------------------
# Floating-point tolerance in mm to accommodate single-decimal rounding across 12 months
# (Each monthly value may have ±0.05 mm rounding error; 12 * 0.05 = 0.6 mm max theoretical)
CONSISTENCY_TOLERANCE_MM: float = 0.5

# Expected time range
MIN_EXPECTED_YEAR: int = 1901
MAX_EXPECTED_YEAR: int = 2015
EXPECTED_SUBDIVISIONS_COUNT: int = 36

# Statistical anomaly & feature settings
DEFAULT_BASELINE_YEARS: Tuple[int, int] = (1971, 2000)  # Standard IMD climatological normal baseline
ROLLING_WINDOWS: List[int] = [5, 10, 20]
MIN_OBSERVATION_COUNT: int = 30  # Minimum valid annual observations required for statistical stability

RANDOM_SEED: int = 42
