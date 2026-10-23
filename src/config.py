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
REPORTS_DIR: Path = BASE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
LOGS_DIR: Path = BASE_DIR / "logs"
TESTS_DIR: Path = BASE_DIR / "tests"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR]:
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

# Report paths
INITIAL_REPORT_PATH: Path = REPORTS_DIR / "initial_data_quality_report.md"
PREPROCESSING_REPORT_PATH: Path = REPORTS_DIR / "preprocessing_report.md"
VALIDATION_REPORT_PATH: Path = REPORTS_DIR / "validation_report.md"
DATA_QUALITY_SUMMARY_PATH: Path = REPORTS_DIR / "data_quality_summary.csv"
EDA_REPORT_PATH: Path = REPORTS_DIR / "eda_report.md"

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
