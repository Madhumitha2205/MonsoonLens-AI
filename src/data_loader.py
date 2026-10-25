"""
MonsoonLens AI - Robust Backend Data Access and Validation Layer.

Provides clean, safe, and reproducible data loading, schema validation,
unit verification, and missing value checks across all processed datasets.
Returns copies of datasets to prevent in-place mutation and provides actionable
error handling.
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANNUAL_PATH,
    CLEANED_WIDE_PATH,
    DATA_QUALITY_SUMMARY_PATH,
    METADATA_PATH,
    MONTH_COLUMNS,
    MONTHLY_LONG_PATH,
    QUALITY_FLAGS_PATH,
    SEASON_COLUMNS,
    SEASONAL_PATH,
)

# -----------------------------------------------------------------------------
# Expected Schemas and Types
# -----------------------------------------------------------------------------
EXPECTED_ANNUAL_COLUMNS: List[str] = [
    "SUBDIVISION",
    "YEAR",
    "ANNUAL_SOURCE",
    "ANNUAL_CALCULATED",
    "ANNUAL_VALIDATED",
    "ANNUAL_MISSING_MONTHS",
    "ANNUAL_QUALITY_FLAG",
]

EXPECTED_MONTHLY_COLUMNS: List[str] = [
    "SUBDIVISION",
    "YEAR",
    "MONTH",
    "MONTH_NUMBER",
    "DATE",
    "RAINFALL",
    "IS_MISSING",
    "QUALITY_FLAG",
]

EXPECTED_SEASONAL_COLUMNS: List[str] = [
    "SUBDIVISION",
    "YEAR",
    "SEASON",
    "SEASON_NUMBER",
    "SEASON_NAME",
    "RAINFALL_SOURCE",
    "RAINFALL_CALCULATED",
    "RAINFALL_VALIDATED",
    "QUALITY_FLAG",
]

EXPECTED_WIDE_COLUMNS: List[str] = (
    ["SUBDIVISION", "YEAR"]
    + MONTH_COLUMNS
    + ["ANNUAL_SOURCE"]
    + [f"{s}_SOURCE" for s in SEASON_COLUMNS]
    + ["ANNUAL_CALCULATED"]
    + [f"{s}_CALCULATED" for s in SEASON_COLUMNS]
    + ["MISSING_MONTHS_COUNT", "ANNUAL_CONSISTENT"]
)


# -----------------------------------------------------------------------------
# Validation Utilities
# -----------------------------------------------------------------------------
def validate_dataframe_schema(
    df: pd.DataFrame,
    expected_columns: List[str],
    dataset_name: str = "Dataset"
) -> None:
    """
    Validate that all required columns are present in the dataframe.
    Raises ValueError with descriptive actionable error message if missing.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError(f"{dataset_name} must be a valid pandas DataFrame.")

    if df.empty:
        raise ValueError(f"{dataset_name} is completely empty (0 records).")

    missing_cols = [c for c in expected_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_rainfall_values(
    df: Union[pd.DataFrame, pd.Series],
    rainfall_cols: Optional[Union[str, List[str]]] = None,
    dataset_name: str = "Dataset"
) -> Dict[str, Any]:
    """
    Audit rainfall measurements for validity:
    - Confirms values are numeric
    - Flags any negative rainfall values (raises ValueError if negative values found)
    - Flags any infinite values (+inf / -inf)
    - Distinguishes missing values (NaN) from valid zero rainfall.
    """
    if isinstance(df, pd.Series):
        cname = rainfall_cols if isinstance(rainfall_cols, str) else (df.name or "RAINFALL")
        df = pd.DataFrame({cname: df})
        rainfall_cols = [cname]
    elif rainfall_cols is None:
        rainfall_cols = [c for c in df.columns if "RAIN" in c.upper() or "ANNUAL" in c.upper()]
    elif isinstance(rainfall_cols, str):
        rainfall_cols = [rainfall_cols]

    report: Dict[str, Any] = {
        "is_valid": True,
        "negative_count": 0,
        "infinite_count": 0,
        "missing_count": 0,
        "zero_count": 0,
        "details": {}
    }

    for col in rainfall_cols:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        nan_count = int(series.isna().sum())
        inf_count = int(np.isinf(series).sum())
        neg_count = int((series < 0).sum())
        zero_count = int((series == 0).sum())

        col_valid = (inf_count == 0) and (neg_count == 0)
        if not col_valid:
            report["is_valid"] = False

        report["negative_count"] += neg_count
        report["infinite_count"] += inf_count
        report["missing_count"] += nan_count
        report["zero_count"] += zero_count

        report["details"][col] = {
            "negative_count": neg_count,
            "infinite_count": inf_count,
            "missing_count": nan_count,
            "zero_count": zero_count,
            "min": float(series.min()) if not series.dropna().empty else None,
            "max": float(series.max()) if not series.dropna().empty else None,
        }

    if report["negative_count"] > 0:
        raise ValueError(
            f"{dataset_name} contains {report['negative_count']} negative values, which is physically impossible for rainfall."
        )

    return report


def check_subdivision_continuity(
    annual_df: pd.DataFrame,
    subdivision: Optional[str] = None
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Audit calendar year continuity for a specific subdivision or across all subdivisions.
    Returns record count, year bounds, and missing calendar years.
    """
    if subdivision is not None:
        sub_df = annual_df[annual_df["SUBDIVISION"] == subdivision].sort_values("YEAR")
        if sub_df.empty:
            return {
                "subdivision": subdivision,
                "record_count": 0,
                "is_continuous": False,
                "missing_years": [],
                "missing_years_count": 0,
            }

        years = sub_df["YEAR"].astype(int).tolist()
        full_span = set(range(min(years), max(years) + 1))
        missing_years = sorted(list(full_span - set(years)))

        return {
            "subdivision": subdivision,
            "record_count": len(years),
            "min_year": min(years),
            "max_year": max(years),
            "is_continuous": len(missing_years) == 0,
            "missing_years": missing_years,
            "missing_years_count": len(missing_years),
        }

    # If subdivision is None, audit all subdivisions and return a DataFrame indexed by SUBDIVISION
    records = []
    for sub, grp in annual_df.groupby("SUBDIVISION"):
        years = grp["YEAR"].dropna().astype(int).tolist()
        if not years:
            records.append({
                "SUBDIVISION": sub,
                "record_count": 0,
                "min_year": None,
                "max_year": None,
                "is_continuous": False,
                "missing_year_count": 0,
                "missing_years": [],
            })
            continue
        full_span = set(range(min(years), max(years) + 1))
        missing_years = sorted(list(full_span - set(years)))
        records.append({
            "SUBDIVISION": sub,
            "record_count": len(years),
            "min_year": min(years),
            "max_year": max(years),
            "is_continuous": len(missing_years) == 0,
            "missing_year_count": len(missing_years),
            "missing_years": missing_years,
        })
    return pd.DataFrame(records).set_index("SUBDIVISION")


# -----------------------------------------------------------------------------
# Data Loaders (Returning Fresh Copies)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _read_csv_cached(file_path_str: str) -> pd.DataFrame:
    """Internal cached reader reading raw CSV string paths."""
    p = Path(file_path_str)
    if not p.exists():
        raise FileNotFoundError(f"Processed dataset file not found at: {p.resolve()}")
    return pd.read_csv(p)


def load_annual_rainfall(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load validated annual rainfall dataset.
    Returns a fresh DataFrame copy.
    """
    target = path or ANNUAL_PATH
    df = _read_csv_cached(str(target)).copy()
    validate_dataframe_schema(df, EXPECTED_ANNUAL_COLUMNS, "Annual Rainfall Dataset")
    return df


def load_monthly_rainfall(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load tidy long monthly rainfall dataset.
    Returns a fresh DataFrame copy.
    """
    target = path or MONTHLY_LONG_PATH
    df = _read_csv_cached(str(target)).copy()
    validate_dataframe_schema(df, EXPECTED_MONTHLY_COLUMNS, "Monthly Rainfall Dataset")
    return df


def load_seasonal_rainfall(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load tidy seasonal rainfall dataset.
    Returns a fresh DataFrame copy.
    """
    target = path or SEASONAL_PATH
    df = _read_csv_cached(str(target)).copy()
    validate_dataframe_schema(df, EXPECTED_SEASONAL_COLUMNS, "Seasonal Rainfall Dataset")
    return df


def load_cleaned_wide(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load cleaned wide rainfall dataset.
    Returns a fresh DataFrame copy.
    """
    target = path or CLEANED_WIDE_PATH
    df = _read_csv_cached(str(target)).copy()
    validate_dataframe_schema(df, EXPECTED_WIDE_COLUMNS, "Cleaned Wide Rainfall Dataset")
    return df


def load_quality_flags(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load quality flags audit log.
    Returns a fresh DataFrame copy.
    """
    target = path or QUALITY_FLAGS_PATH
    df = _read_csv_cached(str(target)).copy()
    expected_flags_cols = ["subdivision", "year", "field", "original_value", "flag_type", "severity", "reason"]
    validate_dataframe_schema(df, expected_flags_cols, "Quality Flags Dataset")
    return df


def load_dataset_metadata(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load JSON provenance metadata file.
    Returns metadata dictionary.
    """
    target = path or METADATA_PATH
    if not target.exists():
        raise FileNotFoundError(f"Metadata file missing at: {target.resolve()}")
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def get_available_subdivisions(annual_df: Optional[pd.DataFrame] = None) -> List[str]:
    """Return sorted unique subdivision names available in the annual dataset."""
    df = annual_df if annual_df is not None else load_annual_rainfall()
    return sorted(df["SUBDIVISION"].dropna().unique().tolist())


# Aliases for compatibility
load_annual_data = load_annual_rainfall
load_monthly_data = load_monthly_rainfall
load_seasonal_data = load_seasonal_rainfall
load_cleaned_wide_data = load_cleaned_wide
