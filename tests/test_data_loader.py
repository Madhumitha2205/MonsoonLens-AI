"""
Tests for src/data_loader.py - Data access layer and schema/range validators.
"""

import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_annual_data,
    load_monthly_data,
    load_seasonal_data,
    load_cleaned_wide_data,
    validate_dataframe_schema,
    validate_rainfall_values,
    check_subdivision_continuity,
)


def test_validate_dataframe_schema_success():
    """Verify that schema validation succeeds when all expected columns are present."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    # Should not raise
    validate_dataframe_schema(df, ["A", "B"], "test_dataset")


def test_validate_dataframe_schema_missing_columns():
    """Verify that schema validation raises ValueError when columns are missing."""
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe_schema(df, ["A", "B", "C"], "test_dataset")


def test_validate_rainfall_values_valid():
    """Verify rainfall validation succeeds for physically reasonable rainfall values."""
    s = pd.Series([0.0, 150.5, 1200.0, np.nan, 3500.0])
    validate_rainfall_values(s, "TEST_RAINFALL")


def test_validate_rainfall_values_negative_raises():
    """Verify rainfall validation raises ValueError on physically impossible negative values."""
    s = pd.Series([100.0, -5.0, 200.0])
    with pytest.raises(ValueError, match="negative values"):
        validate_rainfall_values(s, "TEST_RAINFALL")


def test_check_subdivision_continuity():
    """Verify continuity checker identifies both continuous and interrupted historical series."""
    # Continuous series
    df_cont = pd.DataFrame({
        "SUBDIVISION": ["TEST_A"] * 5,
        "YEAR": [2000, 2001, 2002, 2003, 2004],
        "ANNUAL": [1000] * 5,
    })
    cont_summary = check_subdivision_continuity(df_cont)
    assert cont_summary.loc["TEST_A", "is_continuous"] == True
    assert cont_summary.loc["TEST_A", "missing_year_count"] == 0

    # Interrupted series (e.g. WWII gap)
    df_gap = pd.DataFrame({
        "SUBDIVISION": ["TEST_B"] * 4,
        "YEAR": [1940, 1941, 1946, 1947],
        "ANNUAL": [1000] * 4,
    })
    gap_summary = check_subdivision_continuity(df_gap)
    assert gap_summary.loc["TEST_B", "is_continuous"] == False
    assert gap_summary.loc["TEST_B", "missing_year_count"] == 4  # 1942, 1943, 1944, 1945


def test_load_annual_data_non_mutating():
    """Verify that load_annual_data returns a copy and is not mutated by downstream modifications."""
    df1 = load_annual_data()
    orig_shape = df1.shape
    df1["MUTATION_TEST"] = 999.0

    df2 = load_annual_data()
    assert "MUTATION_TEST" not in df2.columns
    assert df2.shape == orig_shape
    assert "ANNUAL_VALIDATED" in df2.columns


def test_load_all_processed_datasets():
    """Verify that annual, monthly, seasonal, and wide datasets load successfully with valid schemas."""
    monthly = load_monthly_data()
    assert not monthly.empty
    assert "RAINFALL" in monthly.columns

    seasonal = load_seasonal_data()
    assert not seasonal.empty
    assert "RAINFALL_VALIDATED" in seasonal.columns

    wide = load_cleaned_wide_data()
    assert not wide.empty
    assert "JAN" in wide.columns and "DEC" in wide.columns
