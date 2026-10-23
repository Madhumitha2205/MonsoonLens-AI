"""
MonsoonLens AI - Automated Unit and Integration Test Suite.

Verifies preprocessing integrity, schema validation, missingness preservation,
calculation correctness, group isolation, and continuity handling.
Compatible with standard library unittest and pytest runners.
"""

import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANNUAL_PATH,
    CLEANED_WIDE_PATH,
    CONSISTENCY_TOLERANCE_MM,
    METADATA_PATH,
    MONTH_COLUMNS,
    MONTHLY_LONG_PATH,
    QUALITY_FLAGS_PATH,
    RAW_DATA_PATH,
    RAW_EXPECTED_COLUMNS,
    SEASON_COLUMNS,
    SEASON_MAPPING,
    SEASONAL_PATH,
)
from src.feature_engineering import (
    add_annual_time_series_features,
    add_seasonal_proportions_and_anomalies,
    prepare_changepoint_series,
    validate_continuity_and_sort,
)
from src.preprocess import (
    build_annual_dataset,
    build_cleaned_wide_dataset,
    build_monthly_long_dataset,
    build_seasonal_dataset,
    calculate_derived_aggregations,
    clean_subdivision_name,
    load_and_validate_raw,
    standardize_schema_and_types,
)


class TestRawDatasetProperties(unittest.TestCase):
    """Tests for raw dataset loading and schema guarantees."""

    def test_raw_csv_exists_and_readable(self):
        """Ensure raw CSV exists and can be loaded without errors."""
        self.assertTrue(RAW_DATA_PATH.exists(), f"Raw CSV missing: {RAW_DATA_PATH}")
        df = load_and_validate_raw(RAW_DATA_PATH)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 4116)

    def test_raw_dataset_has_all_expected_columns(self):
        """Ensure all 19 standard columns are present in raw dataset."""
        df = load_and_validate_raw(RAW_DATA_PATH)
        for col in RAW_EXPECTED_COLUMNS:
            self.assertIn(col, df.columns, f"Required column missing: {col}")


class TestDataCleaningAndTypes(unittest.TestCase):
    """Tests for text cleaning, data type conversion, and invalid value detection."""

    def test_clean_subdivision_name(self):
        """Ensure whitespace trimming and multi-space normalization."""
        self.assertEqual(clean_subdivision_name("  KERALA  "), "KERALA")
        self.assertEqual(clean_subdivision_name("ANDAMAN  &   NICOBAR ISLANDS"), "ANDAMAN & NICOBAR ISLANDS")
        self.assertEqual(clean_subdivision_name("COASTAL KARNATAKA\t"), "COASTAL KARNATAKA")

    def test_negative_rainfall_detection_on_synthetic_data(self):
        """Verify that negative rainfall values are detected and flagged."""
        synthetic_df = pd.DataFrame({
            "SUBDIVISION": ["TEST_REGION"],
            "YEAR": [2000],
            "JAN": [-15.0],  # Invalid negative rainfall
            "FEB": [10.0], "MAR": [10.0], "APR": [10.0], "MAY": [10.0], "JUN": [10.0],
            "JUL": [10.0], "AUG": [10.0], "SEP": [10.0], "OCT": [10.0], "NOV": [10.0], "DEC": [10.0],
            "ANNUAL": [105.0],
            "Jan-Feb": [-5.0], "Mar-May": [30.0], "Jun-Sep": [40.0], "Oct-Dec": [30.0]
        })
        cleaned, flags = standardize_schema_and_types(synthetic_df)
        flag_types = [f["flag_type"] for f in flags]
        self.assertIn("NEGATIVE_RAINFALL", flag_types)

    def test_unconvertible_numeric_detection(self):
        """Verify that non-numeric corrupt strings are detected and flagged."""
        synthetic_df = pd.DataFrame({
            "SUBDIVISION": ["TEST_REGION"],
            "YEAR": [2000],
            "JAN": ["CORRUPT_STR"],
            "FEB": [10.0], "MAR": [10.0], "APR": [10.0], "MAY": [10.0], "JUN": [10.0],
            "JUL": [10.0], "AUG": [10.0], "SEP": [10.0], "OCT": [10.0], "NOV": [10.0], "DEC": [10.0],
            "ANNUAL": [110.0],
            "Jan-Feb": [10.0], "Mar-May": [30.0], "Jun-Sep": [40.0], "Oct-Dec": [30.0]
        })
        cleaned, flags = standardize_schema_and_types(synthetic_df)
        flag_types = [f["flag_type"] for f in flags]
        self.assertIn("UNCONVERTIBLE_NUMERIC", flag_types)
        # Verify it became NaN instead of 0.0
        self.assertTrue(np.isnan(cleaned.loc[0, "JAN"]))


class TestMissingValueHandlingAndCalculations(unittest.TestCase):
    """Tests guaranteeing that missing values are never zero-filled and partial sums are rejected."""

    def test_missing_value_preservation(self):
        """Ensure missing values remain NaN and are not replaced with 0 or mean."""
        df_wide = pd.read_csv(CLEANED_WIDE_PATH)
        df_long = pd.read_csv(MONTHLY_LONG_PATH)

        # In Andaman & Nicobar 1923, March was NA in raw data
        row_1923 = df_wide[(df_wide["SUBDIVISION"] == "ANDAMAN & NICOBAR ISLANDS") & (df_wide["YEAR"] == 1923)]
        self.assertEqual(len(row_1923), 1)
        self.assertTrue(np.isnan(row_1923["MAR"].values[0]))
        self.assertNotEqual(row_1923["MAR"].values[0], 0.0)

        # In long format, ensure IS_MISSING == 1 and RAINFALL is NaN
        long_1923_mar = df_long[
            (df_long["SUBDIVISION"] == "ANDAMAN & NICOBAR ISLANDS")
            & (df_long["YEAR"] == 1923)
            & (df_long["MONTH"] == "MAR")
        ]
        self.assertEqual(len(long_1923_mar), 1)
        self.assertEqual(long_1923_mar["IS_MISSING"].values[0], 1)
        self.assertTrue(pd.isna(long_1923_mar["RAINFALL"].values[0]))

    def test_rejection_of_partial_sums(self):
        """Ensure calculated annual and seasonal totals are NaN when any month is missing."""
        synthetic_df = pd.DataFrame({
            "SUBDIVISION": ["TEST_REGION"],
            "YEAR": [2000],
            "JAN": [100.0],
            "FEB": [np.nan],  # Missing FEB!
            "MAR": [50.0], "APR": [50.0], "MAY": [50.0],
            "JUN": [200.0], "JUL": [200.0], "AUG": [200.0], "SEP": [200.0],
            "OCT": [80.0], "NOV": [80.0], "DEC": [80.0],
            "ANNUAL": [np.nan],
            "Jan-Feb": [np.nan], "Mar-May": [150.0], "Jun-Sep": [800.0], "Oct-Dec": [240.0]
        })
        derived_df, _ = calculate_derived_aggregations(synthetic_df)

        # Jan-Feb_CALCULATED must be NaN because FEB is missing
        self.assertTrue(np.isnan(derived_df.loc[0, "Jan-Feb_CALCULATED"]))
        # ANNUAL_CALCULATED must be NaN because FEB is missing (never use partial sum 1190.0!)
        self.assertTrue(np.isnan(derived_df.loc[0, "ANNUAL_CALCULATED"]))
        # Mar-May_CALCULATED must be 150.0 because all 3 months are present
        self.assertEqual(derived_df.loc[0, "Mar-May_CALCULATED"], 150.0)


class TestProcessedDatasetsStructure(unittest.TestCase):
    """Tests ensuring output dimensions, schemas, and date formats are accurate."""

    def test_cleaned_wide_structure(self):
        df = pd.read_csv(CLEANED_WIDE_PATH)
        self.assertEqual(len(df), 4116)
        expected_cols = ["SUBDIVISION", "YEAR"] + MONTH_COLUMNS + ["ANNUAL_SOURCE", "ANNUAL_CALCULATED"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_monthly_long_structure_and_dates(self):
        df = pd.read_csv(MONTHLY_LONG_PATH)
        # Exactly 4116 * 12 = 49,392 rows
        self.assertEqual(len(df), 49392)
        # Validate date format YYYY-MM-01
        sample_dates = df["DATE"].sample(100, random_state=42)
        for d in sample_dates:
            self.assertRegex(d, r"^\d{4}-\d{2}-01$")

    def test_seasonal_dataset_dimensions(self):
        df = pd.read_csv(SEASONAL_PATH)
        # Exactly 4116 * 4 = 16,464 rows
        self.assertEqual(len(df), 16464)
        seasons_set = set(df["SEASON"].unique())
        self.assertEqual(seasons_set, set(SEASON_COLUMNS))

    def test_annual_dataset_dimensions(self):
        df = pd.read_csv(ANNUAL_PATH)
        self.assertEqual(len(df), 4116)
        self.assertIn("ANNUAL_VALIDATED", df.columns)


class TestFeatureEngineeringIntegrity(unittest.TestCase):
    """Tests verifying time-series features, group isolation, and continuity rules."""

    def test_continuity_check_detects_year_gaps(self):
        """Ensure missing calendar years are flagged as non-consecutive."""
        test_df = pd.DataFrame({
            "SUBDIVISION": ["REGION_A", "REGION_A", "REGION_A"],
            "YEAR": [1908, 1910, 1911],  # 1909 is missing!
            "ANNUAL_VALIDATED": [1000.0, 1200.0, 1100.0]
        })
        checked = validate_continuity_and_sort(test_df)
        # For 1910, year diff is 2, so IS_CONSECUTIVE_YEAR must be False
        row_1910 = checked[checked["YEAR"] == 1910].iloc[0]
        self.assertFalse(row_1910["IS_CONSECUTIVE_YEAR"])
        # For 1911, year diff is 1, so IS_CONSECUTIVE_YEAR must be True
        row_1911 = checked[checked["YEAR"] == 1911].iloc[0]
        self.assertTrue(row_1911["IS_CONSECUTIVE_YEAR"])

    def test_lag_not_calculated_across_year_gap(self):
        """Ensure lag features are NaN when preceding record is not consecutive year."""
        test_df = pd.DataFrame({
            "SUBDIVISION": ["REGION_A", "REGION_A", "REGION_A"],
            "YEAR": [1908, 1910, 1911],  # 1909 is missing
            "ANNUAL_VALIDATED": [1000.0, 1200.0, 1100.0]
        })
        feat_df = add_annual_time_series_features(test_df, rolling_windows=[5], baseline_years=None)
        # 1910 should NOT take 1908 as previous year!
        row_1910 = feat_df[feat_df["YEAR"] == 1910].iloc[0]
        self.assertTrue(np.isnan(row_1910["PREV_YEAR_RAINFALL"]))
        self.assertTrue(np.isnan(row_1910["ONE_YEAR_DIFF"]))

        # 1911 should take 1910 as previous year (1200.0)
        row_1911 = feat_df[feat_df["YEAR"] == 1911].iloc[0]
        self.assertEqual(row_1911["PREV_YEAR_RAINFALL"], 1200.0)
        self.assertEqual(row_1911["ONE_YEAR_DIFF"], -100.0)

    def test_rolling_features_group_isolation(self):
        """Ensure rolling statistics never bleed across subdivisions."""
        test_df = pd.DataFrame({
            "SUBDIVISION": ["SUB_A"] * 5 + ["SUB_B"] * 5,
            "YEAR": list(range(2001, 2006)) + list(range(2001, 2006)),
            "ANNUAL_VALIDATED": [100.0] * 5 + [1000.0] * 5
        })
        feat_df = add_annual_time_series_features(test_df, rolling_windows=[3], baseline_years=None)
        # The rolling mean of SUB_B at its first entry (2001) must NOT include SUB_A's 100.0
        sub_b_first = feat_df[(feat_df["SUBDIVISION"] == "SUB_B") & (feat_df["YEAR"] == 2001)].iloc[0]
        # At window=3 with min_periods=2, index 0 of SUB_B has only 1 period, so it should be NaN
        self.assertTrue(np.isnan(sub_b_first["ROLLING_MEAN_3Y"]))

        # At index 2 (2003) of SUB_B, rolling mean of [1000, 1000, 1000] must be 1000.0
        sub_b_third = feat_df[(feat_df["SUBDIVISION"] == "SUB_B") & (feat_df["YEAR"] == 2003)].iloc[0]
        self.assertEqual(sub_b_third["ROLLING_MEAN_3Y"], 1000.0)

    def test_changepoint_series_preparation(self):
        """Ensure changepoint series extractor correctly flags missing years and length."""
        ann_df = pd.read_csv(ANNUAL_PATH)
        # Test on Andaman & Nicobar Islands (has missing years)
        sub_df, status = prepare_changepoint_series(ann_df, "ANDAMAN & NICOBAR ISLANDS", min_observations=30)
        self.assertTrue(status["is_sufficient_length"])
        self.assertEqual(status["missing_years_count"], 5)
        self.assertEqual(status["missing_calendar_years"], [1909, 1943, 1944, 1945, 1948])


if __name__ == "__main__":
    unittest.main()
