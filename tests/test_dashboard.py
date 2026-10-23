"""
MonsoonLens AI - Unit and Integration Tests for Frontend & Dashboard Utilities.

Validates data loading, schema enforcement, filtering logic, statistical aggregation,
safe missing value handling, and Plotly figure generation.
Compatible with pytest and unittest.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard_utils import (
    compute_filtered_summary,
    create_annual_timeseries_plot,
    create_monthly_climatology_plot,
    create_rainfall_heatmap,
    create_seasonal_breakdown_plot,
    create_subdivision_ranking_plot,
    filter_annual_data,
    filter_monthly_data,
    filter_seasonal_data,
    format_rainfall_mm,
    get_unique_subdivisions,
    load_annual_rainfall,
    load_cleaned_wide,
    load_dataset_metadata,
    load_monthly_rainfall,
    load_quality_flags,
    load_seasonal_rainfall,
    validate_required_columns,
)


class TestDashboardDataLoaders(unittest.TestCase):
    """Test loading and column schema enforcement of processed datasets."""

    def test_load_dataset_metadata(self):
        meta = load_dataset_metadata()
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta.get("project_name"), "MonsoonLens AI")
        self.assertIn("source_data", meta)
        self.assertEqual(meta["source_data"]["subdivisions_count"], 36)

    def test_load_annual_rainfall_schema(self):
        df = load_annual_rainfall()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4116)
        expected = ["SUBDIVISION", "YEAR", "ANNUAL_SOURCE", "ANNUAL_CALCULATED", "ANNUAL_VALIDATED", "ANNUAL_QUALITY_FLAG"]
        for col in expected:
            self.assertIn(col, df.columns)

    def test_load_monthly_rainfall_schema(self):
        df = load_monthly_rainfall()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 49392)
        expected = ["SUBDIVISION", "YEAR", "MONTH", "MONTH_NUMBER", "DATE", "RAINFALL", "IS_MISSING"]
        for col in expected:
            self.assertIn(col, df.columns)

    def test_load_seasonal_rainfall_schema(self):
        df = load_seasonal_rainfall()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 16464)
        expected = ["SUBDIVISION", "YEAR", "SEASON", "SEASON_NAME", "RAINFALL_VALIDATED"]
        for col in expected:
            self.assertIn(col, df.columns)

    def test_load_cleaned_wide_schema(self):
        df = load_cleaned_wide()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4116)
        self.assertIn("SUBDIVISION", df.columns)
        self.assertIn("YEAR", df.columns)
        self.assertIn("JAN", df.columns)
        self.assertIn("DEC", df.columns)

    def test_load_quality_flags_schema(self):
        df = load_quality_flags()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 453)
        expected = ["subdivision", "year", "field", "flag_type", "severity", "reason"]
        for col in expected:
            self.assertIn(col, df.columns)


class TestValidationAndFormatting(unittest.TestCase):
    """Test validation utilities, error triggers, and string formatters."""

    def test_validate_required_columns_success(self):
        sample_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        # Should execute silently without exception
        validate_required_columns(sample_df, ["A", "B"], "test_dataset")

    def test_validate_required_columns_failure(self):
        sample_df = pd.DataFrame({"A": [1, 2]})
        with self.assertRaises(ValueError) as ctx:
            validate_required_columns(sample_df, ["A", "B", "C"], "test_dataset")
        self.assertIn("missing required columns", str(ctx.exception))
        self.assertIn("['B', 'C']", str(ctx.exception))

    def test_format_rainfall_mm(self):
        self.assertEqual(format_rainfall_mm(1411.01), "1,411.0 mm")
        self.assertEqual(format_rainfall_mm(0.0), "0.0 mm")
        self.assertEqual(format_rainfall_mm(np.nan), "N/A")
        self.assertEqual(format_rainfall_mm(None), "N/A")


class TestDataFilteringAndAggregations(unittest.TestCase):
    """Test pure filtering logic and statistical aggregators."""

    @classmethod
    def setUpClass(cls):
        cls.ann_df = load_annual_rainfall()
        cls.monthly_df = load_monthly_rainfall()
        cls.seasonal_df = load_seasonal_rainfall()

    def test_filter_annual_data(self):
        filtered = filter_annual_data(
            self.ann_df,
            year_range=(2000, 2010),
            subdivisions=["KERALA", "PUNJAB"],
        )
        self.assertEqual(filtered["YEAR"].min(), 2000)
        self.assertEqual(filtered["YEAR"].max(), 2010)
        self.assertEqual(set(filtered["SUBDIVISION"].unique()), {"KERALA", "PUNJAB"})
        self.assertEqual(len(filtered), 22)  # 11 years * 2 subdivisions

    def test_filter_monthly_data(self):
        filtered = filter_monthly_data(
            self.monthly_df,
            year_range=(2010, 2012),
            subdivisions=["KERALA"],
            months=["JUL", "AUG"],
        )
        self.assertEqual(len(filtered), 6)  # 3 years * 1 sub * 2 months
        self.assertEqual(set(filtered["MONTH"].unique()), {"JUL", "AUG"})
        self.assertTrue((filtered["SUBDIVISION"] == "KERALA").all())

    def test_filter_seasonal_data(self):
        filtered = filter_seasonal_data(
            self.seasonal_df,
            year_range=(1990, 1999),
            subdivisions=["BIHAR"],
            seasons=["Jun-Sep"],
        )
        self.assertEqual(len(filtered), 10)  # 10 years * 1 season
        self.assertEqual(filtered["SEASON"].iloc[0], "Jun-Sep")

    def test_compute_filtered_summary(self):
        s = pd.Series([100.0, 200.0, 300.0, np.nan])
        stats = compute_filtered_summary(s)
        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["mean"], 200.0)
        self.assertAlmostEqual(stats["median"], 200.0)
        self.assertAlmostEqual(stats["min"], 100.0)
        self.assertAlmostEqual(stats["max"], 300.0)
        self.assertAlmostEqual(stats["cv_percent"], 50.0)

    def test_compute_filtered_summary_empty(self):
        empty_s = pd.Series([], dtype=float)
        stats = compute_filtered_summary(empty_s)
        self.assertEqual(stats["count"], 0)
        self.assertTrue(np.isnan(stats["mean"]))

    def test_get_unique_subdivisions(self):
        subs = get_unique_subdivisions(self.ann_df)
        self.assertEqual(len(subs), 36)
        self.assertEqual(subs, sorted(subs))
        self.assertIn("KERALA", subs)
        self.assertIn("WEST RAJASTHAN", subs)


class TestPlotGenerators(unittest.TestCase):
    """Test that Plotly chart generators produce valid Figure objects without errors."""

    @classmethod
    def setUpClass(cls):
        cls.ann_df = load_annual_rainfall()
        cls.monthly_df = load_monthly_rainfall()
        cls.seasonal_df = load_seasonal_rainfall()

    def test_create_annual_timeseries_plot(self):
        fig = create_annual_timeseries_plot(
            self.ann_df,
            title="Test Annual Series",
            subdivision_name="All India",
            show_rolling=True,
            show_trendline=True,
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_create_monthly_climatology_plot(self):
        monthly_stats = (
            self.monthly_df.groupby(["MONTH_NUMBER", "MONTH"])["RAINFALL"]
            .agg(mean_rainfall="mean")
            .reset_index()
        )
        monthly_stats["percent_of_annual"] = [5.0] * 12
        fig = create_monthly_climatology_plot(monthly_stats, title="Test Climatology")
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_create_seasonal_breakdown_plot(self):
        seasonal_stats = pd.DataFrame({
            "SEASON": ["Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"],
            "SEASON_NAME": ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"],
            "mean_rainfall": [40.0, 130.0, 1050.0, 190.0],
        })
        fig = create_seasonal_breakdown_plot(seasonal_stats, title="Test Season")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)

    def test_create_subdivision_ranking_plot(self):
        sub_stats = pd.DataFrame({
            "SUBDIVISION": ["KERALA", "RAJASTHAN"],
            "mean_annual_mm": [3000.0, 300.0],
        })
        fig = create_subdivision_ranking_plot(sub_stats, metric_col="mean_annual_mm")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)

    def test_create_rainfall_heatmap(self):
        df = pd.DataFrame({
            "YEAR": [2000, 2000, 2001, 2001],
            "MONTH": ["JAN", "FEB", "JAN", "FEB"],
            "RAINFALL": [10.0, 20.0, 30.0, 40.0],
        })
        fig = create_rainfall_heatmap(df, "YEAR", "MONTH", "RAINFALL")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)


if __name__ == "__main__":
    unittest.main()
