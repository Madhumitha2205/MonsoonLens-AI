"""
MonsoonLens AI - Unit Tests for Historical vs Recent Comparator & Adapter.

Validates:
1. Subdivision name standardization, alias resolution, and unknown subdivision detection.
2. Duplicate record detection and rejection (never silently selects the first record).
3. Preservation of missing observations (never replaces NaN with zero or partial sums).
4. ALL-INDIA baseline calculation as unweighted mean with limitation notes.
5. Strict separation of ALL-INDIA from the 36-subdivision ranking.
6. IMD departure classification, safe missing metric handling, and CSV/Markdown generation.
"""

import unittest
import numpy as np
import pandas as pd

from src.historical_recent_comparator import (
    HistoricalRecentComparator,
    classify_imd_departure,
)
from src.recent_rainfall_adapter import (
    CANONICAL_SUBDIVISIONS,
    RecentRainfallAdapter,
)


class TestRecentRainfallAdapter(unittest.TestCase):
    """Test suite for RecentRainfallAdapter validation and harmonization."""

    def test_standardize_subdivision_name_canonical_and_aliases(self):
        self.assertEqual(RecentRainfallAdapter.standardize_subdivision_name("BIHAR"), "BIHAR")
        self.assertEqual(RecentRainfallAdapter.standardize_subdivision_name("kerala  "), "KERALA")
        self.assertEqual(RecentRainfallAdapter.standardize_subdivision_name("  ALL INDIA "), "ALL-INDIA")
        self.assertEqual(RecentRainfallAdapter.standardize_subdivision_name("ODISHA"), "ORISSA")
        self.assertEqual(RecentRainfallAdapter.standardize_subdivision_name("MARATHWADA"), "MATATHWADA")
        self.assertEqual(RecentRainfallAdapter.standardize_subdivision_name("RAYALASEEMA"), "RAYALSEEMA")
        self.assertEqual(RecentRainfallAdapter.standardize_subdivision_name("A & N ISLANDS"), "ANDAMAN & NICOBAR ISLANDS")

    def test_unknown_subdivision_reporting(self):
        self.assertTrue(RecentRainfallAdapter.is_known_subdivision("BIHAR"))
        self.assertTrue(RecentRainfallAdapter.is_known_subdivision("ODISHA"))  # Maps to ORISSA
        self.assertTrue(RecentRainfallAdapter.is_known_subdivision("ALL-INDIA"))
        self.assertFalse(RecentRainfallAdapter.is_known_subdivision("ATLANTIS"))
        self.assertFalse(RecentRainfallAdapter.is_known_subdivision("CENTRAL ASIA"))

        raw_df = pd.DataFrame({
            "SUBDIVISION": ["BIHAR", "UNKNOWN_REGION_XYZ"],
            "YEAR": [2020, 2020],
            "ANNUAL": [1100.0, 500.0],
        })
        val = RecentRainfallAdapter.validate_dataset(raw_df)
        self.assertTrue(any("UNKNOWN_REGION_XYZ" in w for w in val["warnings"]))

        # Check harmonization filters out unknown region
        harmonized = RecentRainfallAdapter.harmonize_dataset(raw_df, filter_unknown_subdivisions=True)
        self.assertIn("BIHAR", harmonized["SUBDIVISION"].values)
        self.assertNotIn("UNKNOWN_REGION_XYZ", harmonized["SUBDIVISION"].values)

    def test_duplicate_records_detection_and_rejection(self):
        # Create dataset with duplicate (SUBDIVISION, YEAR) records
        raw_df = pd.DataFrame({
            "SUBDIVISION": ["BIHAR", "BIHAR"],
            "YEAR": [2020, 2020],
            "ANNUAL": [1100.0, 1300.0],
        })
        val = RecentRainfallAdapter.validate_dataset(raw_df)
        self.assertFalse(val["is_valid"])
        self.assertGreater(val["duplicates_count"], 0)
        self.assertTrue(any("duplicate" in err.lower() for err in val["errors"]))

        # harmonize_dataset with reject_duplicates=True must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            RecentRainfallAdapter.harmonize_dataset(raw_df, reject_duplicates=True)
        self.assertIn("duplicate", str(ctx.exception).lower())

        # get_observation_value must reject duplicates rather than silently picking the first
        with self.assertRaises(ValueError) as ctx:
            RecentRainfallAdapter.get_observation_value(raw_df, "BIHAR", 2020, "Annual", "ANNUAL")
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_missing_values_preservation_and_partial_sums(self):
        # Jan-Feb with FEB missing: Jan-Feb must be NaN, NOT equal to JAN alone
        raw_df = pd.DataFrame({
            "SUBDIVISION": ["KERALA"],
            "YEAR": [2021],
            "JAN": [15.0],
            "FEB": [np.nan],
            "MAR": [30.0],
            "APR": [40.0],
            "MAY": [50.0],
            "JUN": [600.0],
            "JUL": [700.0],
            "AUG": [500.0],
            "SEP": [200.0],
            "OCT": [150.0],
            "NOV": [50.0],
            "DEC": [10.0],
        })
        harmonized = RecentRainfallAdapter.harmonize_dataset(raw_df)
        kerala_row = harmonized.iloc[0]

        # Winter season (Jan-Feb) has 1 missing month -> must be NaN (partial sums rejected)
        self.assertTrue(np.isnan(kerala_row["Jan-Feb"]))
        # Annual has 1 missing month -> must be NaN
        self.assertTrue(np.isnan(kerala_row["ANNUAL"]))
        # Monsoon season (Jun-Sep) has all 4 months present -> must be computed
        self.assertEqual(kerala_row["Jun-Sep"], 2000.0)

    def test_negative_rainfall_detection(self):
        raw_df = pd.DataFrame({
            "SUBDIVISION": ["BIHAR"],
            "YEAR": [2020],
            "ANNUAL": [-50.0],
        })
        val = RecentRainfallAdapter.validate_dataset(raw_df)
        self.assertFalse(val["is_valid"])
        self.assertTrue(any("negative" in err.lower() for err in val["errors"]))


class TestHistoricalRecentComparator(unittest.TestCase):
    """Test suite for HistoricalRecentComparator baseline calculations and departure metrics."""

    @classmethod
    def setUpClass(cls):
        cls.comparator = HistoricalRecentComparator()

    def test_all_india_baseline_selection_and_limitation_note(self):
        s_ai, meta_ai = self.comparator.get_historical_series("ALL-INDIA", "Annual", "ANNUAL")
        self.assertIsInstance(s_ai, pd.Series)
        self.assertEqual(len(s_ai), 115)
        self.assertTrue(meta_ai["is_all_india"])
        self.assertFalse(meta_ai["is_official_area_weighted"])
        self.assertIn("unweighted", meta_ai["baseline_label"].lower())
        self.assertIn("unweighted arithmetic mean", meta_ai["limitation_note"].lower())

        # Compare with individual subdivision
        s_bihar, meta_bihar = self.comparator.get_historical_series("BIHAR", "Annual", "ANNUAL")
        self.assertFalse(meta_bihar["is_all_india"])
        self.assertEqual(meta_bihar["limitation_note"], "")

    def test_compare_all_subdivisions_strictly_36_regions(self):
        from src.data_gov_in_client import DataGovInClient
        client = DataGovInClient()
        recent_df, _ = client.get_recent_rainfall_dataset(force_live=False)
        harmonized = RecentRainfallAdapter.harmonize_dataset(recent_df)

        comp_df = self.comparator.compare_all_subdivisions(
            recent_df=harmonized,
            year=2018,
            period_type="Annual",
            period_value="ANNUAL",
        )

        self.assertIsInstance(comp_df, pd.DataFrame)
        # MUST strictly contain 36 subdivisions - ALL-INDIA MUST NOT BE PRESENT!
        self.assertEqual(len(comp_df), 36)
        self.assertNotIn("ALL-INDIA", comp_df["SUBDIVISION"].values)
        for expected_sub in CANONICAL_SUBDIVISIONS:
            self.assertIn(expected_sub, comp_df["SUBDIVISION"].values)

    def test_get_all_india_comparison_standalone(self):
        from src.data_gov_in_client import DataGovInClient
        client = DataGovInClient()
        recent_df, _ = client.get_recent_rainfall_dataset(force_live=False)
        harmonized = RecentRainfallAdapter.harmonize_dataset(recent_df)

        ai_comp = self.comparator.get_all_india_comparison(
            recent_df=harmonized,
            year=2018,
            period_type="Annual",
            period_value="ANNUAL",
        )
        self.assertIsInstance(ai_comp, dict)
        self.assertTrue(ai_comp["baseline_meta"]["is_all_india"])
        self.assertIn("limitation_note", ai_comp["baseline_meta"])
        self.assertIn("unweighted arithmetic mean", ai_comp["baseline_meta"]["limitation_note"].lower())

    def test_missing_observation_departure_handling(self):
        s = pd.Series([1000.0, 1100.0, 900.0])
        # Test None observation
        dep_none = self.comparator.compute_departure(None, s)
        self.assertTrue(dep_none["is_missing"])
        self.assertTrue(np.isnan(dep_none["recent_value"]))
        self.assertTrue(np.isnan(dep_none["absolute_departure_mm"]))
        self.assertTrue(np.isnan(dep_none["percentage_departure"]))
        self.assertEqual(dep_none["imd_category"]["key"], "MISSING")
        self.assertEqual(dep_none["imd_category"]["label"], "Observation Missing")

        # Test np.nan observation
        dep_nan = self.comparator.compute_departure(np.nan, s)
        self.assertTrue(dep_nan["is_missing"])
        self.assertEqual(dep_nan["imd_category"]["key"], "MISSING")

    def test_classify_imd_departure_categories(self):
        self.assertEqual(classify_imd_departure(75.0)["key"], "LARGE_EXCESS")
        self.assertEqual(classify_imd_departure(35.0)["key"], "EXCESS")
        self.assertEqual(classify_imd_departure(10.0)["key"], "NORMAL")
        self.assertEqual(classify_imd_departure(-15.0)["key"], "NORMAL")
        self.assertEqual(classify_imd_departure(-35.0)["key"], "DEFICIENT")
        self.assertEqual(classify_imd_departure(-65.0)["key"], "LARGE_DEFICIENT")
        self.assertEqual(classify_imd_departure(-100.0, recent_value=0.0)["key"], "NO_RAIN")
        self.assertEqual(classify_imd_departure(None)["key"], "MISSING")
        self.assertEqual(classify_imd_departure(np.nan)["key"], "MISSING")

    def test_markdown_summary_with_missing_and_all_india(self):
        s = pd.Series([1000.0, 1100.0, 900.0])

        # Missing observation
        dep_missing = self.comparator.compute_departure(None, s)
        md_missing = self.comparator.generate_markdown_summary(
            subdivision="BIHAR",
            year=2020,
            period_type="Annual",
            period_value="ANNUAL",
            metrics=dep_missing,
        )
        self.assertIn("Missing", md_missing)
        self.assertIn("No valid rainfall observation was recorded", md_missing)

        # ALL-INDIA with limitation note
        _, meta_ai = self.comparator.get_historical_series("ALL-INDIA", "Annual", "ANNUAL")
        dep_ai = self.comparator.compute_departure(1150.0, s, baseline_meta=meta_ai)
        md_ai = self.comparator.generate_markdown_summary(
            subdivision="ALL-INDIA",
            year=2020,
            period_type="Annual",
            period_value="ANNUAL",
            metrics=dep_ai,
        )
        self.assertIn("National Aggregate Baseline Limitation", md_ai)
        self.assertIn("unweighted arithmetic mean", md_ai)


if __name__ == "__main__":
    unittest.main()
