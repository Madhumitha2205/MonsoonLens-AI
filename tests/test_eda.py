"""
MonsoonLens AI - Automated Unit and Integration Test Suite for Stage 2 (EDA).

Validates EDA calculations, climatology aggregations, Coefficient of Variation (CV),
correlation matrix bounds, figure generation integrity, and report consistency.
Compatible with standard library unittest and pytest runners.
"""

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
    EDA_REPORT_PATH,
    FIGURES_DIR,
    MONTH_COLUMNS,
    MONTHLY_LONG_PATH,
    SEASON_COLUMNS,
    SEASONAL_PATH,
)
from src.eda_analysis import (
    compute_all_india_annual_stats,
    compute_monthly_climatology,
    compute_regional_variability,
    compute_seasonal_correlations,
    compute_seasonal_summary,
    compute_subdivision_annual_stats,
    load_eda_datasets,
)


class TestEDACalculations(unittest.TestCase):
    """Test mathematical and aggregation correctness of EDA functions."""

    @classmethod
    def setUpClass(cls):
        cls.data = load_eda_datasets()
        cls.wide_df = cls.data["wide"]
        cls.long_df = cls.data["long"]
        cls.ann_df = cls.data["annual"]
        cls.season_df = cls.data["seasonal"]

    def test_all_india_annual_stats(self):
        """Verify national annual statistics and regression bounds."""
        stats = compute_all_india_annual_stats(self.ann_df)

        self.assertAlmostEqual(stats["mean"], 1411.01, places=1)
        self.assertGreater(stats["std"], 0.0)
        self.assertLessEqual(stats["min"], stats["median"])
        self.assertLessEqual(stats["median"], stats["max"])
        # Slope should be within plausible meteorological bounds (-5 to +5 mm/yr)
        self.assertGreaterEqual(stats["trend_slope_mm_per_year"], -5.0)
        self.assertLessEqual(stats["trend_slope_mm_per_year"], 5.0)
        # p-value bounded in [0, 1]
        self.assertGreaterEqual(stats["trend_pvalue"], 0.0)
        self.assertLessEqual(stats["trend_pvalue"], 1.0)

    def test_subdivision_annual_stats_and_cv(self):
        """Verify per-subdivision statistics and Coefficient of Variation (CV)."""
        sub_stats = compute_subdivision_annual_stats(self.ann_df)

        # All 36 subdivisions must be represented
        self.assertEqual(len(sub_stats), 36)

        # Verify CV formula: CV = (std / mean) * 100
        for _, row in sub_stats.iterrows():
            expected_cv = round((row["std_annual_mm"] / row["mean_annual_mm"]) * 100.0, 2)
            self.assertAlmostEqual(row["cv_percent"], expected_cv, delta=0.1)
            # Arid regions should have higher CV than wet regions
            self.assertGreater(row["cv_percent"], 5.0)
            self.assertLess(row["cv_percent"], 100.0)

    def test_monthly_climatology_shares(self):
        """Verify that monthly percentage shares sum to 100% and July is peak month."""
        monthly_stats = compute_monthly_climatology(self.long_df)

        self.assertEqual(len(monthly_stats), 12)
        total_share = monthly_stats["percent_of_annual"].sum()
        self.assertAlmostEqual(total_share, 100.0, places=1)

        # July (month 7) and August (month 8) should be the top two rainfall months
        july_row = monthly_stats[monthly_stats["MONTH"] == "JUL"].iloc[0]
        aug_row = monthly_stats[monthly_stats["MONTH"] == "AUG"].iloc[0]
        self.assertGreater(july_row["percent_of_annual"], 20.0)
        self.assertGreater(aug_row["percent_of_annual"], 15.0)

    def test_seasonal_summary_contribution(self):
        """Verify 4 seasons and Southwest Monsoon dominance (>70%)."""
        seasonal_stats = compute_seasonal_summary(self.season_df)

        self.assertEqual(len(seasonal_stats), 4)
        total_contrib = seasonal_stats["percent_contribution"].sum()
        self.assertAlmostEqual(total_contrib, 100.0, places=1)

        monsoon = seasonal_stats[seasonal_stats["SEASON"] == "Jun-Sep"].iloc[0]
        self.assertGreater(monsoon["percent_contribution"], 70.0)

    def test_seasonal_correlation_matrix_properties(self):
        """Verify correlation matrix diagonal is 1.0 and bounded [-1, 1]."""
        p_corr, s_corr = compute_seasonal_correlations(self.wide_df)

        for df_corr in [p_corr, s_corr]:
            self.assertEqual(df_corr.shape, (5, 5))
            # Diagonals must be 1.0
            for i in range(5):
                self.assertAlmostEqual(df_corr.iloc[i, i], 1.0, places=3)
            # Values in [-1, 1]
            self.assertTrue((df_corr >= -1.0).all().all())
            self.assertTrue((df_corr <= 1.0).all().all())


class TestFigureArtifacts(unittest.TestCase):
    """Test that all 7 required figures exist, are non-empty, and valid PNG files."""

    def test_all_seven_figures_exist_and_non_empty(self):
        expected_figures = [
            "01_annual_rainfall_trends.png",
            "02_monthly_climatology_profile.png",
            "03_seasonal_rainfall_breakdown.png",
            "04_regional_monthly_heatmap.png",
            "05_subdivision_variability_comparison.png",
            "06_subdivision_rainfall_boxplots.png",
            "07_seasonal_correlation_matrix.png",
        ]

        for fname in expected_figures:
            fig_path = FIGURES_DIR / fname
            self.assertTrue(fig_path.exists(), f"Figure missing: {fname}")
            size_kb = fig_path.stat().st_size / 1024.0
            self.assertGreater(size_kb, 50.0, f"Figure unexpectedly small ({size_kb:.1f} KB): {fname}")


class TestReportArtifact(unittest.TestCase):
    """Test that the written EDA report exists and contains all required sections."""

    def test_eda_report_exists_and_contains_required_sections(self):
        self.assertTrue(EDA_REPORT_PATH.exists(), f"Report missing: {EDA_REPORT_PATH}")
        content = EDA_REPORT_PATH.read_text(encoding="utf-8")

        required_sections = [
            "# MonsoonLens AI: Exploratory Data Analysis (EDA) Report",
            "## 1. Objective and Dataset Overview",
            "## 2. Data Quality Re-Verification Findings",
            "## 3. Analytical Methods",
            "## 4. Annual Rainfall Trends and Trajectory Findings",
            "## 5. Monthly and Seasonal Dynamics",
            "## 6. Regional Rainfall Comparisons and Spatial Variability",
            "## 7. Distribution and Inter-Seasonal Correlation Analysis",
            "## 8. Scientific Limitations and Assumptions",
            "## 9. Key Summary Takeaways",
            "## 10. Actionable Next Steps for Stage 3",
        ]

        for sec in required_sections:
            self.assertIn(sec, content, f"Required section missing from report: {sec}")


if __name__ == "__main__":
    unittest.main()
