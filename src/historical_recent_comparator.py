"""
MonsoonLens AI - Historical vs Recent Rainfall Comparison & Analytics Engine.

Provides statistical baseline computation (1901-2015 historical normal),
departure metrics (absolute mm, percentage %, standardized z-score),
historical empirical percentile ranking, and IMD categorical classifications:
- Large Excess: >= +60%
- Excess: +20% to +59%
- Normal: -19% to +19%
- Deficient: -20% to -59%
- Large Deficient: <= -60%
- No Rain: -100% (when observed rainfall is 0.0 mm and normal > 0)
- Observation Missing: when recent observation is NaN

Enforces strict scientific integrity:
- Separates national aggregate comparisons from the 36-subdivision ranking.
- Explicitly labels unweighted national subdivision means with limitation notices.
- Robustly handles missing observations without zero-imputation or formatting crashes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.config import IMD_CATEGORIES
from src.data_loader import (
    load_annual_data,
    load_monthly_data,
    load_seasonal_data,
)
from src.recent_rainfall_adapter import (
    CANONICAL_SUBDIVISIONS,
    RecentRainfallAdapter,
)

logger = logging.getLogger(__name__)


def classify_imd_departure(pct_departure: Optional[float], recent_value: Optional[float] = None) -> Dict[str, Any]:
    """
    Classify percentage departure according to India Meteorological Department standards.

    Parameters
    ----------
    pct_departure : Optional[float]
        Percentage departure relative to climatological baseline.
    recent_value : Optional[float]
        Actual observed rainfall value in mm.

    Returns
    -------
    Dict[str, Any]
        Dictionary with category key, human-readable label, and visual color code.
    """
    if pct_departure is None or pd.isna(pct_departure) or (recent_value is not None and pd.isna(recent_value)):
        return {
            "key": "MISSING",
            "label": "Observation Missing",
            "color": "#94a3b8",
            "description": "No valid observation recorded (NaN)",
        }

    # Special case: completely dry where normal rainfall occurs
    if recent_value is not None and recent_value == 0.0 and pct_departure <= -99.9:
        return {
            "key": "NO_RAIN",
            "label": "No Rain",
            "color": IMD_CATEGORIES["NO_RAIN"]["color"],
            "description": "Zero rainfall recorded (100% deficiency)",
        }

    if pct_departure >= 60.0:
        return {
            "key": "LARGE_EXCESS",
            "label": "Large Excess",
            "color": IMD_CATEGORIES["LARGE_EXCESS"]["color"],
            "description": "Rainfall >= 60% above normal",
        }
    elif pct_departure >= 20.0:
        return {
            "key": "EXCESS",
            "label": "Excess",
            "color": IMD_CATEGORIES["EXCESS"]["color"],
            "description": "Rainfall 20% to 59% above normal",
        }
    elif pct_departure >= -19.99:
        return {
            "key": "NORMAL",
            "label": "Normal",
            "color": IMD_CATEGORIES["NORMAL"]["color"],
            "description": "Rainfall within +/- 19% of normal",
        }
    elif pct_departure >= -59.99:
        return {
            "key": "DEFICIENT",
            "label": "Deficient",
            "color": IMD_CATEGORIES["DEFICIENT"]["color"],
            "description": "Rainfall 20% to 59% below normal",
        }
    else:
        return {
            "key": "LARGE_DEFICIENT",
            "label": "Large Deficient",
            "color": IMD_CATEGORIES["LARGE_DEFICIENT"]["color"],
            "description": "Rainfall >= 60% below normal",
        }


class HistoricalRecentComparator:
    """Computes historical baselines and evaluates recent rainfall departures."""

    def __init__(
        self,
        annual_df: Optional[pd.DataFrame] = None,
        monthly_df: Optional[pd.DataFrame] = None,
        seasonal_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize comparator with 1901-2015 historical processed datasets.
        Loads from disk if not explicitly passed.
        """
        self.annual_df = annual_df if annual_df is not None else load_annual_data()
        self.monthly_df = monthly_df if monthly_df is not None else load_monthly_data()
        self.seasonal_df = seasonal_df if seasonal_df is not None else load_seasonal_data()

    def get_historical_series(
        self,
        subdivision: str,
        period_type: str,
        period_value: str,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Extract historical 1901-2015 time series for a given subdivision and period.

        Parameters
        ----------
        subdivision : str
            Canonical subdivision name or 'ALL-INDIA'.
        period_type : str
            'Annual', 'Seasonal', or 'Monthly'.
        period_value : str
            'ANNUAL', season name (e.g., 'Jun-Sep'), or month name (e.g., 'JUL').

        Returns
        -------
        Tuple[pd.Series, Dict[str, Any]]
            Series of historical rainfall values indexed by YEAR, and metadata describing baseline construction.
        """
        ptype = period_type.strip().lower()
        std_sub = RecentRainfallAdapter.standardize_subdivision_name(subdivision)
        is_all_india = (std_sub == "ALL-INDIA")

        meta: Dict[str, Any] = {
            "subdivision": std_sub,
            "period_type": period_type,
            "period_value": period_value,
            "is_all_india": is_all_india,
            "is_official_area_weighted": False,
            "baseline_label": "Official Subdivisional Series (1901-2015)" if not is_all_india else "Unweighted Mean of 36 Subdivisions (1901-2015)",
            "limitation_note": (
                "Calculated as the unweighted arithmetic mean across 36 subdivisions. "
                "Because subdivisional land areas differ significantly in geographic extent, "
                "this figure should not be treated as the official IMD area-weighted national total."
                if is_all_india else ""
            ),
        }

        if ptype == "annual":
            if is_all_india:
                grouped = (
                    self.annual_df.groupby("YEAR")["ANNUAL_VALIDATED"]
                    .mean()
                    .dropna()
                )
                return grouped, meta
            else:
                sub_df = self.annual_df[self.annual_df["SUBDIVISION"] == std_sub]
                s = sub_df.set_index("YEAR")["ANNUAL_VALIDATED"].dropna()
                return s, meta

        elif ptype in ["seasonal", "season"]:
            if is_all_india:
                s_df = self.seasonal_df[self.seasonal_df["SEASON"] == period_value]
                grouped = (
                    s_df.groupby("YEAR")["RAINFALL_VALIDATED"]
                    .mean()
                    .dropna()
                )
                return grouped, meta
            else:
                s_df = self.seasonal_df[
                    (self.seasonal_df["SUBDIVISION"] == std_sub)
                    & (self.seasonal_df["SEASON"] == period_value)
                ]
                s = s_df.set_index("YEAR")["RAINFALL_VALIDATED"].dropna()
                return s, meta

        elif ptype in ["monthly", "month"]:
            m_val = period_value.upper()
            if is_all_india:
                m_df = self.monthly_df[self.monthly_df["MONTH"] == m_val]
                grouped = (
                    m_df.groupby("YEAR")["RAINFALL"]
                    .mean()
                    .dropna()
                )
                return grouped, meta
            else:
                m_df = self.monthly_df[
                    (self.monthly_df["SUBDIVISION"] == std_sub)
                    & (self.monthly_df["MONTH"] == m_val)
                ]
                s = m_df.set_index("YEAR")["RAINFALL"].dropna()
                return s, meta

        else:
            raise ValueError(f"Unsupported period_type: {period_type}")

    def compute_baseline_statistics(self, historical_series: pd.Series) -> Dict[str, float]:
        """
        Compute parametric and non-parametric summary statistics on the historical baseline.

        Returns
        -------
        Dict[str, float]
            Dictionary containing mean, median, std, iqr, q25, q75, p10, p90, min, max, count.
        """
        s = historical_series.dropna()
        if len(s) == 0:
            return {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "iqr": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "p10": np.nan,
                "p90": np.nan,
                "min": np.nan,
                "max": np.nan,
                "count": 0,
            }

        mean_val = float(s.mean())
        median_val = float(s.median())
        std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        p10 = float(s.quantile(0.10))
        p90 = float(s.quantile(0.90))

        return {
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "std": round(std_val, 2),
            "iqr": round(q75 - q25, 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "p10": round(p10, 2),
            "p90": round(p90, 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "count": int(len(s)),
        }

    def compute_departure(
        self,
        recent_value: Optional[float],
        historical_series: pd.Series,
        baseline_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute departure metrics comparing recent value against the historical series.

        Parameters
        ----------
        recent_value : Optional[float]
            Recent observed rainfall in mm (may be NaN or None).
        historical_series : pd.Series
            Historical rainfall observations (1901-2015).
        baseline_meta : Optional[Dict[str, Any]]
            Metadata describing baseline construction.

        Returns
        -------
        Dict[str, Any]
            Detailed departure metrics, z-score, percentile rank, and IMD classification.
        """
        baseline = self.compute_baseline_statistics(historical_series)
        mean_hist = baseline["mean"]
        std_hist = baseline["std"]

        if recent_value is None or pd.isna(recent_value) or pd.isna(mean_hist):
            return {
                "recent_value": np.nan,
                "baseline_mean": mean_hist,
                "baseline_median": baseline["median"],
                "baseline_std": std_hist,
                "absolute_departure_mm": np.nan,
                "percentage_departure": np.nan,
                "standardized_z_score": np.nan,
                "percentile_rank": np.nan,
                "imd_category": classify_imd_departure(None, None),
                "baseline_stats": baseline,
                "is_missing": True,
                "baseline_meta": baseline_meta or {},
            }

        # Absolute departure: Delta = R_recent - mu_hist
        abs_dep = round(float(recent_value - mean_hist), 2)

        # Percentage departure: Delta_% = (Delta / mu_hist) * 100
        if abs(mean_hist) > 1e-4:
            pct_dep = round(float((abs_dep / mean_hist) * 100.0), 2)
        else:
            pct_dep = 0.0 if abs_dep == 0.0 else (100.0 if abs_dep > 0 else -100.0)

        # Standardized departure (z-score): z = (R_recent - mu_hist) / sigma_hist
        if std_hist > 1e-4:
            z_score = round(float((recent_value - mean_hist) / std_hist), 2)
        else:
            z_score = 0.0

        # Empirical percentile rank (0 to 100)
        valid_hist = historical_series.dropna()
        if len(valid_hist) > 0:
            pct_rank = round(float((valid_hist < recent_value).mean() * 100.0), 1)
        else:
            pct_rank = np.nan

        imd_cat = classify_imd_departure(pct_dep, recent_value)

        return {
            "recent_value": round(float(recent_value), 2),
            "baseline_mean": mean_hist,
            "baseline_median": baseline["median"],
            "baseline_std": std_hist,
            "absolute_departure_mm": abs_dep,
            "percentage_departure": pct_dep,
            "standardized_z_score": z_score,
            "percentile_rank": pct_rank,
            "imd_category": imd_cat,
            "baseline_stats": baseline,
            "is_missing": False,
            "baseline_meta": baseline_meta or {},
        }

    def compare_all_subdivisions(
        self,
        recent_df: pd.DataFrame,
        year: int,
        period_type: str,
        period_value: str,
    ) -> pd.DataFrame:
        """
        Generate comparison table strictly across the 36 canonical IMD subdivisions.
        Excludes ALL-INDIA aggregate row from ranking and distributions.

        Parameters
        ----------
        recent_df : pd.DataFrame
            Standardized recent observations.
        year : int
            Recent observation year.
        period_type : str
            'Annual', 'Seasonal', or 'Monthly'.
        period_value : str
            'ANNUAL', season name, or month name.

        Returns
        -------
        pd.DataFrame
            Table with 36 rows covering all canonical subdivisions.
        """
        rows = []
        for sub in CANONICAL_SUBDIVISIONS:
            recent_val = RecentRainfallAdapter.get_observation_value(
                recent_df,
                subdivision=sub,
                year=year,
                period_type=period_type,
                period_value=period_value,
            )

            try:
                hist_series, meta = self.get_historical_series(
                    subdivision=sub,
                    period_type=period_type,
                    period_value=period_value,
                )
                dep = self.compute_departure(recent_val, hist_series, meta)
            except Exception as e:
                logger.warning(f"Error comparing {sub}: {e}")
                continue

            rows.append({
                "SUBDIVISION": sub,
                "RECENT_RAINFALL": dep["recent_value"],
                "BASELINE_MEAN": dep["baseline_mean"],
                "ABSOLUTE_DEPARTURE_MM": dep["absolute_departure_mm"],
                "PERCENTAGE_DEPARTURE": dep["percentage_departure"],
                "Z_SCORE": dep["standardized_z_score"],
                "PERCENTILE_RANK": dep["percentile_rank"],
                "IMD_CATEGORY": dep["imd_category"]["key"],
                "CATEGORY_LABEL": dep["imd_category"]["label"],
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="PERCENTAGE_DEPARTURE", ascending=False).reset_index(drop=True)
        return df

    def get_all_india_comparison(
        self,
        recent_df: pd.DataFrame,
        year: int,
        period_type: str,
        period_value: str,
    ) -> Dict[str, Any]:
        """
        Compute standalone ALL-INDIA comparison separate from the 36-subdivision ranking.
        """
        recent_val = RecentRainfallAdapter.get_observation_value(
            recent_df,
            subdivision="ALL-INDIA",
            year=year,
            period_type=period_type,
            period_value=period_value,
        )
        hist_series, meta = self.get_historical_series(
            subdivision="ALL-INDIA",
            period_type=period_type,
            period_value=period_value,
        )
        return self.compute_departure(recent_val, hist_series, meta)

    @staticmethod
    def generate_comparison_csv(comparison_df: pd.DataFrame) -> str:
        """Convert comparison DataFrame into formatted CSV string for export."""
        return comparison_df.to_csv(index=False)

    @staticmethod
    def generate_markdown_summary(
        subdivision: str,
        year: int,
        period_type: str,
        period_value: str,
        metrics: Dict[str, Any],
        comparison_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate a structured, non-causal markdown summary report.
        Handles missing observations safely and restricts national breakdown to the 36 subdivisions.
        """
        cat = metrics.get("imd_category", {})
        cat_label = cat.get("label", "Unknown")
        cat_desc = cat.get("description", "")
        meta = metrics.get("baseline_meta", {})
        limitation_note = meta.get("limitation_note", "")

        is_missing = metrics.get("is_missing", False) or pd.isna(metrics.get("recent_value"))

        recent_val_str = f"{metrics['recent_value']:.1f} mm" if not is_missing else "Missing (NaN)"
        base_mean_str = f"{metrics['baseline_mean']:.1f} mm" if pd.notna(metrics.get("baseline_mean")) else "N/A"
        base_std_str = f"{metrics['baseline_std']:.1f} mm" if pd.notna(metrics.get("baseline_std")) else "N/A"
        base_med_str = f"{metrics['baseline_median']:.1f} mm" if pd.notna(metrics.get("baseline_median")) else "N/A"

        abs_dep = metrics.get("absolute_departure_mm")
        pct_dep = metrics.get("percentage_departure")
        z_score = metrics.get("standardized_z_score")
        pct_rank = metrics.get("percentile_rank")

        if not is_missing and pd.notna(abs_dep) and pd.notna(pct_dep):
            dep_sign = "+" if abs_dep >= 0 else ""
            pct_sign = "+" if pct_dep >= 0 else ""
            dep_str = f"{dep_sign}{abs_dep:.1f} mm ({pct_sign}{pct_dep:.1f}%)"
            z_str = f"{z_score:+.2f} \\sigma" if pd.notna(z_score) else "N/A"
            rank_str = f"{pct_rank:.1f}th Percentile" if pd.notna(pct_rank) else "N/A"
        else:
            dep_str = "N/A (Missing observation)"
            z_str = "N/A"
            rank_str = "N/A"

        baseline_type_label = meta.get("baseline_label", "115-Year Baseline (1901-2015)")

        md = f"""# Historical vs Recent Rainfall Monitor Report

**Target Region:** {subdivision}  
**Observation Period:** {period_type} - {period_value} ({year})  
**Baseline Dataset:** {baseline_type_label}  
**Provenance Status:** Local cached dataset - provenance not independently verified  

---

## 1. Climatological Metrics

| Metric | Observed Value | Climatological Baseline | Departure (\\Delta) | IMD Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Rainfall** | **{recent_val_str}** | **{base_mean_str}** | **{dep_str}** | **{cat_label}** |
| **Standardized $z$-Score** | {z_str} | 0.00 \\sigma | Baseline Std: {base_std_str} | - |
| **Historical Percentile** | {rank_str} | 50.0th Percentile | Median: {base_med_str} | - |

"""
        if limitation_note:
            md += f"""> [!NOTE]
> **National Aggregate Baseline Limitation:**  
> {limitation_note}

"""

        if is_missing:
            md += f"""## 2. Statistical Interpretation
No valid rainfall observation was recorded for **{subdivision}** during **{period_value} of {year}** in the available dataset. The value is preserved as missing (`NaN`). No departure or IMD classification can be computed.
"""
        else:
            md += f"""## 2. Statistical Interpretation
During the {period_type.lower()} period **{period_value} of {year}**, **{subdivision}** received **{recent_val_str}** of precipitation.
Compared to the baseline normal of **{base_mean_str}**, this represents an absolute departure of **{dep_str}**.

According to India Meteorological Department (IMD) meteorological classification standards:
- **Category:** **{cat_label}** ({cat_desc})
- **Standardized Anomaly:** {z_str} standard deviations from the historical mean.
- **Percentile Ranking:** Greater than **{rank_str}** of all recorded baseline years.
"""

        if comparison_df is not None and not comparison_df.empty:
            # Strictly filter to canonical 36 subdivisions
            clean_sub_df = comparison_df[comparison_df["SUBDIVISION"] != "ALL-INDIA"]
            total_valid_subs = len(clean_sub_df)
            cat_counts = clean_sub_df["CATEGORY_LABEL"].value_counts().to_dict()

            md += f"\n## 3. Subdivisional Distribution Summary (Across 36 Meteorological Subdivisions)\n\n"
            md += f"Breakdown across the {total_valid_subs} canonical IMD subdivisions for this period:\n\n"
            for label, count in cat_counts.items():
                pct = (count / total_valid_subs) * 100.0 if total_valid_subs > 0 else 0.0
                md += f"- **{label}:** {count} subdivisions ({pct:.1f}%)\n"

        md += """
---

## 4. Scientific Disclaimer & Climatological Scope
> [!NOTE]
> - **Statistical Departure vs Surface Hydrology:** Rainfall departures reflect atmospheric precipitation deviations from long-term climatological means. They do **not** model surface flood inundation, river runoff, or soil moisture drought. Flooding and drought depend heavily on topography, soil saturation, drainage infrastructure, and storm intensity.
> - **Provenance Notice:** The local cached dataset provenance has not been independently verified against official live IMD registry feeds.
"""
        return md
