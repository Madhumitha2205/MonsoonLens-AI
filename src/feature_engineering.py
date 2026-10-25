"""
MonsoonLens AI - Feature Engineering and Time-Series Preparation Module.

Generates scientifically defensible time-series, climatological, and anomaly features
without modifying or corrupting primary cleaned datasets.
Enforces strict chronological sorting, group-isolated subdivision computations,
continuity validation (preventing lag calculation across missing calendar years),
backward-looking rolling windows (preventing future lookahead leakage), and
configurable climatological baselines.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_BASELINE_YEARS,
    MIN_OBSERVATION_COUNT,
    ROLLING_WINDOWS,
    SEASON_COLUMNS,
)


def validate_continuity_and_sort(
    df: pd.DataFrame,
    group_col: str = "SUBDIVISION",
    year_col: str = "YEAR"
) -> pd.DataFrame:
    """
    Ensure the dataset is strictly sorted by group and year.
    Adds a boolean indicator `IS_CONSECUTIVE_YEAR` marking whether the current record
    is exactly 1 calendar year after the preceding record for the same subdivision.
    """
    sorted_df = df.sort_values(by=[group_col, year_col]).copy()
    # Compute year difference within each subdivision
    year_diff = sorted_df.groupby(group_col)[year_col].diff()
    sorted_df["YEAR_DIFF_FROM_PREV"] = year_diff
    # A year is strictly consecutive only if year_diff == 1
    sorted_df["IS_CONSECUTIVE_YEAR"] = year_diff == 1
    return sorted_df


def add_annual_time_series_features(
    annual_df: pd.DataFrame,
    value_col: str = "ANNUAL_VALIDATED",
    rolling_windows: List[int] = ROLLING_WINDOWS,
    baseline_years: Optional[Tuple[int, int]] = DEFAULT_BASELINE_YEARS,
) -> pd.DataFrame:
    """
    Compute time-series features on annual rainfall series.
    Features:
    - Lag-1 previous year rainfall (only if strictly consecutive calendar year)
    - 1-year absolute difference (only if strictly consecutive)
    - 1-year percentage change (only if previous year > 0)
    - Backward-looking rolling means (5, 10, 20 years)
    - Backward-looking rolling standard deviations (5, 10, 20 years)
    - Climatological baseline mean and standard deviation
    - Absolute anomaly (rainfall - baseline_mean)
    - Standardized anomaly ((rainfall - baseline_mean) / baseline_std)
    """
    df = validate_continuity_and_sort(annual_df, group_col="SUBDIVISION", year_col="YEAR")
    feature_df = df.copy()

    # 1. Lag features with continuity enforcement
    prev_val = feature_df.groupby("SUBDIVISION")[value_col].shift(1)
    # If not consecutive calendar year, set lag to NaN to prevent false lag across time gaps
    consecutive_mask = feature_df["IS_CONSECUTIVE_YEAR"] == True
    feature_df["PREV_YEAR_RAINFALL"] = np.where(consecutive_mask, prev_val, np.nan)

    # 2. 1-year difference
    feature_df["ONE_YEAR_DIFF"] = feature_df[value_col] - feature_df["PREV_YEAR_RAINFALL"]

    # 3. 1-year percentage change (only valid when previous year is positive)
    valid_pct_mask = consecutive_mask & (feature_df["PREV_YEAR_RAINFALL"] > 0)
    feature_df["PERCENT_CHANGE_PREV_YEAR"] = np.where(
        valid_pct_mask,
        ((feature_df[value_col] - feature_df["PREV_YEAR_RAINFALL"]) / feature_df["PREV_YEAR_RAINFALL"]) * 100.0,
        np.nan
    ).round(2)

    # 4. Backward-looking rolling statistics (strictly isolated per subdivision)
    # Using closed='left' or default right-edge backward window (includes current year, no future data)
    for window in rolling_windows:
        min_p = max(2, window // 2)  # Require at least 50% valid years in window
        # Rolling mean
        roll_mean = (
            feature_df.groupby("SUBDIVISION")[value_col]
            .rolling(window=window, min_periods=min_p)
            .mean()
            .reset_index(level=0, drop=True)
            .round(2)
        )
        feature_df[f"ROLLING_MEAN_{window}Y"] = roll_mean

        # Rolling std
        roll_std = (
            feature_df.groupby("SUBDIVISION")[value_col]
            .rolling(window=window, min_periods=min_p)
            .std()
            .reset_index(level=0, drop=True)
            .round(2)
        )
        feature_df[f"ROLLING_STD_{window}Y"] = roll_std

    # 5. Climatological Baseline Anomalies
    # Compute baseline reference distribution per subdivision
    if baseline_years:
        b_start, b_end = baseline_years
        baseline_slice = feature_df[
            (feature_df["YEAR"] >= b_start) & (feature_df["YEAR"] <= b_end)
        ]
    else:
        baseline_slice = feature_df

    baseline_stats = (
        baseline_slice.groupby("SUBDIVISION")[value_col]
        .agg(baseline_mean="mean", baseline_std="std", baseline_count="count")
        .reset_index()
    )

    feature_df = feature_df.merge(baseline_stats, on="SUBDIVISION", how="left")

    # Absolute anomaly: rainfall - baseline_mean
    feature_df["ANNUAL_ANOMALY"] = (feature_df[value_col] - feature_df["baseline_mean"]).round(2)

    # Standardized anomaly (Z-score relative to baseline): anomaly / baseline_std
    # Guard against zero or null standard deviation
    std_safe = np.where(feature_df["baseline_std"] > 1e-4, feature_df["baseline_std"], np.nan)
    feature_df["ANNUAL_STANDARDIZED_ANOMALY"] = (
        (feature_df[value_col] - feature_df["baseline_mean"]) / std_safe
    ).round(3)

    return feature_df


def add_seasonal_proportions_and_anomalies(
    seasonal_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    baseline_years: Optional[Tuple[int, int]] = DEFAULT_BASELINE_YEARS,
) -> pd.DataFrame:
    """
    Augment seasonal records with:
    - Proportion of annual rainfall (SEASON_RAINFALL / ANNUAL_TOTAL)
    - Seasonal anomalies relative to subdivision-season climatological baseline
    """
    # Merge annual total for proportion calculation
    ann_lookup = annual_df[["SUBDIVISION", "YEAR", "ANNUAL_VALIDATED"]].copy()
    merged = seasonal_df.merge(ann_lookup, on=["SUBDIVISION", "YEAR"], how="left")

    # Seasonal proportion of annual rainfall (only when annual > 0 and valid)
    valid_ann = merged["ANNUAL_VALIDATED"] > 0
    merged["SEASONAL_PROPORTION_OF_ANNUAL"] = np.where(
        valid_ann & merged["RAINFALL_VALIDATED"].notnull(),
        (merged["RAINFALL_VALIDATED"] / merged["ANNUAL_VALIDATED"]).round(4),
        np.nan
    )

    # Climatological baseline by subdivision and season
    if baseline_years:
        b_start, b_end = baseline_years
        b_slice = merged[(merged["YEAR"] >= b_start) & (merged["YEAR"] <= b_end)]
    else:
        b_slice = merged

    season_baseline = (
        b_slice.groupby(["SUBDIVISION", "SEASON"])["RAINFALL_VALIDATED"]
        .agg(season_baseline_mean="mean", season_baseline_std="std")
        .reset_index()
    )

    merged = merged.merge(season_baseline, on=["SUBDIVISION", "SEASON"], how="left")
    merged["SEASONAL_ANOMALY"] = (
        merged["RAINFALL_VALIDATED"] - merged["season_baseline_mean"]
    ).round(2)

    std_safe = np.where(merged["season_baseline_std"] > 1e-4, merged["season_baseline_std"], np.nan)
    merged["SEASONAL_STANDARDIZED_ANOMALY"] = (
        (merged["RAINFALL_VALIDATED"] - merged["season_baseline_mean"]) / std_safe
    ).round(3)

    return merged


def add_monthly_anomalies_and_climatology(
    monthly_long_df: pd.DataFrame,
    baseline_years: Optional[Tuple[int, int]] = DEFAULT_BASELINE_YEARS,
) -> pd.DataFrame:
    """
    Augment monthly records with monthly climatology:
    - Calendar month baseline mean and std per subdivision
    - Monthly anomaly (RAINFALL - monthly_baseline_mean)
    - Monthly standardized anomaly
    """
    df = monthly_long_df.copy()

    if baseline_years:
        b_start, b_end = baseline_years
        b_slice = df[(df["YEAR"] >= b_start) & (df["YEAR"] <= b_end)]
    else:
        b_slice = df

    monthly_baseline = (
        b_slice.groupby(["SUBDIVISION", "MONTH_NUMBER"])["RAINFALL"]
        .agg(monthly_baseline_mean="mean", monthly_baseline_std="std")
        .reset_index()
    )

    df = df.merge(monthly_baseline, on=["SUBDIVISION", "MONTH_NUMBER"], how="left")
    df["MONTHLY_ANOMALY"] = (df["RAINFALL"] - df["monthly_baseline_mean"]).round(2)

    std_safe = np.where(df["monthly_baseline_std"] > 1e-4, df["monthly_baseline_std"], np.nan)
    df["MONTHLY_STANDARDIZED_ANOMALY"] = (
        (df["RAINFALL"] - df["monthly_baseline_mean"]) / std_safe
    ).round(3)

    return df


def compute_median_absolute_deviation(series: pd.Series) -> Tuple[float, float]:
    """
    Compute robust center (median) and spread (MAD) of a series.
    MAD = median(|X - median(X)|) * 1.4826 (for normal distribution consistency)
    """
    clean_series = series.dropna()
    if clean_series.empty:
        return np.nan, np.nan
    med = float(clean_series.median())
    mad = float(np.median(np.abs(clean_series - med))) * 1.4826
    return med, mad


def prepare_changepoint_series(
    annual_df: pd.DataFrame,
    subdivision: str,
    value_col: str = "ANNUAL_VALIDATED",
    min_observations: int = MIN_OBSERVATION_COUNT,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract and validate an annual rainfall time-series for a specific subdivision,
    formatted specifically for change-point algorithms (e.g. Pettitt test, CUSUM, ruptures).
    Verifies time-series length, reports missing years, and preserves chronological order.
    """
    sub_df = annual_df[annual_df["SUBDIVISION"] == subdivision].sort_values("YEAR").copy()
    valid_obs_count = int(sub_df[value_col].notnull().sum())

    all_years = set(range(int(sub_df["YEAR"].min()), int(sub_df["YEAR"].max()) + 1))
    sub_years = set(sub_df["YEAR"].astype(int))
    missing_calendar_years = sorted(list(all_years - sub_years))

    is_sufficient = valid_obs_count >= min_observations

    status = {
        "subdivision": subdivision,
        "total_records": len(sub_df),
        "valid_observations": valid_obs_count,
        "is_sufficient_length": is_sufficient,
        "min_required_observations": min_observations,
        "year_min": int(sub_df["YEAR"].min()),
        "year_max": int(sub_df["YEAR"].max()),
        "missing_calendar_years": missing_calendar_years,
        "missing_years_count": len(missing_calendar_years),
    }

    return sub_df, status


def compute_decadal_summary(annual_df: pd.DataFrame, value_col: str = "ANNUAL_VALIDATED") -> pd.DataFrame:
    """
    Compute decadal rainfall statistics with explicit decade boundaries:
    1901-1910, 1911-1920, ..., 2001-2010, 2011-2015.
    """
    df = annual_df.copy()

    # Assign explicit decade labels
    def get_decade(year: int) -> str:
        decade_start = (year // 10) * 10 + 1
        decade_end = min(decade_start + 9, 2015)
        return f"{decade_start}-{decade_end}"

    df["DECADE"] = df["YEAR"].apply(get_decade)

    summary = (
        df.groupby(["SUBDIVISION", "DECADE"])[value_col]
        .agg(
            record_count="count",
            mean_rainfall="mean",
            median_rainfall="median",
            std_rainfall="std",
            min_rainfall="min",
            max_rainfall="max",
        )
        .round(2)
        .reset_index()
    )
    return summary


class FeatureMatrixResult(pd.DataFrame):
    """DataFrame that safely yields [self, feature_cols] when unpacked as a 2-tuple."""
    _metadata = ["_feature_cols"]

    @property
    def _constructor(self):
        return FeatureMatrixResult

    def __iter__(self):
        if hasattr(self, "_feature_cols") and self._feature_cols is not None:
            return iter([pd.DataFrame(self), self._feature_cols])
        return super().__iter__()


def create_forecasting_feature_matrix(
    annual_df: pd.DataFrame,
    subdivision: Optional[str] = None,
    value_col: str = "ANNUAL_VALIDATED",
    target_col: Optional[str] = None,
    lags: List[int] = [1, 2, 3],
    rolling_windows: List[int] = [3, 5],
    baseline_years: Optional[Tuple[int, int]] = DEFAULT_BASELINE_YEARS,
    include_time_trend: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Generate clean, strictly backward-looking time-series and rolling features
    for annual rainfall forecasting.
    """
    if target_col is not None:
        value_col = target_col

    df = annual_df.copy()
    if subdivision is not None and "SUBDIVISION" in df.columns:
        df = df[df["SUBDIVISION"] == subdivision].copy()

    df = validate_continuity_and_sort(df, group_col="SUBDIVISION", year_col="YEAR")
    feature_cols: List[str] = []

    # 1. Target column
    df["TARGET"] = df[value_col]

    # 2. Lag Features with Continuity Validation
    for k in lags:
        col_name = f"LAG_{k}"
        year_diff_k = df.groupby("SUBDIVISION")["YEAR"].diff(k)
        lag_val = df.groupby("SUBDIVISION")[value_col].shift(k)
        is_continuous_k = year_diff_k == k
        lag_series = np.where(is_continuous_k, lag_val, np.nan)
        df[col_name] = lag_series
        df[f"rainfall_lag_{k}"] = lag_series
        feature_cols.append(col_name)

    # 3. Lag Difference (Lag 1 - Lag 2)
    if 1 in lags and 2 in lags:
        diff_col = "LAG_DIFF_1_2"
        df[diff_col] = df["LAG_1"] - df["LAG_2"]
        df["rainfall_diff_1"] = df[diff_col]
        feature_cols.append(diff_col)

    # 4. Backward-Looking Rolling Statistics on Prior Observations (Shift 1)
    prior_series = df.groupby("SUBDIVISION")[value_col].shift(1)

    for w in rolling_windows:
        min_p = max(1, w // 2)
        # Prior rolling mean
        roll_mean_col = f"ROLLING_MEAN_{w}Y_PRIOR"
        r_mean = (
            prior_series.groupby(df["SUBDIVISION"])
            .rolling(window=w, min_periods=min_p)
            .mean()
            .reset_index(level=0, drop=True)
            .round(2)
        )
        df[roll_mean_col] = r_mean
        df[f"rolling_mean_{w}"] = r_mean
        feature_cols.append(roll_mean_col)

        # Prior rolling std
        roll_std_col = f"ROLLING_STD_{w}Y_PRIOR"
        r_std = (
            prior_series.groupby(df["SUBDIVISION"])
            .rolling(window=w, min_periods=min_p)
            .std()
            .reset_index(level=0, drop=True)
            .round(2)
        )
        df[roll_std_col] = r_std
        df[f"rolling_std_{w}"] = r_std
        feature_cols.append(roll_std_col)

    # 5. Climatological Baseline
    if baseline_years:
        b_start, b_end = baseline_years
        baseline_slice = df[(df["YEAR"] >= b_start) & (df["YEAR"] <= b_end)]
    else:
        baseline_slice = df

    baseline_means = (
        baseline_slice.groupby("SUBDIVISION")[value_col]
        .mean()
        .rename("BASELINE_MEAN")
        .reset_index()
    )
    df = df.merge(baseline_means, on="SUBDIVISION", how="left")
    feature_cols.append("BASELINE_MEAN")

    # Prior baseline anomaly (using LAG_1 to avoid leakage)
    if "LAG_1" in df.columns:
        df["PRIOR_BASELINE_ANOMALY"] = (df["LAG_1"] - df["BASELINE_MEAN"]).round(2)
        feature_cols.append("PRIOR_BASELINE_ANOMALY")

    # 6. Linear Time Step Trend
    if include_time_trend:
        min_year = int(df["YEAR"].min()) if not df.empty and df["YEAR"].notnull().any() else 1901
        df["TIME_STEP"] = (df["YEAR"] - min_year).astype(float) if not df.empty else pd.Series(dtype=float)
        feature_cols.append("TIME_STEP")

    res = FeatureMatrixResult(df)
    res._feature_cols = feature_cols
    return res


def create_monthly_cyclical_features(
    monthly_df: pd.DataFrame,
    month_col: str = "MONTH_NUMBER"
) -> pd.DataFrame:
    """
    Generate harmonic cyclical representations of calendar month numbers (1-12)
    using sine and cosine transformations:
    - SIN_MONTH / month_sin = sin(2 * pi * month / 12)
    - COS_MONTH / month_cos = cos(2 * pi * month / 12)
    """
    df = monthly_df.copy()
    if month_col not in df.columns and "MONTH_NUMBER" in df.columns:
        m_series = df["MONTH_NUMBER"]
    elif month_col not in df.columns and "MONTH" in df.columns:
        from src.config import MONTH_NUMBERS
        m_series = df["MONTH"].map(MONTH_NUMBERS)
    else:
        m_series = df[month_col]

    month_rad = 2 * np.pi * m_series.astype(float) / 12.0
    sin_vals = np.sin(month_rad).round(4)
    cos_vals = np.cos(month_rad).round(4)

    df["SIN_MONTH"] = sin_vals
    df["COS_MONTH"] = cos_vals
    df["month_sin"] = sin_vals
    df["month_cos"] = cos_vals
    return df


# Aliases
create_annual_features = add_annual_time_series_features
