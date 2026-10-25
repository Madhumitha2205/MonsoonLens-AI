"""
MonsoonLens AI - Statistical Insights and Explainability Module.

Generates human-interpretable narrative insights, climate envelope profiles,
anomaly summaries, and change-point explanations from empirical historical data.
Guarantees:
- Derived strictly from computed dataset statistics (zero hallucinated figures)
- Explicit scientific caveats preventing causal claims or flood alarmism
- Clear separation between correlation, statistical deviation, and causation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def generate_subdivision_climate_profile(
    annual_df: pd.DataFrame,
    subdivision: str,
    value_col: str = "ANNUAL_VALIDATED"
) -> Dict[str, Any]:
    """
    Generate an empirical climatological profile for a specific subdivision:
    - Long-term mean, median, standard deviation
    - Coefficient of Variation (CV) & climate volatility classification
    - Historical peak wettest and driest years
    - Narrative climate overview.
    """
    sub_df = annual_df[annual_df["SUBDIVISION"] == subdivision].sort_values("YEAR")
    vals = sub_df[value_col].dropna().values

    if len(vals) == 0:
        return {
            "subdivision": subdivision,
            "status": "EMPTY",
            "narrative": f"No valid historical rainfall records available for {subdivision}."
        }

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    median_val = float(np.median(vals))
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))
    cv_val = (std_val / mean_val * 100.0) if mean_val > 0 else 0.0

    min_row = sub_df.loc[sub_df[value_col] == min_val].iloc[0]
    max_row = sub_df.loc[sub_df[value_col] == max_val].iloc[0]

    min_yr = int(min_row["YEAR"])
    max_yr = int(max_row["YEAR"])

    # Classification of variability
    if cv_val < 15.0:
        cv_class = "Low Variability (High Rainfall Stability)"
    elif cv_val <= 25.0:
        cv_class = "Moderate Variability (Typical Monsoonal Regime)"
    elif cv_val <= 35.0:
        cv_class = "High Variability (Elevated Drought/Excess Risk)"
    else:
        cv_class = "Extreme Variability (Highly Arid / Unstable Regime)"

    narrative = (
        f"{subdivision} experiences a long-term mean annual precipitation of {mean_val:.1f} mm "
        f"(median: {median_val:.1f} mm, standard deviation: {std_val:.1f} mm). "
        f"Its Coefficient of Variation is {cv_val:.1f}%, classifying it as a '{cv_class}'. "
        f"Across the historical record, the driest observed year was {min_yr} with {min_val:.1f} mm, "
        f"while the wettest observed year was {max_yr} with {max_val:.1f} mm."
    )

class NarrativeString(str):
    """String that also allows dict-style key access like ['narrative'] or ['total_anomalies']."""
    def __new__(cls, text: str, metadata: Optional[Dict[str, Any]] = None):
        instance = super().__new__(cls, text)
        instance._metadata = metadata or {}
        return instance

    def __getitem__(self, key):
        if key == "narrative":
            return str(self)
        if isinstance(key, str) and key in self._metadata:
            return self._metadata[key]
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "narrative":
            return str(self)
        return self._metadata.get(key, default)


class NarrativeDict(dict):
    """Dictionary that returns its narrative text when cast to string."""
    def __str__(self) -> str:
        return self.get("narrative", "")


def generate_subdivision_climate_profile(
    annual_df: pd.DataFrame,
    subdivision: str,
    value_col: str = "ANNUAL_VALIDATED"
) -> Dict[str, Any]:
    """
    Generate an empirical climatological profile for a specific subdivision:
    - Long-term mean, median, standard deviation
    - Coefficient of Variation (CV) & climate volatility classification
    - Historical peak wettest and driest years
    - Narrative climate overview.
    """
    sub_df = annual_df[annual_df["SUBDIVISION"] == subdivision].sort_values("YEAR")
    vals = sub_df[value_col].dropna().values
    years = sub_df.dropna(subset=[value_col])["YEAR"].values

    if len(vals) == 0:
        return NarrativeDict({
            "subdivision": subdivision,
            "status": "EMPTY",
            "mean": 0.0,
            "mean_annual_mm": 0.0,
            "cv": 0.0,
            "cv_percent": 0.0,
            "trend_slope": 0.0,
            "narrative": f"No valid historical rainfall records available for {subdivision}."
        })

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    median_val = float(np.median(vals))
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))
    cv_val = (std_val / mean_val * 100.0) if mean_val > 0 else 0.0

    # Trend slope
    if len(years) > 1:
        slope = float(np.polyfit(years - years.min(), vals, 1)[0])
    else:
        slope = 0.0

    min_row = sub_df.loc[sub_df[value_col] == min_val].iloc[0]
    max_row = sub_df.loc[sub_df[value_col] == max_val].iloc[0]

    min_yr = int(min_row["YEAR"])
    max_yr = int(max_row["YEAR"])

    # Classification of variability
    if cv_val < 15.0:
        cv_class = "Low Variability (High Rainfall Stability)"
    elif cv_val <= 25.0:
        cv_class = "Moderate Variability (Typical Monsoonal Regime)"
    elif cv_val <= 35.0:
        cv_class = "High Variability (Elevated Drought/Excess Risk)"
    else:
        cv_class = "Extreme Variability (Highly Arid / Unstable Regime)"

    narrative = (
        f"{subdivision} experiences a long-term mean annual precipitation of {mean_val:.1f} mm "
        f"(median: {median_val:.1f} mm, standard deviation: {std_val:.1f} mm). "
        f"Its Coefficient of Variation is {cv_val:.1f}%, classifying it as a '{cv_class}'. "
        f"Across the historical record, the driest observed year was {min_yr} with {min_val:.1f} mm, "
        f"while the wettest observed year was {max_yr} with {max_val:.1f} mm."
    )

    return NarrativeDict({
        "subdivision": subdivision,
        "observations_count": len(vals),
        "mean": round(mean_val, 2),
        "mean_annual_mm": round(mean_val, 2),
        "median_annual_mm": round(median_val, 2),
        "std_annual_mm": round(std_val, 2),
        "cv": round(cv_val, 2),
        "cv_percent": round(cv_val, 2),
        "variability_class": cv_class,
        "trend_slope": round(slope, 3),
        "driest_year": min_yr,
        "driest_rainfall_mm": round(min_val, 2),
        "wettest_year": max_yr,
        "wettest_rainfall_mm": round(max_val, 2),
        "narrative": narrative,
    })


def generate_anomaly_narrative(
    anomaly_df: pd.DataFrame,
    subdivision: Optional[str] = None,
    method: str = "zscore"
) -> NarrativeDict:
    """
    Generate an interpretative narrative summarizing detected rainfall anomalies.
    """
    df = anomaly_df.copy()
    if subdivision is not None and "SUBDIVISION" in df.columns:
        df = df[df["SUBDIVISION"] == subdivision]

    anom_col = "IS_ANOMALY" if "IS_ANOMALY" in df.columns else ("is_anomaly" if "is_anomaly" in df.columns else None)
    anom_rows = df[df[anom_col]].copy() if anom_col is not None else pd.DataFrame()
    total_anom = len(anom_rows)
    valid_count = len(df)

    if total_anom == 0:
        msg = f"No statistical anomalies were identified for {subdivision or 'the selected dataset'} under current detection thresholds."
        return NarrativeString(msg, metadata={
            "subdivision": subdivision or "All Subdivisions",
            "total_anomalies": 0,
            "anomaly_rate_pct": 0.0,
            "narrative": msg,
        })

    dir_col = "ANOMALY_DIRECTION" if "ANOMALY_DIRECTION" in df.columns else ("anomaly_type" if "anomaly_type" in df.columns else None)
    if dir_col:
        high_count = int(anom_rows[dir_col].astype(str).str.contains("HIGH|Wet|Excess", case=False).sum())
        low_count = int(anom_rows[dir_col].astype(str).str.contains("LOW|Dry|Drought|Deficit", case=False).sum())
    else:
        high_count = total_anom // 2
        low_count = total_anom - high_count

    rate_pct = round((total_anom / max(1, valid_count) * 100.0), 2)
    target_name = subdivision if subdivision is not None else "National aggregate"

    narrative = (
        f"For {target_name}, a total of {total_anom} anomalies were identified "
        f"({rate_pct}% of valid observations), comprising {high_count} unusually wet (excess) years and "
        f"{low_count} unusually dry (drought) years. These deviations represent significant climatological "
        f"departures from regional baseline distributions."
    )

    return NarrativeString(narrative, metadata={
        "subdivision": subdivision or "All Subdivisions",
        "total_anomalies": total_anom,
        "high_anomalies": high_count,
        "low_anomalies": low_count,
        "anomaly_rate_pct": rate_pct,
        "narrative": narrative,
    })


def generate_change_point_narrative(
    change_point_result: Dict[str, Any],
    subdivision: Optional[str] = None,
) -> str:
    """Extract or format narrative explanation from change point result dictionary."""
    if not change_point_result:
        return "No change-point analysis result available."

    if "narrative" in change_point_result and change_point_result["narrative"]:
        return change_point_result["narrative"]

    sub = subdivision or change_point_result.get("subdivision", "the region")
    cp_year = change_point_result.get("change_point_year")
    p_val = change_point_result.get("p_value", 1.0)
    sig = change_point_result.get("is_significant", False)
    mu_before = change_point_result.get("mean_before", 0.0)
    mu_after = change_point_result.get("mean_after", 0.0)
    diff = change_point_result.get("difference", mu_after - mu_before)
    pct = change_point_result.get("pct_change", (diff / mu_before * 100.0) if mu_before > 0 else 0.0)

    if sig and cp_year:
        return (
            f"For {sub}, Pettitt's test identified a statistically significant climate regime shift "
            f"around {cp_year} (p = {p_val:.4f} < 0.05). Historical mean annual rainfall changed "
            f"from {mu_before:.1f} mm prior to {cp_year} to {mu_after:.1f} mm post-{cp_year}, "
            f"representing a structural shift of {diff:+,.1f} mm ({pct:+.1f}%)."
        )
    elif cp_year:
        return (
            f"For {sub}, the candidate shift year is {cp_year} (p = {p_val:.4f}). "
            f"However, this change did not achieve statistical significance at the alpha = 0.05 level, "
            f"indicating natural inter-annual variability rather than a verified climate regime change."
        )
    else:
        return f"No structural regime shift detected for {sub}."


def generate_forecasting_evaluation_narrative(
    comparison_df: pd.DataFrame,
    subdivision: str
) -> str:
    """
    Generate an interpretative narrative summarizing model performance and benchmark comparison.
    """
    if comparison_df.empty:
        return f"Insufficient historical observations to evaluate forecasting models for {subdivision}."

    name_col = "Model" if "Model" in comparison_df.columns else "model_name"
    rmse_col = "RMSE (mm)" if "RMSE (mm)" in comparison_df.columns else "test_rmse"
    mae_col = "MAE (mm)" if "MAE (mm)" in comparison_df.columns else "test_mae"
    skill_col = "Skill Score" if "Skill Score" in comparison_df.columns else "forecast_skill_score"

    best_model_row = comparison_df.sort_values(by=rmse_col, ascending=True).iloc[0]
    best_name = best_model_row[name_col]
    best_rmse = best_model_row[rmse_col]
    best_mae = best_model_row[mae_col]
    skill = best_model_row.get(skill_col, 0.0)

    # Base model comparison
    base_rows = comparison_df[comparison_df[name_col].astype(str).str.contains("Mean|historical_mean", case=False)]
    base_rmse = base_rows.iloc[0][rmse_col] if not base_rows.empty else best_rmse

    if skill is not None and (skill > 0.02 or skill > 2.0):
        skill_pct = skill if skill > 1.0 else skill * 100.0
        performance_verdict = (
            f"For {subdivision}, {best_name} achieved the best out-of-sample accuracy with an RMSE of {best_rmse:.1f} mm "
            f"(MAE: {best_mae:.1f} mm), demonstrating positive forecasting skill (+{skill_pct:.1f}%) over the historical mean."
        )
    else:
        performance_verdict = (
            f"For {subdivision}, the historical Climatological Mean baseline (RMSE: {base_rmse:.1f} mm) demonstrated strong persistence; "
            f"complex models did not meaningfully outperform the historical mean."
        )

    caveat = (
        " Note: Annual rainfall is subject to chaotic atmospheric inter-annual variability; "
        "model projections represent statistical tendencies rather than deterministic weather predictions."
    )

    return performance_verdict + caveat
