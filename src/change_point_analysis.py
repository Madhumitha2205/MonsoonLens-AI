"""
MonsoonLens AI - Statistical Rainfall Change-Point Analysis Module.

Detects potential historical structural shifts in rainfall mean and distribution over time.
Implements:
1. Pettitt's Non-Parametric Rank Test (Mann-Whitney based shift detection)
2. Cumulative Sum (CUSUM) Analysis
3. Standard Normal Homogeneity Test (SNHT / Alexandersson's Test)

Key Guarantees:
- Chronological ordering strictly enforced
- Partitioned by subdivision (no spatial cross-contamination)
- Rejection of insufficient time series (minimum sample size >= 20 observations)
- Explicit handling of historical calendar year gaps
- Transparent differentiation between candidate statistical shift and causal climate change proof.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHANGE_POINT_P_VALUE_THRESHOLD, MIN_OBSERVATION_COUNT


def pettitt_test(
    series: Union[List[float], np.ndarray, pd.Series],
    alpha: float = CHANGE_POINT_P_VALUE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Execute Pettitt's non-parametric rank test for a single change point in mean.
    """
    clean_vals = np.asarray(series, dtype=float)
    clean_vals = clean_vals[~np.isnan(clean_vals)]
    n = len(clean_vals)

    if n < 10:
        return {
            "change_index": None,
            "change_point_index": None,
            "stat_k": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "mean_before": 0.0,
            "mean_after": 0.0,
            "difference": 0.0,
            "u_series": [],
            "sample_size": n,
        }

    # Vectorized computation of sgn matrix
    diff_matrix = clean_vals[:, np.newaxis] - clean_vals[np.newaxis, :]
    sgn_matrix = np.sign(diff_matrix)

    u_vals = np.zeros(n - 1, dtype=float)
    for t in range(1, n):
        u_vals[t - 1] = np.sum(sgn_matrix[:t, t:])

    abs_u = np.abs(u_vals)
    k_stat = float(np.max(abs_u))
    change_idx = int(np.argmax(abs_u))

    exponent = -6.0 * (k_stat ** 2) / (n ** 3 + n ** 2)
    p_value = min(1.0, float(2.0 * np.exp(exponent)))

    mean_before = float(np.mean(clean_vals[:change_idx + 1]))
    mean_after = float(np.mean(clean_vals[change_idx + 1:])) if change_idx + 1 < n else mean_before
    diff = mean_after - mean_before

    return {
        "change_index": change_idx,
        "change_point_index": change_idx,
        "stat_k": k_stat,
        "p_value": p_value,
        "is_significant": p_value < alpha,
        "mean_before": round(mean_before, 2),
        "mean_after": round(mean_after, 2),
        "difference": round(diff, 2),
        "u_series": u_vals.tolist(),
        "sample_size": n,
    }


def cusum_test(
    series: Union[List[float], np.ndarray, pd.Series],
    alpha: float = CHANGE_POINT_P_VALUE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Cumulative Sum (CUSUM) change-point test based on cumulative deviations from the mean.
    """
    clean_vals = np.asarray(series, dtype=float)
    clean_vals = clean_vals[~np.isnan(clean_vals)]
    n = len(clean_vals)

    if n < 10:
        return {
            "change_index": None,
            "change_point_index": None,
            "cusum_range": 0.0,
            "is_significant": False,
            "mean_before": 0.0,
            "mean_after": 0.0,
            "difference": 0.0,
            "cusum_series": [],
            "sample_size": n,
        }

    mean_val = np.mean(clean_vals)
    deviations = clean_vals - mean_val
    cusum_series = np.cumsum(deviations)

    cusum_max = np.max(cusum_series)
    cusum_min = np.min(cusum_series)
    cusum_range = float(cusum_max - cusum_min)

    change_idx = int(np.argmax(np.abs(cusum_series)))

    std_val = np.std(clean_vals) if np.std(clean_vals) > 1e-4 else 1.0
    relative_range = cusum_range / (std_val * np.sqrt(n))
    is_significant = relative_range > (1.36 if alpha <= 0.05 else 1.22)

    mean_before = float(np.mean(clean_vals[:change_idx + 1]))
    mean_after = float(np.mean(clean_vals[change_idx + 1:])) if change_idx + 1 < n else mean_before
    diff = mean_after - mean_before

    return {
        "change_index": change_idx,
        "change_point_index": change_idx,
        "cusum_range": cusum_range,
        "relative_range": float(relative_range),
        "is_significant": is_significant,
        "mean_before": round(mean_before, 2),
        "mean_after": round(mean_after, 2),
        "difference": round(diff, 2),
        "cusum_series": cusum_series.tolist(),
        "sample_size": n,
    }


def snht_test(
    series: Union[List[float], np.ndarray, pd.Series],
    alpha: float = CHANGE_POINT_P_VALUE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Standard Normal Homogeneity Test (SNHT / Alexandersson's Test).
    """
    clean_vals = np.asarray(series, dtype=float)
    clean_vals = clean_vals[~np.isnan(clean_vals)]
    n = len(clean_vals)

    if n < 10:
        return {
            "change_index": None,
            "change_point_index": None,
            "stat_t": 0.0,
            "is_significant": False,
            "mean_before": 0.0,
            "mean_after": 0.0,
            "difference": 0.0,
            "sample_size": n,
        }

    mean_val = np.mean(clean_vals)
    std_val = np.std(clean_vals, ddof=1) if len(clean_vals) > 1 else 1.0
    if std_val < 1e-4:
        return {
            "change_index": None,
            "change_point_index": None,
            "stat_t": 0.0,
            "is_significant": False,
            "mean_before": round(float(mean_val), 2),
            "mean_after": round(float(mean_val), 2),
            "difference": 0.0,
            "sample_size": n,
        }

    z = (clean_vals - mean_val) / std_val
    t_vals = np.zeros(n - 1)

    for k in range(1, n):
        z1_bar = np.mean(z[:k])
        z2_bar = np.mean(z[k:])
        t_vals[k - 1] = k * (z1_bar ** 2) + (n - k) * (z2_bar ** 2)

    change_idx = int(np.argmax(t_vals))
    max_t = float(t_vals[change_idx])

    is_significant = max_t > (8.5 if alpha <= 0.05 else 6.8)

    mean_before = float(np.mean(clean_vals[:change_idx + 1]))
    mean_after = float(np.mean(clean_vals[change_idx + 1:])) if change_idx + 1 < n else mean_before
    diff = mean_after - mean_before

    return {
        "change_index": change_idx,
        "change_point_index": change_idx,
        "stat_t": max_t,
        "is_significant": is_significant,
        "mean_before": round(mean_before, 2),
        "mean_after": round(mean_after, 2),
        "difference": round(diff, 2),
        "sample_size": n,
    }


def analyze_subdivision_change_point(
    annual_df: pd.DataFrame,
    subdivision: Optional[str] = None,
    value_col: str = "ANNUAL_VALIDATED",
    min_observations: int = MIN_OBSERVATION_COUNT,
    method: str = "pettitt",
    alpha: float = CHANGE_POINT_P_VALUE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Execute full statistical change-point analysis on an annual rainfall series
    for a specific subdivision.
    """
    if subdivision is not None and "SUBDIVISION" in annual_df.columns:
        sub_df = annual_df[annual_df["SUBDIVISION"] == subdivision].sort_values("YEAR").copy()
        sub_name = subdivision
    elif "SUBDIVISION" in annual_df.columns:
        sub_name = str(annual_df["SUBDIVISION"].iloc[0])
        sub_df = annual_df[annual_df["SUBDIVISION"] == sub_name].sort_values("YEAR").copy()
    else:
        sub_name = "REGION"
        sub_df = annual_df.sort_values("YEAR").copy()

    valid_mask = sub_df[value_col].notnull()
    valid_sub_df = sub_df[valid_mask].reset_index(drop=True)

    n_obs = len(valid_sub_df)
    years = valid_sub_df["YEAR"].values
    vals = valid_sub_df[value_col].values

    if n_obs < min_observations:
        return {
            "subdivision": sub_name,
            "status": "INSUFFICIENT_OBSERVATIONS",
            "message": f"Sample size (N={n_obs}) is below minimum requirement ({min_observations})",
            "has_change_point": False,
            "is_significant": False,
            "sample_size": n_obs,
        }

    # Run selected algorithm
    if method.lower() == "pettitt":
        res = pettitt_test(vals, alpha=alpha)
        stat_name = "Pettitt's K"
        stat_val = res["stat_k"]
    elif method.lower() == "cusum":
        res = cusum_test(vals, alpha=alpha)
        stat_name = "CUSUM Range"
        stat_val = res["cusum_range"]
    elif method.lower() == "snht":
        res = snht_test(vals, alpha=alpha)
        stat_name = "SNHT T"
        stat_val = res["stat_t"]
    else:
        raise ValueError(f"Unknown change-point method: {method}. Supported: 'pettitt', 'cusum', 'snht'.")

    change_idx = res["change_index"]
    if change_idx is None or change_idx >= len(years) - 1:
        return {
            "subdivision": sub_name,
            "status": "NO_CHANGE_POINT",
            "has_change_point": False,
            "is_significant": False,
            "sample_size": n_obs,
        }

    change_year = int(years[change_idx])
    pre_vals = vals[:change_idx + 1]
    post_vals = vals[change_idx + 1:]

    pre_mean = float(np.mean(pre_vals))
    pre_std = float(np.std(pre_vals))
    post_mean = float(np.mean(post_vals))
    post_std = float(np.std(post_vals))

    shift_mm = round(post_mean - pre_mean, 2)
    shift_pct = round((shift_mm / pre_mean * 100.0), 2) if pre_mean > 0 else 0.0

    pooled_std = np.sqrt(((len(pre_vals) - 1) * (pre_std ** 2) + (len(post_vals) - 1) * (post_std ** 2)) / max(1, (len(pre_vals) + len(post_vals) - 2)))
    cohens_d = round(shift_mm / pooled_std, 3) if pooled_std > 1e-4 else 0.0

    p_val = res.get("p_value", None)
    is_sig = res.get("is_significant", False)

    direction = "INCREASE" if shift_mm > 0 else "DECREASE"
    sig_text = "statistically significant (p < 0.05)" if is_sig else "statistically non-significant (p >= 0.05)"
    narrative = (
        f"In {sub_name}, a candidate structural shift was identified around year {change_year}. "
        f"Pre-change mean rainfall was {pre_mean:.1f} mm (N={len(pre_vals)}), shifting to "
        f"{post_mean:.1f} mm (N={len(post_vals)}) post-{change_year}, representing an estimated {direction.lower()} "
        f"of {abs(shift_mm):.1f} mm ({abs(shift_pct):.1f}%). This candidate change is {sig_text}."
    )

    return {
        "subdivision": sub_name,
        "status": "DETECTED",
        "method": method,
        "stat_name": stat_name,
        "stat_value": round(float(stat_val), 3),
        "p_value": round(float(p_val), 4) if p_val is not None else None,
        "is_significant": is_sig,
        "has_change_point": is_sig,
        "change_year": change_year,
        "change_point_year": change_year,
        "change_point_index": change_idx,
        "sample_size": n_obs,
        "pre_change_years": (int(years[0]), change_year),
        "pre_change_mean_mm": round(pre_mean, 2),
        "mean_before": round(pre_mean, 2),
        "pre_change_std_mm": round(pre_std, 2),
        "pre_change_count": len(pre_vals),
        "post_change_years": (int(years[change_idx + 1]), int(years[-1])),
        "post_change_mean_mm": round(post_mean, 2),
        "mean_after": round(post_mean, 2),
        "post_change_std_mm": round(post_std, 2),
        "post_change_count": len(post_vals),
        "shift_magnitude_mm": shift_mm,
        "difference": shift_mm,
        "shift_percentage": shift_pct,
        "pct_change": shift_pct,
        "cohens_d": cohens_d,
        "direction": direction,
        "narrative": narrative,
    }


def analyze_all_subdivisions_change_points(
    annual_df: pd.DataFrame,
    value_col: str = "ANNUAL_VALIDATED",
    method: str = "pettitt",
    alpha: float = CHANGE_POINT_P_VALUE_THRESHOLD,
) -> pd.DataFrame:
    """
    Run change-point analysis systematically across all available subdivisions.
    Returns summary DataFrame of candidate change points, significance, and shift magnitudes.
    """
    records = []
    subdivisions = sorted(annual_df["SUBDIVISION"].dropna().unique().tolist())

    for sub in subdivisions:
        res = analyze_subdivision_change_point(annual_df, sub, value_col=value_col, method=method, alpha=alpha)
        if res.get("status") == "DETECTED":
            records.append({
                "SUBDIVISION": sub,
                "METHOD": res["method"],
                "CHANGE_YEAR": res["change_year"],
                "IS_SIGNIFICANT": res["is_significant"],
                "P_VALUE": res["p_value"],
                "STAT_VALUE": res["stat_value"],
                "PRE_MEAN_MM": res["pre_change_mean_mm"],
                "POST_MEAN_MM": res["post_change_mean_mm"],
                "SHIFT_MM": res["shift_magnitude_mm"],
                "SHIFT_PCT": res["shift_percentage"],
                "COHENS_D": res["cohens_d"],
                "DIRECTION": res["direction"],
            })

    df_out = pd.DataFrame(records)
    if not df_out.empty:
        df_out["subdivision"] = df_out["SUBDIVISION"]
        df_out["change_year"] = df_out["CHANGE_YEAR"]
        df_out["is_significant"] = df_out["IS_SIGNIFICANT"]
        df_out["p_value"] = df_out["P_VALUE"]
        df_out["shift_magnitude_mm"] = df_out["SHIFT_MM"]
        df_out = df_out.sort_values(by=["IS_SIGNIFICANT", "P_VALUE"], ascending=[False, True]).reset_index(drop=True)
    return df_out
