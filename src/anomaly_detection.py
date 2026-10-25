"""
MonsoonLens AI - Multi-Method Rainfall Anomaly Detection Module.

Identifies rainfall observations that differ substantially from historical reference distributions.
Implements:
1. IQR-Based Detection (Tukey's Fences)
2. Z-Score / Standardized Deviation Detection
3. Isolation Forest (Unsupervised Tree Ensemble)

Key Guarantees:
- Subdivision-isolated and seasonality-aware reference baselines
- Explicit handling of missing observations (never treated as zero)
- Safe handling of zero variance and small sample groups
- Uniform output schema across all detection strategies
- Clear differentiation between statistical outliers and causal disaster claims.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANOMALY_IFOREST_CONTAMINATION,
    ANOMALY_IQR_MULTIPLIER,
    ANOMALY_Z_SCORE_THRESHOLD,
    RANDOM_SEED,
)


def detect_anomalies_iqr(
    df: pd.DataFrame,
    value_col: str = "ANNUAL_VALIDATED",
    group_cols: Optional[Union[str, List[str]]] = "SUBDIVISION",
    multiplier: float = ANOMALY_IQR_MULTIPLIER,
    factor: Optional[float] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Detect statistical outliers using the Interquartile Range (Tukey's Fences).
    Fences: [Q1 - multiplier * IQR, Q3 + multiplier * IQR]
    Baseline is the group median.
    """
    if factor is not None:
        multiplier = factor

    if isinstance(group_cols, str):
        group_cols = [group_cols]

    out = df.copy()
    valid_mask = out[value_col].notnull()

    # Initialize result columns
    out["BASELINE_RAINFALL"] = np.nan
    out["DEVIATION_FROM_BASELINE"] = np.nan
    out["STANDARDIZED_DEVIATION"] = np.nan
    out["IS_ANOMALY"] = False
    out["ANOMALY_DIRECTION"] = "NORMAL"
    out["DETECTION_METHOD"] = "IQR"
    out["ANOMALY_SCORE"] = 0.0
    out["EXPLANATION"] = "Normal observation within expected range"

    if not valid_mask.any():
        _attach_normalized_anomaly_columns(out)
        return out

    if group_cols is None or not group_cols or not all(c in out.columns for c in group_cols):
        groups = [(None, out[valid_mask])]
    else:
        groups = out[valid_mask].groupby(group_cols)

    for keys, grp in groups:
        idx = grp.index
        vals = grp[value_col].astype(float)
        n_obs = len(vals)

        if n_obs < 4:
            out.loc[idx, "EXPLANATION"] = f"Insufficient sample size (N={n_obs} < 4) for reliable IQR"
            continue

        q1 = float(vals.quantile(0.25))
        q3 = float(vals.quantile(0.75))
        iqr = q3 - q1
        med = float(vals.median())

        lower_fence = max(0.0, q1 - multiplier * iqr)
        upper_fence = q3 + multiplier * iqr

        out.loc[idx, "BASELINE_RAINFALL"] = med
        out.loc[idx, "DEVIATION_FROM_BASELINE"] = (vals - med).round(2)

        # Standardized deviation relative to IQR (normalized spread)
        iqr_safe = iqr if iqr > 1e-4 else 1.0
        out.loc[idx, "STANDARDIZED_DEVIATION"] = ((vals - med) / iqr_safe).round(3)

        high_mask = vals > upper_fence
        low_mask = vals < lower_fence

        # High anomalies
        if high_mask.any():
            h_idx = idx[high_mask]
            out.loc[h_idx, "IS_ANOMALY"] = True
            out.loc[h_idx, "ANOMALY_DIRECTION"] = "HIGH"
            score = ((vals.loc[h_idx] - upper_fence) / iqr_safe).round(2)
            out.loc[h_idx, "ANOMALY_SCORE"] = score
            for i in h_idx:
                v = out.loc[i, value_col]
                out.loc[i, "EXPLANATION"] = (
                    f"Rainfall ({v:.1f} mm) exceeds upper fence ({upper_fence:.1f} mm, "
                    f"Q3+{multiplier}*IQR) by {v - upper_fence:.1f} mm"
                )

        # Low anomalies
        if low_mask.any():
            l_idx = idx[low_mask]
            out.loc[l_idx, "IS_ANOMALY"] = True
            out.loc[l_idx, "ANOMALY_DIRECTION"] = "LOW"
            score = ((lower_fence - vals.loc[l_idx]) / iqr_safe).round(2)
            out.loc[l_idx, "ANOMALY_SCORE"] = score
            for i in l_idx:
                v = out.loc[i, value_col]
                out.loc[i, "EXPLANATION"] = (
                    f"Rainfall ({v:.1f} mm) falls below lower fence ({lower_fence:.1f} mm, "
                    f"Q1-{multiplier}*IQR) by {lower_fence - v:.1f} mm"
                )

    _attach_normalized_anomaly_columns(out)
    return out


def detect_anomalies_zscore(
    df: pd.DataFrame,
    value_col: str = "ANNUAL_VALIDATED",
    group_cols: Optional[Union[str, List[str]]] = "SUBDIVISION",
    threshold: float = ANOMALY_Z_SCORE_THRESHOLD,
    z_threshold: Optional[float] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Detect statistical anomalies using standardized Z-scores relative to historical mean and std.
    Anomaly defined when |Z| > threshold.
    Safely handles constant groups and zero standard deviations.
    """
    if z_threshold is not None:
        threshold = z_threshold

    if isinstance(group_cols, str):
        group_cols = [group_cols]

    out = df.copy()
    valid_mask = out[value_col].notnull()

    out["BASELINE_RAINFALL"] = np.nan
    out["DEVIATION_FROM_BASELINE"] = np.nan
    out["STANDARDIZED_DEVIATION"] = np.nan
    out["IS_ANOMALY"] = False
    out["ANOMALY_DIRECTION"] = "NORMAL"
    out["DETECTION_METHOD"] = "Z-Score"
    out["ANOMALY_SCORE"] = 0.0
    out["EXPLANATION"] = "Normal observation within expected range"

    if not valid_mask.any():
        _attach_normalized_anomaly_columns(out)
        return out

    if group_cols is None or not group_cols or not all(c in out.columns for c in group_cols):
        groups = [(None, out[valid_mask])]
    else:
        groups = out[valid_mask].groupby(group_cols)

    for keys, grp in groups:
        idx = grp.index
        vals = grp[value_col].astype(float)
        n_obs = len(vals)

        if n_obs < 3:
            out.loc[idx, "EXPLANATION"] = f"Insufficient sample size (N={n_obs} < 3) for Z-Score"
            continue

        mean_val = float(vals.mean())
        std_val = float(vals.std())

        out.loc[idx, "BASELINE_RAINFALL"] = round(mean_val, 2)
        out.loc[idx, "DEVIATION_FROM_BASELINE"] = (vals - mean_val).round(2)

        if std_val < 1e-4:
            out.loc[idx, "STANDARDIZED_DEVIATION"] = 0.0
            out.loc[idx, "EXPLANATION"] = "Zero or near-zero variance in group"
            continue

        z_scores = ((vals - mean_val) / std_val).round(3)
        out.loc[idx, "STANDARDIZED_DEVIATION"] = z_scores

        high_mask = z_scores > threshold
        low_mask = z_scores < -threshold

        if high_mask.any():
            h_idx = idx[high_mask]
            out.loc[h_idx, "IS_ANOMALY"] = True
            out.loc[h_idx, "ANOMALY_DIRECTION"] = "HIGH"
            out.loc[h_idx, "ANOMALY_SCORE"] = z_scores.loc[h_idx].abs()
            for i in h_idx:
                v = out.loc[i, value_col]
                z = z_scores.loc[i]
                out.loc[i, "EXPLANATION"] = (
                    f"Rainfall ({v:.1f} mm) is {z:+.2f} standard deviations above group mean ({mean_val:.1f} mm)"
                )

        if low_mask.any():
            l_idx = idx[low_mask]
            out.loc[l_idx, "IS_ANOMALY"] = True
            out.loc[l_idx, "ANOMALY_DIRECTION"] = "LOW"
            out.loc[l_idx, "ANOMALY_SCORE"] = z_scores.loc[l_idx].abs()
            for i in l_idx:
                v = out.loc[i, value_col]
                z = z_scores.loc[i]
                out.loc[i, "EXPLANATION"] = (
                    f"Rainfall ({v:.1f} mm) is {z:+.2f} standard deviations below group mean ({mean_val:.1f} mm)"
                )

    _attach_normalized_anomaly_columns(out)
    return out


def _attach_normalized_anomaly_columns(df: pd.DataFrame) -> None:
    """Helper to attach uniform lowercase alias columns for caller convenience."""
    df["is_anomaly"] = df["IS_ANOMALY"]
    df["z_score"] = df["STANDARDIZED_DEVIATION"]
    df["iqr_distance"] = df["ANOMALY_SCORE"]
    df["anomaly_score"] = df["ANOMALY_SCORE"]
    df["anomaly_type"] = df["ANOMALY_DIRECTION"].map({
        "HIGH": "Extreme Wet",
        "LOW": "Severe Drought",
        "NORMAL": "Normal",
    }).fillna(df["ANOMALY_DIRECTION"])


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    value_col: str = "ANNUAL_VALIDATED",
    group_cols: Optional[Union[str, List[str]]] = "SUBDIVISION",
    contamination: float = ANOMALY_IFOREST_CONTAMINATION,
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Detect multivariate anomalies using an unsupervised Isolation Forest.
    Operates per reference group using observed rainfall and standardized deviation.
    Requires minimum sample size of 15 observations per group.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    out = df.copy()
    valid_mask = out[value_col].notnull()

    out["BASELINE_RAINFALL"] = np.nan
    out["DEVIATION_FROM_BASELINE"] = np.nan
    out["STANDARDIZED_DEVIATION"] = np.nan
    out["IS_ANOMALY"] = False
    out["ANOMALY_DIRECTION"] = "NORMAL"
    out["DETECTION_METHOD"] = "Isolation Forest"
    out["ANOMALY_SCORE"] = 0.0
    out["EXPLANATION"] = "Normal observation within expected distribution"

    if not valid_mask.any():
        _attach_normalized_anomaly_columns(out)
        return out

    if group_cols is None or not group_cols or not all(c in out.columns for c in group_cols):
        groups = [(None, out[valid_mask])]
    else:
        groups = out[valid_mask].groupby(group_cols)

    for keys, grp in groups:
        idx = grp.index
        vals = grp[value_col].astype(float)
        n_obs = len(vals)

        mean_val = float(vals.mean())
        std_val = float(vals.std()) if len(vals) > 1 else 0.0
        std_safe = std_val if std_val > 1e-4 else 1.0

        out.loc[idx, "BASELINE_RAINFALL"] = round(mean_val, 2)
        out.loc[idx, "DEVIATION_FROM_BASELINE"] = (vals - mean_val).round(2)
        out.loc[idx, "STANDARDIZED_DEVIATION"] = ((vals - mean_val) / std_safe).round(3)

        if n_obs < 15:
            out.loc[idx, "EXPLANATION"] = f"Insufficient sample size (N={n_obs} < 15) for Isolation Forest"
            continue

        # Prepare 2D feature matrix: [value, standardized_deviation]
        X = np.column_stack([
            vals.values,
            ((vals.values - mean_val) / std_safe)
        ])

        try:
            iso = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100
            )
            preds = iso.fit_predict(X)  # -1 for anomaly, 1 for normal
            scores = -iso.decision_function(X)  # Higher score = more anomalous

            is_anom = (preds == -1)
            out.loc[idx, "IS_ANOMALY"] = is_anom
            out.loc[idx, "ANOMALY_SCORE"] = np.round(scores, 3)

            for i, anom, v in zip(idx, is_anom, vals):
                if anom:
                    direction = "HIGH" if v > mean_val else "LOW"
                    out.loc[i, "ANOMALY_DIRECTION"] = direction
                    out.loc[i, "EXPLANATION"] = (
                        f"Unsupervised Isolation Forest flagged {direction} anomaly "
                        f"({v:.1f} mm vs group mean {mean_val:.1f} mm; anomaly score: {out.loc[i, 'ANOMALY_SCORE']:.3f})"
                    )
        except Exception as e:
            out.loc[idx, "EXPLANATION"] = f"Isolation Forest failed: {str(e)}"

    _attach_normalized_anomaly_columns(out)
    return out


def detect_anomalies(
    df: pd.DataFrame,
    method: str = "zscore",
    value_col: str = "ANNUAL_VALIDATED",
    group_cols: Optional[Union[str, List[str]]] = "SUBDIVISION",
    subdivision_col: Optional[str] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Unified entry point for anomaly detection.
    """
    grp_cols = subdivision_col if subdivision_col is not None else group_cols
    method_clean = method.lower().replace("-", "").replace("_", "")

    if method_clean in ["zscore", "z"]:
        threshold = kwargs.get("threshold", kwargs.get("z_threshold", ANOMALY_Z_SCORE_THRESHOLD))
        return detect_anomalies_zscore(df, value_col=value_col, group_cols=grp_cols, threshold=threshold)

    elif method_clean in ["iqr", "tukey"]:
        multiplier = kwargs.get("multiplier", kwargs.get("factor", ANOMALY_IQR_MULTIPLIER))
        return detect_anomalies_iqr(df, value_col=value_col, group_cols=grp_cols, multiplier=multiplier)

    elif method_clean in ["iforest", "isolationforest", "forest"]:
        contamination = kwargs.get("contamination", ANOMALY_IFOREST_CONTAMINATION)
        rand_state = kwargs.get("random_state", RANDOM_SEED)
        return detect_anomalies_isolation_forest(df, value_col=value_col, group_cols=grp_cols, contamination=contamination, random_state=rand_state)

    else:
        raise ValueError(
            f"Unsupported anomaly detection method: '{method}'. "
            f"Supported methods: 'zscore', 'iqr', 'isolation_forest'."
        )


def summarize_anomalies(anomaly_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary metrics of detected anomalies.
    """
    val_col = "OBSERVED_RAINFALL" if "OBSERVED_RAINFALL" in anomaly_df.columns else "ANNUAL_VALIDATED"
    valid_obs = int(anomaly_df[val_col].notnull().sum()) if val_col in anomaly_df.columns else len(anomaly_df)
    anom_mask = anomaly_df["IS_ANOMALY"] if "IS_ANOMALY" in anomaly_df.columns else anomaly_df["is_anomaly"]
    total_anomalies = int(anom_mask.sum())
    anomaly_pct = round((total_anomalies / max(1, valid_obs) * 100.0), 2) if valid_obs > 0 else 0.0

    dir_col = "ANOMALY_DIRECTION" if "ANOMALY_DIRECTION" in anomaly_df.columns else "anomaly_type"
    high_anomalies = int(anomaly_df[dir_col].astype(str).str.contains("HIGH|Wet|Excess", case=False).sum()) if dir_col in anomaly_df.columns else total_anomalies // 2
    low_anomalies = int(anomaly_df[dir_col].astype(str).str.contains("LOW|Dry|Drought|Deficit", case=False).sum()) if dir_col in anomaly_df.columns else total_anomalies - high_anomalies

    if "SUBDIVISION" in anomaly_df.columns:
        sub_counts = (
            anomaly_df[anom_mask]
            .groupby("SUBDIVISION")[anomaly_df.columns[0]]
            .count()
            .sort_values(ascending=False)
            .to_dict()
        )
    else:
        sub_counts = {}

    return {
        "valid_observations": valid_obs,
        "total_observations": valid_obs,
        "total_anomalies": total_anomalies,
        "anomaly_count": total_anomalies,
        "anomaly_percentage": anomaly_pct,
        "high_anomalies_count": high_anomalies,
        "excess_count": high_anomalies,
        "low_anomalies_count": low_anomalies,
        "drought_count": low_anomalies,
        "subdivisions_with_anomalies_count": len(sub_counts),
        "anomalies_by_subdivision": sub_counts,
    }


def get_top_anomalous_years(
    anomaly_df: pd.DataFrame,
    subdivision: Optional[str] = None,
    top_n: int = 10,
    n: Optional[int] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Return top anomalies or (top_droughts, top_floods) if n is specified."""
    count = n if n is not None else top_n
    anom_col = "IS_ANOMALY" if "IS_ANOMALY" in anomaly_df.columns else "is_anomaly"
    df = anomaly_df[anomaly_df[anom_col]].copy()
    if subdivision is not None and "SUBDIVISION" in df.columns:
        df = df[df["SUBDIVISION"] == subdivision]

    if n is not None:
        dir_col = "ANOMALY_DIRECTION" if "ANOMALY_DIRECTION" in df.columns else "anomaly_type"
        score_col = "ANOMALY_SCORE" if "ANOMALY_SCORE" in df.columns else "anomaly_score"
        droughts = df[df[dir_col].astype(str).str.contains("LOW|Drought|Deficit", case=False)].sort_values(score_col, ascending=False).head(count)
        floods = df[df[dir_col].astype(str).str.contains("HIGH|Wet|Excess", case=False)].sort_values(score_col, ascending=False).head(count)
        return droughts, floods

    cols = [
        "SUBDIVISION",
        "YEAR",
        "ANNUAL_VALIDATED" if "ANNUAL_VALIDATED" in df.columns else "RAINFALL",
        "BASELINE_RAINFALL",
        "DEVIATION_FROM_BASELINE",
        "STANDARDIZED_DEVIATION",
        "ANOMALY_DIRECTION",
        "DETECTION_METHOD",
        "ANOMALY_SCORE",
        "EXPLANATION",
    ]
    available_cols = [c for c in cols if c in df.columns]
    score_col = "ANOMALY_SCORE" if "ANOMALY_SCORE" in df.columns else "anomaly_score"
    return df.sort_values(by=score_col, ascending=False).head(count)[available_cols].reset_index(drop=True)
