"""
Tests for src/anomaly_detection.py - Z-score, IQR, Isolation Forest, and summary aggregations.
"""

import pytest
import numpy as np
import pandas as pd

from src.anomaly_detection import (
    detect_anomalies,
    detect_anomalies_iqr,
    detect_anomalies_isolation_forest,
    detect_anomalies_zscore,
    get_top_anomalous_years,
    summarize_anomalies,
)


@pytest.fixture
def sample_rainfall_df():
    """Create synthetic rainfall dataset with deliberate extreme anomalies."""
    np.random.seed(42)
    normal_rain = np.random.normal(loc=1000.0, scale=100.0, size=50)
    years = list(range(1960, 2010))

    df = pd.DataFrame({
        "SUBDIVISION": ["TEST_REGION"] * 50,
        "YEAR": years,
        "ANNUAL_VALIDATED": normal_rain,
    })

    # Inject extreme wet anomaly at 1975 (4.5 sigma)
    df.loc[df["YEAR"] == 1975, "ANNUAL_VALIDATED"] = 1550.0
    # Inject extreme drought anomaly at 1987 (-4.5 sigma)
    df.loc[df["YEAR"] == 1987, "ANNUAL_VALIDATED"] = 450.0

    return df


def test_detect_anomalies_zscore(sample_rainfall_df):
    """Verify Z-score anomaly detection identifies injected extreme years."""
    res_df = detect_anomalies_zscore(sample_rainfall_df, value_col="ANNUAL_VALIDATED", threshold=2.5)

    assert "z_score" in res_df.columns
    assert "is_anomaly" in res_df.columns
    assert "anomaly_type" in res_df.columns

    # 1975 should be extreme wet / excess
    row_1975 = res_df[res_df["YEAR"] == 1975].iloc[0]
    assert row_1975["is_anomaly"] == True
    assert row_1975["anomaly_type"] in ["Extreme Wet", "Excess Rainfall"]
    assert row_1975["z_score"] > 2.5

    # 1987 should be severe drought / deficit
    row_1987 = res_df[res_df["YEAR"] == 1987].iloc[0]
    assert row_1987["is_anomaly"] == True
    assert row_1987["anomaly_type"] in ["Severe Drought", "Deficit Rainfall"]
    assert row_1987["z_score"] < -2.5


def test_detect_anomalies_iqr(sample_rainfall_df):
    """Verify IQR anomaly detection identifies outliers beyond 1.5x IQR."""
    res_df = detect_anomalies_iqr(sample_rainfall_df, value_col="ANNUAL_VALIDATED", factor=1.5)

    assert "iqr_distance" in res_df.columns
    assert "is_anomaly" in res_df.columns

    row_1975 = res_df[res_df["YEAR"] == 1975].iloc[0]
    assert row_1975["is_anomaly"] == True
    assert row_1975["anomaly_type"] == "Extreme Wet"

    row_1987 = res_df[res_df["YEAR"] == 1987].iloc[0]
    assert row_1987["is_anomaly"] == True
    assert row_1987["anomaly_type"] == "Severe Drought"


def test_detect_anomalies_isolation_forest(sample_rainfall_df):
    """Verify Isolation Forest flags isolated multivariate/univariate points."""
    res_df = detect_anomalies_isolation_forest(
        sample_rainfall_df,
        value_col="ANNUAL_VALIDATED",
        contamination=0.05,
        random_state=42,
    )

    assert "anomaly_score" in res_df.columns
    assert "is_anomaly" in res_df.columns
    assert res_df["is_anomaly"].sum() > 0


def test_subdivision_grouped_detection():
    """Verify detect_anomalies groups by subdivision so arid baselines are not confused with wet ones."""
    arid_rain = np.random.normal(loc=300.0, scale=30.0, size=30)
    wet_rain = np.random.normal(loc=3000.0, scale=200.0, size=30)

    df_arid = pd.DataFrame({"SUBDIVISION": ["ARID"] * 30, "YEAR": list(range(1980, 2010)), "ANNUAL_VALIDATED": arid_rain})
    df_wet = pd.DataFrame({"SUBDIVISION": ["WET"] * 30, "YEAR": list(range(1980, 2010)), "ANNUAL_VALIDATED": wet_rain})
    combined = pd.concat([df_arid, df_wet], ignore_index=True)

    # When grouped by subdivision, ARID's 300mm should NOT be flagged as an anomaly
    res = detect_anomalies(combined, method="zscore", subdivision_col="SUBDIVISION", threshold=2.5)
    arid_anom_count = res[res["SUBDIVISION"] == "ARID"]["is_anomaly"].sum()
    assert arid_anom_count <= 2  # At most statistical outliers within ARID, not all 30!


def test_summarize_and_top_anomalies(sample_rainfall_df):
    """Verify summary dictionary and top anomalies retrieval."""
    res_df = detect_anomalies_zscore(sample_rainfall_df, threshold=2.0)
    summary = summarize_anomalies(res_df)

    assert "total_observations" in summary
    assert "anomaly_count" in summary
    assert "anomaly_percentage" in summary
    assert "drought_count" in summary
    assert "excess_count" in summary

    top_droughts, top_floods = get_top_anomalous_years(res_df, n=3)
    assert not top_droughts.empty
    assert not top_floods.empty
