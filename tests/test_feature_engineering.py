"""
Tests for src/feature_engineering.py - Lag, rolling, cyclical, and leakage-free feature matrices.
"""

import pytest
import numpy as np
import pandas as pd

from src.feature_engineering import (
    create_annual_features,
    create_forecasting_feature_matrix,
    create_monthly_cyclical_features,
)


def test_create_monthly_cyclical_features():
    """Verify sine/cosine transformation for monthly cyclical encoding."""
    df = pd.DataFrame({"MONTH_NUMBER": [1, 3, 6, 9, 12]})
    feat_df = create_monthly_cyclical_features(df, month_col="MONTH_NUMBER")

    assert "month_sin" in feat_df.columns
    assert "month_cos" in feat_df.columns

    # Verify bounds [-1.0, 1.0]
    assert feat_df["month_sin"].between(-1.0, 1.0).all()
    assert feat_df["month_cos"].between(-1.0, 1.0).all()

    # Verify cycle property: distance between month 1 and month 12 in sin/cos space is close
    m1 = feat_df.iloc[0][["month_sin", "month_cos"]].values
    m12 = feat_df.iloc[-1][["month_sin", "month_cos"]].values
    dist = np.linalg.norm(m1 - m12)
    assert dist < 0.6  # Dec and Jan are close adjacent months


def test_create_forecasting_feature_matrix_no_leakage():
    """Verify that rolling features are shifted by 1 so the target year is never in the predictors."""
    # Synthetic series where Year N has rainfall N * 100
    df = pd.DataFrame({
        "SUBDIVISION": ["TEST_REGION"] * 10,
        "YEAR": list(range(2000, 2010)),
        "ANNUAL_VALIDATED": [float(i * 100) for i in range(1, 11)],
    })

    feat_df = create_forecasting_feature_matrix(
        df,
        target_col="ANNUAL_VALIDATED",
        lags=[1, 2],
        rolling_windows=[3],
    )

    # For Year 2003 (index 3), value is 400.
    # Prior values are: 2000: 100, 2001: 2002, 2002: 300
    # Lag 1 must be 300 (2002), Lag 2 must be 200 (2001)
    row_2003 = feat_df[feat_df["YEAR"] == 2003].iloc[0]
    assert row_2003["rainfall_lag_1"] == 300.0
    assert row_2003["rainfall_lag_2"] == 200.0

    # Rolling mean 3 prior must be mean(100, 200, 300) = 200.0, NOT including 400!
    assert np.isclose(row_2003["rolling_mean_3"], 200.0)


def test_create_forecasting_feature_matrix_gap_handling():
    """Verify that lag features properly evaluate to NaN across multi-year temporal gaps."""
    df_gap = pd.DataFrame({
        "SUBDIVISION": ["ISLANDS"] * 4,
        "YEAR": [1940, 1941, 1946, 1947],
        "ANNUAL_VALIDATED": [1500.0, 1600.0, 1400.0, 1550.0],
    })

    feat_df = create_forecasting_feature_matrix(df_gap, lags=[1], rolling_windows=[3])

    # 1946 should NOT have 1941 as its 1-year lag because 1946 - 1941 = 5 != 1
    row_1946 = feat_df[feat_df["YEAR"] == 1946].iloc[0]
    assert pd.isna(row_1946["rainfall_lag_1"])

    # 1947 should have 1946 as its 1-year lag (1947 - 1946 = 1)
    row_1947 = feat_df[feat_df["YEAR"] == 1947].iloc[0]
    assert row_1947["rainfall_lag_1"] == 1400.0
