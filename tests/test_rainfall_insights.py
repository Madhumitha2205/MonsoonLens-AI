"""
Tests for src/rainfall_insights.py - Narrative generators and climate profile summaries.
"""

import pytest
import numpy as np
import pandas as pd

from src.rainfall_insights import (
    generate_anomaly_narrative,
    generate_change_point_narrative,
    generate_forecasting_evaluation_narrative,
    generate_subdivision_climate_profile,
)


@pytest.fixture
def mock_sub_df():
    """Create sample subdivision series."""
    return pd.DataFrame({
        "SUBDIVISION": ["TEST_SUB"] * 20,
        "YEAR": list(range(1990, 2010)),
        "ANNUAL_VALIDATED": [1000.0 + i * 10 for i in range(20)],
    })


def test_generate_subdivision_climate_profile(mock_sub_df):
    """Verify climate profile statistical dictionary creation."""
    prof = generate_subdivision_climate_profile(mock_sub_df, "TEST_SUB")
    assert prof["subdivision"] == "TEST_SUB"
    assert prof["mean"] > 1000.0
    assert prof["wettest_year"] == 2009
    assert prof["driest_year"] == 1990
    assert prof["trend_slope"] > 0


def test_generate_anomaly_narrative():
    """Verify natural language narrative for anomaly detection."""
    df = pd.DataFrame({
        "YEAR": [1970, 1980, 1990],
        "ANNUAL_VALIDATED": [500.0, 1200.0, 1800.0],
        "is_anomaly": [True, False, True],
        "anomaly_type": ["Severe Drought", "Normal", "Extreme Wet"],
    })

    text = generate_anomaly_narrative(df, "TEST_SUB", method="zscore")
    assert isinstance(text, str)
    assert "TEST_SUB" in text
    assert "2 anomalies" in text or "drought" in text.lower()


def test_generate_change_point_narrative():
    """Verify natural language explanation for detected change points."""
    cp_res = {
        "change_point_year": 1979,
        "is_significant": True,
        "p_value": 0.003,
        "mean_before": 1200.0,
        "mean_after": 1050.0,
        "difference": -150.0,
        "pct_change": -12.5,
    }

    text = generate_change_point_narrative(cp_res, "TEST_SUB")
    assert "1979" in text
    assert "statistically significant" in text
    assert "12.5%" in text or "150.0" in text


def test_generate_forecasting_evaluation_narrative():
    """Verify narrative synthesis from model comparison dataframe."""
    comp_df = pd.DataFrame([
        {"model_name": "historical_mean", "test_mae": 150.0, "test_rmse": 180.0, "forecast_skill_score": 0.0},
        {"model_name": "random_forest", "test_mae": 120.0, "test_rmse": 140.0, "forecast_skill_score": 22.2},
    ])

    text = generate_forecasting_evaluation_narrative(comp_df, "TEST_SUB")
    assert "TEST_SUB" in text
    assert "random_forest" in text
    assert "skill" in text.lower()
