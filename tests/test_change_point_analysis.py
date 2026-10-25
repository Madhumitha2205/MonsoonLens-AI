"""
Tests for src/change_point_analysis.py - Pettitt's test, CUSUM, SNHT, and regime shift analyzers.
"""

import pytest
import numpy as np
import pandas as pd

from src.change_point_analysis import (
    analyze_all_subdivisions_change_points,
    analyze_subdivision_change_point,
    cusum_test,
    pettitt_test,
    snht_test,
)


@pytest.fixture
def synthetic_step_change():
    """Create synthetic time series with a distinct step change at index 30."""
    np.random.seed(42)
    # 30 years at mean 600 mm, followed by 30 years at mean 900 mm
    regime1 = np.random.normal(loc=600.0, scale=40.0, size=30)
    regime2 = np.random.normal(loc=900.0, scale=40.0, size=30)
    data = np.concatenate([regime1, regime2])
    years = list(range(1950, 2010))

    df = pd.DataFrame({
        "SUBDIVISION": ["STEP_REGION"] * 60,
        "YEAR": years,
        "ANNUAL_VALIDATED": data,
    })
    return df, data


def test_pettitt_test_detects_step_change(synthetic_step_change):
    """Verify Pettitt's rank test correctly detects synthetic regime transition at index ~30."""
    _, data = synthetic_step_change
    res = pettitt_test(data, alpha=0.05)

    assert res["change_point_index"] is not None
    assert abs(res["change_point_index"] - 30) <= 2
    assert res["p_value"] < 0.01
    assert res["is_significant"] == True
    assert res["mean_after"] > res["mean_before"]


def test_cusum_test_detects_step_change(synthetic_step_change):
    """Verify CUSUM cumulative deviations test detects step change."""
    _, data = synthetic_step_change
    res = cusum_test(data, alpha=0.05)

    assert res["change_point_index"] is not None
    assert abs(res["change_point_index"] - 30) <= 3
    assert res["is_significant"] == True


def test_snht_test_detects_step_change(synthetic_step_change):
    """Verify SNHT test detects step change."""
    _, data = synthetic_step_change
    res = snht_test(data, alpha=0.05)

    assert res["change_point_index"] is not None
    assert abs(res["change_point_index"] - 30) <= 3
    assert res["is_significant"] == True


def test_pettitt_test_no_change():
    """Verify stationary noise does not falsely trigger a strong significant change point."""
    np.random.seed(123)
    stationary = np.random.normal(loc=1000.0, scale=50.0, size=50)
    res = pettitt_test(stationary, alpha=0.01)
    # Either not significant or p_value >= 0.01
    assert (not res["is_significant"]) or (res["p_value"] >= 0.01)


def test_analyze_subdivision_change_point(synthetic_step_change):
    """Verify integration analyzer maps change point index to exact calendar year."""
    df, _ = synthetic_step_change
    res = analyze_subdivision_change_point(df, method="pettitt", value_col="ANNUAL_VALIDATED")

    assert res["change_point_year"] in [1979, 1980, 1981]
    assert res["is_significant"] == True
    assert "mean_before" in res and "mean_after" in res
    assert res["difference"] > 0
