"""
Tests for src/rainfall_forecasting.py - Baseline and ML forecasting pipelines and horizon projection.
"""

import pytest
import numpy as np
import pandas as pd

from src.rainfall_forecasting import (
    RainfallForecastingPipeline,
    compare_all_models_on_subdivision,
)


@pytest.fixture
def subdivision_time_series():
    """Create continuous 60-year rainfall series for a single subdivision."""
    np.random.seed(42)
    rain = np.random.normal(loc=1200.0, scale=150.0, size=60)
    years = list(range(1956, 2016))
    return pd.DataFrame({
        "SUBDIVISION": ["KERALA_TEST"] * 60,
        "YEAR": years,
        "ANNUAL_VALIDATED": rain,
    })


def test_pipeline_temporal_split(subdivision_time_series):
    """Verify strictly out-of-time temporal split (no future leakage)."""
    pipe = RainfallForecastingPipeline(subdivision_name="KERALA_TEST", annual_df=subdivision_time_series)
    pipe.fit_and_compare(subdivision_time_series, test_years=15)

    assert len(pipe.test_df) == 15
    assert pipe.train_df["YEAR"].max() < pipe.test_df["YEAR"].min()


def test_baseline_models_fit(subdivision_time_series):
    """Verify historical mean, naive, and rolling baselines generate predictions."""
    pipe = RainfallForecastingPipeline(subdivision_name="KERALA_TEST", annual_df=subdivision_time_series)
    comp_df = pipe.fit_and_compare(subdivision_time_series, test_years=15)

    model_names = comp_df["model_name"].tolist()
    assert "historical_mean" in model_names
    assert "naive" in model_names
    assert "rolling_mean_5" in model_names

    # Check metrics are valid numbers
    for _, row in comp_df.iterrows():
        assert row["test_mae"] > 0
        assert row["test_rmse"] > 0


def test_ml_models_fit(subdivision_time_series):
    """Verify linear trend and random forest models train and evaluate without error."""
    pipe = RainfallForecastingPipeline(subdivision_name="KERALA_TEST", annual_df=subdivision_time_series)
    comp_df = pipe.fit_and_compare(subdivision_time_series, test_years=15)

    model_names = comp_df["model_name"].tolist()
    assert "linear_trend" in model_names
    assert "random_forest" in model_names


def test_generate_future_forecast_horizon(subdivision_time_series):
    """Verify multi-year future recursive horizon generates predictions and bounds."""
    pipe = RainfallForecastingPipeline(subdivision_name="KERALA_TEST", annual_df=subdivision_time_series)
    pipe.fit_and_compare(subdivision_time_series, test_years=15)

    horizon_res = pipe.generate_future_forecast(model_name="random_forest", horizon_years=5)
    forecasts = horizon_res["forecasts"]

    assert len(forecasts) == 5
    last_year = subdivision_time_series["YEAR"].max()

    for i, fc in enumerate(forecasts):
        assert fc["year"] == last_year + 1 + i
        assert fc["predicted_annual"] > 0
        assert fc["lower_bound"] <= fc["predicted_annual"] <= fc["upper_bound"]


def test_insufficient_data_handling():
    """Verify pipeline returns empty comparison DataFrame gracefully when sample size is insufficient."""
    short_df = pd.DataFrame({
        "SUBDIVISION": ["SHORT_REGION"] * 10,
        "YEAR": list(range(2000, 2010)),
        "ANNUAL_VALIDATED": [1000.0] * 10,
    })
    pipe = RainfallForecastingPipeline(subdivision_name="SHORT_REGION", annual_df=short_df)
    comp_df = pipe.fit_and_compare(short_df)
    assert comp_df.empty
    assert "model_name" in comp_df.columns

