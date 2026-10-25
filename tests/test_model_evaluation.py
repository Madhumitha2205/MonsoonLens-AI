"""
Tests for src/model_evaluation.py - MAE, RMSE, MAPE, R2, Forecast Skill Score, and model comparison.
"""

import pytest
import numpy as np
import pandas as pd

from src.model_evaluation import (
    calculate_forecast_skill_score,
    calculate_mae,
    calculate_mape,
    calculate_r2,
    calculate_rmse,
    compare_models,
    evaluate_forecast,
)


def test_error_metrics_calculation():
    """Verify standard regression evaluation metrics on known numerical arrays."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])

    mae = calculate_mae(y_true, y_pred)
    assert np.isclose(mae, 10.0)

    rmse = calculate_rmse(y_true, y_pred)
    assert np.isclose(rmse, 10.0)

    mape = calculate_mape(y_true, y_pred)
    # Errors: 10/100 (10%), 10/200 (5%), 10/300 (3.333%), 10/400 (2.5%) -> mean = 5.208%
    assert 5.0 < mape < 5.5

    r2 = calculate_r2(y_true, y_pred)
    assert r2 > 0.95


def test_forecast_skill_score():
    """Verify Forecast Skill Score behavior relative to reference benchmark."""
    # Perfect model
    perf_skill = calculate_forecast_skill_score(rmse_model=0.0, rmse_reference=100.0)
    assert np.isclose(perf_skill, 1.0) or np.isclose(perf_skill, 100.0)

    # Identical to reference
    ref_skill = calculate_forecast_skill_score(rmse_model=100.0, rmse_reference=100.0)
    assert np.isclose(ref_skill, 0.0)

    # Worse than reference
    worse_skill = calculate_forecast_skill_score(rmse_model=150.0, rmse_reference=100.0)
    assert np.isclose(worse_skill, -0.5) or np.isclose(worse_skill, -50.0)

    # Kwargs alias support (rmse_baseline, benchmark_rmse)
    kw_skill = calculate_forecast_skill_score(rmse_model=80.0, rmse_baseline=100.0)
    assert np.isclose(kw_skill, 0.2)



def test_evaluate_forecast():
    """Verify evaluate_forecast returns complete dictionary of formatted metrics."""
    y_true = np.array([1000.0, 1200.0, 1100.0])
    y_pred = np.array([1020.0, 1180.0, 1120.0])

    res = evaluate_forecast(y_true, y_pred, model_name="TestModel")
    assert res["model_name"] == "TestModel"
    assert "mae" in res
    assert "rmse" in res
    assert "mape" in res
    assert "r2" in res
    assert res["n_eval"] == 3


def test_compare_models():
    """Verify compare_models generates sorted comparison dataframe with skill scores."""
    results = [
        {"model_name": "historical_mean", "train_mae": 100.0, "test_mae": 120.0, "test_rmse": 150.0, "test_mape": 10.0, "test_r2": 0.0},
        {"model_name": "random_forest", "train_mae": 50.0, "test_mae": 90.0, "test_rmse": 120.0, "test_mape": 7.5, "test_r2": 0.35},
        {"model_name": "naive", "train_mae": 110.0, "test_mae": 140.0, "test_rmse": 180.0, "test_mape": 12.0, "test_r2": -0.2},
    ]

    comp_df = compare_models(results, benchmark_name="historical_mean")
    assert "forecast_skill_score" in comp_df.columns

    # Random forest has RMSE 120 vs reference 150 -> Skill = (1 - 120/150)*100 = 20.0%
    rf_row = comp_df[comp_df["model_name"] == "random_forest"].iloc[0]
    assert np.isclose(rf_row["forecast_skill_score"], 20.0)
