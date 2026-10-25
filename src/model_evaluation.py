"""
MonsoonLens AI - Model Evaluation and Comparison Module.

Provides reproducible evaluation metrics, benchmark comparisons, and skill scores
for time-series forecasting models.
Guarantees:
- Safe handling of zero denominators (MAPE)
- Zero division protection for R²
- Dimension mismatch checks
- Transparent comparison against climatological baselines.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true: Union[List[float], np.ndarray, pd.Series], y_pred: Union[List[float], np.ndarray, pd.Series]) -> float:
    """Calculate Mean Absolute Error (MAE) in millimeters."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    _validate_inputs(yt, yp)
    return float(mean_absolute_error(yt, yp))


def calculate_rmse(y_true: Union[List[float], np.ndarray, pd.Series], y_pred: Union[List[float], np.ndarray, pd.Series]) -> float:
    """Calculate Root Mean Squared Error (RMSE) in millimeters."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    _validate_inputs(yt, yp)
    return float(np.sqrt(mean_squared_error(yt, yp)))


def calculate_mape(
    y_true: Union[List[float], np.ndarray, pd.Series],
    y_pred: Union[List[float], np.ndarray, pd.Series],
    epsilon: float = 1e-4
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE) in %.
    Safely filters out observations where y_true == 0 to prevent division by zero.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    _validate_inputs(yt, yp)

    valid_mask = np.abs(yt) > epsilon
    if not np.any(valid_mask):
        return np.nan

    abs_pct_errors = np.abs((yt[valid_mask] - yp[valid_mask]) / yt[valid_mask]) * 100.0
    return float(np.mean(abs_pct_errors))


def calculate_r2(y_true: Union[List[float], np.ndarray, pd.Series], y_pred: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Calculate Coefficient of Determination (R²).
    Returns NaN if y_true has zero variance.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    _validate_inputs(yt, yp)

    if np.var(yt) < 1e-6:
        return np.nan

    return float(r2_score(yt, yp))


def calculate_forecast_skill_score(
    model_rmse: Optional[float] = None,
    baseline_rmse: Optional[float] = None,
    **kwargs
) -> float:
    """
    Calculate forecast skill score relative to a reference baseline:
    Skill Score = 1 - (RMSE_model / RMSE_baseline)
    - Positive score: Model outperforms baseline
    - Zero: Model performance matches baseline
    - Negative: Model performs worse than baseline
    """
    m_rmse = model_rmse if model_rmse is not None else kwargs.get("rmse_model", 0.0)
    b_rmse = (
        baseline_rmse
        if baseline_rmse is not None
        else kwargs.get("rmse_baseline", kwargs.get("rmse_reference", kwargs.get("benchmark_rmse", 1.0)))
    )
    if b_rmse < 1e-6:
        return 0.0
    return round(float(1.0 - (m_rmse / b_rmse)), 4)


def evaluate_forecast(
    y_true: Union[List[float], np.ndarray, pd.Series],
    y_pred: Union[List[float], np.ndarray, pd.Series],
    model_name: str = "Model",
    baseline_rmse: Optional[float] = None,
    evaluation_years: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Calculate full evaluation profile for a model's predictions.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    _validate_inputs(yt, yp)

    mae_val = calculate_mae(yt, yp)
    rmse_val = calculate_rmse(yt, yp)
    mape_val = calculate_mape(yt, yp)
    r2_val = calculate_r2(yt, yp)

    skill_score = None
    if baseline_rmse is not None:
        skill_score = calculate_forecast_skill_score(rmse_val, baseline_rmse)

    return {
        "model_name": model_name,
        "n_eval_observations": len(yt),
        "n_eval": len(yt),
        "evaluation_period": evaluation_years,
        "mae_mm": round(mae_val, 2),
        "rmse_mm": round(rmse_val, 2),
        "mape_pct": round(mape_val, 2) if not np.isnan(mape_val) else None,
        "r2_score": round(r2_val, 4) if not np.isnan(r2_val) else None,
        "mae": round(mae_val, 2),
        "rmse": round(rmse_val, 2),
        "mape": round(mape_val, 2) if not np.isnan(mape_val) else None,
        "r2": round(r2_val, 4) if not np.isnan(r2_val) else None,
        "skill_score_vs_baseline": skill_score,
    }


def compare_models(
    results_list: List[Dict[str, Any]],
    baseline_model_name: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Compile multiple model evaluation results into a sorted comparative DataFrame.
    """
    base_name = baseline_model_name or kwargs.get("benchmark_name", "historical_mean")
    if not results_list:
        return pd.DataFrame()

    df = pd.DataFrame(results_list)
    # If forecast_skill_score not present, compute it relative to base_name
    if "forecast_skill_score" not in df.columns:
        rmse_col = "rmse_mm" if "rmse_mm" in df.columns else ("test_rmse" if "test_rmse" in df.columns else None)
        name_col = "model_name" if "model_name" in df.columns else "Model"
        if rmse_col and name_col in df.columns:
            base_rows = df[df[name_col] == base_name]
            if not base_rows.empty:
                b_rmse = float(base_rows.iloc[0][rmse_col])
                df["forecast_skill_score"] = ((1.0 - (df[rmse_col] / b_rmse)) * 100.0).round(2)

    if "rmse_mm" in df.columns:
        df = df.sort_values(by="rmse_mm", ascending=True).reset_index(drop=True)
    elif "test_rmse" in df.columns:
        df = df.sort_values(by="test_rmse", ascending=True).reset_index(drop=True)
    return df


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Internal validator for input arrays."""
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Cannot evaluate predictions on empty arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, but y_pred has {len(y_pred)} elements."
        )
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Evaluation arrays contain NaN values. Clean predictions before evaluation.")
