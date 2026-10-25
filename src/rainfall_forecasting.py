"""
MonsoonLens AI - Rainfall Time-Series Forecasting Module.

Provides defensible statistical and machine-learning forecasting on historical annual rainfall.
Implements:
1. Baselines:
   - Climatological Mean Baseline
   - Naive (Persistence) Baseline
   - Rolling Window Average Baseline
2. Predictive Models:
   - Linear Trend Forecaster
   - Random Forest Regressor (with lag and rolling features)
3. Multi-Step Horizon Projections with empirical residual uncertainty intervals.

Strict Methodological Rules:
- Enforces chronological train/test splitting (NO random shuffling)
- Prevents feature lookahead leakage (all rolling features backward-looking)
- Clearly communicates assumptions, bounds, and limitations.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_TRAIN_END_YEAR,
    MIN_OBSERVATION_COUNT,
    RANDOM_SEED,
)
from src.feature_engineering import create_forecasting_feature_matrix
from src.model_evaluation import evaluate_forecast


class RainfallForecastingPipeline:
    """
    Chronological forecasting and benchmark evaluation pipeline
    for subdivisional annual rainfall series.
    """

    def __init__(
        self,
        subdivision: Optional[str] = None,
        annual_df: Optional[pd.DataFrame] = None,
        value_col: str = "ANNUAL_VALIDATED",
        train_end_year: int = DEFAULT_TRAIN_END_YEAR,
        random_state: int = RANDOM_SEED,
        subdivision_name: Optional[str] = None,
    ):
        sub_name = subdivision or subdivision_name or "ALL_INDIA"
        self.subdivision = sub_name
        self.value_col = value_col
        self.train_end_year = train_end_year
        self.random_state = random_state

        if annual_df is None:
            from src.data_loader import load_annual_rainfall
            annual_df = load_annual_rainfall()

        if "SUBDIVISION" in annual_df.columns and sub_name in annual_df["SUBDIVISION"].values:
            sub_df = annual_df[annual_df["SUBDIVISION"] == sub_name].sort_values("YEAR").copy()
        else:
            sub_df = annual_df.sort_values("YEAR").copy()

        self.raw_data = sub_df[sub_df[value_col].notnull()].reset_index(drop=True)

        # Generate feature matrix
        self.feature_df, self.feature_cols = create_forecasting_feature_matrix(
            self.raw_data,
            subdivision=sub_name,
            value_col=value_col,
            lags=[1, 2, 3],
            rolling_windows=[3, 5],
            include_time_trend=True,
        )

        # Drop rows where lag features are NaN due to initial boundary or historical gaps
        self.model_data = self.feature_df.dropna(subset=self.feature_cols + ["TARGET"]).reset_index(drop=True)

        self._train_df = self.model_data[self.model_data["YEAR"] <= self.train_end_year].copy()
        self._test_df = self.model_data[self.model_data["YEAR"] > self.train_end_year].copy()

    @property
    def train_df(self) -> pd.DataFrame:
        """Training split dataframe."""
        return self._train_df

    @property
    def test_df(self) -> pd.DataFrame:
        """Testing split dataframe."""
        return self._test_df

    @property
    def has_sufficient_data(self) -> bool:
        """Verify minimum sample sizes for train and test splits."""
        return len(self._train_df) >= 20 and len(self._test_df) >= 5

    def get_split_summary(self) -> Dict[str, Any]:
        """Return information about train and test splits."""
        return {
            "subdivision": self.subdivision,
            "train_observations": len(self._train_df),
            "train_years": (int(self._train_df["YEAR"].min()), int(self._train_df["YEAR"].max())) if not self._train_df.empty else None,
            "test_observations": len(self._test_df),
            "test_years": (int(self._test_df["YEAR"].min()), int(self._test_df["YEAR"].max())) if not self._test_df.empty else None,
            "features_used": self.feature_cols,
        }

    # -------------------------------------------------------------------------
    # Baseline Evaluators
    # -------------------------------------------------------------------------
    def evaluate_climatological_mean_baseline(self) -> Dict[str, Any]:
        """Predicts historical training mean for all test years."""
        train_mean = float(self._train_df["TARGET"].mean())
        y_true = self._test_df["TARGET"].values
        y_pred = np.full(len(y_true), train_mean)

        eval_res = evaluate_forecast(
            y_true, y_pred,
            model_name="Climatological Mean Baseline",
            baseline_rmse=None,
            evaluation_years=(int(self._test_df["YEAR"].min()), int(self._test_df["YEAR"].max()))
        )
        eval_res["predictions"] = y_pred
        eval_res["years"] = self._test_df["YEAR"].values
        return eval_res

    def evaluate_naive_persistence_baseline(self) -> Dict[str, Any]:
        """Predicts latest observed value (LAG_1)."""
        y_true = self._test_df["TARGET"].values
        y_pred = self._test_df["LAG_1"].values

        eval_res = evaluate_forecast(
            y_true, y_pred,
            model_name="Naive Persistence Baseline",
            baseline_rmse=None,
            evaluation_years=(int(self._test_df["YEAR"].min()), int(self._test_df["YEAR"].max()))
        )
        eval_res["predictions"] = y_pred
        eval_res["years"] = self._test_df["YEAR"].values
        return eval_res

    def evaluate_rolling_mean_baseline(self, window: int = 5) -> Dict[str, Any]:
        """Predicts prior rolling mean."""
        col_name = f"ROLLING_MEAN_{window}Y_PRIOR"
        if col_name not in self._test_df.columns:
            col_name = "ROLLING_MEAN_3Y_PRIOR"

        y_true = self._test_df["TARGET"].values
        y_pred = self._test_df[col_name].values

        eval_res = evaluate_forecast(
            y_true, y_pred,
            model_name=f"Rolling {window}-Year Mean Baseline",
            baseline_rmse=None,
            evaluation_years=(int(self._test_df["YEAR"].min()), int(self._test_df["YEAR"].max()))
        )
        eval_res["predictions"] = y_pred
        eval_res["years"] = self._test_df["YEAR"].values
        return eval_res

    # -------------------------------------------------------------------------
    # Predictive Models
    # -------------------------------------------------------------------------
    def evaluate_linear_trend_model(self) -> Dict[str, Any]:
        """Linear regression model using time step trend."""
        X_train = self._train_df[["TIME_STEP"]]
        y_train = self._train_df["TARGET"]
        X_test = self._test_df[["TIME_STEP"]]
        y_true = self._test_df["TARGET"].values

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        eval_res = evaluate_forecast(
            y_true, y_pred,
            model_name="Linear Trend Forecaster",
            evaluation_years=(int(self._test_df["YEAR"].min()), int(self._test_df["YEAR"].max()))
        )
        eval_res["model_obj"] = model
        eval_res["predictions"] = y_pred
        eval_res["years"] = self._test_df["YEAR"].values
        return eval_res

    def evaluate_random_forest_model(self) -> Dict[str, Any]:
        """Random Forest Regressor trained on lag, rolling, and baseline features."""
        X_train = self._train_df[self.feature_cols]
        y_train = self._train_df["TARGET"]
        X_test = self._test_df[self.feature_cols]
        y_true = self._test_df["TARGET"].values

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=4,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        eval_res = evaluate_forecast(
            y_true, y_pred,
            model_name="Random Forest Regressor",
            evaluation_years=(int(self._test_df["YEAR"].min()), int(self._test_df["YEAR"].max()))
        )
        eval_res["model_obj"] = model
        eval_res["predictions"] = y_pred
        eval_res["years"] = self._test_df["YEAR"].values
        eval_res["feature_importances"] = dict(zip(self.feature_cols, model.feature_importances_.round(4)))
        return eval_res

    # -------------------------------------------------------------------------
    # Comprehensive Evaluation Benchmark
    # -------------------------------------------------------------------------
    def evaluate_all_models(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Train and evaluate all models and baselines against the test period.
        Calculates skill scores relative to the Climatological Mean Baseline.
        """
        if not self.has_sufficient_data:
            return pd.DataFrame(), {"error": "Insufficient historical data for train/test split"}

        # 1. Base Climatological Mean
        base_res = self.evaluate_climatological_mean_baseline()
        base_rmse = base_res["rmse_mm"]
        base_res["skill_score_vs_baseline"] = 0.0

        # 2. Naive Persistence
        naive_res = self.evaluate_naive_persistence_baseline()
        naive_res["skill_score_vs_baseline"] = round(1.0 - (naive_res["rmse_mm"] / base_rmse), 4) if base_rmse > 0 else 0.0

        # 3. Rolling Mean
        rolling_res = self.evaluate_rolling_mean_baseline(window=5)
        rolling_res["skill_score_vs_baseline"] = round(1.0 - (rolling_res["rmse_mm"] / base_rmse), 4) if base_rmse > 0 else 0.0

        # 4. Linear Trend
        linear_res = self.evaluate_linear_trend_model()
        linear_res["skill_score_vs_baseline"] = round(1.0 - (linear_res["rmse_mm"] / base_rmse), 4) if base_rmse > 0 else 0.0

        # 5. Random Forest
        rf_res = self.evaluate_random_forest_model()
        rf_res["skill_score_vs_baseline"] = round(1.0 - (rf_res["rmse_mm"] / base_rmse), 4) if base_rmse > 0 else 0.0

        all_models = [base_res, naive_res, rolling_res, linear_res, rf_res]
        summary_records = []
        for m in all_models:
            summary_records.append({
                "Model": m["model_name"],
                "MAE (mm)": m["mae_mm"],
                "RMSE (mm)": m["rmse_mm"],
                "MAPE (%)": m["mape_pct"],
                "R² Score": m["r2_score"],
                "Skill Score": m["skill_score_vs_baseline"],
            })

        summary_df = pd.DataFrame(summary_records).sort_values("RMSE (mm)", ascending=True).reset_index(drop=True)
        return summary_df, {m["model_name"]: m for m in all_models}

    # -------------------------------------------------------------------------
    # Future Projections
    # -------------------------------------------------------------------------
    def generate_future_forecast(
        self,
        horizon: int = DEFAULT_FORECAST_HORIZON,
        model_type: str = "random_forest",
        model_name: Optional[str] = None,
        horizon_years: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate future rainfall estimates for requested horizon (e.g. next 5-10 years)
        beyond the latest historical observation in the dataset (2015).
        
        Includes empirical prediction uncertainty intervals (± 1.96 * residual_RMSE).
        """
        m_type = (model_name or model_type).lower()
        h = horizon_years if horizon_years is not None else horizon

        latest_year = int(self.raw_data["YEAR"].max())
        future_years = [latest_year + i for i in range(1, h + 1)]

        # Fit model on 100% of available historical observations
        X_full = self.model_data[self.feature_cols]
        y_full = self.model_data["TARGET"]

        if m_type == "linear_trend":
            X_time = self.model_data[["TIME_STEP"]]
            model = LinearRegression().fit(X_time, y_full)
            residuals = y_full - model.predict(X_time)
            res_std = float(np.std(residuals))

            min_year = int(self.model_data["YEAR"].min())
            future_steps = pd.DataFrame({"TIME_STEP": [yr - min_year for yr in future_years]})
            preds = model.predict(future_steps)

        else:
            # Random forest projection using recursive lag updating
            model = RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_split=4, random_state=self.random_state
            )
            model.fit(X_full, y_full)
            residuals = y_full - model.predict(X_full)
            res_std = float(np.std(residuals))

            # Start from latest historical row
            last_row = self.model_data.iloc[-1].copy()
            preds = []
            current_lags = [last_row["TARGET"], last_row["LAG_1"], last_row["LAG_2"]]
            min_year = int(self.model_data["YEAR"].min())

            for yr in future_years:
                row_dict = {
                    "LAG_1": current_lags[0],
                    "LAG_2": current_lags[1],
                    "LAG_3": current_lags[2],
                    "LAG_DIFF_1_2": current_lags[0] - current_lags[1],
                    "ROLLING_MEAN_3Y_PRIOR": np.mean(current_lags[:3]),
                    "ROLLING_STD_3Y_PRIOR": np.std(current_lags[:3]),
                    "ROLLING_MEAN_5Y_PRIOR": last_row.get("ROLLING_MEAN_5Y_PRIOR", current_lags[0]),
                    "ROLLING_STD_5Y_PRIOR": last_row.get("ROLLING_STD_5Y_PRIOR", 10.0),
                    "BASELINE_MEAN": last_row["BASELINE_MEAN"],
                    "PRIOR_BASELINE_ANOMALY": current_lags[0] - last_row["BASELINE_MEAN"],
                    "TIME_STEP": float(yr - min_year),
                }
                x_feat = pd.DataFrame([row_dict])[self.feature_cols]
                p_val = float(model.predict(x_feat)[0])
                preds.append(p_val)
                # Shift lags
                current_lags = [p_val, current_lags[0], current_lags[1]]

            preds = np.array(preds)

        # 95% Empirical Prediction Interval based on residual spread
        lower_bound = np.maximum(0.0, preds - 1.96 * res_std)
        upper_bound = preds + 1.96 * res_std

        forecast_records = []
        for yr, p, lb, ub in zip(future_years, preds, lower_bound, upper_bound):
            forecast_records.append({
                "year": yr,
                "predicted_annual": round(float(p), 2),
                "lower_bound": round(float(lb), 2),
                "upper_bound": round(float(ub), 2),
            })

        return {
            "subdivision": self.subdivision,
            "model_type": m_type,
            "horizon": h,
            "forecast_years": future_years,
            "forecasts": forecast_records,
            "predicted_rainfall_mm": [round(float(p), 2) for p in preds],
            "prediction_interval_lower_mm": [round(float(l), 2) for l in lower_bound],
            "prediction_interval_upper_mm": [round(float(u), 2) for u in upper_bound],
            "residual_std_mm": round(res_std, 2),
            "disclaimer": (
                "These projections represent statistical pattern continuations based strictly on historical "
                "subdivisional data. They do not constitute operational seasonal weather predictions or deterministic forecasts."
            ),
        }

    def fit_and_compare(
        self,
        sub_df: Optional[pd.DataFrame] = None,
        test_years: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fit and evaluate all models, supporting custom sub_df and test_years parameters.
        Returns standardized comparison DataFrame with lowercase metric columns.
        """
        if sub_df is not None:
            self.raw_data = sub_df[sub_df[self.value_col].notnull()].reset_index(drop=True)
            self.feature_df, self.feature_cols = create_forecasting_feature_matrix(
                self.raw_data,
                subdivision=self.subdivision,
                value_col=self.value_col,
                lags=[1, 2, 3],
                rolling_windows=[3, 5],
                include_time_trend=True,
            )
            self.model_data = self.feature_df.dropna(subset=self.feature_cols + ["TARGET"]).reset_index(drop=True)

        if test_years is not None and not self.model_data.empty:
            max_yr = int(self.model_data["YEAR"].max())
            self.train_end_year = max_yr - test_years
            self._train_df = self.model_data[self.model_data["YEAR"] <= self.train_end_year].copy()
            self._test_df = self.model_data[self.model_data["YEAR"] > self.train_end_year].copy()

        summary_df, _ = self.evaluate_all_models()
        if summary_df.empty or "Model" not in summary_df.columns:
            return pd.DataFrame(columns=["model_name", "test_mae", "test_rmse", "test_mape", "test_r2", "forecast_skill_score"])

        # Add normalized column aliases
        model_name_map = {
            "Climatological Mean Baseline": "historical_mean",
            "Historical Mean": "historical_mean",
            "Naive Persistence Baseline": "naive",
            "Naive": "naive",
            "5-Year Trailing Rolling Mean Baseline": "rolling_mean_5",
            "Rolling 5-Year Mean Baseline": "rolling_mean_5",
            "Linear Trend Regression": "linear_trend",
            "Linear Trend Forecaster": "linear_trend",
            "Random Forest Regressor": "random_forest",
        }
        summary_df["model_name"] = summary_df["Model"].map(model_name_map).fillna(summary_df["Model"])
        summary_df["test_mae"] = summary_df["MAE (mm)"]
        summary_df["test_rmse"] = summary_df["RMSE (mm)"]
        summary_df["test_mape"] = summary_df["MAPE (%)"]
        summary_df["test_r2"] = summary_df["R² Score"]
        summary_df["forecast_skill_score"] = summary_df["Skill Score"] * 100.0
        return summary_df


def compare_all_models_on_subdivision(
    annual_df: pd.DataFrame,
    subdivision: str,
    train_end_year: int = DEFAULT_TRAIN_END_YEAR,
) -> pd.DataFrame:
    """Benchmark all models on a given subdivision and return comparison DataFrame."""
    pipe = RainfallForecastingPipeline(subdivision=subdivision, annual_df=annual_df, train_end_year=train_end_year)
    return pipe.fit_and_compare()
