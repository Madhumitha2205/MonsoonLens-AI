# MonsoonLens AI: Backend System Architecture Report

## 1. Architectural Philosophy
The MonsoonLens AI backend is designed with a layered, decoupled architecture:
1. **Data Access Layer (`src/data_loader.py`):** Schema enforcement, non-mutating loading, and caching.
2. **Feature Engineering Layer (`src/feature_engineering.py`):** Leakage-free lag and rolling feature extraction.
3. **Analytics Engines:**
   - `src/anomaly_detection.py`: Z-Score, IQR, and Isolation Forest.
   - `src/change_point_analysis.py`: Pettitt's test, CUSUM, SNHT.
   - `src/rainfall_forecasting.py`: Chronological forecasting and baselines.
   - `src/model_evaluation.py`: Mathematical metrics and skill score benchmarks.
   - `src/rainfall_insights.py`: Explainable narrative generator.
4. **Orchestrator (`src/backend_pipeline.py`):** Unified facade consumed by the Streamlit frontend.
5. **Presentation Layer (`pages/`, `app.py`):** Interactive multi-page dashboard.

---

## 2. Data Flow Diagram
```
Raw CSV -> Preprocessing -> Processed CSVs -> Data Loader -> Backend Pipeline -> Streamlit UI
                                                                     |
                                             +-----------------------+-----------------------+
                                             |                       |                       |
                                      Anomaly Detection      Change-Point Analysis       Forecasting
```
