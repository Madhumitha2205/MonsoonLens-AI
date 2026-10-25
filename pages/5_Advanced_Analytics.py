"""
MonsoonLens AI - Advanced Analytics: Anomaly Detection, Change-Point Analysis & Forecasting.

Provides interactive, end-to-end access to:
1. Multi-method statistical and ML anomaly detection (IQR, Z-Score, Isolation Forest).
2. Non-parametric climate regime shift and change-point analysis (Pettitt, CUSUM, SNHT).
3. Leakage-free baseline and machine learning forecasting models with horizon projections.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import (
    detect_anomalies,
    detect_anomalies_iqr,
    detect_anomalies_isolation_forest,
    detect_anomalies_zscore,
    get_top_anomalous_years,
    summarize_anomalies,
)
from src.change_point_analysis import (
    analyze_all_subdivisions_change_points,
    analyze_subdivision_change_point,
    cusum_test,
    pettitt_test,
    snht_test,
)
from src.dashboard_utils import (
    COLOR_AMBER_ACCENT,
    COLOR_CLOUD_WHITE,
    COLOR_CYAN_ACCENT,
    COLOR_GREEN_ACCENT,
    COLOR_MUTED_SLATE,
    COLOR_NAVY_CARD,
    COLOR_PRIMARY_BLUE,
    COLOR_ROSE_ACCENT,
    apply_custom_css,
    create_anomaly_scatter_plot,
    create_change_point_plot,
    create_forecast_projection_plot,
    get_csv_download_button,
    get_unique_subdivisions,
    load_annual_rainfall,
    render_header,
    render_sidebar_branding,
)
from src.rainfall_forecasting import (
    RainfallForecastingPipeline,
    compare_all_models_on_subdivision,
)
from src.rainfall_insights import (
    generate_anomaly_narrative,
    generate_change_point_narrative,
    generate_forecasting_evaluation_narrative,
)

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced Analytics - MonsoonLens AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar_branding()

render_header(
    title="Advanced Climate Analytics & Machine Learning",
    subtitle="Detect historical rainfall anomalies, uncover structural regime shifts, and benchmark predictive forecasting models across India.",
    icon="⚡",
)

# -----------------------------------------------------------------------------
# Load Base Data
# -----------------------------------------------------------------------------
try:
    annual_df = load_annual_rainfall()
except Exception as e:
    st.error(f"⚠️ Error loading annual rainfall dataset: {e}")
    st.stop()

all_subdivisions = get_unique_subdivisions(annual_df)

# -----------------------------------------------------------------------------
# Tabs Interface
# -----------------------------------------------------------------------------
tab_anomaly, tab_changepoint, tab_forecasting = st.tabs([
    "🚨 Anomaly Detection",
    "📈 Regime Shifts & Change Points",
    "🔮 Forecasting & Projections",
])

# =============================================================================
# TAB 1: ANOMALY DETECTION
# =============================================================================
with tab_anomaly:
    st.markdown("### 🚨 Historical Rainfall Anomaly Detection")
    st.markdown(
        "Detect unusually dry (meteorological drought) and unusually wet (excess rainfall) years using "
        "robust parametric, non-parametric, and machine learning methods."
    )

    acol1, acol2, acol3 = st.columns([1.5, 1.2, 1.3])
    with acol1:
        anomaly_sub = st.selectbox(
            "Select Subdivision for Anomaly Analysis",
            options=["All India (All Subdivisions)"] + all_subdivisions,
            index=0,
            key="anomaly_sub_select",
        )
    with acol2:
        anomaly_method = st.selectbox(
            "Detection Method",
            options=["Z-Score (Standard Deviation)", "IQR (Interquartile Range)", "Isolation Forest (Machine Learning)"],
            index=0,
            key="anomaly_method_select",
        )
    with acol3:
        if "Z-Score" in anomaly_method:
            z_thresh = st.slider("Z-Score Threshold (σ)", min_value=1.5, max_value=3.5, value=2.0, step=0.1)
            method_key = "zscore"
            method_kwargs = {"threshold": z_thresh}
        elif "IQR" in anomaly_method:
            iqr_factor = st.slider("IQR Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
            method_key = "iqr"
            method_kwargs = {"factor": iqr_factor}
        else:
            contam = st.slider("Contamination Rate", min_value=0.01, max_value=0.15, value=0.05, step=0.01)
            method_key = "isolation_forest"
            method_kwargs = {"contamination": contam, "random_state": 42}

    # Filter data
    if anomaly_sub == "All India (All Subdivisions)":
        target_df = annual_df.copy()
        # Run detection per subdivision to respect regional climate baselines
        detected_df = detect_anomalies(
            target_df,
            method=method_key,
            subdivision_col="SUBDIVISION",
            value_col="ANNUAL_VALIDATED",
            **method_kwargs,
        )
    else:
        target_df = annual_df[annual_df["SUBDIVISION"] == anomaly_sub].copy()
        detected_df = detect_anomalies(
            target_df,
            method=method_key,
            subdivision_col=None,
            value_col="ANNUAL_VALIDATED",
            **method_kwargs,
        )

    summary = summarize_anomalies(detected_df)

    # Anomaly KPI Metrics
    mkpi1, mkpi2, mkpi3, mkpi4 = st.columns(4)
    with mkpi1:
        st.metric("Total Records Evaluated", f"{summary['total_observations']:,}")
    with mkpi2:
        st.metric("Total Anomalies Detected", f"{summary['anomaly_count']:,}", delta=f"{summary['anomaly_percentage']}% of total")
    with mkpi3:
        st.metric("Dry Anomalies (Drought)", f"{summary['drought_count']:,}")
    with mkpi4:
        st.metric("Wet Anomalies (Excess)", f"{summary['excess_count']:,}")

    # Plot
    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
    if anomaly_sub != "All India (All Subdivisions)":
        fig_anomaly = create_anomaly_scatter_plot(
            detected_df,
            value_col="ANNUAL_VALIDATED",
            subdivision_name=anomaly_sub,
            method_name=anomaly_method.split(" ")[0],
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)

        # Narrative
        narrative = generate_anomaly_narrative(detected_df, anomaly_sub, method=method_key)
        st.markdown(
            f"""
            <div class="dashboard-card" style="margin-top: 12px; margin-bottom: 20px;">
                <h4 style="color: {COLOR_CYAN_ACCENT}; margin-top: 0;">🧠 Climate Anomaly Narrative</h4>
                <p style="color: {COLOR_CLOUD_WHITE}; font-size: 0.95rem; line-height: 1.6;">{narrative}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Aggregated annual rate chart across India
        agg_by_yr = detected_df.groupby("YEAR")["is_anomaly"].agg(["count", "sum"]).reset_index()
        agg_by_yr["rate"] = (agg_by_yr["sum"] / agg_by_yr["count"]) * 100
        fig_all_anom = px.bar(
            agg_by_yr,
            x="YEAR",
            y="sum",
            title="<b>Frequency of Anomalous Subdivisions Per Year Across India</b>",
            labels={"sum": "Number of Subdivisions Flagged", "YEAR": "Year"},
            color="sum",
            color_continuous_scale="Reds",
        )
        fig_all_anom.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=COLOR_CLOUD_WHITE))
        st.plotly_chart(fig_all_anom, use_container_width=True)

    # Anomaly Explorer Table
    anom_only = detected_df[detected_df["is_anomaly"]].sort_values(by="YEAR", ascending=False)
    st.markdown(f"##### 📋 Detected Anomaly Records ({len(anom_only)} records)")
    
    tcol1, tcol2 = st.columns([3, 1])
    with tcol1:
        st.caption("Showing verified anomaly events with deviation flags.")
    with tcol2:
        get_csv_download_button(anom_only, "monsoonlens_anomalies.csv", "Export Anomalies CSV")

    display_cols = [c for c in ["SUBDIVISION", "YEAR", "ANNUAL_VALIDATED", "z_score", "anomaly_type", "iqr_distance", "anomaly_score"] if c in anom_only.columns]
    st.dataframe(anom_only[display_cols], use_container_width=True, height=280, hide_index=True)


# =============================================================================
# TAB 2: REGIME SHIFTS & CHANGE-POINT ANALYSIS
# =============================================================================
with tab_changepoint:
    st.markdown("### 📈 Climate Regime Shift & Change-Point Analysis")
    st.markdown(
        "Evaluate whether a region's historical rainfall underwent a statistically significant, structural transition "
        "rather than random year-to-year variation, using **Pettitt's non-parametric rank test**, **CUSUM**, and **SNHT**."
    )

    cp_col1, cp_col2, cp_col3 = st.columns([1.5, 1.2, 1.3])
    with cp_col1:
        cp_sub = st.selectbox(
            "Select Subdivision for Change-Point Analysis",
            options=all_subdivisions,
            index=all_subdivisions.index("PUNJAB") if "PUNJAB" in all_subdivisions else 0,
            key="cp_sub_select",
        )
    with cp_col2:
        cp_method = st.selectbox(
            "Change-Point Test Method",
            options=["Pettitt's Test (Recommended - Non-Parametric)", "CUSUM (Cumulative Sum)", "SNHT (Standard Normal Homogeneity Test)"],
            index=0,
            key="cp_method_select",
        )
    with cp_col3:
        alpha_thresh = st.select_slider(
            "Significance Threshold (α)",
            options=[0.01, 0.05, 0.10],
            value=0.05,
            format_func=lambda x: f"α = {x} ({int((1-x)*100)}% Confidence)",
        )

    # Run analysis for selected subdivision
    sub_annual = annual_df[annual_df["SUBDIVISION"] == cp_sub].sort_values("YEAR")
    valid_sub = sub_annual.dropna(subset=["ANNUAL_VALIDATED"])

    if "Pettitt" in cp_method:
        res = pettitt_test(valid_sub["ANNUAL_VALIDATED"].values, alpha=alpha_thresh)
    elif "CUSUM" in cp_method:
        res = cusum_test(valid_sub["ANNUAL_VALIDATED"].values, alpha=alpha_thresh)
    else:
        res = snht_test(valid_sub["ANNUAL_VALIDATED"].values, alpha=alpha_thresh)

    # Attach year mapping
    idx = res.get("change_point_index")
    cp_year = int(valid_sub["YEAR"].iloc[idx]) if idx is not None and 0 <= idx < len(valid_sub) else None
    res["change_point_year"] = cp_year
    res["subdivision"] = cp_sub

    # Display KPI metrics
    ckpi1, ckpi2, ckpi3, ckpi4, ckpi5 = st.columns(5)
    with ckpi1:
        st.metric("Detected Shift Year", f"{cp_year}" if cp_year else "None")
    with ckpi2:
        p_val = res.get("p_value", 1.0)
        p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
        st.metric("p-value", p_str)
    with ckpi3:
        sig_text = "Significant (p < α)" if res.get("is_significant") else "Not Significant"
        st.metric("Statistical Verdict", sig_text)
    with ckpi4:
        st.metric("Mean Before Shift", f"{res.get('mean_before', 0):,.1f} mm")
    with ckpi5:
        delta_val = res.get("difference", 0)
        st.metric("Mean After Shift", f"{res.get('mean_after', 0):,.1f} mm", delta=f"{delta_val:+,.1f} mm")

    # Plot
    fig_cp = create_change_point_plot(valid_sub, res, value_col="ANNUAL_VALIDATED")
    st.plotly_chart(fig_cp, use_container_width=True)

    # Narrative Insight
    cp_narrative = generate_change_point_narrative(res, cp_sub)
    st.markdown(
        f"""
        <div class="dashboard-card" style="margin-top: 12px; margin-bottom: 24px;">
            <h4 style="color: {COLOR_AMBER_ACCENT}; margin-top: 0;">🔬 Scientific Regime Shift Narrative</h4>
            <p style="color: {COLOR_CLOUD_WHITE}; font-size: 0.95rem; line-height: 1.6;">{cp_narrative}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Overview Across All Subdivisions
    with st.expander("🌐 View All-India Change-Point Summary Table (All 36 Subdivisions)", expanded=False):
        @st.cache_data(show_spinner=False)
        def get_cached_all_shifts():
            return analyze_all_subdivisions_change_points(annual_df, method="pettitt", alpha=0.05)

        all_shifts_df = get_cached_all_shifts()
        scol1, scol2 = st.columns([3, 1])
        with scol1:
            sig_shifts = all_shifts_df[all_shifts_df["is_significant"]]
            st.markdown(
                f"**{len(sig_shifts)} out of 36 subdivisions** exhibit statistically significant climate regime shifts (p < 0.05)."
            )
        with scol2:
            get_csv_download_button(all_shifts_df, "monsoonlens_regime_shifts_all_india.csv", "Export All Shifts CSV")
        st.dataframe(all_shifts_df, use_container_width=True, height=320, hide_index=True)


# =============================================================================
# TAB 3: RAINFALL FORECASTING & HORIZON PROJECTION
# =============================================================================
with tab_forecasting:
    st.markdown("### 🔮 Annual Rainfall Forecasting & Model Benchmarking")
    st.markdown(
        "Evaluate leakage-free forecasting models on out-of-time test periods (1996–2015) and project "
        "multi-year future scenarios with empirical residual uncertainty intervals."
    )

    fcol1, fcol2, fcol3 = st.columns([1.5, 1.2, 1.3])
    with fcol1:
        fc_sub = st.selectbox(
            "Select Subdivision for Forecasting",
            options=all_subdivisions,
            index=all_subdivisions.index("KERALA") if "KERALA" in all_subdivisions else 0,
            key="fc_sub_select",
        )
    with fcol2:
        model_family = st.selectbox(
            "Model Architecture",
            options=["Random Forest Regressor (ML)", "Linear Trend OLS", "5-Year Rolling Mean (Baseline)", "Naive Prior Year (Baseline)", "Historical Mean (Benchmark)"],
            index=0,
            key="fc_model_select",
        )
    with fcol3:
        forecast_horizon = st.slider(
            "Forecast Horizon (Years Ahead)",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of future years to project beyond the historical record.",
        )

    # Map selected model
    model_key_map = {
        "Random Forest Regressor (ML)": "random_forest",
        "Linear Trend OLS": "linear_trend",
        "5-Year Rolling Mean (Baseline)": "rolling_mean_5",
        "Naive Prior Year (Baseline)": "naive",
        "Historical Mean (Benchmark)": "historical_mean",
    }
    selected_model_key = model_key_map[model_family]

    # Benchmark all models for this subdivision
    sub_data = annual_df[annual_df["SUBDIVISION"] == fc_sub].sort_values("YEAR")
    pipeline = RainfallForecastingPipeline(subdivision_name=fc_sub)
    comp_df = pipeline.fit_and_compare(sub_data, test_years=20)

    # Performance metrics
    matched_row = comp_df[comp_df["model_name"] == selected_model_key]
    if not matched_row.empty:
        m_res = matched_row.iloc[0]
        fkpi1, fkpi2, fkpi3, fkpi4, fkpi5 = st.columns(5)
        with fkpi1:
            st.metric("Test MAE", f"{m_res['test_mae']:.1f} mm")
        with fkpi2:
            st.metric("Test RMSE", f"{m_res['test_rmse']:.1f} mm")
        with fkpi3:
            st.metric("Test MAPE", f"{m_res['test_mape']:.1f}%")
        with fkpi4:
            st.metric("Test R²", f"{m_res['test_r2']:.3f}")
        with fkpi5:
            skill = m_res.get("forecast_skill_score", 0.0)
            skill_delta = f"{skill:+.1f}% vs Mean"
            st.metric("Forecast Skill Score", f"{skill:.1f}%", delta=skill_delta)

    # Future projection
    fc_future = pipeline.generate_future_forecast(
        model_name=selected_model_key,
        horizon_years=forecast_horizon,
    )

    # Plot Projection
    sub_hist = annual_df[annual_df["SUBDIVISION"] == fc_sub].sort_values("YEAR")
    fig_fc = create_forecast_projection_plot(
        history_df=sub_hist,
        forecast_res=fc_future,
        subdivision_name=fc_sub,
        model_name=model_family.split(" (")[0],
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Future Forecast Table
    proj_preds = fc_future.get("forecasts", [])
    if proj_preds:
        st.markdown(f"##### 📅 Projected Scenarios ({forecast_horizon} Years Ahead)")
        p_df = pd.DataFrame(proj_preds)
        p_df["predicted_annual"] = p_df["predicted_annual"].round(1)
        if "lower_bound" in p_df.columns:
            p_df["lower_bound"] = p_df["lower_bound"].round(1)
            p_df["upper_bound"] = p_df["upper_bound"].round(1)
        
        pcol1, pcol2 = st.columns([3, 1])
        with pcol1:
            st.caption("Forecast estimates based strictly on historical dynamics and residual uncertainty.")
        with pcol2:
            get_csv_download_button(p_df, f"forecast_{fc_sub.lower().replace(' ', '_')}.csv", "Export Horizon CSV")

        st.dataframe(p_df, use_container_width=True, hide_index=True)

    # Model Comparison Table
    st.markdown("##### 📊 Full Model Benchmark Comparison (Out-of-Time Test Set: 1996–2015)")
    st.dataframe(
        comp_df[["model_name", "test_mae", "test_rmse", "test_mape", "test_r2", "forecast_skill_score"]],
        use_container_width=True,
        hide_index=True,
    )

    # Narrative
    fc_narrative = generate_forecasting_evaluation_narrative(comp_df, fc_sub)
    st.markdown(
        f"""
        <div class="dashboard-card" style="margin-top: 12px; margin-bottom: 20px;">
            <h4 style="color: {COLOR_GREEN_ACCENT}; margin-top: 0;">📋 Model Performance Narrative</h4>
            <p style="color: {COLOR_CLOUD_WHITE}; font-size: 0.95rem; line-height: 1.6;">{fc_narrative}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Disclaimer
    st.markdown(
        """
        <div class="scientific-callout">
            <b>⚠️ Scientific & Operational Disclaimer:</b><br>
            Multi-year precipitation forecasting in tropical monsoon systems is inherently constrained by chaotic ocean-atmosphere 
            teleconnections (e.g. El Niño-Southern Oscillation, Indian Ocean Dipole, Madden-Julian Oscillation). These models are designed 
            to quantify climate baselines, trend momentum, and multi-year planning scenarios — <b>not deterministic severe storm or flash flood warnings</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
