"""
MonsoonLens AI - Main Application and Executive Overview.

Landing page providing high-level executive KPIs, national annual trends,
monthly climatology, seasonal contributions, regional previews, and scientific caveats.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    create_annual_timeseries_plot,
    create_monthly_climatology_plot,
    create_seasonal_breakdown_plot,
    format_rainfall_mm,
    load_annual_rainfall,
    load_cleaned_wide,
    load_dataset_metadata,
    load_monthly_rainfall,
    load_quality_flags,
    load_seasonal_rainfall,
    render_header,
    render_sidebar_branding,
)
from src.eda_analysis import (
    compute_all_india_annual_stats,
    compute_monthly_climatology,
    compute_seasonal_summary,
    compute_subdivision_annual_stats,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MonsoonLens AI - Overview",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar_branding()

# -----------------------------------------------------------------------------
# Data Loading & Graceful Error Handling
# -----------------------------------------------------------------------------
try:
    with st.spinner("Loading processed meteorological datasets..."):
        metadata = load_dataset_metadata()
        annual_df = load_annual_rainfall()
        monthly_df = load_monthly_rainfall()
        seasonal_df = load_seasonal_rainfall()
        wide_df = load_cleaned_wide()
        flags_df = load_quality_flags()
except Exception as e:
    st.error(
        f"⚠️ **Failed to load processed datasets**: {e}\n\n"
        "Please ensure that data preprocessing has been run (`python src/preprocess.py`) "
        "and that the CSV artifacts exist under `data/processed/`."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Compute Core Statistics from Real Data
# -----------------------------------------------------------------------------
national_stats = compute_all_india_annual_stats(annual_df)
monthly_stats = compute_monthly_climatology(monthly_df)
seasonal_stats = compute_seasonal_summary(seasonal_df)
subdivision_stats = compute_subdivision_annual_stats(annual_df)

min_year = int(annual_df["YEAR"].min())
max_year = int(annual_df["YEAR"].max())
num_years = max_year - min_year + 1
num_subdivisions = annual_df["SUBDIVISION"].nunique()
num_records = len(monthly_df)
national_mean = national_stats["mean"]
total_flags = len(flags_df)

# Top 5 wettest and driest subdivisions
wettest_5 = subdivision_stats.head(5)
driest_5 = subdivision_stats.tail(5).iloc[::-1]

# -----------------------------------------------------------------------------
# Header Section
# -----------------------------------------------------------------------------
render_header(
    title="MonsoonLens AI",
    subtitle="Historical Rainfall Pattern Analysis Using Machine Learning & Empirical Climatology",
    icon="🌧️",
)

st.markdown(
    """
    **MonsoonLens AI** provides empirical analysis of historical Indian rainfall gauge observations 
    spanning **115 continuous years (1901–2015)** across India's **36 meteorological subdivisions**. 
    Explore multi-decadal trends, seasonal dynamics, spatial variation, and granular data quality audits.
    """
)

# -----------------------------------------------------------------------------
# Executive KPI Row
# -----------------------------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

with kpi_col1:
    st.metric(
        label="Temporal Coverage",
        value=f"{num_years} Years",
        help="Continuous historical timeline from 1901 to 2015",
    )
with kpi_col2:
    st.metric(
        label="Subdivisions",
        value=f"{num_subdivisions} Regions",
        help="Standard India Meteorological Department (IMD) subdivisions",
    )
with kpi_col3:
    st.metric(
        label="Monthly Records",
        value=f"{num_records:,}",
        help="Tidy long format observations across all subdivisions and months",
    )
with kpi_col4:
    st.metric(
        label="National Mean Annual",
        value=format_rainfall_mm(national_mean),
        help="All-India mean annual rainfall across all 36 subdivisions (1901–2015)",
    )
with kpi_col5:
    monsoon_row = seasonal_stats[seasonal_stats["SEASON"] == "Jun-Sep"].iloc[0]
    st.metric(
        label="Monsoon Dominance",
        value=f"{monsoon_row['percent_contribution']:.1f}%",
        help="Share of annual rainfall delivered during the Southwest Monsoon (Jun-Sep)",
    )
with kpi_col6:
    st.metric(
        label="Quality Flags",
        value=f"{total_flags}",
        help="Granular audit flags (missing gauge observations, extremes, rounding)",
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Primary Interactive Overview Visualizations
# -----------------------------------------------------------------------------
tab_trend, tab_climatology, tab_seasons, tab_regions = st.tabs([
    "📈 National Annual Trend",
    "📅 Monthly Climatology",
    "🌦️ Seasonal Distribution",
    "🗺️ Regional Contrast Preview",
])

with tab_trend:
    st.markdown(
        """
        ##### All-India Annual Rainfall Trajectory (1901–2015)
        The chart illustrates historical annual rainfall averaged across all 36 subdivisions.
        The yellow curve shows a **10-year rolling average**, while the dashed red line represents the **ordinary least squares (OLS) linear trend**.
        """
    )
    col_plot, col_meta = st.columns([3, 1])
    with col_plot:
        fig_trend = create_annual_timeseries_plot(
            annual_df,
            title="All-India Annual Rainfall Series",
            subdivision_name="National Average",
            show_rolling=True,
            rolling_window=10,
            show_trendline=True,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_meta:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div style="font-size: 0.8rem; color: {COLOR_MUTED_SLATE}; font-weight: 600;">STATISTICAL METRICS</div>
                <div style="margin-top: 10px; font-size: 0.9rem; line-height: 1.8;">
                    <b>Mean:</b> {national_stats['mean']:.1f} mm<br>
                    <b>Median:</b> {national_stats['median']:.1f} mm<br>
                    <b>Std Dev:</b> {national_stats['std']:.1f} mm<br>
                    <b>Min Year:</b> {national_stats['min_year']} ({national_stats['min']:.1f} mm)<br>
                    <b>Max Year:</b> {national_stats['max_year']} ({national_stats['max']:.1f} mm)<br>
                    <b>Trend Slope:</b> {national_stats['trend_slope_mm_per_year']:+.2f} mm/yr<br>
                    <b>p-value:</b> {national_stats['trend_pvalue']:.4f} {"(Not Significant)" if national_stats['trend_pvalue'] >= 0.05 else "(Significant)"}<br>
                    <b>R²:</b> {national_stats['trend_r_squared']:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_climatology:
    st.markdown(
        """
        ##### Monthly Rainfall Climatology Profile
        Historical average rainfall for each calendar month across India. The overlaid white curve displays each month's percentage share of the annual total.
        """
    )
    fig_clim = create_monthly_climatology_plot(
        monthly_stats,
        title="All-India Monthly Climatology Distribution",
        subtitle="Chronological monthly mean rainfall (mm) and percentage contribution",
    )
    st.plotly_chart(fig_clim, use_container_width=True)

with tab_seasons:
    st.markdown(
        """
        ##### Seasonal Rainfall Breakdown (IMD Standard Definitions)
        Distribution of precipitation across the four standard meteorological seasons:
        **Winter** (Jan–Feb), **Pre-Monsoon** (Mar–May), **Southwest Monsoon** (Jun–Sep), and **Post-Monsoon** (Oct–Dec).
    """
    )
    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        fig_season = create_seasonal_breakdown_plot(seasonal_stats, title="Mean Rainfall by Season (mm)")
        st.plotly_chart(fig_season, use_container_width=True)
    with col_s2:
        st.dataframe(
            seasonal_stats[["SEASON_NAME", "mean_rainfall", "std_rainfall", "percent_contribution"]].rename(
                columns={
                    "SEASON_NAME": "Season",
                    "mean_rainfall": "Mean (mm)",
                    "std_rainfall": "Std Dev (mm)",
                    "percent_contribution": "Share (%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_regions:
    st.markdown(
        """
        ##### Regional Divergence: Wettest vs. Driest Subdivisions
        India experiences profound spatial precipitation disparity. Below is a comparison between the 5 highest-rainfall subdivisions and the 5 most arid subdivisions.
        """
    )
    col_wet, col_dry = st.columns(2)
    with col_wet:
        st.markdown(f"<span style='color: {COLOR_CYAN_ACCENT}; font-weight: 700;'>🌧️ Top 5 Wettest Subdivisions</span>", unsafe_allow_html=True)
        fig_wet = go.Figure(
            go.Bar(
                x=wettest_5["mean_annual_mm"],
                y=wettest_5["SUBDIVISION"],
                orientation="h",
                marker=dict(color=COLOR_PRIMARY_BLUE),
                hovertemplate="<b>%{y}:</b> %{x:,.1f} mm<extra></extra>",
            )
        )
        fig_wet.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_CLOUD_WHITE),
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Mean Annual Rainfall (mm)",
            yaxis=dict(autorange="reversed"),
            height=280,
        )
        st.plotly_chart(fig_wet, use_container_width=True)

    with col_dry:
        st.markdown(f"<span style='color: {COLOR_AMBER_ACCENT}; font-weight: 700;'>☀️ Top 5 Driest Subdivisions</span>", unsafe_allow_html=True)
        fig_dry = go.Figure(
            go.Bar(
                x=driest_5["mean_annual_mm"],
                y=driest_5["SUBDIVISION"],
                orientation="h",
                marker=dict(color=COLOR_AMBER_ACCENT),
                hovertemplate="<b>%{y}:</b> %{x:,.1f} mm<extra></extra>",
            )
        )
        fig_dry.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_CLOUD_WHITE),
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Mean Annual Rainfall (mm)",
            yaxis=dict(autorange="reversed"),
            height=280,
        )
        st.plotly_chart(fig_dry, use_container_width=True)

st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Key Findings & Empirical Insights
# -----------------------------------------------------------------------------
st.subheader("🔍 Key Empirical Findings")

col_f1, col_f2 = st.columns(2)

with col_f1:
    st.markdown(
        f"""
        <div class="dashboard-card">
            <h4 style="color: {COLOR_CYAN_ACCENT}; margin-top: 0;">1. Peak Historical Extremes</h4>
            <p style="color: {COLOR_CLOUD_WHITE}; line-height: 1.6; font-size: 0.95rem;">
                Across 115 years, the wettest all-India year was <b>1917</b> (1,725.6 mm, +22.3% above the 1,411.0 mm baseline), 
                driven by intense monsoon depressions. Conversely, the driest year was <b>1972</b> (1,152.8 mm, -18.3% below normal), 
                corresponding to severe countrywide drought conditions.
            </p>
            <h4 style="color: {COLOR_CYAN_ACCENT}; margin-top: 16px;">2. Southwest Monsoon Concentration</h4>
            <p style="color: {COLOR_CLOUD_WHITE}; line-height: 1.6; font-size: 0.95rem;">
                The 4-month Southwest Monsoon window (June to September) supplies <b>74.6%</b> of national annual rainfall. 
                <b>July</b> (23.8%) and <b>August</b> (20.6%) form the meteorological core, collectively generating more than 
                44% of total precipitation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_f2:
    st.markdown(
        f"""
        <div class="dashboard-card">
            <h4 style="color: {COLOR_GREEN_ACCENT}; margin-top: 0;">3. Spatial Rainfall Disparity</h4>
            <p style="color: {COLOR_CLOUD_WHITE}; line-height: 1.6; font-size: 0.95rem;">
                Annual mean rainfall varies by an order of magnitude: <b>Coastal Karnataka</b> (3,408.0 mm) and 
                <b>Arunachal Pradesh</b> (3,418.8 mm) receive over 11 times the annual rainfall of 
                <b>West Rajasthan</b> (292.6 mm).
            </p>
            <h4 style="color: {COLOR_AMBER_ACCENT}; margin-top: 16px;">4. Variability & Aridity Correlation</h4>
            <p style="color: {COLOR_CLOUD_WHITE}; line-height: 1.6; font-size: 0.95rem;">
                Arid subdivisions demonstrate significantly higher relative year-to-year variability: 
                West Rajasthan exhibits a Coefficient of Variation (CV) of <b>38.3%</b>, whereas humid subdivisions like 
                Assam & Meghalaya demonstrate stable precipitation with a CV of only <b>11.8%</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Scientific Integrity and Limitations Callout
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="scientific-callout">
        <b>💡 Scientific & Machine Learning Integrity Notes:</b><br>
        1. <b>Descriptive vs. Predictive</b>: All metrics, trends, and charts presented above represent descriptive historical observations from ground-truth rain-gauge recordings. They do not represent model predictions or extrapolate future precipitation.<br>
        2. <b>Preservation of Missing Observations</b>: Missing rainfall values represent historical gauge gaps (e.g. WWII years in Andaman & Nicobar) and are strictly preserved as NaN. They are never zero-filled or artificially imputed.<br>
        3. <b>Correlation ≠ Causation</b>: Long-term trend lines show empirical slope coefficients and must not be interpreted as definitive causal indicators of regional climate shifts without dynamical meteorological modeling.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Quick Navigation Links
# -----------------------------------------------------------------------------
st.subheader("🧭 Explore the Dashboard Pages")

nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)

with nav_col1:
    st.markdown(
        f"""
        <div class="dashboard-card" style="text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">📊</div>
            <h4 style="color: {COLOR_CLOUD_WHITE}; margin: 0 0 8px 0;">Rainfall Explorer</h4>
            <p style="color: {COLOR_MUTED_SLATE}; font-size: 0.85rem; margin-bottom: 12px;">
                Interactive filtering by year, subdivision, month, and season. Heatmaps, time series, and CSV downloads.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col2:
    st.markdown(
        f"""
        <div class="dashboard-card" style="text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🗺️</div>
            <h4 style="color: {COLOR_CLOUD_WHITE}; margin: 0 0 8px 0;">Regional Analysis</h4>
            <p style="color: {COLOR_MUTED_SLATE}; font-size: 0.85rem; margin-bottom: 12px;">
                Subdivision-level rankings, Coefficient of Variation (CV%), multi-region comparisons, and heatmaps.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col3:
    st.markdown(
        f"""
        <div class="dashboard-card" style="text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🌦️</div>
            <h4 style="color: {COLOR_CLOUD_WHITE}; margin: 0 0 8px 0;">Seasonal Insights</h4>
            <p style="color: {COLOR_MUTED_SLATE}; font-size: 0.85rem; margin-bottom: 12px;">
                4-season breakdowns, Southwest Monsoon dominance, boxplots, and inter-seasonal correlation matrices.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col4:
    st.markdown(
        f"""
        <div class="dashboard-card" style="text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🛡️</div>
            <h4 style="color: {COLOR_CLOUD_WHITE}; margin: 0 0 8px 0;">Quality & Reports</h4>
            <p style="color: {COLOR_MUTED_SLATE}; font-size: 0.85rem; margin-bottom: 12px;">
                Granular audit flags, data inventory, metadata, EDA report preview, static figures, and export center.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col5:
    st.markdown(
        f"""
        <div class="dashboard-card" style="text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">⚡</div>
            <h4 style="color: {COLOR_CYAN_ACCENT}; margin: 0 0 8px 0;">Advanced Analytics</h4>
            <p style="color: {COLOR_MUTED_SLATE}; font-size: 0.85rem; margin-bottom: 12px;">
                Anomaly detection (IQR/Z-Score/IForest), Pettitt/CUSUM regime shifts, and ML forecasting horizons.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
