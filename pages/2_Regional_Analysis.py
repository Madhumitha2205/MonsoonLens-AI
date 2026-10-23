"""
MonsoonLens AI - Regional & Subdivisional Analysis.

Provides deep geographical and subdivisional comparisons across India's 36 meteorological
subdivisions without fabricating coordinates: ranked annual means, variability (CV%),
multi-subdivision comparisons, subdivision-by-month heatmap, and statistical tables.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MONTH_COLUMNS
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
    create_subdivision_ranking_plot,
    format_rainfall_mm,
    get_csv_download_button,
    get_unique_subdivisions,
    load_annual_rainfall,
    load_monthly_rainfall,
    render_header,
    render_sidebar_branding,
)
from src.eda_analysis import compute_subdivision_annual_stats

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Regional Analysis - MonsoonLens AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar_branding()

render_header(
    title="Regional & Subdivisional Rainfall Analysis",
    subtitle="Evaluate spatial disparity, rainfall variability, and multi-region comparisons across India's 36 meteorological subdivisions.",
    icon="🗺️",
)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
try:
    annual_df = load_annual_rainfall()
    monthly_df = load_monthly_rainfall()
except Exception as e:
    st.error(f"⚠️ Error loading datasets: {e}")
    st.stop()

subdivision_stats = compute_subdivision_annual_stats(annual_df)
all_subdivisions = get_unique_subdivisions(annual_df)

min_year = int(annual_df["YEAR"].min())
max_year = int(annual_df["YEAR"].max())

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.markdown(f"<h3 style='color: {COLOR_CLOUD_WHITE}; margin-bottom: 12px;'>🗺️ Regional Controls</h3>", unsafe_allow_html=True)

selected_years = st.sidebar.slider(
    "Observation Timeline",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
    help="Time period over which to compute regional rankings and comparisons.",
)

# Multi-region comparison selector
default_compare = [
    "COASTAL KARNATAKA",
    "WEST RAJASTHAN",
    "ASSAM & MEGHALAYA",
    "KERALA",
]
selected_compare_subs = st.sidebar.multiselect(
    "Compare Subdivisions",
    options=all_subdivisions,
    default=[s for s in default_compare if s in all_subdivisions],
    help="Select two or more subdivisions to contrast directly in monthly and annual comparisons.",
)

# -----------------------------------------------------------------------------
# Filter Data for Selected Timeline
# -----------------------------------------------------------------------------
filtered_annual = annual_df[
    (annual_df["YEAR"] >= selected_years[0]) & (annual_df["YEAR"] <= selected_years[1])
].copy()

filtered_monthly = monthly_df[
    (monthly_df["YEAR"] >= selected_years[0]) & (monthly_df["YEAR"] <= selected_years[1])
].copy()

# Recalculate stats for the selected timeframe if not default
if selected_years != (min_year, max_year):
    active_sub_stats = compute_subdivision_annual_stats(filtered_annual)
else:
    active_sub_stats = subdivision_stats.copy()

# -----------------------------------------------------------------------------
# Key Regional Summary Metrics
# -----------------------------------------------------------------------------
wettest_region = active_sub_stats.iloc[0]
driest_region = active_sub_stats.iloc[-1]
highest_cv_region = active_sub_stats.sort_values("cv_percent", ascending=False).iloc[0]
lowest_cv_region = active_sub_stats.sort_values("cv_percent", ascending=True).iloc[0]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(
        label="Wettest Subdivision",
        value=wettest_region["SUBDIVISION"].title(),
        delta=f"{wettest_region['mean_annual_mm']:,.1f} mm/yr",
        delta_color="normal",
        help="Subdivision with highest mean annual precipitation in the selected period",
    )
with kpi2:
    st.metric(
        label="Driest Subdivision",
        value=driest_region["SUBDIVISION"].title(),
        delta=f"{driest_region['mean_annual_mm']:,.1f} mm/yr",
        delta_color="inverse",
        help="Subdivision with lowest mean annual precipitation in the selected period",
    )
with kpi3:
    st.metric(
        label="Highest Variability (CV%)",
        value=highest_cv_region["SUBDIVISION"].title(),
        delta=f"CV: {highest_cv_region['cv_percent']:.1f}%",
        delta_color="inverse",
        help="Subdivision with greatest relative year-to-year rainfall swings",
    )
with kpi4:
    st.metric(
        label="Most Stable Precipitation",
        value=lowest_cv_region["SUBDIVISION"].title(),
        delta=f"CV: {lowest_cv_region['cv_percent']:.1f}%",
        delta_color="normal",
        help="Subdivision with lowest relative interannual variation",
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Visualizations Tabs
# -----------------------------------------------------------------------------
tab_rankings, tab_variability, tab_compare, tab_heatmap = st.tabs([
    "📊 Mean Rainfall Rankings",
    "📈 Rainfall Variability (CV%)",
    "⚖️ Multi-Subdivision Comparison",
    "🔥 Subdivision × Month Heatmap",
])

with tab_rankings:
    st.markdown(
        f"""
        ##### Ranked Annual Rainfall Across All 36 Subdivisions ({selected_years[0]}–{selected_years[1]})
        Subdivisions sorted in descending order of mean annual rainfall. Notice how coastal and mountainous subdivisions 
        dominate the top tiers, whereas interior and north-western desert tracts occupy the lower tiers.
        """
    )
    col_rank_ctrl1, col_rank_ctrl2 = st.columns([1, 4])
    with col_rank_ctrl1:
        top_n = st.selectbox("Display Count", options=[10, 20, 36], index=2)
        sort_order = st.radio("Sort Order", options=["Wettest First", "Driest First"], index=0)

    fig_rank = create_subdivision_ranking_plot(
        active_sub_stats,
        metric_col="mean_annual_mm",
        metric_name="Mean Annual Rainfall (mm)",
        top_n=top_n,
        ascending=(sort_order == "Driest First"),
    )
    st.plotly_chart(fig_rank, use_container_width=True)

with tab_variability:
    st.markdown(
        f"""
        ##### Coefficient of Variation (CV%) Across Subdivisions ({selected_years[0]}–{selected_years[1]})
        $$\\text{{CV}} = \\left( \\frac{{\\sigma}}{{\\mu}} \\right) \\times 100\\%$$
        CV measures relative dispersion. Arid regions characteristically possess high CV because small absolute deviations 
        represent large percentages of their low baseline totals.
        """
    )
    st.markdown(
        """
        <div class="scientific-warning">
            <b>⚠️ Analytical Note:</b> A high Coefficient of Variation describes empirical variability in historical observations. 
            It does not automatically demonstrate increased anthropogenic climate vulnerability without dynamical hydrological modeling.
        </div>
        """,
        unsafe_allow_html=True,
    )
    cv_sorted = active_sub_stats.sort_values("cv_percent", ascending=False)
    fig_cv = go.Figure(
        go.Bar(
            x=cv_sorted["cv_percent"],
            y=cv_sorted["SUBDIVISION"],
            orientation="h",
            marker=dict(
                color=cv_sorted["cv_percent"],
                colorscale="Reds",
                colorbar=dict(title=dict(text="CV (%)", font=dict(color=COLOR_CLOUD_WHITE))),
            ),
            hovertemplate="<b>%{y}</b><br><b>CV:</b> %{x:.2f}%<br><b>Mean:</b> %{customdata[0]:,.1f} mm<extra></extra>",
            customdata=cv_sorted[["mean_annual_mm"]].values,
        )
    )
    fig_cv.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_CLOUD_WHITE),
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis_title="Coefficient of Variation (CV %)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=10, color=COLOR_MUTED_SLATE)),
        height=800,
    )
    st.plotly_chart(fig_cv, use_container_width=True)

with tab_compare:
    st.markdown("##### Multi-Subdivision Direct Contrast")
    if len(selected_compare_subs) < 2:
        st.info("💡 Please select at least two subdivisions in the sidebar under 'Compare Subdivisions' to view side-by-side comparisons.")
    else:
        st.markdown(f"Comparing: **{', '.join(selected_compare_subs)}**")

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("###### Monthly Climatology Comparison")
            comp_monthly = (
                filtered_monthly[filtered_monthly["SUBDIVISION"].isin(selected_compare_subs)]
                .groupby(["SUBDIVISION", "MONTH_NUMBER", "MONTH"])["RAINFALL"]
                .mean()
                .reset_index()
                .sort_values("MONTH_NUMBER")
            )
            fig_comp_m = px.line(
                comp_monthly,
                x="MONTH",
                y="RAINFALL",
                color="SUBDIVISION",
                markers=True,
                category_orders={"MONTH": MONTH_COLUMNS},
            )
            fig_comp_m.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLOR_CLOUD_WHITE),
                xaxis_title="Month",
                yaxis_title="Mean Rainfall (mm)",
                margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(orientation="h", y=-0.25),
                height=420,
            )
            st.plotly_chart(fig_comp_m, use_container_width=True)

        with comp_col2:
            st.markdown("###### Annual Trajectory Comparison")
            comp_annual = filtered_annual[filtered_annual["SUBDIVISION"].isin(selected_compare_subs)]
            fig_comp_ann = px.line(
                comp_annual,
                x="YEAR",
                y="ANNUAL_VALIDATED",
                color="SUBDIVISION",
            )
            fig_comp_ann.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLOR_CLOUD_WHITE),
                xaxis_title="Year",
                yaxis_title="Annual Rainfall (mm)",
                margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(orientation="h", y=-0.25),
                height=420,
            )
            st.plotly_chart(fig_comp_ann, use_container_width=True)

with tab_heatmap:
    st.markdown("##### Subdivision × Month Climatology Matrix")
    st.markdown(
        "Matrix displaying mean rainfall for each of the 36 subdivisions across the 12 calendar months. "
        "Notice the Southwest Monsoon surge (June to September) concentrated across western coastal and northeastern subdivisions."
    )

    sub_month_matrix = (
        filtered_monthly.groupby(["SUBDIVISION", "MONTH_NUMBER", "MONTH"])["RAINFALL"]
        .mean()
        .reset_index()
    )
    matrix_pivot = sub_month_matrix.pivot(index="SUBDIVISION", columns="MONTH", values="RAINFALL")
    # Sort months chronologically
    chronological_cols = [m for m in MONTH_COLUMNS if m in matrix_pivot.columns]
    matrix_pivot = matrix_pivot[chronological_cols]

    fig_matrix = go.Figure(
        data=go.Heatmap(
            z=matrix_pivot.values,
            x=matrix_pivot.columns.tolist(),
            y=matrix_pivot.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(
                title=dict(text="Rainfall (mm)", font=dict(color=COLOR_CLOUD_WHITE)),
                tickfont=dict(color=COLOR_MUTED_SLATE),
            ),
            hovertemplate="<b>%{y}</b><br><b>%{x}:</b> %{z:,.1f} mm<extra></extra>",
        )
    )
    fig_matrix.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_CLOUD_WHITE),
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(title="Calendar Month", tickfont=dict(color=COLOR_MUTED_SLATE)),
        yaxis=dict(title="Subdivision", autorange="reversed", tickfont=dict(size=9, color=COLOR_MUTED_SLATE)),
        height=850,
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

# -----------------------------------------------------------------------------
# Detailed Regional Statistics Table & CSV Export
# -----------------------------------------------------------------------------
st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
st.subheader("📋 Subdivisional Statistical Summary Table")

col_tbl_info, col_tbl_dl = st.columns([3, 1])
with col_tbl_info:
    st.markdown("Descriptive statistics, dispersion metrics, and empirical trend slopes for all 36 subdivisions:")
with col_tbl_dl:
    get_csv_download_button(
        df=active_sub_stats,
        filename=f"monsoonlens_regional_stats_{selected_years[0]}_{selected_years[1]}.csv",
        label="Export Regional Stats CSV",
    )

display_stats_df = active_sub_stats.rename(
    columns={
        "SUBDIVISION": "Subdivision",
        "observations_count": "Observations",
        "mean_annual_mm": "Mean (mm)",
        "std_annual_mm": "Std Dev (mm)",
        "cv_percent": "CV (%)",
        "median_annual_mm": "Median (mm)",
        "iqr_annual_mm": "IQR (mm)",
        "min_annual_mm": "Min (mm)",
        "max_annual_mm": "Max (mm)",
        "trend_slope_mm_per_year": "Slope (mm/yr)",
        "trend_pvalue": "p-value",
        "is_trend_significant_p05": "p < 0.05",
    }
)

st.dataframe(
    display_stats_df,
    use_container_width=True,
    height=400,
    hide_index=True,
)
