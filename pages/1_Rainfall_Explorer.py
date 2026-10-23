"""
MonsoonLens AI - Interactive Rainfall Explorer.

Provides dynamic, user-driven filtering across years, subdivisions, months, and seasons
with synchronized time series, monthly profiles, heatmaps, boxplots, and CSV downloads.
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

from src.config import MONTH_COLUMNS, SEASON_COLUMNS, SEASON_DESCRIPTIONS
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
    create_rainfall_heatmap,
    filter_annual_data,
    filter_monthly_data,
    format_rainfall_mm,
    get_csv_download_button,
    get_unique_subdivisions,
    load_annual_rainfall,
    load_monthly_rainfall,
    render_header,
    render_sidebar_branding,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Rainfall Explorer - MonsoonLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar_branding()

render_header(
    title="Interactive Rainfall Explorer",
    subtitle="Slice and explore historical rainfall observations across custom timelines, regions, and seasons.",
    icon="🔍",
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

all_subdivisions = get_unique_subdivisions(annual_df)
min_year = int(annual_df["YEAR"].min())
max_year = int(annual_df["YEAR"].max())

# -----------------------------------------------------------------------------
# Sidebar Interactive Filters
# -----------------------------------------------------------------------------
st.sidebar.markdown(f"<h3 style='color: {COLOR_CLOUD_WHITE}; margin-bottom: 12px;'>🔍 Filter Controls</h3>", unsafe_allow_html=True)

# 1. Reset Filters Button
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.session_state["filter_subdivision"] = "All India (Average)"
    st.session_state["filter_year_range"] = (min_year, max_year)
    st.session_state["filter_months"] = MONTH_COLUMNS
    st.session_state["filter_season"] = "All Seasons"
    st.rerun()

# 2. Subdivision Selector
sub_options = ["All India (Average)"] + all_subdivisions
selected_sub = st.sidebar.selectbox(
    "Meteorological Subdivision",
    options=sub_options,
    index=0,
    key="filter_subdivision",
    help="Select an individual subdivision or analyze the all-India aggregated average.",
)

# 3. Year Range Slider
selected_years = st.sidebar.slider(
    "Year Range",
    min_value=min_year,
    max_value=max_year,
    value=st.session_state.get("filter_year_range", (min_year, max_year)),
    step=1,
    key="filter_year_range",
    help="Select the historical observation period to investigate.",
)

# 4. Season Filter (optional quick selector)
season_filter_options = ["All Seasons"] + [f"{k} ({SEASON_DESCRIPTIONS[k]})" for k in SEASON_COLUMNS]
selected_season_str = st.sidebar.selectbox(
    "Season Focus",
    options=season_filter_options,
    index=0,
    key="filter_season",
    help="Filter observations belonging to specific meteorological seasons.",
)

# Determine months based on season if filtered
season_month_map = {
    "Jan-Feb": ["JAN", "FEB"],
    "Mar-May": ["MAR", "APR", "MAY"],
    "Jun-Sep": ["JUN", "JUL", "AUG", "SEP"],
    "Oct-Dec": ["OCT", "NOV", "DEC"],
}

season_key = selected_season_str.split(" ")[0] if selected_season_str != "All Seasons" else None
default_months = season_month_map.get(season_key, MONTH_COLUMNS)

# 5. Month Multiselect
selected_months = st.sidebar.multiselect(
    "Months to Include",
    options=MONTH_COLUMNS,
    default=default_months,
    key="filter_months",
    help="Select one or more individual calendar months.",
)

if not selected_months:
    st.warning("⚠️ No months selected. Automatically defaulting to all 12 calendar months.")
    selected_months = MONTH_COLUMNS

# -----------------------------------------------------------------------------
# Apply Filters to Datasets
# -----------------------------------------------------------------------------
is_all_india = (selected_sub == "All India (Average)")
sub_list = None if is_all_india else [selected_sub]

# Filter Annual DF
filtered_annual = filter_annual_data(
    annual_df,
    year_range=selected_years,
    subdivisions=sub_list,
)

# Filter Monthly DF
filtered_monthly = filter_monthly_data(
    monthly_df,
    year_range=selected_years,
    subdivisions=sub_list,
    months=selected_months,
)

# -----------------------------------------------------------------------------
# Active Filters Summary Banner
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="background: {COLOR_NAVY_CARD}; border: 1px solid #1E293B; border-radius: 8px; padding: 10px 16px; margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 15px; align-items: center;">
        <span style="color: {COLOR_MUTED_SLATE}; font-size: 0.85rem; font-weight: 600;">ACTIVE FILTERS:</span>
        <span class="badge-tag">Region: {selected_sub}</span>
        <span class="badge-tag">Years: {selected_years[0]} – {selected_years[1]} ({selected_years[1] - selected_years[0] + 1} yrs)</span>
        <span class="badge-tag">Months: {', '.join(selected_months) if len(selected_months) < 12 else 'All 12 Months'}</span>
        {f'<span class="badge-tag">Season: {season_key}</span>' if season_key else ''}
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered_monthly.empty:
    st.warning("⚠️ No observations match the current filter combination. Please adjust your selections.")
    st.stop()

# -----------------------------------------------------------------------------
# Dynamic Summary KPIs
# -----------------------------------------------------------------------------
valid_monthly = filtered_monthly["RAINFALL"].dropna()
mean_monthly_val = valid_monthly.mean() if not valid_monthly.empty else np.nan
max_monthly_val = valid_monthly.max() if not valid_monthly.empty else np.nan
missing_records = int(filtered_monthly["IS_MISSING"].sum())
total_records = len(filtered_monthly)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric(
        label="Filtered Records",
        value=f"{total_records:,}",
        help="Total rows returned by active filters",
    )
with kpi2:
    st.metric(
        label="Mean Monthly Rainfall",
        value=format_rainfall_mm(mean_monthly_val),
        help="Average monthly rainfall across the filtered observations",
    )
with kpi3:
    st.metric(
        label="Max Monthly Observation",
        value=format_rainfall_mm(max_monthly_val),
        help="Highest single-month rainfall observed within the filtered slice",
    )
with kpi4:
    missing_pct = (missing_records / total_records * 100.0) if total_records > 0 else 0.0
    st.metric(
        label="Missing Observations",
        value=f"{missing_records:,}",
        delta=f"{missing_pct:.2f}%" if missing_records > 0 else "0.00%",
        delta_color="inverse",
        help="Recorded gaps preserved as NaN without imputation",
    )
with kpi5:
    # Compute annual mean for the selected region across the year range
    if not filtered_annual.empty:
        ann_mean = filtered_annual["ANNUAL_VALIDATED"].dropna().mean()
        st.metric(
            label="Mean Annual Total",
            value=format_rainfall_mm(ann_mean),
            help="Average annual sum across valid years in selection",
        )
    else:
        st.metric(label="Mean Annual Total", value="N/A")

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
tab_ts, tab_monthly, tab_heatmap, tab_dist = st.tabs([
    "📈 Annual Time Series",
    "📊 Monthly Profile",
    "🔥 Year × Month Heatmap",
    "📦 Distribution & Boxplots",
])

with tab_ts:
    st.markdown(f"##### Annual Rainfall Trajectory: {selected_sub} ({selected_years[0]}–{selected_years[1]})")
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        rolling_w = st.selectbox("Rolling Average Window", options=[5, 10, 20], index=1)
        show_trend = st.checkbox("Show OLS Linear Trendline", value=True)

    fig_ann = create_annual_timeseries_plot(
        filtered_annual,
        title="Annual Validated Rainfall",
        subdivision_name=selected_sub,
        show_rolling=True,
        rolling_window=rolling_w,
        show_trendline=show_trend,
    )
    st.plotly_chart(fig_ann, use_container_width=True)

with tab_monthly:
    st.markdown(f"##### Monthly Climatology for {selected_sub} ({selected_years[0]}–{selected_years[1]})")
    # Aggregate monthly stats for the filtered subset
    sub_monthly_stats = (
        filtered_monthly.groupby(["MONTH_NUMBER", "MONTH"])["RAINFALL"]
        .agg(mean_rainfall="mean", std_rainfall="std")
        .round(2)
        .reset_index()
        .sort_values("MONTH_NUMBER")
    )
    tot_m = sub_monthly_stats["mean_rainfall"].sum()
    if tot_m > 0:
        sub_monthly_stats["percent_of_annual"] = ((sub_monthly_stats["mean_rainfall"] / tot_m) * 100.0).round(2)

    fig_m = create_monthly_climatology_plot(
        sub_monthly_stats,
        title=f"Monthly Rainfall Profile: {selected_sub}",
        subtitle=f"Average precipitation (mm) and contribution for selected period",
    )
    st.plotly_chart(fig_m, use_container_width=True)

with tab_heatmap:
    st.markdown("##### Temporal Heatmap: Monthly Rainfall Intensity Across Years")
    st.markdown("Color intensity reflects monthly rainfall in mm. Dark blue / purple indicates low precipitation; bright yellow indicates extreme monsoon rainfall.")

    # If All India, aggregate monthly rainfall by year and month
    if is_all_india:
        hm_df = filtered_monthly.groupby(["YEAR", "MONTH", "MONTH_NUMBER"])["RAINFALL"].mean().reset_index()
    else:
        hm_df = filtered_monthly[["YEAR", "MONTH", "MONTH_NUMBER", "RAINFALL"]].copy()

    # Sort months chronologically
    hm_pivot = hm_df.pivot(index="YEAR", columns="MONTH", values="RAINFALL")
    # Reindex columns to chronological order that exist in selection
    chronological_cols = [m for m in MONTH_COLUMNS if m in hm_pivot.columns]
    hm_pivot = hm_pivot[chronological_cols]

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=hm_pivot.values,
            x=hm_pivot.columns.tolist(),
            y=hm_pivot.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(
                title=dict(text="Rainfall (mm)", font=dict(color=COLOR_CLOUD_WHITE)),
                tickfont=dict(color=COLOR_MUTED_SLATE),
            ),
            hovertemplate="<b>Year:</b> %{y}<br><b>Month:</b> %{x}<br><b>Rainfall:</b> %{z:,.1f} mm<extra></extra>",
        )
    )
    fig_hm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_CLOUD_WHITE),
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(title="Month", tickfont=dict(color=COLOR_MUTED_SLATE)),
        yaxis=dict(title="Year", autorange="reversed", tickfont=dict(color=COLOR_MUTED_SLATE)),
        height=max(400, min(800, len(hm_pivot) * 6)),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

with tab_dist:
    st.markdown("##### Monthly Rainfall Distribution & Outlier Detection")
    st.markdown("Boxplots illustrate median, interquartile ranges (IQR), and extreme precipitation values.")

    # Boxplot of rainfall by month
    fig_box = px.box(
        filtered_monthly.dropna(subset=["RAINFALL"]),
        x="MONTH",
        y="RAINFALL",
        category_orders={"MONTH": MONTH_COLUMNS},
        color="MONTH",
        color_discrete_sequence=[COLOR_PRIMARY_BLUE, COLOR_CYAN_ACCENT, COLOR_AMBER_ACCENT, COLOR_GREEN_ACCENT],
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_CLOUD_WHITE),
        showlegend=False,
        xaxis_title="Month",
        yaxis_title="Rainfall (mm)",
        margin=dict(l=40, r=40, t=30, b=40),
        height=420,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------------------------------------------------------
# Filtered Data Table & Download Section
# -----------------------------------------------------------------------------
st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
st.subheader("📋 Filtered Observations Table")

display_cols = ["SUBDIVISION", "YEAR", "MONTH", "DATE", "RAINFALL", "IS_MISSING", "QUALITY_FLAG"]
table_df = filtered_monthly[display_cols].copy()
table_df["RAINFALL"] = table_df["RAINFALL"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else "NaN (Missing)")

col_table_head, col_dl = st.columns([3, 1])
with col_table_head:
    st.markdown(f"Displaying **{len(table_df):,}** records matching current filters:")
with col_dl:
    get_csv_download_button(
        df=filtered_monthly[display_cols],
        filename=f"monsoonlens_filtered_{selected_sub.lower().replace(' ', '_')}_{selected_years[0]}_{selected_years[1]}.csv",
        label="Export Filtered CSV",
    )

st.dataframe(table_df, use_container_width=True, height=320, hide_index=True)
