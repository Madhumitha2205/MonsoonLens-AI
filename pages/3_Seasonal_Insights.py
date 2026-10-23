"""
MonsoonLens AI - Seasonal Insights & Dynamics.

Analyzes rainfall across India's four meteorological seasons: Winter (Jan-Feb),
Pre-Monsoon (Mar-May), Southwest Monsoon (Jun-Sep), and Post-Monsoon (Oct-Dec),
profiling seasonal contributions, multi-decadal trends, and inter-seasonal correlations.
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

from src.config import SEASON_COLUMNS, SEASON_DESCRIPTIONS
from src.dashboard_utils import (
    COLOR_AMBER_ACCENT,
    COLOR_CLOUD_WHITE,
    COLOR_CYAN_ACCENT,
    COLOR_GREEN_ACCENT,
    COLOR_MUTED_SLATE,
    COLOR_NAVY_CARD,
    COLOR_PRIMARY_BLUE,
    COLOR_ROSE_ACCENT,
    SEASON_COLORS,
    apply_custom_css,
    create_seasonal_breakdown_plot,
    filter_seasonal_data,
    format_rainfall_mm,
    get_csv_download_button,
    get_unique_subdivisions,
    load_cleaned_wide,
    load_seasonal_rainfall,
    render_header,
    render_sidebar_branding,
)
from src.eda_analysis import compute_seasonal_correlations, compute_seasonal_summary

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Seasonal Insights - MonsoonLens AI",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar_branding()

render_header(
    title="Seasonal Rainfall Insights & Dynamics",
    subtitle="Examine precipitation dynamics across India's 4 meteorological seasons, multi-decadal trajectories, and correlation structures.",
    icon="🌦️",
)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
try:
    seasonal_df = load_seasonal_rainfall()
    wide_df = load_cleaned_wide()
except Exception as e:
    st.error(f"⚠️ Error loading datasets: {e}")
    st.stop()

all_subdivisions = get_unique_subdivisions(seasonal_df)
min_year = int(seasonal_df["YEAR"].min())
max_year = int(seasonal_df["YEAR"].max())

# -----------------------------------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------------------------------
st.sidebar.markdown(f"<h3 style='color: {COLOR_CLOUD_WHITE}; margin-bottom: 12px;'>🌦️ Seasonal Controls</h3>", unsafe_allow_html=True)

# 1. Subdivision filter
sub_options = ["All India (Average)"] + all_subdivisions
selected_sub = st.sidebar.selectbox(
    "Select Subdivision",
    options=sub_options,
    index=0,
    help="Examine seasonal distributions for the whole country or an individual subdivision.",
)

# 2. Timeline slider
selected_years = st.sidebar.slider(
    "Observation Timeline",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
    help="Time period over which to compute seasonal summaries and trends.",
)

# 3. Season filter
selected_seasons = st.sidebar.multiselect(
    "Filter Seasons",
    options=SEASON_COLUMNS,
    default=SEASON_COLUMNS,
    format_func=lambda s: f"{s} ({SEASON_DESCRIPTIONS.get(s, '')})",
    help="Select which seasons to visualize in trend and comparison plots.",
)

if not selected_seasons:
    st.warning("⚠️ No seasons selected. Defaulting to all 4 meteorological seasons.")
    selected_seasons = SEASON_COLUMNS

# -----------------------------------------------------------------------------
# Apply Filters
# -----------------------------------------------------------------------------
is_all_india = (selected_sub == "All India (Average)")
sub_list = None if is_all_india else [selected_sub]

filtered_seasonal = filter_seasonal_data(
    seasonal_df,
    year_range=selected_years,
    subdivisions=sub_list,
    seasons=selected_seasons,
)

# Filter wide dataframe for correlations
filtered_wide = wide_df[
    (wide_df["YEAR"] >= selected_years[0]) & (wide_df["YEAR"] <= selected_years[1])
].copy()
if not is_all_india:
    filtered_wide = filtered_wide[filtered_wide["SUBDIVISION"] == selected_sub]

# Recalculate seasonal summary for selected slice
seasonal_summary = compute_seasonal_summary(filtered_seasonal)

# -----------------------------------------------------------------------------
# Seasonal KPI Row
# -----------------------------------------------------------------------------
monsoon_info = seasonal_summary[seasonal_summary["SEASON"] == "Jun-Sep"]
winter_info = seasonal_summary[seasonal_summary["SEASON"] == "Jan-Feb"]
premonsoon_info = seasonal_summary[seasonal_summary["SEASON"] == "Mar-May"]
postmonsoon_info = seasonal_summary[seasonal_summary["SEASON"] == "Oct-Dec"]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    val = monsoon_info["mean_rainfall"].iloc[0] if not monsoon_info.empty else np.nan
    pct = monsoon_info["percent_contribution"].iloc[0] if not monsoon_info.empty else 0.0
    st.metric(
        label="Southwest Monsoon (Jun-Sep)",
        value=format_rainfall_mm(val),
        delta=f"{pct:.1f}% Share",
        help="Primary monsoon season providing majority of annual precipitation",
    )
with kpi2:
    val = postmonsoon_info["mean_rainfall"].iloc[0] if not postmonsoon_info.empty else np.nan
    pct = postmonsoon_info["percent_contribution"].iloc[0] if not postmonsoon_info.empty else 0.0
    st.metric(
        label="Post-Monsoon (Oct-Dec)",
        value=format_rainfall_mm(val),
        delta=f"{pct:.1f}% Share",
        help="Northeast monsoon affecting southern peninsula and retreating monsoon",
    )
with kpi3:
    val = premonsoon_info["mean_rainfall"].iloc[0] if not premonsoon_info.empty else np.nan
    pct = premonsoon_info["percent_contribution"].iloc[0] if not premonsoon_info.empty else 0.0
    st.metric(
        label="Pre-Monsoon / Summer (Mar-May)",
        value=format_rainfall_mm(val),
        delta=f"{pct:.1f}% Share",
        help="Convective thunderstorms and pre-monsoon showers",
    )
with kpi4:
    val = winter_info["mean_rainfall"].iloc[0] if not winter_info.empty else np.nan
    pct = winter_info["percent_contribution"].iloc[0] if not winter_info.empty else 0.0
    st.metric(
        label="Winter (Jan-Feb)",
        value=format_rainfall_mm(val),
        delta=f"{pct:.1f}% Share",
        help="Western disturbances bringing light precipitation to northern regions",
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Visualization Tabs
# -----------------------------------------------------------------------------
tab_shares, tab_trends, tab_boxplots, tab_corr = st.tabs([
    "🥧 Seasonal Contributions",
    "📈 Multi-Decadal Seasonal Trajectories",
    "📦 Seasonal Distributions & Boxplots",
    "🔗 Inter-Seasonal Correlations",
])

with tab_shares:
    st.markdown(f"##### Seasonal Percentage Share: {selected_sub} ({selected_years[0]}–{selected_years[1]})")
    col_pie1, col_pie2 = st.columns([1, 1])

    with col_pie1:
        # Donut Chart of Seasonal Contributions
        fig_donut = go.Figure(
            go.Pie(
                labels=seasonal_summary["SEASON_NAME"],
                values=seasonal_summary["mean_rainfall"],
                hole=0.45,
                marker=dict(colors=[SEASON_COLORS.get(s, COLOR_PRIMARY_BLUE) for s in seasonal_summary["SEASON"]]),
                textinfo="label+percent",
                hovertemplate="<b>%{label}:</b> %{value:,.1f} mm (%{percent})<extra></extra>",
            )
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_CLOUD_WHITE),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            height=380,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_pie2:
        # Bar comparison
        fig_season_bar = create_seasonal_breakdown_plot(
            seasonal_summary,
            title="Mean Rainfall by Season (mm)",
        )
        st.plotly_chart(fig_season_bar, use_container_width=True)

with tab_trends:
    st.markdown(f"##### Annual Trajectories by Season: {selected_sub} ({selected_years[0]}–{selected_years[1]})")
    st.markdown("Track the long-term interannual trajectory of each meteorological season.")

    # Aggregate by year and season
    if is_all_india:
        agg_season_series = (
            filtered_seasonal.groupby(["YEAR", "SEASON", "SEASON_NAME"])["RAINFALL_VALIDATED"]
            .mean()
            .reset_index()
        )
    else:
        agg_season_series = filtered_seasonal[["YEAR", "SEASON", "SEASON_NAME", "RAINFALL_VALIDATED"]].copy()

    fig_trends = px.line(
        agg_season_series,
        x="YEAR",
        y="RAINFALL_VALIDATED",
        color="SEASON_NAME",
        color_discrete_map={
            "Winter": SEASON_COLORS["Jan-Feb"],
            "Pre-Monsoon / Summer": SEASON_COLORS["Mar-May"],
            "Southwest Monsoon": SEASON_COLORS["Jun-Sep"],
            "Post-Monsoon / Northeast Monsoon": SEASON_COLORS["Oct-Dec"],
        },
    )
    fig_trends.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_CLOUD_WHITE),
        xaxis_title="Year",
        yaxis_title="Seasonal Rainfall (mm)",
        margin=dict(l=30, r=30, t=30, b=30),
        legend=dict(orientation="h", y=-0.2),
        height=450,
    )
    st.plotly_chart(fig_trends, use_container_width=True)

with tab_boxplots:
    st.markdown(f"##### Seasonal Dispersion & Spread: {selected_sub} ({selected_years[0]}–{selected_years[1]})")
    st.markdown("Compare the variability, median, and outlier spreads of each season.")

    fig_sbox = px.box(
        filtered_seasonal.dropna(subset=["RAINFALL_VALIDATED"]),
        x="SEASON_NAME",
        y="RAINFALL_VALIDATED",
        color="SEASON_NAME",
        color_discrete_map={
            "Winter": SEASON_COLORS["Jan-Feb"],
            "Pre-Monsoon / Summer": SEASON_COLORS["Mar-May"],
            "Southwest Monsoon": SEASON_COLORS["Jun-Sep"],
            "Post-Monsoon / Northeast Monsoon": SEASON_COLORS["Oct-Dec"],
        },
    )
    fig_sbox.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_CLOUD_WHITE),
        showlegend=False,
        xaxis_title="Meteorological Season",
        yaxis_title="Seasonal Rainfall (mm)",
        margin=dict(l=30, r=30, t=30, b=30),
        height=420,
    )
    st.plotly_chart(fig_sbox, use_container_width=True)

with tab_corr:
    st.markdown("##### Inter-Seasonal Correlation Matrix (Pearson & Spearman)")
    st.markdown(
        "Investigates whether rainfall in one season correlates with rainfall in other seasons or with the annual total. "
        "Notice the Southwest Monsoon exhibits high positive correlation ($r > 0.90$) with annual rainfall, "
        "confirming that year-to-year annual fluctuations are overwhelmingly driven by the monsoon."
    )

    if len(filtered_wide) < 10:
        st.warning("⚠️ Insufficient continuous observations in current slice to compute stable correlation matrices.")
    else:
        pearson_corr, spearman_corr = compute_seasonal_correlations(filtered_wide)

        corr_type = st.radio("Correlation Method", options=["Pearson (Linear)", "Spearman (Rank-Order)"], horizontal=True)
        active_corr = pearson_corr if "Pearson" in corr_type else spearman_corr

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=active_corr.values,
                x=active_corr.columns.tolist(),
                y=active_corr.index.tolist(),
                colorscale="Blues",
                zmin=-0.2,
                zmax=1.0,
                text=np.round(active_corr.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=12, color=COLOR_CLOUD_WHITE),
                colorbar=dict(title=dict(text="Correlation (r)", font=dict(color=COLOR_CLOUD_WHITE))),
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
            )
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_CLOUD_WHITE),
            margin=dict(l=40, r=40, t=30, b=40),
            xaxis=dict(tickangle=-15, tickfont=dict(size=10, color=COLOR_MUTED_SLATE)),
            yaxis=dict(tickfont=dict(size=10, color=COLOR_MUTED_SLATE)),
            height=450,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# Seasonal Summary Table & Download
# -----------------------------------------------------------------------------
st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
st.subheader("📋 Seasonal Summary Statistics")

col_tbl_head, col_dl = st.columns([3, 1])
with col_tbl_head:
    st.markdown("Empirical statistical summary for each season across the selected period:")
with col_dl:
    get_csv_download_button(
        df=seasonal_summary,
        filename=f"monsoonlens_seasonal_summary_{selected_sub.lower().replace(' ', '_')}_{selected_years[0]}_{selected_years[1]}.csv",
        label="Export Seasonal CSV",
    )

display_s_table = seasonal_summary.rename(
    columns={
        "SEASON_NAME": "Season",
        "SEASON": "Code",
        "mean_rainfall": "Mean (mm)",
        "std_rainfall": "Std Dev (mm)",
        "median_rainfall": "Median (mm)",
        "min_rainfall": "Min (mm)",
        "max_rainfall": "Max (mm)",
        "percent_contribution": "Share of Annual (%)",
    }
)
st.dataframe(
    display_s_table[["Season", "Code", "Mean (mm)", "Std Dev (mm)", "Median (mm)", "Min (mm)", "Max (mm)", "Share of Annual (%)"]],
    use_container_width=True,
    hide_index=True,
)
