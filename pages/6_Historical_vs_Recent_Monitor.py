"""
MonsoonLens AI - Historical vs Recent Rainfall Monitor.

Tracks and benchmarks recent rainfall observations from data.gov.in
against the 115-year climatological baseline (1901-2015).

Features:
- Clear provenance labeling distinguishing live API queries from unverified local cache.
- Parametric and empirical baseline departure metrics (absolute mm, %, z-score, percentile).
- Separation of 36-subdivision rankings from national aggregate comparisons.
- Clear limitation notes for unweighted national subdivision means.
- Safe missing value preservation (never converts missing observations to zero).
- Standard IMD categorical departure classifications.
- Export center for CSV tables and Markdown summary reports.
"""

from __future__ import annotations

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

from src.config import (
    MONTH_COLUMNS,
    SEASON_COLUMNS,
    SEASON_DESCRIPTIONS,
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
    create_baseline_vs_recent_distribution_plot,
    create_departure_bar_chart,
    get_csv_download_button,
    render_header,
    render_sidebar_branding,
)
from src.data_gov_in_client import DataGovInClient
from src.historical_recent_comparator import HistoricalRecentComparator
from src.recent_rainfall_adapter import (
    CANONICAL_SUBDIVISIONS,
    RecentRainfallAdapter,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Historical vs Recent Monitor - MonsoonLens AI",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar_branding()

render_header(
    title="Historical vs Recent Rainfall Monitor",
    subtitle="Evaluate recent rainfall observations against 115 years of climatological baseline data (1901-2015).",
    icon=":satellite:",
)

# -----------------------------------------------------------------------------
# Initialize Client & Comparator
# -----------------------------------------------------------------------------
client = DataGovInClient()
try:
    comparator = HistoricalRecentComparator()
except Exception as e:
    st.error(f"Error initializing historical baseline comparator: {e}")
    st.stop()

# Session State for Live Refresh
if "force_live_refresh" not in st.session_state:
    st.session_state.force_live_refresh = False

# Fetch or Load Recent Dataset
try:
    with st.spinner("Accessing recent rainfall dataset..."):
        should_force_live = st.session_state.force_live_refresh
        st.session_state.force_live_refresh = False
        raw_recent_df, source_info = client.get_recent_rainfall_dataset(
            force_live=should_force_live
        )
        recent_df = RecentRainfallAdapter.harmonize_dataset(
            raw_recent_df,
            reject_duplicates=False,  # Checked via validate_dataset
            filter_unknown_subdivisions=True,
        )
        val_report = RecentRainfallAdapter.validate_dataset(raw_recent_df)
except Exception as e:
    st.error(
        f"**Could not load recent rainfall data**: {e}\n\n"
        "Please ensure `data/external/data_gov_in_subdivision_recent.csv` exists or verify network settings."
    )
    st.stop()

# Display Live API Fallback or Incomplete Warnings
if source_info.get("fallback_from_live_failure"):
    st.warning(
        f":warning: **Live API Query Fallback:** {source_info.get('message')}"
    )
elif source_info.get("is_incomplete"):
    st.info(
        f":information_source: **Live API Incomplete Dataset Notice:** {source_info.get('message')}"
    )

# -----------------------------------------------------------------------------
# Provenance & Data Source Status Card
# -----------------------------------------------------------------------------
is_live = source_info.get("is_live", False)
if is_live:
    badge_color = "#10b981"
    badge_label = source_info.get("provenance_status", "Live API Response (data.gov.in) - Unverified public OGD feed")
else:
    badge_color = "#f59e0b"
    badge_label = source_info.get("provenance_status", "Local cached dataset - provenance not independently verified")

metadata = source_info.get("metadata", {}) or {}
catalog_url = "https://data.gov.in/catalog/rainfall-india"
resource_id = source_info.get("resource_id", "fb5e2a33-3e34-4087-83a1-f0e46f2748d0")
units_str = source_info.get("units", "Millimeters (mm)")
source_update_date = source_info.get("source_update_date", "Not available in offline cache")
available_years = sorted(list(recent_df["YEAR"].unique()))

with st.container():
    st.markdown(
        f"""
        <div class="dashboard-card" style="margin-bottom: 20px; border-left: 4px solid {badge_color};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span style="background-color: {badge_color}; color: #ffffff; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">
                        {badge_label}
                    </span>
                    <h4 style="color: {COLOR_CLOUD_WHITE}; margin: 8px 0 4px 0;">Rainfall Observation Registry</h4>
                    <p style="color: {COLOR_MUTED_SLATE}; font-size: 0.85rem; margin: 0 0 6px 0;">
                        Publisher Claimed: <b>India Meteorological Department (IMD)</b> | Portal: <a href="{catalog_url}" target="_blank" style="color: {COLOR_CYAN_ACCENT}; text-decoration: underline;">data.gov.in</a>
                    </p>
                    <p style="color: #cbd5e1; font-size: 0.80rem; margin: 0; max-width: 780px;">
                        {"Live query executed successfully." if is_live else "Notice: The local cached dataset is provided as an offline development sample. Its provenance and exact correspondence to official IMD release archives have not been independently verified."}
                    </p>
                </div>
                <div style="text-align: right; font-size: 0.82rem; color: {COLOR_MUTED_SLATE}; line-height: 1.6;">
                    <div><b>Candidate Resource ID:</b> <code style="color: {COLOR_CYAN_ACCENT};">{resource_id}</code></div>
                    <div><b>Observed Years in Cache:</b> {min(available_years)} to {max(available_years)}</div>
                    <div><b>Source Catalog Update Date:</b> {source_update_date}</div>
                    <div><b>Units:</b> {units_str}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if val_report.get("errors") or val_report.get("warnings"):
    with st.expander("Dataset Quality & Validation Notices", expanded=bool(val_report.get("errors"))):
        for error in val_report.get("errors", []):
            st.error(f"Validation error: {error}")

        for warning in val_report.get("warnings", []):
            st.warning(f"Notice: {warning}")


# -----------------------------------------------------------------------------
# API Configuration Expander
# -----------------------------------------------------------------------------
with st.expander("data.gov.in API Configuration & Live Query Settings", expanded=False):
    st.markdown(
        """
        The Open Government Data Platform India (`data.gov.in`) provides REST APIs for registered users.
        
        **How to configure live API queries:**
        1. Register and generate an API key at [data.gov.in](https://data.gov.in/).
        2. Set the key in your local `.env` file:
           ```bash
           DATA_GOV_IN_API_KEY="your_api_key_here"
           ```
        3. Or add it to `.streamlit/secrets.toml`:
           ```toml
           DATA_GOV_IN_API_KEY = "your_api_key_here"
           ```
        *API keys are never logged or displayed in cleartext. When no API key is provided or the network is unreachable, the system uses the local cached dataset.*
        """
    )
    col_k1, col_k2 = st.columns([3, 1])
    with col_k1:
        st.write(
            f"**Current API Key Status:** {'Configured (Key detected)' if client.has_api_key() else 'Not configured (Operating in local cached mode)'}"
        )
    with col_k2:
        if st.button("Refresh from Live API", key="btn_refresh_live"):
            st.session_state.force_live_refresh = True
            st.rerun()

# -----------------------------------------------------------------------------
# Interactive Selection Controls
# -----------------------------------------------------------------------------
st.markdown("### Select Analysis Parameters")

ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns(4)

with ctrl_c1:
    subdivision_options = ["ALL-INDIA"] + CANONICAL_SUBDIVISIONS
    selected_subdivision = st.selectbox(
        "Subdivision / Region",
        options=subdivision_options,
        index=0,
        help="Select any of the 36 canonical IMD subdivisions or the All-India national unweighted aggregate.",
    )

with ctrl_c2:
    period_type_map = {
        "Annual": "Annual",
        "Southwest Monsoon (Jun-Sep)": "Seasonal",
        "Post-Monsoon (Oct-Dec)": "Seasonal",
        "Pre-Monsoon / Summer (Mar-May)": "Seasonal",
        "Winter (Jan-Feb)": "Seasonal",
        "Monthly Breakdown": "Monthly",
    }
    selected_period_label = st.selectbox(
        "Temporal Resolution",
        options=list(period_type_map.keys()),
        index=0,
        help="Choose annual total, seasonal aggregate, or individual calendar month.",
    )

period_type = period_type_map[selected_period_label]

with ctrl_c3:
    if period_type == "Annual":
        period_value = "ANNUAL"
        st.selectbox("Period", options=["Full Calendar Year"], index=0, disabled=True)
    elif period_type == "Seasonal":
        if "Jun-Sep" in selected_period_label:
            period_value = "Jun-Sep"
        elif "Oct-Dec" in selected_period_label:
            period_value = "Oct-Dec"
        elif "Mar-May" in selected_period_label:
            period_value = "Mar-May"
        else:
            period_value = "Jan-Feb"
        st.selectbox("Season", options=[period_value], index=0, disabled=True)
    else:
        period_value = st.selectbox(
            "Month",
            options=MONTH_COLUMNS,
            index=6,  # July default
            help="Select calendar month (JAN to DEC).",
        )

with ctrl_c4:
    selected_year = st.selectbox(
        "Recent Observation Year",
        options=sorted(available_years, reverse=True),
        index=0,
        help="Observation year from available dataset records.",
    )

# -----------------------------------------------------------------------------
# ALL-INDIA Limitation Callout
# -----------------------------------------------------------------------------
if selected_subdivision == "ALL-INDIA":
    st.info(
        "**National Aggregate Baseline Limitation:** "
        "The historical baseline for ALL-INDIA is calculated as the **unweighted arithmetic mean** "
        "across the 36 meteorological subdivisions (1901-2015), because the matching official area-weighted "
        "historical series is not present in the local processed dataset. "
        "Since subdivisional land areas differ significantly in geographic extent, this figure should "
        "**not** be treated as the official IMD national rainfall total."
    )

# -----------------------------------------------------------------------------
# Execution & Analysis
# -----------------------------------------------------------------------------
try:
    hist_series, baseline_meta = comparator.get_historical_series(
        subdivision=selected_subdivision,
        period_type=period_type,
        period_value=period_value,
    )
    recent_val = RecentRainfallAdapter.get_observation_value(
        recent_df,
        subdivision=selected_subdivision,
        year=selected_year,
        period_type=period_type,
        period_value=period_value,
    )
    metrics = comparator.compute_departure(recent_val, hist_series, baseline_meta)
except Exception as e:
    st.error(f"Error computing departure comparison: {e}")
    st.stop()

# Check for missing observation
is_obs_missing = metrics.get("is_missing", False) or pd.isna(metrics.get("recent_value"))
if is_obs_missing:
    st.warning(
        f"No valid rainfall observation is available for **{selected_subdivision}** in "
        f"**{period_value} {selected_year}** in the available dataset. "
        "The value is preserved as missing (NaN). No departure or IMD classification is computed."
    )

# -----------------------------------------------------------------------------
# Executive KPI Metrics
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"### Departure Analysis: **{selected_subdivision}** ({selected_year} - {period_value})"
)

kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)

recent_obs = metrics["recent_value"]
hist_mean = metrics["baseline_mean"]
abs_dep = metrics["absolute_departure_mm"]
pct_dep = metrics["percentage_departure"]
z_score = metrics["standardized_z_score"]
pct_rank = metrics["percentile_rank"]
imd_cat = metrics["imd_category"]

with kpi_c1:
    st.metric(
        label=f"Recent Observed ({selected_year})",
        value=f"{recent_obs:,.1f} mm" if not is_obs_missing else "Missing (NaN)",
        help="Rainfall observation recorded in the dataset.",
    )

with kpi_c2:
    st.metric(
        label="Baseline Normal (Mean)",
        value=f"{hist_mean:,.1f} mm" if pd.notna(hist_mean) else "N/A",
        help=baseline_meta.get("baseline_label", "115-year baseline (1901-2015)"),
    )

with kpi_c3:
    if not is_obs_missing and pd.notna(abs_dep):
        sign_str = "+" if abs_dep >= 0 else ""
        delta_str = f"{sign_str}{pct_dep:.1f}%" if pd.notna(pct_dep) else None
        st.metric(
            label="Absolute Departure",
            value=f"{sign_str}{abs_dep:,.1f} mm",
            delta=delta_str,
            help="Absolute departure: Delta = Observed - Baseline Mean.",
        )
    else:
        st.metric(label="Absolute Departure", value="N/A", help="Observation is missing.")

with kpi_c4:
    if not is_obs_missing and pd.notna(z_score):
        st.metric(
            label="Standardized z-Score",
            value=f"{z_score:+.2f} σ",
            help="Standardized deviation: (Observed - Baseline Mean) / Baseline Std.",
        )
    else:
        st.metric(label="Standardized z-Score", value="N/A", help="Observation is missing.")

with kpi_c5:
    if not is_obs_missing and pd.notna(pct_rank):
        st.metric(
            label="Historical Percentile",
            value=f"{pct_rank:.1f}th",
            help="Empirical percentile rank of the recent observation in the 1901-2015 distribution.",
        )
    else:
        st.metric(label="Historical Percentile", value="N/A", help="Observation is missing.")

with kpi_c6:
    st.markdown(
        f"""
        <div style="background-color: {COLOR_NAVY_CARD}; padding: 12px; border-radius: 8px; border: 1px solid #1e293b; text-align: center;">
            <div style="font-size: 0.75rem; color: {COLOR_MUTED_SLATE}; text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">IMD Classification</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: {imd_cat['color']};">
                {imd_cat['label']}
            </div>
            <div style="font-size: 0.72rem; color: {COLOR_MUTED_SLATE}; margin-top: 2px;">
                {imd_cat['description']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
st.markdown("### Visual Climatology & Regional Benchmarks")

tab_viz1, tab_viz2 = st.tabs([
    "Baseline Distribution vs Recent Observation",
    "36-Subdivisions National Departure Ranking",
])

with tab_viz1:
    fig_dist = create_baseline_vs_recent_distribution_plot(
        historical_series=hist_series,
        recent_value=recent_obs if not is_obs_missing else None,
        label=f"{selected_subdivision} ({period_value})",
        recent_year=selected_year,
        unit="mm",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab_viz2:
    with st.spinner("Calculating departures across the 36 canonical subdivisions..."):
        # Strictly compares the 36 subdivisions (ALL-INDIA is excluded)
        subdivisions_comp_df = comparator.compare_all_subdivisions(
            recent_df=recent_df,
            year=selected_year,
            period_type=period_type,
            period_value=period_value,
        )

    fig_bars = create_departure_bar_chart(
        comparison_df=subdivisions_comp_df,
        title=f"Subdivisional Rainfall Departure from Normal - {period_value} ({selected_year})",
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    # Subdivisional Distribution Summary
    valid_sub_df = subdivisions_comp_df[subdivisions_comp_df["SUBDIVISION"] != "ALL-INDIA"]
    total_subs = len(valid_sub_df)
    cat_counts = valid_sub_df["CATEGORY_LABEL"].value_counts().to_dict()

    cat_cols = st.columns(len(cat_counts) if cat_counts else 1)
    for idx, (label, count) in enumerate(cat_counts.items()):
        pct = (count / total_subs) * 100.0 if total_subs > 0 else 0.0
        with cat_cols[idx % len(cat_cols)]:
            st.metric(label=label, value=f"{count} / {total_subs}", delta=f"{pct:.1f}% of regions")

# -----------------------------------------------------------------------------
# Scientific Context & Non-Causal Explanation
# -----------------------------------------------------------------------------
st.markdown("### Climatological Context & Non-Causal Interpretation")

if not is_obs_missing and pd.notna(recent_obs) and pd.notna(hist_mean):
    st.markdown(
        f"""
        During **{period_value} of {selected_year}**, **{selected_subdivision}** recorded **{recent_obs:,.1f} mm** of precipitation.
        Compared to the historical baseline normal (**{hist_mean:,.1f} mm**), the region exhibited a departure of 
        **{abs_dep:+,.1f} mm ({pct_dep:+.1f}%)**, placing it in the **{imd_cat['label']}** category under IMD operational standards.
        """
    )
else:
    st.markdown(
        f"Rainfall observation for **{selected_subdivision}** during **{period_value} {selected_year}** "
        "is missing (`NaN`). Statistical departures cannot be evaluated."
    )

st.info(
    """
    **Scientific Integrity & Non-Causal Scope:**
    - **Departure vs Surface Inundation:** Statistical rainfall departure measures atmospheric precipitation relative to long-term averages. 
      It does **not** model surface flood inundation or hydrological runoff, which depend heavily on terrain slope, antecedent soil moisture, 
      river basin storage, and local drainage infrastructure.
    - **Atmospheric Dryness vs Agricultural Drought:** A negative rainfall departure indicates meteorological deficiency, but agricultural 
      or socioeconomic drought is mediated by irrigation capacity, reservoir reserves, and crop phenology.
    - **Provenance Notice:** The local cached dataset provenance has not been independently verified against official live IMD registry feeds.
    """
)

# -----------------------------------------------------------------------------
# Comparison Data Table & Download Center
# -----------------------------------------------------------------------------
st.markdown("### Download Reports & Data")

d_col1, d_col2 = st.columns(2)

with d_col1:
    # Full CSV Comparison (36 subdivisions strictly)
    csv_data = comparator.generate_comparison_csv(subdivisions_comp_df)
    csv_filename = f"monsoonlens_recent_comparison_{period_value}_{selected_year}.csv"
    st.download_button(
        label="Download Subdivisional Departures (CSV)",
        data=csv_data,
        file_name=csv_filename,
        mime="text/csv",
        use_container_width=True,
    )

with d_col2:
    # Markdown Summary Report
    md_summary = comparator.generate_markdown_summary(
        subdivision=selected_subdivision,
        year=selected_year,
        period_type=period_type,
        period_value=period_value,
        metrics=metrics,
        comparison_df=subdivisions_comp_df,
    )
    md_filename = f"monsoonlens_monitor_report_{selected_subdivision.replace(' ', '_')}_{selected_year}.md"
    st.download_button(
        label="Download Climatological Summary Report (Markdown)",
        data=md_summary,
        file_name=md_filename,
        mime="text/markdown",
        use_container_width=True,
    )

# Detailed data table expander
with st.expander("View 36-Subdivisions Comparative Table", expanded=False):
    st.dataframe(
        subdivisions_comp_df,
        column_config={
            "SUBDIVISION": "Subdivision",
            "RECENT_RAINFALL": st.column_config.NumberColumn("Observed (mm)", format="%.1f"),
            "BASELINE_MEAN": st.column_config.NumberColumn("Baseline Normal (mm)", format="%.1f"),
            "ABSOLUTE_DEPARTURE_MM": st.column_config.NumberColumn("Departure (mm)", format="%+.1f"),
            "PERCENTAGE_DEPARTURE": st.column_config.NumberColumn("Departure (%)", format="%+.1f%%"),
            "Z_SCORE": st.column_config.NumberColumn("z-Score (σ)", format="%+.2f"),
            "PERCENTILE_RANK": st.column_config.NumberColumn("Percentile", format="%.1f%%"),
            "CATEGORY_LABEL": "IMD Classification",
        },
        use_container_width=True,
        hide_index=True,
    )
