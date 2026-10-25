"""
MonsoonLens AI - Data Quality, Audit Trail, and Research Reports.

Provides comprehensive audit transparency: dataset inventory, granular quality flags,
metadata viewer, embedded publication figures, interactive Markdown reports, and download center.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANNUAL_PATH,
    ANOMALY_REPORT_PATH,
    BACKEND_ARCHITECTURE_REPORT_PATH,
    CHANGE_POINT_REPORT_PATH,
    CLEANED_WIDE_PATH,
    DATA_QUALITY_SUMMARY_PATH,
    EDA_REPORT_PATH,
    FIGURES_DIR,
    FORECASTING_REPORT_PATH,
    HISTORICAL_RECENT_REPORT_PATH,
    METADATA_PATH,
    MODEL_EVALUATION_REPORT_PATH,
    MONTHLY_LONG_PATH,
    PREPROCESSING_REPORT_PATH,
    QUALITY_FLAGS_PATH,
    SEASONAL_PATH,
    VALIDATION_REPORT_PATH,
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
    get_csv_download_button,
    get_unique_subdivisions,
    load_annual_rainfall,
    load_cleaned_wide,
    load_dataset_metadata,
    load_monthly_rainfall,
    load_quality_flags,
    load_seasonal_rainfall,
    render_header,
    render_sidebar_branding,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Quality & Reports - MonsoonLens AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar_branding()

render_header(
    title="Data Quality, Audit Logs & Research Reports",
    subtitle="Inspect dataset provenance, audit flags, scientific assumptions, static publication figures, and download research artifacts.",
    icon="🛡️",
)

# -----------------------------------------------------------------------------
# Load Data & Metadata
# -----------------------------------------------------------------------------
try:
    metadata = load_dataset_metadata()
    flags_df = load_quality_flags()
except Exception as e:
    st.error(f"⚠️ Error loading quality artifacts: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Top Level Audit KPIs
# -----------------------------------------------------------------------------
total_flags = len(flags_df)
missing_obs_count = len(flags_df[flags_df["flag_type"] == "MISSING_OBSERVATION"])
extremes_count = len(flags_df[flags_df["flag_type"].str.contains("EXTREME", na=False)])
tolerance_count = len(flags_df[flags_df["flag_type"].str.contains("TOLERANCE", na=False)])

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(
        label="Total Audit Flags",
        value=f"{total_flags:,}",
        help="Total granular flags recorded across missing observations, consistency checks, and statistical extremes",
    )
with kpi2:
    st.metric(
        label="Missing Gauge Records",
        value=f"{missing_obs_count}",
        help="Historical gauge recording gaps preserved as NaN (e.g. WWII in Andaman & Nicobar)",
    )
with kpi3:
    st.metric(
        label="Extreme Meteorological Events",
        value=f"{extremes_count}",
        help="Legitimate weather extremes exceeding 99.5th percentile preserved without truncation",
    )
with kpi4:
    st.metric(
        label="Rounding Adjustments",
        value=f"{tolerance_count}",
        help="Single-decimal floating point discrepancies within ±0.5 mm tolerance",
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs: Inventory, Flags Explorer, Reports, Figures, Download Center
# -----------------------------------------------------------------------------
tab_inventory, tab_flags, tab_reports, tab_figures, tab_downloads = st.tabs([
    "📦 Dataset Inventory",
    "🔍 Granular Quality Flags",
    "📑 Project Research Reports",
    "🖼️ Static Figures Gallery",
    "📥 Artifact Download Center",
])

# -----------------------------------------------------------------------------
# Tab 1: Dataset Inventory
# -----------------------------------------------------------------------------
with tab_inventory:
    st.markdown("##### Processed Dataset Files & Architectural Provenance")
    st.markdown(
        "All processed datasets are generated reproducibly by `src/preprocess.py` and validated by `src/validate_data.py`. "
        "Raw input files are strictly immutable."
    )

    inventory_items = [
        {
            "File": "cleaned_wide.csv",
            "Path": str(CLEANED_WIDE_PATH.relative_to(PROJECT_ROOT)),
            "Rows": 4116,
            "Columns": 26,
            "Size (KB)": round(CLEANED_WIDE_PATH.stat().st_size / 1024, 1) if CLEANED_WIDE_PATH.exists() else "Missing",
            "Description": "Standardized wide dataset with 12 constituent months, source totals, and consistency flags.",
        },
        {
            "File": "monthly_rainfall_long.csv",
            "Path": str(MONTHLY_LONG_PATH.relative_to(PROJECT_ROOT)),
            "Rows": 49392,
            "Columns": 8,
            "Size (KB)": round(MONTHLY_LONG_PATH.stat().st_size / 1024, 1) if MONTHLY_LONG_PATH.exists() else "Missing",
            "Description": "Tidy monthly time series (subdivision-year-month), standard ISO dates, missing flags.",
        },
        {
            "File": "annual_rainfall.csv",
            "Path": str(ANNUAL_PATH.relative_to(PROJECT_ROOT)),
            "Rows": 4116,
            "Columns": 7,
            "Size (KB)": round(ANNUAL_PATH.stat().st_size / 1024, 1) if ANNUAL_PATH.exists() else "Missing",
            "Description": "Annual series comparing source-reported and monthly-calculated totals with validation flags.",
        },
        {
            "File": "seasonal_rainfall.csv",
            "Path": str(SEASONAL_PATH.relative_to(PROJECT_ROOT)),
            "Rows": 16464,
            "Columns": 9,
            "Size (KB)": round(SEASONAL_PATH.stat().st_size / 1024, 1) if SEASONAL_PATH.exists() else "Missing",
            "Description": "Tidy seasonal rainfall across Winter, Pre-Monsoon, Monsoon, and Post-Monsoon.",
        },
        {
            "File": "quality_flags.csv",
            "Path": str(QUALITY_FLAGS_PATH.relative_to(PROJECT_ROOT)),
            "Rows": len(flags_df),
            "Columns": 7,
            "Size (KB)": round(QUALITY_FLAGS_PATH.stat().st_size / 1024, 1) if QUALITY_FLAGS_PATH.exists() else "Missing",
            "Description": "Granular audit log of missing observations, consistency checks, and extreme values.",
        },
        {
            "File": "dataset_metadata.json",
            "Path": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
            "Rows": "N/A",
            "Columns": "N/A",
            "Size (KB)": round(METADATA_PATH.stat().st_size / 1024, 1) if METADATA_PATH.exists() else "Missing",
            "Description": "Machine-readable provenance, environment specs, cleaning policies, and schema dictionary.",
        },
    ]

    st.dataframe(pd.DataFrame(inventory_items), use_container_width=True, hide_index=True)

    with st.expander("🔍 View Raw Machine-Readable Metadata (JSON)"):
        st.json(metadata)

# -----------------------------------------------------------------------------
# Tab 2: Granular Quality Flags
# -----------------------------------------------------------------------------
with tab_flags:
    st.markdown("##### Granular Quality Flags & Audit Trail (`quality_flags.csv`)")
    st.markdown(
        "Every single observation that diverged from strict validation criteria is recorded here. "
        "Filter by flag type, severity, or subdivision."
    )

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        flag_types = ["All Types"] + sorted(flags_df["flag_type"].unique().tolist())
        selected_ftype = st.selectbox("Filter Flag Type", options=flag_types, index=0)
    with fcol2:
        severities = ["All Severities"] + sorted(flags_df["severity"].unique().tolist())
        selected_sev = st.selectbox("Filter Severity", options=severities, index=0)
    with fcol3:
        flag_subs = ["All Subdivisions"] + sorted(flags_df["subdivision"].unique().tolist())
        selected_fsub = st.selectbox("Filter Subdivision", options=flag_subs, index=0)

    # Apply filters
    filtered_flags = flags_df.copy()
    if selected_ftype != "All Types":
        filtered_flags = filtered_flags[filtered_flags["flag_type"] == selected_ftype]
    if selected_sev != "All Severities":
        filtered_flags = filtered_flags[filtered_flags["severity"] == selected_sev]
    if selected_fsub != "All Subdivisions":
        filtered_flags = filtered_flags[filtered_flags["subdivision"] == selected_fsub]

    col_flag_count, col_flag_dl = st.columns([3, 1])
    with col_flag_count:
        st.markdown(f"Displaying **{len(filtered_flags)}** quality flags:")
    with col_flag_dl:
        get_csv_download_button(filtered_flags, "monsoonlens_quality_flags_filtered.csv", "Export Filtered Flags")

    st.dataframe(filtered_flags, use_container_width=True, height=360, hide_index=True)

    st.markdown(
        """
        <div class="scientific-callout">
            <b>Historical Observation Gaps Identified:</b><br>
            • <b>Andaman & Nicobar Islands:</b> WWII interruption (1943–1945), 1909, 1948.<br>
            • <b>Arunachal Pradesh:</b> Gauge recording commenced in 1916 (1901–1915 absent), missing 1954–1956.<br>
            • <b>Lakshadweep:</b> Missing observation in 1918.<br>
            <i>All 33 other subdivisions have complete 115-year continuous records without a single missing month.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Tab 3: Project Research Reports
# -----------------------------------------------------------------------------
with tab_reports:
    st.markdown("##### Project Stage Reports & Scientific Documentation")
    st.markdown("Explore the detailed Markdown documentation generated by each stage of MonsoonLens AI.")

    report_file_map = {
        "Exploratory Data Analysis Report (Stage 2)": EDA_REPORT_PATH,
        "Historical vs Recent Rainfall Monitor Report": HISTORICAL_RECENT_REPORT_PATH,
        "Anomaly Detection Research Report": ANOMALY_REPORT_PATH,
        "Change-Point Analysis Research Report": CHANGE_POINT_REPORT_PATH,
        "Rainfall Forecasting Pipeline Report": FORECASTING_REPORT_PATH,
        "Model Evaluation & Benchmark Report": MODEL_EVALUATION_REPORT_PATH,
        "Backend Architecture Documentation": BACKEND_ARCHITECTURE_REPORT_PATH,
        "Preprocessing & Cleaning Report (Stage 1)": PREPROCESSING_REPORT_PATH,
        "Post-Processing Validation Report": VALIDATION_REPORT_PATH,
    }

    report_choice = st.selectbox(
        "Select Report to View",
        options=list(report_file_map.keys()),
        index=0,
    )
    target_report_path = report_file_map[report_choice]

    if target_report_path.exists():
        content = target_report_path.read_text(encoding="utf-8")
        st.download_button(
            label=f"📥 Download {target_report_path.name}",
            data=content,
            file_name=target_report_path.name,
            mime="text/markdown",
        )
        st.markdown(f"<div class='dashboard-card' style='max-height: 600px; overflow-y: auto;'>", unsafe_allow_html=True)
        st.markdown(content)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Report not found at: {target_report_path}")

# -----------------------------------------------------------------------------
# Tab 4: Static Figures Gallery
# -----------------------------------------------------------------------------
with tab_figures:
    st.markdown("##### Publication-Grade Figures (`reports/figures/`)")
    st.markdown("High-resolution static figures generated during Stage 2 Exploratory Data Analysis.")

    figure_files = [
        ("01_annual_rainfall_trends.png", "Figure 1: All-India Annual Rainfall Trends (1901–2015) with 10-Yr Rolling Dynamics and OLS Fit"),
        ("02_monthly_climatology_profile.png", "Figure 2: Chronological Monthly Climatology Distribution and Percentage Share"),
        ("03_seasonal_rainfall_breakdown.png", "Figure 3: Seasonal Precipitation Breakdown and Southwest Monsoon Dominance"),
        ("04_regional_monthly_heatmap.png", "Figure 4: Spatial-Temporal Climatology Heatmap Across All 36 Subdivisions"),
        ("05_subdivision_variability_comparison.png", "Figure 5: Coefficient of Variation (CV%) Ranking Highlighting Arid vs. Humid Dispersion"),
        ("06_subdivision_rainfall_boxplots.png", "Figure 6: Rainfall Distribution Boxplots Highlighting Regional Outlier Spreads"),
        ("07_seasonal_correlation_matrix.png", "Figure 7: Inter-Seasonal and Annual Correlation Heatmap"),
    ]

    fig_choice = st.selectbox(
        "Select Figure to Inspect",
        options=[f[1] for f in figure_files],
        index=0,
    )

    # Find the corresponding file
    matched_filename = [f[0] for f in figure_files if f[1] == fig_choice][0]
    matched_path = FIGURES_DIR / matched_filename

    if matched_path.exists():
        st.image(str(matched_path), caption=fig_choice, use_container_width=True)
        # Download button for the image
        with open(matched_path, "rb") as img_file:
            st.download_button(
                label=f"📥 Download High-Res Image ({matched_filename})",
                data=img_file.read(),
                file_name=matched_filename,
                mime="image/png",
            )
    else:
        st.warning(f"⚠️ Image not found at: {matched_path}")

# -----------------------------------------------------------------------------
# Tab 5: Download Center
# -----------------------------------------------------------------------------
with tab_downloads:
    st.markdown("##### Research Artifacts & Processed Data Download Center")
    st.markdown("Download any of the cleaned, validated datasets in CSV format for offline modeling and academic research.")

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <h4 style="color: {COLOR_CYAN_ACCENT}; margin-top: 0;">📊 Processed Tabular Datasets</h4>
                <p style="color: {COLOR_MUTED_SLATE}; font-size: 0.85rem;">Cleaned, standardized, and validated CSV files ready for machine learning pipelines:</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Cleaned Wide CSV
        if CLEANED_WIDE_PATH.exists():
            wide_bytes = CLEANED_WIDE_PATH.read_bytes()
            st.download_button(
                label="📥 Download cleaned_wide.csv (4,116 rows)",
                data=wide_bytes,
                file_name="cleaned_wide.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # 2. Monthly Rainfall Long CSV
        if MONTHLY_LONG_PATH.exists():
            monthly_bytes = MONTHLY_LONG_PATH.read_bytes()
            st.download_button(
                label="📥 Download monthly_rainfall_long.csv (49,392 rows)",
                data=monthly_bytes,
                file_name="monthly_rainfall_long.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # 3. Annual Rainfall CSV
        if ANNUAL_PATH.exists():
            ann_bytes = ANNUAL_PATH.read_bytes()
            st.download_button(
                label="📥 Download annual_rainfall.csv (4,116 rows)",
                data=ann_bytes,
                file_name="annual_rainfall.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # 4. Seasonal Rainfall CSV
        if SEASONAL_PATH.exists():
            season_bytes = SEASONAL_PATH.read_bytes()
            st.download_button(
                label="📥 Download seasonal_rainfall.csv (16,464 rows)",
                data=season_bytes,
                file_name="seasonal_rainfall.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # 5. Quality Flags CSV
        if QUALITY_FLAGS_PATH.exists():
            flags_bytes = QUALITY_FLAGS_PATH.read_bytes()
            st.download_button(
                label="📥 Download quality_flags.csv (453 flags)",
                data=flags_bytes,
                file_name="quality_flags.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with dl_col2:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <h4 style="color: {COLOR_GREEN_ACCENT}; margin-top: 0;">📑 Metadata & Research Reports</h4>
                <p style="color: {COLOR_MUTED_SLATE}; font-size: 0.85rem;">Comprehensive provenance logs and empirical analysis reports:</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metadata JSON
        if METADATA_PATH.exists():
            meta_bytes = METADATA_PATH.read_bytes()
            st.download_button(
                label="📥 Download dataset_metadata.json",
                data=meta_bytes,
                file_name="dataset_metadata.json",
                mime="application/json",
                use_container_width=True,
            )

        # EDA Report
        if EDA_REPORT_PATH.exists():
            eda_bytes = EDA_REPORT_PATH.read_bytes()
            st.download_button(
                label="📥 Download eda_report.md",
                data=eda_bytes,
                file_name="eda_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Preprocessing Report
        if PREPROCESSING_REPORT_PATH.exists():
            prep_bytes = PREPROCESSING_REPORT_PATH.read_bytes()
            st.download_button(
                label="📥 Download preprocessing_report.md",
                data=prep_bytes,
                file_name="preprocessing_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Anomaly Report
        if ANOMALY_REPORT_PATH.exists():
            st.download_button(
                label="📥 Download anomaly_detection_report.md",
                data=ANOMALY_REPORT_PATH.read_bytes(),
                file_name="anomaly_detection_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Change Point Report
        if CHANGE_POINT_REPORT_PATH.exists():
            st.download_button(
                label="📥 Download change_point_analysis_report.md",
                data=CHANGE_POINT_REPORT_PATH.read_bytes(),
                file_name="change_point_analysis_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Forecasting Report
        if FORECASTING_REPORT_PATH.exists():
            st.download_button(
                label="📥 Download forecasting_report.md",
                data=FORECASTING_REPORT_PATH.read_bytes(),
                file_name="forecasting_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Model Evaluation Report
        if MODEL_EVALUATION_REPORT_PATH.exists():
            st.download_button(
                label="📥 Download model_evaluation_report.md",
                data=MODEL_EVALUATION_REPORT_PATH.read_bytes(),
                file_name="model_evaluation_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Backend Architecture
        if BACKEND_ARCHITECTURE_REPORT_PATH.exists():
            st.download_button(
                label="📥 Download backend_architecture.md",
                data=BACKEND_ARCHITECTURE_REPORT_PATH.read_bytes(),
                file_name="backend_architecture.md",
                mime="text/markdown",
                use_container_width=True,
            )
