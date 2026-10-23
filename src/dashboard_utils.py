"""
MonsoonLens AI - Streamlit Dashboard Utilities.

Provides reusable cached dataset loaders, schema validators, statistical aggregators,
Plotly chart generators, and UI components for the MonsoonLens AI web application.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    ANNUAL_PATH,
    CLEANED_WIDE_PATH,
    DATA_QUALITY_SUMMARY_PATH,
    EDA_REPORT_PATH,
    FIGURES_DIR,
    METADATA_PATH,
    MONTH_COLUMNS,
    MONTH_NUMBERS,
    MONTHLY_LONG_PATH,
    PREPROCESSING_REPORT_PATH,
    QUALITY_FLAGS_PATH,
    REPORTS_DIR,
    SEASON_COLUMNS,
    SEASON_DESCRIPTIONS,
    SEASONAL_PATH,
    VALIDATION_REPORT_PATH,
)

# -----------------------------------------------------------------------------
# Color Palette and Styling Constants
# -----------------------------------------------------------------------------
COLOR_MIDNIGHT = "#0B1220"
COLOR_NAVY_CARD = "#132238"
COLOR_PRIMARY_BLUE = "#3182CE"
COLOR_CYAN_ACCENT = "#38BDF8"
COLOR_CLOUD_WHITE = "#F8FAFC"
COLOR_MUTED_SLATE = "#94A3B8"
COLOR_GREEN_ACCENT = "#34D399"
COLOR_AMBER_ACCENT = "#FBBF24"
COLOR_ROSE_ACCENT = "#F87171"
COLOR_GRID = "#1E293B"

SEASON_COLORS: Dict[str, str] = {
    "Jan-Feb": "#38BDF8",       # Cyan (Winter)
    "Mar-May": "#FBBF24",       # Amber (Pre-Monsoon/Summer)
    "Jun-Sep": "#3182CE",       # Deep Blue (Southwest Monsoon)
    "Oct-Dec": "#34D399",       # Green (Post-Monsoon/Northeast Monsoon)
}

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLOR_CLOUD_WHITE, family="sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        bgcolor="rgba(19, 34, 56, 0.7)",
        bordercolor=COLOR_GRID,
        borderwidth=1,
    ),
)


def apply_figure_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent scientific theme and axis grid lines across Plotly figures."""
    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, tickfont=dict(color=COLOR_MUTED_SLATE))
    fig.update_yaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, tickfont=dict(color=COLOR_MUTED_SLATE))
    return fig


# -----------------------------------------------------------------------------
# Reusable Cached Data Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset_metadata() -> Dict[str, Any]:
    """Load JSON metadata describing dataset coverage, cleaning specifications, and audit logs."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Dataset metadata file not found at: {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_annual_rainfall() -> pd.DataFrame:
    """Load annual rainfall dataset containing source, calculated, and validated totals."""
    if not ANNUAL_PATH.exists():
        raise FileNotFoundError(f"Annual rainfall dataset not found at: {ANNUAL_PATH}")
    df = pd.read_csv(ANNUAL_PATH)
    validate_required_columns(
        df,
        ["SUBDIVISION", "YEAR", "ANNUAL_VALIDATED", "ANNUAL_QUALITY_FLAG"],
        "annual_rainfall.csv"
    )
    return df


@st.cache_data(show_spinner=False)
def load_monthly_rainfall() -> pd.DataFrame:
    """Load tidy monthly rainfall dataset (subdivision, year, month, date, rainfall, quality)."""
    if not MONTHLY_LONG_PATH.exists():
        raise FileNotFoundError(f"Monthly rainfall dataset not found at: {MONTHLY_LONG_PATH}")
    df = pd.read_csv(MONTHLY_LONG_PATH)
    validate_required_columns(
        df,
        ["SUBDIVISION", "YEAR", "MONTH", "MONTH_NUMBER", "DATE", "RAINFALL", "IS_MISSING"],
        "monthly_rainfall_long.csv"
    )
    return df


@st.cache_data(show_spinner=False)
def load_seasonal_rainfall() -> pd.DataFrame:
    """Load tidy seasonal rainfall series across 4 meteorological seasons."""
    if not SEASONAL_PATH.exists():
        raise FileNotFoundError(f"Seasonal rainfall dataset not found at: {SEASONAL_PATH}")
    df = pd.read_csv(SEASONAL_PATH)
    validate_required_columns(
        df,
        ["SUBDIVISION", "YEAR", "SEASON", "SEASON_NAME", "RAINFALL_VALIDATED"],
        "seasonal_rainfall.csv"
    )
    return df


@st.cache_data(show_spinner=False)
def load_cleaned_wide() -> pd.DataFrame:
    """Load primary wide cleaned dataset with 12 months, source & calculated totals."""
    if not CLEANED_WIDE_PATH.exists():
        raise FileNotFoundError(f"Cleaned wide dataset not found at: {CLEANED_WIDE_PATH}")
    df = pd.read_csv(CLEANED_WIDE_PATH)
    validate_required_columns(
        df,
        ["SUBDIVISION", "YEAR"] + MONTH_COLUMNS,
        "cleaned_wide.csv"
    )
    return df


@st.cache_data(show_spinner=False)
def load_quality_flags() -> pd.DataFrame:
    """Load data-quality audit flags log."""
    if not QUALITY_FLAGS_PATH.exists():
        raise FileNotFoundError(f"Quality flags dataset not found at: {QUALITY_FLAGS_PATH}")
    df = pd.read_csv(QUALITY_FLAGS_PATH)
    validate_required_columns(
        df,
        ["subdivision", "year", "field", "flag_type", "severity", "reason"],
        "quality_flags.csv"
    )
    return df


# -----------------------------------------------------------------------------
# Validation and Quality Checks
# -----------------------------------------------------------------------------
def validate_required_columns(df: pd.DataFrame, required_columns: List[str], dataset_name: str) -> None:
    """Ensure required columns are present in the dataframe; raises ValueError if missing."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


# -----------------------------------------------------------------------------
# Data Filtering and Statistical Transformations
# -----------------------------------------------------------------------------
def get_unique_subdivisions(df: pd.DataFrame) -> List[str]:
    """Return sorted unique meteorological subdivisions from a dataframe."""
    if "SUBDIVISION" in df.columns:
        return sorted(df["SUBDIVISION"].dropna().unique().tolist())
    if "subdivision" in df.columns:
        return sorted(df["subdivision"].dropna().unique().tolist())
    return []


def filter_annual_data(
    df: pd.DataFrame,
    year_range: Tuple[int, int],
    subdivisions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter annual rainfall dataframe by year range and optional subdivisions."""
    mask = (df["YEAR"] >= year_range[0]) & (df["YEAR"] <= year_range[1])
    if subdivisions:
        mask &= df["SUBDIVISION"].isin(subdivisions)
    return df.loc[mask].copy()


def filter_monthly_data(
    df: pd.DataFrame,
    year_range: Tuple[int, int],
    subdivisions: Optional[List[str]] = None,
    months: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter monthly rainfall dataframe by year range, subdivisions, and months."""
    mask = (df["YEAR"] >= year_range[0]) & (df["YEAR"] <= year_range[1])
    if subdivisions:
        mask &= df["SUBDIVISION"].isin(subdivisions)
    if months:
        mask &= df["MONTH"].isin(months)
    return df.loc[mask].copy()


def filter_seasonal_data(
    df: pd.DataFrame,
    year_range: Tuple[int, int],
    subdivisions: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter seasonal rainfall dataframe by year range, subdivisions, and seasons."""
    mask = (df["YEAR"] >= year_range[0]) & (df["YEAR"] <= year_range[1])
    if subdivisions:
        mask &= df["SUBDIVISION"].isin(subdivisions)
    if seasons:
        mask &= df["SEASON"].isin(seasons)
    return df.loc[mask].copy()


def compute_filtered_summary(series: pd.Series) -> Dict[str, float]:
    """
    Compute summary metrics on a rainfall series without imputing missing values.
    Returns count, mean, std, median, min, max, iqr, and cv_percent.
    """
    clean_series = series.dropna()
    n = len(clean_series)
    if n == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            "cv_percent": np.nan,
        }

    mean_val = float(clean_series.mean())
    std_val = float(clean_series.std()) if n > 1 else 0.0
    median_val = float(clean_series.median())
    q25 = float(clean_series.quantile(0.25))
    q75 = float(clean_series.quantile(0.75))
    iqr_val = q75 - q25
    min_val = float(clean_series.min())
    max_val = float(clean_series.max())
    cv_val = (std_val / mean_val * 100.0) if mean_val > 0 else np.nan

    return {
        "count": n,
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "iqr": iqr_val,
        "cv_percent": cv_val,
    }


def format_rainfall_mm(val: Optional[float], decimals: int = 1) -> str:
    """Format numeric rainfall in mm safely."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:,.{decimals}f} mm"


# -----------------------------------------------------------------------------
# Visual Component & UI Helpers
# -----------------------------------------------------------------------------
def apply_custom_css() -> None:
    """Inject polished scientific styling into Streamlit."""
    custom_css = f"""
    <style>
    /* Metric container styling */
    div[data-testid="stMetric"] {{
        background: {COLOR_NAVY_CARD};
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }}
    div[data-testid="stMetric"]:hover {{
        border-color: {COLOR_CYAN_ACCENT};
        transition: border-color 0.2s ease-in-out;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {COLOR_MUTED_SLATE} !important;
        font-weight: 500;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-testid="stMetricValue"] {{
        color: {COLOR_CLOUD_WHITE} !important;
        font-weight: 700;
        font-size: 1.6rem;
    }}

    /* Card styling */
    .dashboard-card {{
        background: {COLOR_NAVY_CARD};
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }}

    /* Section badges */
    .badge-tag {{
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: {COLOR_CYAN_ACCENT};
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }}

    /* Scientific note boxes */
    .scientific-callout {{
        background: rgba(19, 34, 56, 0.85);
        border-left: 4px solid {COLOR_CYAN_ACCENT};
        border-radius: 0 8px 8px 0;
        padding: 12px 18px;
        margin: 14px 0;
        font-size: 0.9rem;
        color: {COLOR_CLOUD_WHITE};
    }}
    .scientific-warning {{
        background: rgba(30, 20, 10, 0.85);
        border-left: 4px solid {COLOR_AMBER_ACCENT};
        border-radius: 0 8px 8px 0;
        padding: 12px 18px;
        margin: 14px 0;
        font-size: 0.9rem;
        color: #FEF3C7;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, icon: str = "🌧️") -> None:
    """Render a consistent page header with branding and sub-description."""
    st.markdown(
        f"""
        <div style="margin-bottom: 24px;">
            <div class="badge-tag">MonsoonLens AI &bull; Meteorological Analytics</div>
            <h1 style="color: {COLOR_CLOUD_WHITE}; font-size: 2.2rem; font-weight: 800; margin: 0 0 6px 0; display: flex; align-items: center; gap: 10px;">
                <span>{icon}</span> {title}
            </h1>
            <p style="color: {COLOR_MUTED_SLATE}; font-size: 1.05rem; margin: 0; line-height: 1.5;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_branding() -> None:
    """Render persistent sidebar brand identity, metadata summary, and project details."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #1E293B; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="font-size: 2rem;">🌧️</div>
                    <div>
                        <div style="font-weight: 800; font-size: 1.25rem; color: {COLOR_CLOUD_WHITE}; letter-spacing: -0.02em;">MonsoonLens AI</div>
                        <div style="font-size: 0.75rem; color: {COLOR_CYAN_ACCENT}; font-weight: 600;">Historical Rainfall Intelligence</div>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.8rem; color: {COLOR_MUTED_SLATE}; line-height: 1.4;">
                    Rigorous, reproducible analysis of 115 years (1901–2015) of Indian rainfall observations across 36 subdivisions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Plotly Chart Creation Functions
# -----------------------------------------------------------------------------
def create_annual_timeseries_plot(
    df: pd.DataFrame,
    title: str = "Historical Annual Rainfall Trend",
    subdivision_name: str = "All India Average",
    show_rolling: bool = True,
    rolling_window: int = 10,
    show_trendline: bool = True,
) -> go.Figure:
    """Create interactive time series chart for annual rainfall with rolling mean and trend line."""
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="No observations available for the selected filters",
            showarrow=False,
            font=dict(color=COLOR_MUTED_SLATE, size=14),
        )
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        return fig

    # Group by year if multiple subdivisions are in df
    if "SUBDIVISION" in df.columns and df["SUBDIVISION"].nunique() > 1:
        yearly = df.groupby("YEAR")["ANNUAL_VALIDATED"].mean().dropna().reset_index()
    else:
        yearly = df[["YEAR", "ANNUAL_VALIDATED"]].dropna().sort_values("YEAR").reset_index(drop=True)

    years = yearly["YEAR"].values
    rainfall = yearly["ANNUAL_VALIDATED"].values

    # 1. Raw annual rainfall series
    fig.add_trace(
        go.Scatter(
            x=years,
            y=rainfall,
            mode="lines+markers",
            name="Annual Rainfall",
            line=dict(color=COLOR_PRIMARY_BLUE, width=1.5),
            marker=dict(size=4, color=COLOR_CYAN_ACCENT),
            hovertemplate="<b>Year:</b> %{x}<br><b>Rainfall:</b> %{y:,.1f} mm<extra></extra>",
        )
    )

    # 2. Moving average line
    if show_rolling and len(yearly) >= rolling_window:
        yearly[f"MA_{rolling_window}"] = yearly["ANNUAL_VALIDATED"].rolling(
            window=rolling_window, min_periods=max(2, rolling_window // 2)
        ).mean()
        fig.add_trace(
            go.Scatter(
                x=yearly["YEAR"],
                y=yearly[f"MA_{rolling_window}"],
                mode="lines",
                name=f"{rolling_window}-Year Rolling Average",
                line=dict(color=COLOR_AMBER_ACCENT, width=2.5),
                hovertemplate=f"<b>Year:</b> %{{x}}<br><b>{rolling_window}-Yr Mean:</b> %{{y:,.1f}} mm<extra></extra>",
            )
        )

    # 3. OLS linear trendline
    if show_trendline and len(years) > 2:
        poly_coeff = np.polyfit(years, rainfall, deg=1)
        trend_y = np.polyval(poly_coeff, years)
        slope_val = poly_coeff[0]
        fig.add_trace(
            go.Scatter(
                x=years,
                y=trend_y,
                mode="lines",
                name=f"Linear Trend ({slope_val:+.2f} mm/yr)",
                line=dict(color=COLOR_ROSE_ACCENT, width=1.8, dash="dot"),
                hovertemplate="<b>Year:</b> %{x}<br><b>Trend:</b> %{y:,.1f} mm<extra></extra>",
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text=f"<b>{title}</b> ({subdivision_name})", font=dict(size=16)),
        xaxis_title="Year",
        yaxis_title="Annual Rainfall (mm)",
        hovermode="x unified",
        height=450,
    )
    return fig


def create_monthly_climatology_plot(
    monthly_stats: pd.DataFrame,
    title: str = "Monthly Climatology Profile",
    subtitle: str = "Average monthly rainfall distribution (mm)",
) -> go.Figure:
    """Create interactive bar chart with chronological months and percentage share annotations."""
    fig = go.Figure()

    if monthly_stats.empty:
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        return fig

    # Map month to season color
    month_to_season_color = {
        "JAN": COLOR_CYAN_ACCENT, "FEB": COLOR_CYAN_ACCENT,
        "MAR": COLOR_AMBER_ACCENT, "APR": COLOR_AMBER_ACCENT, "MAY": COLOR_AMBER_ACCENT,
        "JUN": COLOR_PRIMARY_BLUE, "JUL": COLOR_PRIMARY_BLUE, "AUG": COLOR_PRIMARY_BLUE, "SEP": COLOR_PRIMARY_BLUE,
        "OCT": COLOR_GREEN_ACCENT, "NOV": COLOR_GREEN_ACCENT, "DEC": COLOR_GREEN_ACCENT,
    }
    bar_colors = [month_to_season_color.get(m, COLOR_PRIMARY_BLUE) for m in monthly_stats["MONTH"]]

    fig.add_trace(
        go.Bar(
            x=monthly_stats["MONTH"],
            y=monthly_stats["mean_rainfall"],
            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            name="Mean Rainfall (mm)",
            hovertemplate="<b>Month:</b> %{x}<br><b>Mean Rainfall:</b> %{y:,.1f} mm<extra></extra>",
        )
    )

    # Optional line for percentage of annual total if column exists
    if "percent_of_annual" in monthly_stats.columns:
        fig.add_trace(
            go.Scatter(
                x=monthly_stats["MONTH"],
                y=monthly_stats["percent_of_annual"],
                mode="lines+markers",
                name="Share of Annual (%)",
                yaxis="y2",
                line=dict(color=COLOR_CLOUD_WHITE, width=2),
                marker=dict(size=6, color=COLOR_CYAN_ACCENT),
                hovertemplate="<b>Month:</b> %{x}<br><b>Share of Annual:</b> %{y:.1f}%<extra></extra>",
            )
        )

    layout = PLOTLY_LAYOUT_DEFAULTS.copy()
    fig.update_layout(
        **layout,
        title=dict(text=f"<b>{title}</b><br><span style='font-size:12px; color:{COLOR_MUTED_SLATE}'>{subtitle}</span>", font=dict(size=16)),
        xaxis_title="Month",
        yaxis_title="Mean Rainfall (mm)",
        yaxis2=dict(
            title="Share of Annual (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(color=COLOR_MUTED_SLATE),
            range=[0, max(monthly_stats.get("percent_of_annual", [30])) * 1.25],
        ),
        height=420,
    )
    return fig


def create_seasonal_breakdown_plot(
    seasonal_stats: pd.DataFrame,
    title: str = "Seasonal Rainfall Breakdown",
) -> go.Figure:
    """Create interactive bar and donut charts illustrating seasonal contribution."""
    fig = go.Figure()

    if seasonal_stats.empty:
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        return fig

    colors = [SEASON_COLORS.get(s, COLOR_PRIMARY_BLUE) for s in seasonal_stats["SEASON"]]

    fig.add_trace(
        go.Bar(
            x=seasonal_stats["SEASON_NAME"],
            y=seasonal_stats["mean_rainfall"],
            marker=dict(color=colors),
            hovertemplate="<b>Season:</b> %{x}<br><b>Mean Rainfall:</b> %{y:,.1f} mm<extra></extra>",
            name="Mean Rainfall (mm)",
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text=f"<b>{title}</b>", font=dict(size=16)),
        xaxis_title="Meteorological Season",
        yaxis_title="Mean Rainfall (mm)",
        height=400,
    )
    return fig


def create_subdivision_ranking_plot(
    sub_stats_df: pd.DataFrame,
    metric_col: str = "mean_annual_mm",
    metric_name: str = "Mean Annual Rainfall (mm)",
    top_n: int = 36,
    ascending: bool = False,
) -> go.Figure:
    """Create ranked horizontal bar chart of subdivisions by rainfall or CV."""
    df_sorted = sub_stats_df.sort_values(by=metric_col, ascending=ascending).head(top_n)

    fig = go.Figure()

    if df_sorted.empty:
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        return fig

    # Color gradient based on metric values
    vals = df_sorted[metric_col].values
    norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    colors = [
        f"rgb({int(49 + (56 - 49) * v)}, {int(130 + (189 - 130) * v)}, {int(206 + (248 - 206) * v)})"
        for v in norm_vals
    ]

    fig.add_trace(
        go.Bar(
            x=df_sorted[metric_col],
            y=df_sorted["SUBDIVISION"],
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="<b>Subdivision:</b> %{y}<br><b>" + metric_name + ":</b> %{x:,.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text=f"<b>Subdivision Ranking: {metric_name}</b>", font=dict(size=16)),
        xaxis_title=metric_name,
        yaxis=dict(
            autorange="reversed" if ascending else True,
            tickfont=dict(size=10, color=COLOR_MUTED_SLATE),
            gridcolor=COLOR_GRID,
        ),
        height=max(450, len(df_sorted) * 22),
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, tickfont=dict(color=COLOR_MUTED_SLATE))
    return fig


def create_rainfall_heatmap(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    values_col: str,
    title: str = "Rainfall Intensity Heatmap (mm)",
    colorscale: str = "Viridis",
) -> go.Figure:
    """Create interactive heatmap of rainfall intensity across two categorical or time axes."""
    pivot = df.pivot(index=index_col, columns=columns_col, values=values_col)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text="Rainfall (mm)", font=dict(color=COLOR_CLOUD_WHITE)),
                tickfont=dict(color=COLOR_MUTED_SLATE),
            ),
            hovertemplate=f"<b>{index_col}:</b> %{{y}}<br><b>{columns_col}:</b> %{{x}}<br><b>Rainfall:</b> %{{z:,.1f}} mm<extra></extra>",
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text=f"<b>{title}</b>", font=dict(size=16)),
        xaxis=dict(title=columns_col, tickfont=dict(size=10, color=COLOR_MUTED_SLATE), gridcolor=COLOR_GRID),
        yaxis=dict(title=index_col, autorange="reversed", tickfont=dict(size=10, color=COLOR_MUTED_SLATE), gridcolor=COLOR_GRID),
        height=max(450, len(pivot) * 20),
    )
    return fig


def get_csv_download_button(df: pd.DataFrame, filename: str, label: str = "Download Data (CSV)") -> None:
    """Render a Streamlit CSV download button for any provided dataframe."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 {label}",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )
