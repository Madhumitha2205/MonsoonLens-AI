# Historical vs Recent Rainfall Monitor - Research & Technical Report

**Project:** MonsoonLens AI: Historical Rainfall Pattern Analysis Using Machine Learning  
**Module:** Historical vs Recent Rainfall Monitor  
**Date:** October 2026  
**Document Classification:** Climatological Methodology & Engineering Documentation  

---

## 1. Executive Summary & Objective

**MonsoonLens AI** has established an empirical meteorological benchmark covering 115 continuous historical years (1901-2015) across India's 36 India Meteorological Department (IMD) subdivisions. However, climate diagnostic and decision-support systems require evaluating recent meteorological observations against this established long-term normal baseline.

The **Historical vs Recent Rainfall Monitor** benchmarks recent observations against the historical baseline. It provides:
1. **Integration with India's Open Government Data (data.gov.in) Platform**, accessing candidate IMD rain gauge records under the Government Open Data License (GODL) - India.
2. **Dual-Mode Operation**: Queries live REST APIs when credentials are provided, with resilient fallback to a local cached dataset when offline.
3. **Multi-Metric Departure Analysis**, combining absolute departure (mm), percentage departure (%), standardized anomaly ($z$-score), and empirical percentile ranking.
4. **IMD Standard Categorical Classification** (Large Excess, Excess, Normal, Deficient, Large Deficient, and No Rain).
5. **Separation of 36-Subdivision Rankings from National Aggregates**, ensuring national comparisons do not distort regional rankings.
6. **Interactive Multipage Streamlit Application** on `pages/6_Historical_vs_Recent_Monitor.py` featuring Plotly distribution curves, 36-subdivision ranking bar charts, and exportable data reports.

---

## 2. Data Provenance & Verification Status

### 2.1 Confirmed Facts vs. Unverified Assumptions

To ensure complete scientific integrity, this project explicitly distinguishes confirmed source facts from unverified assumptions:

| Attribute | Status / Specification | Notes |
| :--- | :--- | :--- |
| **Historical Baseline (1901-2015)** | **Confirmed** | Primary processed dataset compiled from historical IMD sub-divisional records across 36 subdivisions. |
| **Platform Target** | **Confirmed** | Open Government Data (OGD) Platform India ([https://data.gov.in/](https://data.gov.in/)). |
| **Candidate Resource IDs** | **Unverified Candidate Endpoints** | Configured candidate resource IDs: `fb5e2a33-3e34-4087-83a1-f0e46f2748d0` (Rainfall in India) and `7146eb85-c33f-4436-b75a-25dad6ea9d8a` (All-India Area Weighted Rainfall). Exact API availability and field mapping depend on data.gov.in platform maintenance. |
| **Local Cached Dataset** | **Provenance Not Independently Verified** | Bundled under `data/external/data_gov_in_subdivision_recent.csv`. Acts as an offline sample dataset (2016-2023). Its exact correspondence to official IMD release archives has not been independently verified. |
| **Source Update Date** | **Not Available in Offline Cache** | The date of the source catalog's latest release update is not established from the offline file and must not be confused with the observation years (2016-2023). |
| **Measurement Units** | **Millimeters (mm)** | Standard precipitation depth in mm. |

> [!WARNING]
> **Provenance Disclaimer:** The local CSV cache must not be described as "Verified Official" merely because it is present in the repository. Users requiring official regulatory figures must execute a live authenticated query to data.gov.in or consult verified IMD publications directly.

---

## 3. Client Architecture & Data Validation

### 3.1 Dual-Mode Operation
- **Live Mode:** When `DATA_GOV_IN_API_KEY` is present in the environment or `.streamlit/secrets.toml`, the client queries the data.gov.in REST API. All parameters and API keys are safely URL-encoded (`urllib.parse.urlencode`). API keys are never displayed in cleartext, printed in logs, or surfaced in UI cards.
- **Offline Mode:** When running without an API key or when network connectivity is absent, the client operates deterministically using the local cached dataset, labeled as *"Local cached dataset - provenance not independently verified"*.

### 3.2 Data Validation Pipeline (`RecentRainfallAdapter.validate_dataset`)
The adapter enforces rigorous structural validation before any analytical comparison:
1. **Primary Key Validation:** Validates presence of `SUBDIVISION` and `YEAR`.
2. **Year Range Bounds:** Validates that observation years are positive integers within standard historical/operational ranges (1800-2100).
3. **Duplicate Detection & Rejection:** Rejects duplicate `(SUBDIVISION, YEAR)` records. Duplicates are never silently resolved by picking the first record.
4. **Unknown Subdivision Detection:** Checks all region names against the 36 canonical IMD subdivisions. Unrecognized names are reported as validation warnings and excluded from canonical statistics.
5. **Physical Non-Negativity:** Rejects negative rainfall values.
6. **Strict Non-Imputation:** Preserves missing values as `NaN`. Never replaces missing values with zero or partial sums.

---

## 4. Mathematical Formulation & Climatological Baselines

Let $X_{\text{hist}} = \{x_1, x_2, \dots, x_N\}$ represent the historical time series of rainfall observations for a given subdivision and period from 1901 to 2015 ($N = 115$ years).

### 4.1 Climatological Baseline Statistics
1. **Climatological Mean ($\mu_{\text{hist}}$):**
   $$\mu_{\text{hist}} = \frac{1}{N} \sum_{i=1}^{N} x_i$$
2. **Historical Standard Deviation ($\sigma_{\text{hist}}$):**
   $$\sigma_{\text{hist}} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu_{\text{hist}})^2}$$
3. **Climatological Median ($M_{\text{hist}}$) & Interquartile Range ($IQR$):**
   $$IQR = Q_3 - Q_1$$

### 4.2 Departure Metrics for Recent Observation ($R_{\text{recent}}$)
1. **Absolute Departure ($\Delta$):**
   $$\Delta = R_{\text{recent}} - \mu_{\text{hist}} \quad (\text{in } mm)$$
2. **Percentage Departure ($\Delta_{\%}$):**
   $$\Delta_{\%} = \left( \frac{R_{\text{recent}} - \mu_{\text{hist}}}{\mu_{\text{hist}}} \right) \times 100 \quad (\text{for } \mu_{\text{hist}} > 10^{-4})$$
3. **Standardized Anomaly ($z$-score):**
   $$z = \frac{R_{\text{recent}} - \mu_{\text{hist}}}{\sigma_{\text{hist}}} \quad (\text{for } \sigma_{\text{hist}} > 10^{-4})$$
4. **Empirical Historical Percentile Rank ($P$):**
   $$P = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(x_i < R_{\text{recent}}) \times 100$$
   where $\mathbb{I}(\cdot)$ is the indicator function.

---

## 5. National Aggregate Comparison vs. 36-Subdivision Rankings

> [!IMPORTANT]
> **Separation of ALL-INDIA from Regional Rankings:**
> 1. **Baseline Limitation:** The historical baseline for ALL-INDIA in this repository is calculated as the **unweighted arithmetic mean** across the 36 subdivisions (1901-2015), because the matching official area-weighted historical series is not present in the local processed dataset. Because subdivisional land areas differ significantly in geographic extent, this figure is explicitly labeled as an unweighted mean and must not be described as the official IMD national rainfall total.
> 2. **Exclusion from Subdivisional Rankings:** The national aggregate is strictly excluded from the 36-subdivision ranking chart and category distribution percentages. Regional rankings and percentages sum to 100% across the 36 subdivisions only.

---

## 6. India Meteorological Department (IMD) Departure Scheme

MonsoonLens AI adopts the operational meteorological classification criteria established by the India Meteorological Department:

| IMD Category | Criteria ($\Delta_{\%}$) | Climatological Description | Dashboard Color Code |
| :--- | :--- | :--- | :--- |
| **Large Excess (LE)** | $\Delta_{\%} \ge +60\%$ | Rainfall substantially above historical normal | `#1d4ed8` (Deep Blue) |
| **Excess (E)** | $+20\% \le \Delta_{\%} \le +59\%$ | Rainfall moderately above historical normal | `#0284c7` (Sky Blue) |
| **Normal (N)** | $-19\% \le \Delta_{\%} \le +19\%$ | Rainfall within typical climatological variability | `#10b981` (Emerald Green) |
| **Deficient (D)** | $-59\% \le \Delta_{\%} \le -20\%$ | Rainfall moderately below historical normal | `#f59e0b` (Amber) |
| **Large Deficient (LD)**| $\Delta_{\%} \le -60\%$ | Rainfall substantially below normal | `#ef4444` (Crimson Red) |
| **No Rain (NR)** | $\Delta_{\%} = -100\%$ ($R_{\text{recent}} = 0$) | Total absence of precipitation | `#64748b` (Slate Grey) |
| **Observation Missing** | Observation is `NaN` | No valid observation available | `#94a3b8` (Muted Slate) |

---

## 7. Operational Constraints & Non-Causal Boundary

To maintain scientific integrity, the following caveats must be preserved:

> [!NOTE]
> 1. **Meteorological Departure vs. Surface Flooding:**
>    A positive rainfall departure ($\ge +20\%$) indicates greater atmospheric precipitation than normal. It does **not** model hydrological river discharge, floodplain inundation, or urban flash flooding. Inundation is a function of antecedent moisture, peak hourly rainfall intensity, topography, and stormwater infrastructure.
> 2. **Meteorological Dryness vs. Drought Severity:**
>    A negative departure indicates meteorological deficiency. It does **not** equate directly to agricultural or hydrological drought, which depends on soil moisture retention, reservoir water levels, and groundwater replenishment.
> 3. **Non-Imputation Policy:**
>    Unobserved values are strictly recorded as `NaN` and never imputed with zeros, preventing distortion of baseline statistics.

---

## 8. Exact Steps Required to Independently Verify the Data Source

To independently verify the recent observation dataset against official sources:
1. Register on the Government of India Open Government Data Platform at [https://data.gov.in/](https://data.gov.in/).
2. Navigate to the catalog page: [https://data.gov.in/catalog/rainfall-india](https://data.gov.in/catalog/rainfall-india).
3. Confirm the active resource ID and inspect the data schema preview (verify column names, temporal coverage, and units).
4. Generate an individual API key from your profile dashboard.
5. Set `DATA_GOV_IN_API_KEY` in your `.env` file or Streamlit secrets.
6. Launch the dashboard (`streamlit run app.py`) and click **"Refresh from Live API"** on Page 6 to execute a verified live query directly against the government API.
