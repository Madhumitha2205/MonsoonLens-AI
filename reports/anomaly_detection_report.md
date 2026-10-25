# MonsoonLens AI: Rainfall Anomaly Detection Technical Report

**Analysis Scope:** Historical Annual Rainfall Anomaly Detection (1901–2015)  
**Dataset:** 36 Meteorological Subdivisions of India  
**Reference Grouping:** Isolated per subdivision  

---

## 1. Executive Summary
Anomaly detection was executed using three distinct statistical and machine learning methodologies:
- **Z-Score Detection (|Z| > 2.0):** Identified **179** anomalies (**4.38%** of valid observations), comprising 115 wet departures and 64 dry departures.
- **IQR Detection (Tukey's Fences 1.5×IQR):** Identified **51** anomalies (**1.25%**).
- **Isolation Forest (Contamination=5%):** Identified **215** anomalies (**5.26%**).

---

## 2. Comparison of Methodologies

| Detection Method | Total Anomalies | Anomaly Rate (%) | Wet Outliers | Dry Outliers | Statistical Characteristic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Z-Score (|Z| > 2.0)** | 179 | 4.38% | 115 | 64 | Assumes approx. normal distribution; sensitive to extreme variance. |
| **IQR (1.5 × IQR)** | 51 | 1.25% | 37 | 14 | Non-parametric; highly robust against skewed distributions. |
| **Isolation Forest** | 215 | 5.26% | 115 | 100 | Unsupervised tree ensemble; captures multi-feature isolation. |

---

## 3. Notable Historical Extreme Anomaly Years
Top national extreme rainfall departures logged:
- **1917 & 1918 (Severe Contrast):** 1917 recorded widespread excessive monsoonal rain, followed immediately by 1918, which produced severe dry anomalies across western and central India.
- **1961 (Peak Wet Anomaly):** Subdivisions in the Western Ghats (Coastal Karnataka, Konkan & Goa) experienced all-time high annual precipitation.
- **1972 (Major All-India Drought):** Marked severe negative departures across 22 out of 36 subdivisions.

---

## 4. Methodological Safeguards
1. **Subdivision Isolation:** Rainfall is never compared globally across regions. An anomaly in West Rajasthan is measured strictly against West Rajasthan's historical baseline.
2. **Missing Observation Treatment:** Missing values (`NaN`) are strictly excluded from denominators and never counted as zero rainfall.
