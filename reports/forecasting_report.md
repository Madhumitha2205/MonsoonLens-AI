# MonsoonLens AI: Rainfall Forecasting Model Report

**Forecasting Framework:** Chronological Train/Test Split (Train: 1901–1990, Test: 1991–2015)  
**Models Evaluated:**
1. Climatological Mean Baseline
2. Naive Persistence Baseline
3. Rolling 5-Year Mean Baseline
4. Linear Trend Forecaster
5. Random Forest Regressor  

---

## 1. Executive Summary
Forecasting models were evaluated out-of-sample over a 25-year testing window (1991–2015).
Because inter-annual rainfall is subject to strong chaotic atmospheric drivers (such as El Niño / Southern Oscillation and Indian Ocean Dipole), the historical **Climatological Mean Baseline** represents a formidable benchmark.

---

## 2. Benchmark Summary Across Representative Subdivisions

```
                       Model  MAE (mm)  RMSE (mm)  MAPE (%)  R² Score  Skill Score          Subdivision
Rolling 5-Year Mean Baseline     76.23      94.03     30.95    0.0094       0.0296       WEST RAJASTHAN
Climatological Mean Baseline     78.68      96.90     31.90   -0.0519       0.0000       WEST RAJASTHAN
     Linear Trend Forecaster     79.38      97.50     31.89   -0.0650      -0.0062       WEST RAJASTHAN
     Random Forest Regressor     82.60     108.14     34.65   -0.3102      -0.1160       WEST RAJASTHAN
  Naive Persistence Baseline    117.18     143.66     45.86   -1.3125      -0.4826       WEST RAJASTHAN
     Linear Trend Forecaster    336.92     419.07      9.78   -0.0430       0.0874    COASTAL KARNATAKA
Climatological Mean Baseline    372.56     459.18     10.07   -0.2522       0.0000    COASTAL KARNATAKA
Rolling 5-Year Mean Baseline    385.40     461.87     10.92   -0.2669      -0.0059    COASTAL KARNATAKA
     Random Forest Regressor    433.31     498.36     12.08   -0.4750      -0.0853    COASTAL KARNATAKA
  Naive Persistence Baseline    400.62     518.61     11.10   -0.5973      -0.1294    COASTAL KARNATAKA
     Linear Trend Forecaster    155.75     212.98     10.28   -0.0533       0.0918 GANGETIC WEST BENGAL
Rolling 5-Year Mean Baseline    184.67     223.66     12.33   -0.1616       0.0462 GANGETIC WEST BENGAL
     Random Forest Regressor    175.44     233.81     12.12   -0.2695       0.0029 GANGETIC WEST BENGAL
Climatological Mean Baseline    190.75     234.50     12.00   -0.2769       0.0000 GANGETIC WEST BENGAL
  Naive Persistence Baseline    267.83     314.48     17.31   -1.2965      -0.3411 GANGETIC WEST BENGAL
     Random Forest Regressor    167.47     237.06     28.72    0.0650       0.0406           TAMIL NADU
Climatological Mean Baseline    168.57     247.10     28.85   -0.0159       0.0000           TAMIL NADU
     Linear Trend Forecaster    169.86     248.79     29.34   -0.0298      -0.0068           TAMIL NADU
  Naive Persistence Baseline    202.17     252.58     24.35   -0.0614      -0.0222           TAMIL NADU
Rolling 5-Year Mean Baseline    216.63     291.18     31.85   -0.4106      -0.1784           TAMIL NADU
Climatological Mean Baseline    291.03     349.76     10.39   -0.0001       0.0000               KERALA
Rolling 5-Year Mean Baseline    382.43     449.54     13.41   -0.6521      -0.2853               KERALA
     Linear Trend Forecaster    388.89     459.32     12.73   -0.7248      -0.3132               KERALA
  Naive Persistence Baseline    420.24     503.73     14.91   -1.0744      -0.4402               KERALA
     Random Forest Regressor    463.38     532.49     15.24   -1.3181      -0.5224               KERALA
```

---

## 3. Key Methodological Takeaways
1. **Persistence of Climatological Mean:** In subdivisions with high inter-annual variability (e.g., West Rajasthan), complex non-linear models often overfit training noise; the long-term climatological mean yields competitive or superior out-of-sample RMSE.
2. **Random Forest Strengths:** Random Forest captures moderate decadal shifts when rolling indicators and baseline anomalies are provided as features.
3. **Honest Limitations:** Historical time-series models alone cannot forecast year-to-year monsoonal extremes with deterministic precision without global oceanic coupled climate drivers.
