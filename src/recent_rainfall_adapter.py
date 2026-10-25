"""
MonsoonLens AI - Recent Rainfall Data Adapter & Schema Harmonizer.

Harmonizes raw datasets fetched from data.gov.in (or local cached records)
to conform with MonsoonLens AI's internal schemas and canonical 36 IMD subdivisions.

Features:
- Column case normalization and schema mapping.
- Comprehensive data validation (required columns, year bounds, negative rainfall, duplicates).
- Unknown subdivision detection and reporting (never silently treats unrecognized regions as canonical).
- Duplicate (SUBDIVISION, YEAR) record detection and rejection (never silently selects the first record).
- Strict non-imputation of missing values (preserves NaN, avoids partial sums).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    MONTH_COLUMNS,
    SEASON_COLUMNS,
    SEASON_MAPPING,
)

logger = logging.getLogger(__name__)

# Canonical 36 IMD Subdivisions from 1901-2015 historical dataset
CANONICAL_SUBDIVISIONS: List[str] = [
    "ANDAMAN & NICOBAR ISLANDS",
    "ARUNACHAL PRADESH",
    "ASSAM & MEGHALAYA",
    "BIHAR",
    "CHHATTISGARH",
    "COASTAL ANDHRA PRADESH",
    "COASTAL KARNATAKA",
    "EAST MADHYA PRADESH",
    "EAST RAJASTHAN",
    "EAST UTTAR PRADESH",
    "GANGETIC WEST BENGAL",
    "GUJARAT REGION",
    "HARYANA DELHI & CHANDIGARH",
    "HIMACHAL PRADESH",
    "JAMMU & KASHMIR",
    "JHARKHAND",
    "KERALA",
    "KONKAN & GOA",
    "LAKSHADWEEP",
    "MADHYA MAHARASHTRA",
    "MATATHWADA",
    "NAGA MANI MIZO TRIPURA",
    "NORTH INTERIOR KARNATAKA",
    "ORISSA",
    "PUNJAB",
    "RAYALSEEMA",
    "SAURASHTRA & KUTCH",
    "SOUTH INTERIOR KARNATAKA",
    "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    "TAMIL NADU",
    "TELANGANA",
    "UTTARAKHAND",
    "VIDARBHA",
    "WEST MADHYA PRADESH",
    "WEST RAJASTHAN",
    "WEST UTTAR PRADESH",
]

# Alias map mapping common external naming variations to canonical names
SUBDIVISION_ALIAS_MAP: Dict[str, str] = {
    "ANDAMAN AND NICOBAR ISLANDS": "ANDAMAN & NICOBAR ISLANDS",
    "A & N ISLANDS": "ANDAMAN & NICOBAR ISLANDS",
    "ASSAM AND MEGHALAYA": "ASSAM & MEGHALAYA",
    "ODISHA": "ORISSA",
    "MARATHWADA": "MATATHWADA",
    "RAYALASEEMA": "RAYALSEEMA",
    "KONKAN AND GOA": "KONKAN & GOA",
    "SAURASHTRA AND KUTCH": "SAURASHTRA & KUTCH",
    "SAURASHTRA, KUTCH & DIU": "SAURASHTRA & KUTCH",
    "HARYANA, DELHI & CHANDIGARH": "HARYANA DELHI & CHANDIGARH",
    "HARYANA, DELHI AND CHANDIGARH": "HARYANA DELHI & CHANDIGARH",
    "HARYANA DELHI AND CHANDIGARH": "HARYANA DELHI & CHANDIGARH",
    "SUB-HIMALAYAN WEST BENGAL & SIKKIM": "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    "SUB HIMALAYAN WEST BENGAL AND SIKKIM": "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    "SUB HIMALAYAN WEST BENGAL": "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    "GANGETIC W. BENGAL": "GANGETIC WEST BENGAL",
    "N.M.M.T.": "NAGA MANI MIZO TRIPURA",
    "NAGALAND, MANIPUR, MIZORAM & TRIPURA": "NAGA MANI MIZO TRIPURA",
    "NAGALAND MANIPUR MIZORAM AND TRIPURA": "NAGA MANI MIZO TRIPURA",
    "NAGALAND MANIPUR MIZORAM TRIPURA": "NAGA MANI MIZO TRIPURA",
    "COASTAL AP": "COASTAL ANDHRA PRADESH",
    "TELENGANA": "TELANGANA",
    "UTTARANCHAL": "UTTARAKHAND",
    "ALL INDIA": "ALL-INDIA",
    "INDIA": "ALL-INDIA",
}

VALID_SUBDIVISIONS_SET: Set[str] = set(CANONICAL_SUBDIVISIONS) | {"ALL-INDIA"}


class RecentRainfallAdapter:
    """Adapts, validates, and standardizes recent rainfall data for comparison."""

    @staticmethod
    def standardize_subdivision_name(name: str) -> str:
        """
        Standardize raw subdivision string to canonical IMD name.

        Parameters
        ----------
        name : str
            Raw subdivision name from source dataset.

        Returns
        -------
        str
            Canonical subdivision name or sanitized uppercase string.
        """
        if not isinstance(name, str):
            return ""

        clean = re.sub(r"\s+", " ", name.strip().upper())
        clean = clean.replace("&AMP;", "&")

        if clean in SUBDIVISION_ALIAS_MAP:
            return SUBDIVISION_ALIAS_MAP[clean]

        if clean in CANONICAL_SUBDIVISIONS or clean == "ALL-INDIA":
            return clean

        alt_clean = clean.replace(" AND ", " & ").replace(",", "").replace("-", " ")
        alt_clean = re.sub(r"\s+", " ", alt_clean).strip()
        if alt_clean in CANONICAL_SUBDIVISIONS:
            return alt_clean
        if alt_clean in SUBDIVISION_ALIAS_MAP:
            return SUBDIVISION_ALIAS_MAP[alt_clean]

        return clean

    @classmethod
    def is_known_subdivision(cls, name: str) -> bool:
        """Check whether a subdivision name maps to a known canonical subdivision or ALL-INDIA."""
        std = cls.standardize_subdivision_name(name)
        return std in VALID_SUBDIVISIONS_SET

    @classmethod
    def validate_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Systematically validate the recent rainfall dataset.

        Checks:
        1. Required primary columns (SUBDIVISION, YEAR).
        2. Valid year ranges (1800 to 2100).
        3. Duplicate (SUBDIVISION, YEAR) records.
        4. Unknown subdivision names.
        5. Negative rainfall values.

        Returns
        -------
        Dict[str, Any]
            Validation report dictionary with is_valid, errors, and warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if df.empty:
            errors.append("Dataset is completely empty.")
            return {"is_valid": False, "errors": errors, "warnings": warnings}

        # Check required columns
        cols_upper = {str(c).strip().upper().replace(" ", "_"): c for c in df.columns}
        has_sub = any(k in cols_upper for k in ["SUBDIVISION", "SUB_DIVISION", "REGION"])
        has_yr = any(k in cols_upper for k in ["YEAR", "OBSERVATION_YEAR", "Y"])

        if not has_sub:
            errors.append("Missing required subdivision identifier column ('SUBDIVISION').")
        if not has_yr:
            errors.append("Missing required year column ('YEAR').")

        if errors:
            return {"is_valid": False, "errors": errors, "warnings": warnings}

        # Identify duplicate records
        sub_col = cols_upper.get("SUBDIVISION", cols_upper.get("SUB_DIVISION", cols_upper.get("REGION")))
        yr_col = cols_upper.get("YEAR", cols_upper.get("OBSERVATION_YEAR", cols_upper.get("Y")))

        std_subs = df[sub_col].astype(str).apply(cls.standardize_subdivision_name)
        years_num = pd.to_numeric(df[yr_col], errors="coerce")

        # Invalid years
        invalid_years = years_num[(years_num < 1800) | (years_num > 2100) | years_num.isna()]
        if not invalid_years.empty:
            errors.append(f"Found {len(invalid_years)} records with invalid or out-of-range observation years.")

        # Duplicate subdivision-year combinations
        key_df = pd.DataFrame({"sub": std_subs, "yr": years_num})
        duplicates = key_df[key_df.duplicated(subset=["sub", "yr"], keep=False)]
        if not duplicates.empty:
            dup_samples = duplicates.drop_duplicates().head(5).to_dict(orient="records")
            errors.append(
                f"Found {len(duplicates)} duplicate subdivision-year records. "
                f"Samples: {dup_samples}. Duplicate entries cannot be resolved unambiguously."
            )

        # Unknown subdivision names
        unknown_subs = [s for s in std_subs.unique() if s not in VALID_SUBDIVISIONS_SET and s != ""]
        if unknown_subs:
            warnings.append(
                f"Found {len(unknown_subs)} unrecognized subdivision label(s): {unknown_subs}. "
                "These do not correspond to the canonical 36 IMD subdivisions or ALL-INDIA."
            )

        # Negative rainfall check
        rainfall_cols = [c for c in df.columns if c not in [sub_col, yr_col]]
        neg_count = 0
        for col in rainfall_cols:
            num_vals = pd.to_numeric(df[col], errors="coerce")
            neg_count += int((num_vals < 0).sum())

        if neg_count > 0:
            errors.append(f"Found {neg_count} physically impossible negative rainfall value(s).")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "duplicates_count": len(duplicates),
            "unknown_subdivisions": unknown_subs,
        }

    @classmethod
    def harmonize_dataset(
        cls,
        df: pd.DataFrame,
        reject_duplicates: bool = True,
        filter_unknown_subdivisions: bool = True,
    ) -> pd.DataFrame:
        """
        Harmonize column headers, data types, and subdivision labels with strict validation.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe loaded from data.gov.in API or local file.
        reject_duplicates : bool
            If True, raise ValueError when duplicate (SUBDIVISION, YEAR) records exist.
        filter_unknown_subdivisions : bool
            If True, filter out uncanonical/unrecognized subdivision rows with a warning.

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with columns:
            ['SUBDIVISION', 'YEAR', 'JAN'..'DEC', 'ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
        """
        if df.empty:
            return pd.DataFrame(columns=["SUBDIVISION", "YEAR"] + MONTH_COLUMNS + ["ANNUAL"] + SEASON_COLUMNS)

        val_res = cls.validate_dataset(df)
        if reject_duplicates and val_res["duplicates_count"] > 0:
            raise ValueError(f"Recent data validation error: {'; '.join(val_res['errors'])}")

        res = df.copy()

        # Map column names case-insensitively
        col_rename = {}
        for c in res.columns:
            c_str = str(c).strip()
            c_upper = c_str.upper().replace(" ", "_").replace("-", "_")

            if c_upper in ["SUBDIVISION", "SUB_DIVISION", "SUB_DIVISIONS", "REGION"]:
                col_rename[c] = "SUBDIVISION"
            elif c_upper in ["YEAR", "OBSERVATION_YEAR", "Y"]:
                col_rename[c] = "YEAR"
            elif c_upper in ["ANNUAL", "ANNUAL_TOTAL", "ANNUAL_RAINFALL", "TOT_ANNUAL"]:
                col_rename[c] = "ANNUAL"
            elif c_upper in ["JAN_FEB", "JAN_TO_FEB", "WINTER"]:
                col_rename[c] = "Jan-Feb"
            elif c_upper in ["MAR_MAY", "MAR_TO_MAY", "PRE_MONSOON", "SUMMER"]:
                col_rename[c] = "Mar-May"
            elif c_upper in ["JUN_SEP", "JUN_TO_SEP", "MONSOON", "SOUTHWEST_MONSOON"]:
                col_rename[c] = "Jun-Sep"
            elif c_upper in ["OCT_DEC", "OCT_TO_DEC", "POST_MONSOON"]:
                col_rename[c] = "Oct-Dec"
            else:
                for m in MONTH_COLUMNS:
                    if c_upper == m or c_upper.startswith(f"{m}_"):
                        col_rename[c] = m
                        break

        res.rename(columns=col_rename, inplace=True)

        if "SUBDIVISION" not in res.columns or "YEAR" not in res.columns:
            raise ValueError(
                f"Dataset missing required primary key columns ('SUBDIVISION', 'YEAR'). "
                f"Found columns: {list(res.columns)}"
            )

        # Standardize subdivision names
        res["SUBDIVISION"] = res["SUBDIVISION"].astype(str).apply(cls.standardize_subdivision_name)
        res["YEAR"] = pd.to_numeric(res["YEAR"], errors="coerce")
        res = res.dropna(subset=["YEAR"])
        res["YEAR"] = res["YEAR"].astype(int)

        # Filter out unknown/unrecognized subdivisions if configured
        if filter_unknown_subdivisions:
            res = res[res["SUBDIVISION"].isin(VALID_SUBDIVISIONS_SET)]

        # Convert rainfall columns to numeric, replacing negative numbers with NaN
        for col in MONTH_COLUMNS:
            if col in res.columns:
                res[col] = pd.to_numeric(res[col], errors="coerce")
                res.loc[res[col] < 0, col] = np.nan
            else:
                res[col] = np.nan

        # Recompute seasonal columns if missing or incomplete
        for season_name, constituent_months in SEASON_MAPPING.items():
            if season_name not in res.columns or res[season_name].isna().all():
                # Rejection of partial sums: compute as sum strictly if all constituent months are non-null
                res[season_name] = res[constituent_months].apply(
                    lambda row: row.sum() if row.notna().all() else np.nan, axis=1
                ).round(1)
            else:
                res[season_name] = pd.to_numeric(res[season_name], errors="coerce")
                res.loc[res[season_name] < 0, season_name] = np.nan
                res[season_name] = res[season_name].round(1)

        # Recompute ANNUAL if missing or incomplete
        if "ANNUAL" not in res.columns or res["ANNUAL"].isna().all():
            # Rejection of partial sums: strictly compute if all 12 months are non-null
            res["ANNUAL"] = res[MONTH_COLUMNS].apply(
                lambda row: row.sum() if row.notna().all() else np.nan, axis=1
            ).round(1)
        else:
            res["ANNUAL"] = pd.to_numeric(res["ANNUAL"], errors="coerce")
            res.loc[res["ANNUAL"] < 0, "ANNUAL"] = np.nan
            res["ANNUAL"] = res["ANNUAL"].round(1)

        res = res[res["SUBDIVISION"] != ""]
        res = res.sort_values(by=["SUBDIVISION", "YEAR"]).reset_index(drop=True)

        expected_cols = ["SUBDIVISION", "YEAR"] + MONTH_COLUMNS + ["ANNUAL"] + SEASON_COLUMNS
        present_cols = [c for c in expected_cols if c in res.columns]
        return res[present_cols]

    @classmethod
    def get_observation_value(
        cls,
        df: pd.DataFrame,
        subdivision: str,
        year: int,
        period_type: str,
        period_value: str,
    ) -> Optional[float]:
        """
        Extract observed rainfall value in mm for a given subdivision and period.

        Parameters
        ----------
        df : pd.DataFrame
            Harmonized dataframe.
        subdivision : str
            Subdivision name or 'ALL-INDIA'.
        year : int
            Observation year.
        period_type : str
            'Annual', 'Seasonal', or 'Monthly'.
        period_value : str
            'ANNUAL', season name (e.g., 'Jun-Sep'), or month name (e.g., 'JUL').

        Returns
        -------
        Optional[float]
            Rainfall observation in mm, or None if missing or unavailable.
        """
        std_sub = cls.standardize_subdivision_name(subdivision)
        subset = df[(df["SUBDIVISION"] == std_sub) & (df["YEAR"] == int(year))]

        if subset.empty:
            return None

        if len(subset) > 1:
            raise ValueError(
                f"Multiple duplicate records ({len(subset)}) found for subdivision '{std_sub}' "
                f"and year {year}. Cannot determine observation unambiguously."
            )

        row = subset.iloc[0]

        ptype_clean = period_type.strip().lower()
        if ptype_clean == "annual":
            target_col = "ANNUAL"
        elif ptype_clean in ["seasonal", "season"]:
            target_col = period_value
        elif ptype_clean in ["monthly", "month"]:
            target_col = period_value.upper()
        else:
            target_col = period_value

        if target_col in row and pd.notna(row[target_col]):
            return float(row[target_col])

        return None
