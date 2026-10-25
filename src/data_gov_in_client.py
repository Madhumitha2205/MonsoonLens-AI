"""
MonsoonLens AI - Open Government Data Platform (data.gov.in) Client.

Provides an HTTP client to query data.gov.in REST APIs for Indian rainfall observations.

Features:
- Dual API key discovery (environment variable DATA_GOV_IN_API_KEY and st.secrets).
- Safe parameter URL-encoding via standard library (urllib.parse).
- API key protection: API keys are masked in logs, URLs for display, and exception messages.
- Clear distinction between live API responses and unverified local cached fallbacks.
- 100% mocked offline unit testability.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.parse
import pandas as pd
import requests

from src.config import (
    DATA_GOV_IN_ALL_INDIA_RESOURCE_ID,
    DATA_GOV_IN_BASE_URL,
    DATA_GOV_IN_DEFAULT_RESOURCE_ID,
    DATA_GOV_IN_METADATA_PATH,
    DATA_GOV_IN_REQUEST_TIMEOUT,
    RECENT_RAINFALL_PATH,
)

logger = logging.getLogger(__name__)


def mask_sensitive_url(url: str) -> str:
    """Mask api-key query parameter in URL strings for safe logging and display."""
    return re.sub(r"([?&]api-key=)[^&]+", r"\1***", url, flags=re.IGNORECASE)


def sanitize_error_message(msg: str) -> str:
    """Strip cleartext API key from any error message."""
    return re.sub(r"([?&]api-key=)[^&\s'\"]+", r"\1***", msg, flags=re.IGNORECASE)


class DataGovInClient:
    """Client for querying the Open Government Data (OGD) Platform India (data.gov.in)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DATA_GOV_IN_BASE_URL,
        default_resource_id: str = DATA_GOV_IN_DEFAULT_RESOURCE_ID,
        timeout: int = DATA_GOV_IN_REQUEST_TIMEOUT,
        session: Optional[requests.Session] = None,
    ):
        """
        Initialize the data.gov.in API client.

        Parameters
        ----------
        api_key : Optional[str]
            data.gov.in API key. If omitted, attempts to resolve from
            DATA_GOV_IN_API_KEY environment variable or st.secrets.
        base_url : str
            Base URL endpoint for the OGD platform API.
        default_resource_id : str
            Default candidate resource identifier for subdivision rainfall.
        timeout : int
            Request timeout in seconds.
        session : Optional[requests.Session]
            Optional requests.Session instance for dependency injection and testing.
        """
        self.base_url = base_url.rstrip("/") + "/"
        self.default_resource_id = default_resource_id
        self.timeout = timeout
        self.session = session or requests.Session()
        self.api_key = self._resolve_api_key(api_key)

    @staticmethod
    def _resolve_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
        """Resolve API key with precedence: explicit > env > streamlit secrets."""
        if explicit_key is not None:
            cleaned = str(explicit_key).strip()
            return cleaned if cleaned else None

        env_key = os.environ.get("DATA_GOV_IN_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()

        try:
            import streamlit as st
            if hasattr(st, "secrets") and "DATA_GOV_IN_API_KEY" in st.secrets:
                secret_val = st.secrets["DATA_GOV_IN_API_KEY"]
                if secret_val and str(secret_val).strip():
                    return str(secret_val).strip()
        except Exception:
            pass

        return None

    def _sanitize_string(self, text: str) -> str:
        """Sanitize any text to ensure API key is masked in query params and verbatim text."""
        if not text:
            return ""
        cleaned = sanitize_error_message(str(text))
        if self.api_key and len(self.api_key) >= 4:
            cleaned = cleaned.replace(self.api_key, "***")
        return cleaned

    def has_api_key(self) -> bool:
        """Check whether a non-empty API key is configured."""
        return bool(self.api_key and len(self.api_key.strip()) > 0)

    def build_url(
        self,
        resource_id: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
        filters: Optional[Dict[str, str]] = None,
        mask_key: bool = False,
    ) -> str:
        """
        Construct a safely URL-encoded REST query URL for the resource.

        Parameters
        ----------
        resource_id : Optional[str]
            Target resource UUID. Defaults to self.default_resource_id.
        limit : int
            Page size limit.
        offset : int
            Starting index offset for pagination.
        filters : Optional[Dict[str, str]]
            Key-value filters to apply to the request.
        mask_key : bool
            If True, mask the API key as '***' for display/logging.
        """
        rid = resource_id or self.default_resource_id
        params: List[Tuple[str, str]] = [
            ("format", "json"),
            ("offset", str(offset)),
            ("limit", str(limit)),
        ]

        if self.api_key:
            key_val = "***" if mask_key else self.api_key
            params.append(("api-key", key_val))

        if filters:
            for k, v in sorted(filters.items()):
                params.append((f"filters[{k}]", str(v)))

        query_string = urllib.parse.urlencode(params)
        return f"{self.base_url}{rid}?{query_string}"

    def fetch_resource(
        self,
        resource_id: Optional[str] = None,
        filters: Optional[Dict[str, str]] = None,
        limit: int = 1000,
        offset: int = 0,
        max_pages: int = 5,
    ) -> Dict[str, Any]:
        """
        Fetch dataset records from the data.gov.in API with safe parameter encoding.

        Returns
        -------
        Dict[str, Any]
            Structured dictionary containing status, data DataFrame (if successful),
            metadata, and informative message.
        """
        if not self.has_api_key():
            return {
                "status": "API_KEY_MISSING",
                "data": None,
                "metadata": None,
                "records_count": 0,
                "is_incomplete": False,
                "message": (
                    "data.gov.in API key is not configured. Set DATA_GOV_IN_API_KEY in "
                    "your environment or .streamlit/secrets.toml to enable live API queries."
                ),
            }

        rid = resource_id or self.default_resource_id
        endpoint_url = f"{self.base_url}{rid}"
        all_records: List[Dict[str, Any]] = []
        current_offset = offset
        page = 0
        raw_meta: Dict[str, Any] = {}
        total = 0

        try:
            while page < max_pages:
                params: Dict[str, Any] = {
                    "format": "json",
                    "offset": current_offset,
                    "limit": limit,
                    "api-key": self.api_key,
                }
                if filters:
                    for k, v in filters.items():
                        params[f"filters[{k}]"] = str(v)

                response = self.session.get(endpoint_url, params=params, timeout=self.timeout)

                if response.status_code in (401, 403):
                    return {
                        "status": "AUTH_ERROR",
                        "data": None,
                        "metadata": None,
                        "records_count": 0,
                        "is_incomplete": False,
                        "message": (
                            f"Authentication failed (HTTP {response.status_code}). "
                            "Please check that your DATA_GOV_IN_API_KEY is valid and activated."
                        ),
                    }

                if response.status_code != 200:
                    clean_text = self._sanitize_string(response.text[:200])
                    return {
                        "status": "HTTP_ERROR",
                        "data": None,
                        "metadata": None,
                        "records_count": 0,
                        "is_incomplete": False,
                        "message": f"data.gov.in API returned HTTP {response.status_code}: {clean_text}",
                    }

                payload = response.json()
                total = int(payload.get("total", 0) or 0)
                raw_meta = {
                    "title": payload.get("title", ""),
                    "desc": payload.get("desc", ""),
                    "org_type": payload.get("org_type", ""),
                    "source": payload.get("source", ""),
                    "source_update_date": payload.get("updated_date", payload.get("data_updated_date", "Unknown")),
                    "total": total,
                    "count": payload.get("count", 0),
                    "limit": payload.get("limit", limit),
                    "offset": payload.get("offset", current_offset),
                }

                records = payload.get("records", [])
                if not records:
                    break

                all_records.extend(records)
                page += 1

                if len(all_records) >= total or len(records) < limit or page >= max_pages:
                    break

                current_offset += len(records)

            is_incomplete = bool(total > 0 and len(all_records) < total and page >= max_pages)
            raw_meta["is_incomplete"] = is_incomplete
            raw_meta["total_records_available"] = total
            raw_meta["retrieved_records_count"] = len(all_records)

            if not all_records:
                return {
                    "status": "NO_RECORDS",
                    "data": pd.DataFrame(),
                    "metadata": raw_meta,
                    "records_count": 0,
                    "is_incomplete": False,
                    "message": f"No records returned by data.gov.in live API (Resource: {rid}).",
                }

            df = pd.DataFrame(all_records)

            # Validate live response schema against rainfall domain requirements
            cols_upper = {str(c).strip().upper().replace(" ", "_"): c for c in df.columns}
            has_sub = any(k in cols_upper for k in ["SUBDIVISION", "SUB_DIVISION", "REGION"])
            has_yr = any(k in cols_upper for k in ["YEAR", "OBSERVATION_YEAR", "Y"])
            month_abbrs = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
            has_rainfall = any(
                k in cols_upper
                for k in ["ANNUAL", "ANNUAL_TOTAL", "RAINFALL", "TOT_RAINFALL", "VALUE", "AMOUNT"] + month_abbrs
            )

            missing_fields = []
            if not has_sub:
                missing_fields.append("subdivision identifier ('SUBDIVISION')")
            if not has_yr:
                missing_fields.append("observation year ('YEAR')")
            if not has_rainfall:
                missing_fields.append("rainfall observation values (e.g., 'ANNUAL' or monthly columns)")

            if missing_fields:
                return {
                    "status": "INVALID_SCHEMA",
                    "data": None,
                    "metadata": raw_meta,
                    "records_count": len(df),
                    "is_incomplete": is_incomplete,
                    "message": (
                        f"Live API response schema validation failed: missing {', '.join(missing_fields)}. "
                        f"Received columns: {list(df.columns)}."
                    ),
                }

            if is_incomplete:
                success_msg = (
                    f"Retrieved {len(df)} of {total} records from data.gov.in live API (Resource: {rid}). "
                    f"Notice: Dataset is partial/incomplete because pagination reached maximum page limit ({max_pages} pages)."
                )
            else:
                success_msg = f"Successfully retrieved {len(df)} records from data.gov.in live API (Resource: {rid})."

            return {
                "status": "SUCCESS",
                "data": df,
                "metadata": raw_meta,
                "records_count": len(df),
                "is_incomplete": is_incomplete,
                "message": success_msg,
            }

        except requests.Timeout:
            return {
                "status": "TIMEOUT",
                "data": None,
                "metadata": None,
                "records_count": 0,
                "is_incomplete": False,
                "message": f"Connection to data.gov.in timed out after {self.timeout} seconds.",
            }
        except requests.RequestException as e:
            return {
                "status": "NETWORK_ERROR",
                "data": None,
                "metadata": None,
                "records_count": 0,
                "is_incomplete": False,
                "message": self._sanitize_string(f"Network error connecting to data.gov.in: {str(e)}"),
            }
        except Exception as e:
            return {
                "status": "PARSE_ERROR",
                "data": None,
                "metadata": None,
                "records_count": 0,
                "is_incomplete": False,
                "message": self._sanitize_string(f"Failed to parse API response: {str(e)}"),
            }

    @staticmethod
    def load_cached_metadata(metadata_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load local documentation for candidate data.gov.in resources."""
        path = metadata_path or DATA_GOV_IN_METADATA_PATH
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading metadata from {path}: {e}")
        return {
            "source_platform": "Open Government Data (OGD) Platform India",
            "platform_url": "https://data.gov.in/",
            "provenance_status": "Local cached dataset - provenance not independently verified",
            "license": "GODL-India",
        }

    def load_cached_recent_data(
        self,
        csv_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Load local cached recent rainfall dataset.

        Returns
        -------
        Dict[str, Any]
            Structured status dictionary with DataFrame, unverified provenance label,
            and candidate metadata.
        """
        path = csv_path or RECENT_RAINFALL_PATH
        metadata = self.load_cached_metadata()

        if not path.exists():
            return {
                "status": "CACHE_NOT_FOUND",
                "data": None,
                "metadata": metadata,
                "records_count": 0,
                "message": f"Local cached dataset file not found at {path}.",
            }

        try:
            df = pd.read_csv(path, encoding="utf-8")
            return {
                "status": "SUCCESS",
                "source": "CACHED_OFFLINE",
                "data": df,
                "metadata": metadata,
                "records_count": len(df),
                "provenance_status": "Local cached dataset - provenance not independently verified",
                "message": (
                    f"Loaded {len(df)} records from local cached dataset "
                    "(provenance not independently verified against live IMD registry)."
                ),
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "data": None,
                "metadata": metadata,
                "records_count": 0,
                "message": f"Error reading cached recent rainfall dataset: {e}",
            }

    def get_recent_rainfall_dataset(
        self,
        force_live: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Retrieve recent rainfall data, favoring live API if requested and configured,
        or falling back to the local cached dataset.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Dataframe of recent observations and status/provenance metadata.
        """
        live_failure_info: Optional[Dict[str, Any]] = None

        if force_live:
            live_result = self.fetch_resource()
            if (
                live_result["status"] == "SUCCESS"
                and live_result["data"] is not None
                and not live_result["data"].empty
            ):
                meta = live_result.get("metadata", {}) or {}
                is_incomplete = live_result.get("is_incomplete", False)
                info = {
                    "source": "LIVE_API",
                    "resource_id": self.default_resource_id,
                    "catalog_url": "https://data.gov.in/catalog/rainfall-india",
                    "is_live": True,
                    "provenance_status": "Live API Response (data.gov.in) - Unverified public OGD feed",
                    "source_update_date": meta.get("source_update_date", "Live query"),
                    "units": "Millimeters (mm)",
                    "is_incomplete": is_incomplete,
                    "fallback_from_live_failure": False,
                    "message": live_result["message"],
                    "metadata": meta,
                }
                return live_result["data"], info
            else:
                live_failure_info = live_result

        cached_result = self.load_cached_recent_data()
        if cached_result["status"] == "SUCCESS" and cached_result["data"] is not None:
            meta = cached_result.get("metadata", {}) or {}
            cached_props = meta.get("cached_data_properties", {})

            if live_failure_info:
                live_status = live_failure_info.get("status", "UNKNOWN_ERROR")
                live_msg = live_failure_info.get("message", "Live query failed.")
                msg = (
                    f"Live API query failed ({live_status}: {live_msg}). "
                    "Fell back to local cached dataset (provenance not independently verified)."
                )
                info = {
                    "source": "CACHED_OFFLINE",
                    "resource_id": self.default_resource_id,
                    "catalog_url": "https://data.gov.in/catalog/rainfall-india",
                    "is_live": False,
                    "provenance_status": "Local cached dataset - provenance not independently verified",
                    "source_update_date": cached_props.get("source_update_date", "Not available in offline cache"),
                    "units": cached_props.get("units", "Millimeters (mm)"),
                    "observed_year_range": cached_props.get("observed_year_range", [2016, 2023]),
                    "fallback_from_live_failure": True,
                    "live_failure_status": live_status,
                    "live_failure_message": live_msg,
                    "is_incomplete": False,
                    "message": msg,
                    "metadata": meta,
                }
            else:
                info = {
                    "source": "CACHED_OFFLINE",
                    "resource_id": self.default_resource_id,
                    "catalog_url": "https://data.gov.in/catalog/rainfall-india",
                    "is_live": False,
                    "provenance_status": "Local cached dataset - provenance not independently verified",
                    "source_update_date": cached_props.get("source_update_date", "Not available in offline cache"),
                    "units": cached_props.get("units", "Millimeters (mm)"),
                    "observed_year_range": cached_props.get("observed_year_range", [2016, 2023]),
                    "fallback_from_live_failure": False,
                    "is_incomplete": False,
                    "message": cached_result["message"],
                    "metadata": meta,
                }
            return cached_result["data"], info

        raise FileNotFoundError(
            f"Could not load recent rainfall data from live API or cache ({cached_result.get('message')})"
        )
