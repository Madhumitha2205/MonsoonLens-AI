"""
MonsoonLens AI - Unit Tests for data.gov.in Official API Client.

Validates API key resolution, safe URL encoding, API key masking,
timeout protection, HTTP error handling, and unverified offline cache fallback.
All network calls are strictly 100% mocked (no external HTTP traffic).
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import requests

from src.data_gov_in_client import (
    DataGovInClient,
    mask_sensitive_url,
    sanitize_error_message,
)


class TestDataGovInClient(unittest.TestCase):
    """Test suite for DataGovInClient."""

    def test_client_init_and_key_resolution_explicit(self):
        client = DataGovInClient(api_key="my_test_key_123")
        self.assertEqual(client.api_key, "my_test_key_123")
        self.assertTrue(client.has_api_key())

    @patch.dict(os.environ, {"DATA_GOV_IN_API_KEY": "env_test_key_456"}, clear=False)
    def test_client_init_and_key_resolution_env(self):
        client = DataGovInClient()
        self.assertEqual(client.api_key, "env_test_key_456")
        self.assertTrue(client.has_api_key())

    @patch.dict(os.environ, {}, clear=True)
    def test_client_init_no_key(self):
        client = DataGovInClient()
        self.assertFalse(client.has_api_key())
        self.assertIsNone(client.api_key)

    def test_build_url_safe_encoding_and_masking(self):
        client = DataGovInClient(api_key="secret_key_xyz")
        url_raw = client.build_url(
            resource_id="abc-123",
            limit=50,
            offset=100,
            filters={"subdivision": "ANDAMAN & NICOBAR"},
            mask_key=False,
        )
        self.assertIn("abc-123", url_raw)
        self.assertIn("limit=50", url_raw)
        self.assertIn("offset=100", url_raw)
        self.assertIn("api-key=secret_key_xyz", url_raw)
        # Check safe URL encoding of '&' in filter
        self.assertIn("filters%5Bsubdivision%5D=ANDAMAN+%26+NICOBAR", url_raw)

        # Check masked URL
        url_masked = client.build_url(
            resource_id="abc-123",
            limit=50,
            offset=100,
            filters={"year": "2020"},
            mask_key=True,
        )
        self.assertIn("api-key=%2A%2A%2A", url_masked)
        self.assertNotIn("secret_key_xyz", url_masked)

    def test_mask_sensitive_url_and_sanitize_error(self):
        url = "https://api.data.gov.in/resource/abc?format=json&api-key=SECRET12345&limit=10"
        masked = mask_sensitive_url(url)
        self.assertNotIn("SECRET12345", masked)
        self.assertIn("api-key=***", masked)

        err_msg = "Failed to connect to https://api.data.gov.in?api-key=SUPERSECRET&error=failed"
        sanitized = sanitize_error_message(err_msg)
        self.assertNotIn("SUPERSECRET", sanitized)
        self.assertIn("api-key=***", sanitized)

    def test_fetch_resource_missing_api_key_returns_gracefully(self):
        client = DataGovInClient(api_key="")
        result = client.fetch_resource(resource_id="sample-id")
        self.assertEqual(result["status"], "API_KEY_MISSING")
        self.assertIsNone(result["data"])
        self.assertIn("API key is not configured", result["message"])

    def test_fetch_resource_mocked_success(self):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Subdivision Rainfall",
            "desc": "Rainfall records",
            "updated_date": "2024-01-15",
            "total": 2,
            "count": 2,
            "limit": 1000,
            "offset": 0,
            "records": [
                {"SUBDIVISION": "BIHAR", "YEAR": "2020", "ANNUAL": 1200.5},
                {"SUBDIVISION": "KERALA", "YEAR": "2020", "ANNUAL": 2900.0},
            ],
        }
        mock_session.get.return_value = mock_response

        client = DataGovInClient(api_key="valid_mock_key", session=mock_session)
        result = client.fetch_resource(resource_id="test-res")

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["records_count"], 2)
        self.assertIsInstance(result["data"], pd.DataFrame)
        self.assertEqual(len(result["data"]), 2)
        self.assertIn("BIHAR", result["data"]["SUBDIVISION"].values)
        self.assertEqual(result["metadata"]["source_update_date"], "2024-01-15")

    def test_fetch_resource_mocked_auth_error(self):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_session.get.return_value = mock_response

        client = DataGovInClient(api_key="invalid_mock_key", session=mock_session)
        result = client.fetch_resource()

        self.assertEqual(result["status"], "AUTH_ERROR")
        self.assertIsNone(result["data"])
        self.assertIn("Authentication failed", result["message"])

    def test_fetch_resource_mocked_http_server_error(self):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_session.get.return_value = mock_response

        client = DataGovInClient(api_key="valid_key", session=mock_session)
        result = client.fetch_resource()

        self.assertEqual(result["status"], "HTTP_ERROR")
        self.assertIsNone(result["data"])

    def test_fetch_resource_mocked_timeout(self):
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.Timeout("Connection timed out")

        client = DataGovInClient(api_key="valid_key", session=mock_session)
        result = client.fetch_resource()

        self.assertEqual(result["status"], "TIMEOUT")
        self.assertIsNone(result["data"])
        self.assertIn("timed out", result["message"])

    def test_load_cached_metadata_unverified_provenance(self):
        client = DataGovInClient()
        meta = client.load_cached_metadata()
        self.assertIsInstance(meta, dict)
        self.assertIn("provenance_status", meta)
        self.assertIn("not independently verified", meta["provenance_status"])

        cache_res = client.load_cached_recent_data()
        self.assertEqual(cache_res["status"], "SUCCESS")
        self.assertIsInstance(cache_res["data"], pd.DataFrame)
        self.assertGreater(len(cache_res["data"]), 0)
        self.assertEqual(
            cache_res["provenance_status"],
            "Local cached dataset - provenance not independently verified",
        )

    def test_get_recent_rainfall_dataset_fallback(self):
        client = DataGovInClient(api_key="")
        df, info = client.get_recent_rainfall_dataset(force_live=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(info["is_live"])
        self.assertEqual(info["source"], "CACHED_OFFLINE")
        self.assertEqual(
            info["provenance_status"],
            "Local cached dataset - provenance not independently verified",
        )
        self.assertFalse(info["fallback_from_live_failure"])
        self.assertIn("SUBDIVISION", df.columns)
        self.assertIn("ANNUAL", df.columns)

    def test_key_resolution_precedence(self):
        # 1. Explicit key overrides env key
        with patch.dict(os.environ, {"DATA_GOV_IN_API_KEY": "env_key_abc"}):
            client = DataGovInClient(api_key="explicit_key_xyz")
            self.assertEqual(client.api_key, "explicit_key_xyz")

        # 2. Explicit empty string does not pick up env key
        with patch.dict(os.environ, {"DATA_GOV_IN_API_KEY": "env_key_abc"}):
            client = DataGovInClient(api_key="")
            self.assertIsNone(client.api_key)

        # 3. Streamlit secrets fallback when env is empty
        with patch.dict(os.environ, {}, clear=True):
            mock_st = MagicMock()
            mock_st.secrets = {"DATA_GOV_IN_API_KEY": "secrets_key_999"}
            with patch.dict("sys.modules", {"streamlit": mock_st}):
                client = DataGovInClient()
                self.assertEqual(client.api_key, "secrets_key_999")

    def test_key_masking_verbatim_key_in_string(self):
        client = DataGovInClient(api_key="very_secret_api_key_42")
        # Masking in URL parameter
        url = "https://api.data.gov.in/test?api-key=very_secret_api_key_42&other=1"
        self.assertEqual(client._sanitize_string(url), "https://api.data.gov.in/test?api-key=***&other=1")
        # Masking verbatim token in text
        text_with_key = "Connection failed for user key very_secret_api_key_42 at endpoint"
        sanitized = client._sanitize_string(text_with_key)
        self.assertNotIn("very_secret_api_key_42", sanitized)
        self.assertIn("***", sanitized)

    def test_fetch_resource_incomplete_pagination_detection(self):
        mock_session = MagicMock()
        # Mock responses for 2 pages where total is 2500 but limit is 500 and max_pages is 2
        page1 = {
            "title": "Rainfall",
            "total": 2500,
            "records": [{"SUBDIVISION": f"SUB_{i}", "YEAR": 2020, "ANNUAL": 1000.0} for i in range(500)],
        }
        page2 = {
            "title": "Rainfall",
            "total": 2500,
            "records": [{"SUBDIVISION": f"SUB_{i+500}", "YEAR": 2020, "ANNUAL": 1000.0} for i in range(500)],
        }
        resp1 = MagicMock(status_code=200)
        resp1.json.return_value = page1
        resp2 = MagicMock(status_code=200)
        resp2.json.return_value = page2
        mock_session.get.side_effect = [resp1, resp2]

        client = DataGovInClient(api_key="valid_key", session=mock_session)
        result = client.fetch_resource(limit=500, max_pages=2)

        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(result["is_incomplete"])
        self.assertTrue(result["metadata"]["is_incomplete"])
        self.assertEqual(result["records_count"], 1000)
        self.assertEqual(result["metadata"]["total_records_available"], 2500)
        self.assertIn("partial/incomplete", result["message"])

    def test_fetch_resource_invalid_schema_validation(self):
        mock_session = MagicMock()
        # Response missing required subdivision and rainfall columns
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "title": "Random Table",
            "total": 1,
            "records": [{"YEAR": 2020, "STATION_CODE": "XYZ123"}],
        }
        mock_session.get.return_value = mock_resp

        client = DataGovInClient(api_key="valid_key", session=mock_session)
        result = client.fetch_resource()

        self.assertEqual(result["status"], "INVALID_SCHEMA")
        self.assertIsNone(result["data"])
        self.assertIn("subdivision identifier", result["message"])
        self.assertIn("rainfall observation values", result["message"])

    def test_get_recent_rainfall_dataset_force_live_failure_fallback_diagnostics(self):
        mock_session = MagicMock()
        mock_resp = MagicMock(status_code=500, text="Internal Server Error")
        mock_session.get.return_value = mock_resp

        client = DataGovInClient(api_key="valid_key", session=mock_session)
        df, info = client.get_recent_rainfall_dataset(force_live=True)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(info["is_live"])
        self.assertEqual(info["source"], "CACHED_OFFLINE")
        self.assertTrue(info["fallback_from_live_failure"])
        self.assertEqual(info["live_failure_status"], "HTTP_ERROR")
        self.assertIn("HTTP 500", info["live_failure_message"])
        self.assertIn("Fell back to local cached dataset", info["message"])
        self.assertEqual(
            info["provenance_status"],
            "Local cached dataset - provenance not independently verified",
        )

    def test_get_recent_rainfall_dataset_force_live_success(self):
        mock_session = MagicMock()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "title": "Live Rainfall",
            "total": 1,
            "updated_date": "2025-06-01",
            "records": [{"SUBDIVISION": "KERALA", "YEAR": 2021, "ANNUAL": 2850.5}],
        }
        mock_session.get.return_value = mock_resp

        client = DataGovInClient(api_key="valid_key", session=mock_session)
        df, info = client.get_recent_rainfall_dataset(force_live=True)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(info["is_live"])
        self.assertEqual(info["source"], "LIVE_API")
        self.assertFalse(info["fallback_from_live_failure"])
        self.assertEqual(
            info["provenance_status"],
            "Live API Response (data.gov.in) - Unverified public OGD feed",
        )
        self.assertEqual(info["source_update_date"], "2025-06-01")
        self.assertEqual(len(df), 1)
        self.assertIn("KERALA", df["SUBDIVISION"].values)


if __name__ == "__main__":
    unittest.main()

