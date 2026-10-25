"""
Tests for src/backend_pipeline.py - Backend orchestration facade and report verification.
"""

import pytest
from pathlib import Path

from src.backend_pipeline import BackendPipeline
from src.config import (
    ANOMALY_REPORT_PATH,
    CHANGE_POINT_REPORT_PATH,
    FORECASTING_REPORT_PATH,
    MODEL_EVALUATION_REPORT_PATH,
    BACKEND_ARCHITECTURE_REPORT_PATH,
)


def test_backend_pipeline_initialization():
    """Verify BackendPipeline instantiates and loads base data."""
    pipeline = BackendPipeline()
    assert not pipeline.annual_df.empty
    assert len(pipeline.subdivisions) == 36


def test_backend_reports_exist_and_non_empty():
    """Verify that all 5 technical research and architecture reports exist and contain markdown."""
    reports = [
        ANOMALY_REPORT_PATH,
        CHANGE_POINT_REPORT_PATH,
        FORECASTING_REPORT_PATH,
        MODEL_EVALUATION_REPORT_PATH,
        BACKEND_ARCHITECTURE_REPORT_PATH,
    ]

    for r_path in reports:
        assert r_path.exists(), f"Report missing: {r_path}"
        content = r_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 100, f"Report {r_path.name} is unexpectedly empty or short"
        assert content.startswith("# "), f"Report {r_path.name} does not start with top-level H1 header"
