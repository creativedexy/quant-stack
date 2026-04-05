"""Fixtures for service-layer tests.

Creates temporary parquet files in a temp directory so tests
run without any real market data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_synthetic_ohlcv


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory tree with synthetic parquet files."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)

    tickers = ["TEST_A", "TEST_B", "TEST_C"]

    for i, ticker in enumerate(tickers):
        df = generate_synthetic_ohlcv(
            ticker=ticker, days=252, start_date="2024-01-02", seed=42 + i
        )
        safe_name = ticker.replace(".", "_").replace("^", "idx_")
        df.to_parquet(processed / f"{safe_name}.parquet")

    return tmp_path


@pytest.fixture
def service_config(tmp_data_dir: Path) -> dict:
    """Config dict pointing at the temporary data directory."""
    return {
        "general": {
            "base_currency": "GBP",
            "timezone": "Europe/London",
            "log_level": "WARNING",
            "data_dir": str(tmp_data_dir / "data"),
            "random_seed": 42,
        },
        "universe": {
            "name": "test",
            "tickers": ["TEST_A", "TEST_B", "TEST_C"],
            "benchmark": "^TEST",
        },
        "data": {
            "source": "synthetic",
            "start_date": "2024-01-01",
            "end_date": None,
            "interval": "1d",
            "fields": ["Open", "High", "Low", "Close", "Volume"],
            "adjust_prices": True,
            "output_format": "parquet",
        },
    }


@pytest.fixture
def data_service(service_config: dict, tmp_data_dir: Path):
    """DataService wired to temporary data."""
    from src.services.data_service import DataService

    svc = DataService(config=service_config)
    # Override data directories to point at temp location
    svc._data_dir = tmp_data_dir / "data"
    svc._processed_dir = tmp_data_dir / "data" / "processed"
    svc._raw_dir = tmp_data_dir / "data" / "raw"
    return svc
