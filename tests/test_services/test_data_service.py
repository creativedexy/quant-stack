"""Tests for DataService — the dashboard data access layer."""

from __future__ import annotations

import pandas as pd
import pytest

from src.services.data_service import DataService


class TestDataServiceInit:
    """DataService instantiation."""

    def test_instantiates_with_config(self, service_config: dict) -> None:
        """DataService can be created with an explicit config dict."""
        svc = DataService(config=service_config)
        assert svc._tickers == ["TEST_A", "TEST_B", "TEST_C"]

    def test_instantiates_with_empty_config(self) -> None:
        """DataService falls back to defaults when config is empty."""
        svc = DataService(config={})
        assert svc._tickers == []


class TestGetPrices:
    """Loading price data from parquet files."""

    def test_returns_dataframe_with_correct_structure(
        self, data_service: DataService
    ) -> None:
        """get_prices returns a DataFrame with tickers as columns."""
        prices = data_service.get_prices()
        assert isinstance(prices, pd.DataFrame)
        assert isinstance(prices.index, pd.DatetimeIndex)
        assert set(prices.columns) == {"TEST_A", "TEST_B", "TEST_C"}
        assert len(prices) > 0

    def test_filters_by_tickers(self, data_service: DataService) -> None:
        """get_prices respects the tickers argument."""
        prices = data_service.get_prices(tickers=["TEST_A"])
        assert list(prices.columns) == ["TEST_A"]

    def test_returns_empty_dataframe_when_no_data(
        self, service_config: dict
    ) -> None:
        """get_prices returns empty DataFrame when no parquet files exist."""
        svc = DataService(config=service_config)
        # Point at a non-existent directory
        from pathlib import Path
        svc._processed_dir = Path("/tmp/nonexistent_data_dir_12345")
        svc._raw_dir = Path("/tmp/nonexistent_raw_dir_12345")
        prices = svc.get_prices()
        assert isinstance(prices, pd.DataFrame)
        assert prices.empty

    def test_date_filtering(self, data_service: DataService) -> None:
        """get_prices filters by start and end dates."""
        prices = data_service.get_prices(
            start="2024-06-01", end="2024-08-01"
        )
        assert isinstance(prices, pd.DataFrame)
        if not prices.empty:
            assert prices.index.min() >= pd.Timestamp("2024-06-01")
            assert prices.index.max() <= pd.Timestamp("2024-08-01")


class TestGetLatestPrices:
    """Most recent price per ticker."""

    def test_returns_series_with_one_price_per_ticker(
        self, data_service: DataService
    ) -> None:
        """get_latest_prices returns a Series indexed by ticker."""
        latest = data_service.get_latest_prices()
        assert isinstance(latest, pd.Series)
        assert len(latest) == 3
        assert set(latest.index) == {"TEST_A", "TEST_B", "TEST_C"}
        # Prices should be positive
        assert (latest > 0).all()

    def test_returns_empty_series_when_no_data(
        self, service_config: dict
    ) -> None:
        """get_latest_prices returns empty Series when no data."""
        svc = DataService(config=service_config)
        from pathlib import Path
        svc._processed_dir = Path("/tmp/nonexistent_data_dir_12345")
        svc._raw_dir = Path("/tmp/nonexistent_raw_dir_12345")
        latest = svc.get_latest_prices()
        assert isinstance(latest, pd.Series)
        assert latest.empty


class TestGetDataStatus:
    """Data status summary."""

    def test_returns_all_expected_keys(
        self, data_service: DataService
    ) -> None:
        """get_data_status returns dict with the documented keys."""
        status = data_service.get_data_status()
        expected_keys = {
            "last_updated",
            "tickers_available",
            "date_range",
            "data_source",
            "file_sizes",
        }
        assert set(status.keys()) == expected_keys

    def test_tickers_available_populated(
        self, data_service: DataService
    ) -> None:
        """get_data_status lists available tickers from parquet files."""
        status = data_service.get_data_status()
        assert len(status["tickers_available"]) == 3

    def test_file_sizes_are_positive(
        self, data_service: DataService
    ) -> None:
        """Each file size should be a positive integer."""
        status = data_service.get_data_status()
        for ticker, size in status["file_sizes"].items():
            assert size > 0

    def test_returns_defaults_when_no_data(self) -> None:
        """get_data_status handles missing directories gracefully."""
        svc = DataService(config={})
        from pathlib import Path
        svc._processed_dir = Path("/tmp/nonexistent_data_dir_12345")
        svc._raw_dir = Path("/tmp/nonexistent_raw_dir_12345")
        status = svc.get_data_status()
        assert status["tickers_available"] == []
        assert status["last_updated"] is None
        assert status["date_range"] is None


class TestGetReturns:
    """Returns computation."""

    def test_returns_dataframe(self, data_service: DataService) -> None:
        """get_returns produces a DataFrame of daily returns."""
        returns = data_service.get_returns(window=60)
        assert isinstance(returns, pd.DataFrame)
        assert not returns.empty
        # Returns should be small daily values, not raw prices
        assert returns.abs().max().max() < 1.0


class TestGetFeatures:
    """Feature loading."""

    def test_returns_empty_when_no_features(
        self, data_service: DataService
    ) -> None:
        """get_features returns empty DataFrame when no feature file."""
        features = data_service.get_features("TEST_A")
        assert isinstance(features, pd.DataFrame)
        assert features.empty
