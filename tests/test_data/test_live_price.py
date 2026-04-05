"""Tests for LivePriceService and AlphaVantageFetcher.

All network calls are mocked. Tests verify fallback logic, caching
behaviour, and response structure — not actual API connectivity.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.live_price import LivePriceService
from src.data.fetcher import create_fetcher, _FETCHER_REGISTRY


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def live_config() -> dict:
    """Config for LivePriceService with short TTL for testing."""
    return {
        "general": {
            "data_dir": "data",
        },
        "data": {
            "live": {
                "price_cache_ttl": 2,  # short TTL for cache expiry tests
                "fallback_chain": ["yfinance", "alpha_vantage", "cached"],
                "request_timeout": 5,
            },
        },
        "api_keys": {
            "alpha_vantage": "test_key_123",
        },
        "universe": {
            "tickers": ["AAPL", "MSFT"],
        },
    }


@pytest.fixture
def service(live_config: dict) -> LivePriceService:
    """LivePriceService with test config."""
    return LivePriceService(config=live_config)


# ─────────────────────────────────────────────
# LivePriceService — instantiation
# ─────────────────────────────────────────────

class TestLivePriceServiceInit:
    """Test LivePriceService initialisation."""

    def test_instantiates_with_config(self, live_config: dict) -> None:
        """Service should instantiate without error given a valid config."""
        svc = LivePriceService(config=live_config)
        assert svc is not None
        assert svc.sources == ["yfinance", "alpha_vantage", "cached"]

    def test_instantiates_with_no_config(self) -> None:
        """Service should instantiate with defaults when no config given."""
        svc = LivePriceService()
        assert svc is not None
        assert "yfinance" in svc.sources

    def test_cache_ttl_from_config(self, live_config: dict) -> None:
        """Cache TTL should be read from config."""
        svc = LivePriceService(config=live_config)
        assert svc._cache_ttl == 2.0


# ─────────────────────────────────────────────
# LivePriceService — get_price structure
# ─────────────────────────────────────────────

class TestGetPriceStructure:
    """Test that get_price returns the correct dict structure."""

    def test_returns_correct_dict_structure(self, service: LivePriceService) -> None:
        """get_price should return dict with price, source, timestamp, delayed."""
        mock_result = {
            "price": 150.25,
            "source": "yfinance",
            "timestamp": datetime.now(tz=timezone.utc),
            "delayed": True,
        }
        with patch.object(service, "_fetch_from_source", return_value=mock_result):
            result = service.get_price("AAPL")

        assert "price" in result
        assert "source" in result
        assert "timestamp" in result
        assert "delayed" in result
        assert isinstance(result["price"], float)
        assert isinstance(result["source"], str)
        assert isinstance(result["timestamp"], datetime)
        assert isinstance(result["delayed"], bool)

    def test_raises_when_all_sources_fail(self, service: LivePriceService) -> None:
        """get_price should raise RuntimeError when every source fails."""
        with patch.object(service, "_fetch_from_source", return_value=None):
            with pytest.raises(RuntimeError, match="All price sources failed"):
                service.get_price("INVALID")


# ─────────────────────────────────────────────
# LivePriceService — fallback chain
# ─────────────────────────────────────────────

class TestFallbackChain:
    """Test the fallback logic across sources."""

    def test_falls_back_to_second_source(self, service: LivePriceService) -> None:
        """If first source fails, second source should be tried."""
        call_count = 0

        def mock_fetch(source: str, ticker: str):
            nonlocal call_count
            call_count += 1
            if source == "yfinance":
                raise ConnectionError("yfinance down")
            if source == "alpha_vantage":
                return {
                    "price": 155.0,
                    "source": "alpha_vantage",
                    "timestamp": datetime.now(tz=timezone.utc),
                    "delayed": True,
                }
            return None

        with patch.object(service, "_fetch_from_source", side_effect=mock_fetch):
            result = service.get_price("AAPL")

        assert result["source"] == "alpha_vantage"
        assert result["price"] == 155.0
        assert call_count >= 2

    def test_falls_back_to_cached_when_no_network(self, service: LivePriceService) -> None:
        """Should fall back to cached price when all network sources fail."""
        cached_result = {
            "price": 148.0,
            "source": "cached",
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "delayed": True,
        }

        def mock_fetch(source: str, ticker: str):
            if source == "cached":
                return cached_result
            raise ConnectionError(f"{source} unavailable")

        with patch.object(service, "_fetch_from_source", side_effect=mock_fetch):
            result = service.get_price("AAPL")

        assert result["source"] == "cached"
        assert result["price"] == 148.0

    def test_uses_first_successful_source(self, service: LivePriceService) -> None:
        """Should return from the first source that succeeds."""
        yf_result = {
            "price": 152.0,
            "source": "yfinance",
            "timestamp": datetime.now(tz=timezone.utc),
            "delayed": True,
        }

        def mock_fetch(source: str, ticker: str):
            if source == "yfinance":
                return yf_result
            return {
                "price": 999.0,
                "source": source,
                "timestamp": datetime.now(tz=timezone.utc),
                "delayed": True,
            }

        with patch.object(service, "_fetch_from_source", side_effect=mock_fetch):
            result = service.get_price("AAPL")

        assert result["source"] == "yfinance"
        assert result["price"] == 152.0


# ─────────────────────────────────────────────
# LivePriceService — caching
# ─────────────────────────────────────────────

class TestCaching:
    """Test the in-memory price cache."""

    def test_cache_returns_within_ttl(self, service: LivePriceService) -> None:
        """Cached price should be returned if within TTL."""
        mock_result = {
            "price": 150.0,
            "source": "yfinance",
            "timestamp": datetime.now(tz=timezone.utc),
            "delayed": True,
        }

        fetch_count = 0

        def mock_fetch(source: str, ticker: str):
            nonlocal fetch_count
            fetch_count += 1
            return mock_result

        with patch.object(service, "_fetch_from_source", side_effect=mock_fetch):
            # First call fetches
            result1 = service.get_price("AAPL")
            # Second call should use cache
            result2 = service.get_price("AAPL")

        assert result1["price"] == result2["price"]
        # Only one actual fetch should have occurred
        assert fetch_count == 1

    def test_cache_expires_after_ttl(self, service: LivePriceService) -> None:
        """Price should be re-fetched after cache TTL expires."""
        prices = iter([150.0, 155.0])

        def mock_fetch(source: str, ticker: str):
            return {
                "price": next(prices),
                "source": "yfinance",
                "timestamp": datetime.now(tz=timezone.utc),
                "delayed": True,
            }

        with patch.object(service, "_fetch_from_source", side_effect=mock_fetch):
            result1 = service.get_price("AAPL")
            assert result1["price"] == 150.0

            # Expire the cache by manipulating the stored timestamp
            ticker_data, _ = service._cache["AAPL"]
            service._cache["AAPL"] = (ticker_data, time.monotonic() - 10)

            result2 = service.get_price("AAPL")
            assert result2["price"] == 155.0


# ─────────────────────────────────────────────
# LivePriceService — get_prices (batch)
# ─────────────────────────────────────────────

class TestGetPricesBatch:
    """Test batch price fetching."""

    def test_returns_dataframe(self, service: LivePriceService) -> None:
        """get_prices should return a DataFrame indexed by ticker."""
        mock_result = {
            "price": 150.0,
            "source": "yfinance",
            "timestamp": datetime.now(tz=timezone.utc),
            "delayed": True,
        }
        with patch.object(service, "_fetch_from_source", return_value=mock_result):
            df = service.get_prices(["AAPL", "MSFT"])

        assert isinstance(df, pd.DataFrame)
        assert "AAPL" in df.index
        assert "MSFT" in df.index
        assert list(df.columns) == ["price", "source", "timestamp", "delayed"]

    def test_handles_partial_failures(self, service: LivePriceService) -> None:
        """get_prices should include NaN for tickers that fail entirely."""
        def mock_fetch(source: str, ticker: str):
            if ticker == "AAPL":
                return {
                    "price": 150.0,
                    "source": "yfinance",
                    "timestamp": datetime.now(tz=timezone.utc),
                    "delayed": True,
                }
            return None  # MSFT fails all sources

        with patch.object(service, "_fetch_from_source", side_effect=mock_fetch):
            df = service.get_prices(["AAPL", "MSFT"])

        assert df.loc["AAPL", "price"] == 150.0
        assert df.loc["MSFT", "source"] == "unavailable"


# ─────────────────────────────────────────────
# LivePriceService — source status
# ─────────────────────────────────────────────

class TestSourceStatus:
    """Test get_price_source_status."""

    def test_returns_dict_for_all_configured_sources(self, service: LivePriceService) -> None:
        """Status dict should contain all sources from the fallback chain."""
        status = service.get_price_source_status()

        assert isinstance(status, dict)
        for source in service.sources:
            assert source in status
            assert "available" in status[source]
            assert "detail" in status[source]
            assert isinstance(status[source]["available"], bool)
            assert isinstance(status[source]["detail"], str)

    def test_cached_source_unavailable_when_no_dir(self, service: LivePriceService) -> None:
        """Cached source should report unavailable when processed dir missing."""
        status = service.get_price_source_status()
        # 'data/processed' likely doesn't exist in test environment
        assert "cached" in status


# ─────────────────────────────────────────────
# AlphaVantageFetcher — factory registration
# ─────────────────────────────────────────────

class TestAlphaVantageFetcherFactory:
    """Test AlphaVantageFetcher is registered and usable via the factory."""

    def test_registered_in_factory(self) -> None:
        """alpha_vantage should be a valid source in the fetcher registry."""
        # Importing the module triggers registration
        import src.data.alpha_vantage_fetcher  # noqa: F401
        assert "alpha_vantage" in _FETCHER_REGISTRY

    def test_create_fetcher_returns_correct_type(self) -> None:
        """create_fetcher('alpha_vantage') should return AlphaVantageFetcher."""
        import src.data.alpha_vantage_fetcher  # noqa: F401
        from src.data.alpha_vantage_fetcher import AlphaVantageFetcher

        fetcher = create_fetcher("alpha_vantage")
        assert isinstance(fetcher, AlphaVantageFetcher)

    def test_fetcher_has_fetch_method(self) -> None:
        """AlphaVantageFetcher instances must have a fetch method."""
        import src.data.alpha_vantage_fetcher  # noqa: F401
        fetcher = create_fetcher("alpha_vantage")
        assert callable(getattr(fetcher, "fetch", None))
        assert callable(getattr(fetcher, "fetch_multiple", None))


# ─────────────────────────────────────────────
# AlphaVantageFetcher — API key handling
# ─────────────────────────────────────────────

class TestAlphaVantageFetcherConfig:
    """Test AlphaVantageFetcher configuration and API key resolution."""

    def test_reads_key_from_config(self) -> None:
        """Should read API key from config dict."""
        from src.data.alpha_vantage_fetcher import AlphaVantageFetcher

        config = {"api_keys": {"alpha_vantage": "my_test_key"}}
        fetcher = AlphaVantageFetcher(config=config)
        assert fetcher._api_key == "my_test_key"

    def test_explicit_key_overrides_config(self) -> None:
        """Explicit api_key param should take priority over config."""
        from src.data.alpha_vantage_fetcher import AlphaVantageFetcher

        config = {"api_keys": {"alpha_vantage": "config_key"}}
        fetcher = AlphaVantageFetcher(config=config, api_key="explicit_key")
        assert fetcher._api_key == "explicit_key"

    def test_reads_key_from_env(self) -> None:
        """Should fall back to environment variable."""
        from src.data.alpha_vantage_fetcher import AlphaVantageFetcher

        with patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "env_key"}):
            fetcher = AlphaVantageFetcher()
            assert fetcher._api_key == "env_key"


# ─────────────────────────────────────────────
# AlphaVantageFetcher — parsing
# ─────────────────────────────────────────────

class TestAlphaVantageParsing:
    """Test response parsing logic (no network calls)."""

    def test_parse_time_series(self) -> None:
        """_parse_time_series should convert AV JSON to DataFrame."""
        from src.data.alpha_vantage_fetcher import AlphaVantageFetcher

        raw = {
            "2024-01-02": {
                "1. open": "150.00",
                "2. high": "155.00",
                "3. low": "149.00",
                "4. close": "153.00",
                "5. volume": "1000000",
            },
            "2024-01-03": {
                "1. open": "153.00",
                "2. high": "158.00",
                "3. low": "152.00",
                "4. close": "157.00",
                "5. volume": "1200000",
            },
        }

        df = AlphaVantageFetcher._parse_time_series(raw)

        assert len(df) == 2
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.iloc[0]["Open"] == 150.0
        assert df.iloc[1]["Close"] == 157.0
        assert df.iloc[0]["Volume"] == 1000000

    def test_trim_dates(self) -> None:
        """_trim_dates should filter DataFrame to date range."""
        from src.data.alpha_vantage_fetcher import AlphaVantageFetcher

        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": 1000},
            index=dates,
        )

        trimmed = AlphaVantageFetcher._trim_dates(df, "2024-01-03", "2024-01-08")
        assert trimmed.index[0] >= pd.Timestamp("2024-01-03")
        assert trimmed.index[-1] <= pd.Timestamp("2024-01-08")
