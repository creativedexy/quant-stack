"""Tests for BinancePriceFeed."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data.price_feed import BinancePriceFeed
from src.utils.exceptions import DataFeedError


@pytest.fixture
def config() -> dict:
    return {"binance": {"request_timeout_seconds": 10}}


@pytest.fixture
def feed(mock_binance_client: MagicMock, config: dict) -> BinancePriceFeed:
    return BinancePriceFeed(mock_binance_client, config)


def _make_kline(ts_ms: int, o: float, h: float, low: float, c: float, v: float) -> list:
    """Build a single Binance kline list."""
    return [ts_ms, str(o), str(h), str(low), str(c), str(v), 0, "0", 0, "0", "0", "0"]


def _make_klines(n: int, start_close: float = 50000.0, end_close: float = 50000.0) -> list:
    """Build N klines with linearly interpolated close prices."""
    base_ts = 1700000000000
    closes = [start_close + (end_close - start_close) * i / max(n - 1, 1) for i in range(n)]
    klines = []
    for i in range(n):
        ts = base_ts + i * 3600000
        c = closes[i]
        klines.append(_make_kline(ts, c + 10, c + 50, c - 50, c, 100.0))
    return klines


class TestGetMidprice:
    def test_returns_average_of_bid_ask(self, feed: BinancePriceFeed) -> None:
        """Midprice is (bid + ask) / 2."""
        # Mock returns bid=49900, ask=50100
        midprice = feed.get_midprice("BTCGBP")
        assert midprice == 50000.0


class TestGetOhlcv:
    def test_returns_correct_columns(
        self, feed: BinancePriceFeed, mock_binance_client: MagicMock
    ) -> None:
        """OHLCV DataFrame has all required columns."""
        mock_binance_client.get_klines.return_value = _make_klines(24)
        df = feed.get_ohlcv("BTCGBP", "1h", 24)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) == 24

    def test_raises_on_nan(
        self, feed: BinancePriceFeed, mock_binance_client: MagicMock
    ) -> None:
        """DataFeedError raised when OHLCV data contains NaN."""
        klines = _make_klines(5)
        # Inject NaN by setting close to non-numeric
        klines[2][4] = "NaN"
        mock_binance_client.get_klines.return_value = klines
        with pytest.raises(DataFeedError, match="NaN"):
            feed.get_ohlcv("BTCGBP", "1h", 5)

    def test_all_timestamps_utc_aware(
        self, feed: BinancePriceFeed, mock_binance_client: MagicMock
    ) -> None:
        """All timestamps in returned DataFrame are UTC-aware."""
        mock_binance_client.get_klines.return_value = _make_klines(10)
        df = feed.get_ohlcv("BTCGBP", "1h", 10)
        for ts in df["timestamp"]:
            assert ts.tzinfo is not None, f"Timestamp {ts} is naive (no timezone)"


class TestGetPriceChangePct:
    def test_negative_change(
        self, feed: BinancePriceFeed, mock_binance_client: MagicMock
    ) -> None:
        """Price drop from 50000 to 47500 returns -5.0%."""
        mock_binance_client.get_klines.return_value = _make_klines(24, 50000.0, 47500.0)
        pct = feed.get_price_change_pct("BTCGBP", 24)
        assert abs(pct - (-5.0)) < 0.01

    def test_positive_change(
        self, feed: BinancePriceFeed, mock_binance_client: MagicMock
    ) -> None:
        """Price rise from 40000 to 42000 returns 5.0%."""
        mock_binance_client.get_klines.return_value = _make_klines(24, 40000.0, 42000.0)
        pct = feed.get_price_change_pct("BTCGBP", 24)
        assert abs(pct - 5.0) < 0.01


class TestRetryBehaviour:
    def test_retries_on_timeout(
        self, feed: BinancePriceFeed, mock_binance_client: MagicMock
    ) -> None:
        """First call fails, second succeeds -> result returned."""
        mock_binance_client.get_symbol_ticker.side_effect = [
            Exception("timeout"),
            {"price": "50000.00"},
        ]
        price = feed.get_current_price("BTCGBP")
        assert price == 50000.0
        assert mock_binance_client.get_symbol_ticker.call_count == 2

    def test_raises_after_two_failures(
        self, feed: BinancePriceFeed, mock_binance_client: MagicMock
    ) -> None:
        """Both attempts fail -> DataFeedError raised."""
        mock_binance_client.get_symbol_ticker.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
        ]
        with pytest.raises(DataFeedError, match="failed after"):
            feed.get_current_price("BTCGBP")
