"""Tests for BinanceFundingFeed."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.data.funding_feed import BinanceFundingFeed
from src.utils.exceptions import DataFeedError


@pytest.fixture
def mock_assets():
    """Patch load_assets to return test config."""
    assets = {
        "assets": {
            "BTCGBP": {"funding_proxy": "BTCUSDT"},
            "ETHGBP": {"funding_proxy": "ETHUSDT"},
            "XRPGBP": {"funding_proxy": None},
        }
    }
    with patch("src.utils.config_loader.load_assets", return_value=assets):
        yield assets


@pytest.fixture
def client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def feed(client: MagicMock) -> BinanceFundingFeed:
    return BinanceFundingFeed(client)


class TestGetFundingRate:
    def test_returns_negative_funding_rate(
        self, feed: BinanceFundingFeed, client: MagicMock, mock_assets: dict
    ) -> None:
        """Negative funding rate returned as float."""
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        client.futures_funding_rate.return_value = [
            {"symbol": "BTCUSDT", "fundingRate": "-0.00010000", "fundingTime": now_ms}
        ]
        rate = feed.get_funding_rate("BTCGBP")
        assert rate == pytest.approx(-0.0001)

    def test_xrp_returns_zero_with_warning(
        self, feed: BinanceFundingFeed, mock_assets: dict, caplog: pytest.LogCaptureFixture
    ) -> None:
        """XRPGBP (no proxy) returns 0.0 and logs a warning."""
        with caplog.at_level(logging.WARNING):
            rate = feed.get_funding_rate("XRPGBP")
        assert rate == 0.0

    def test_raises_on_stale_data(
        self, feed: BinanceFundingFeed, client: MagicMock, mock_assets: dict
    ) -> None:
        """Funding rate older than 8.5 hours raises DataFeedError."""
        stale_ms = int((datetime.now(timezone.utc).timestamp() - 9 * 3600) * 1000)
        client.futures_funding_rate.return_value = [
            {"symbol": "BTCUSDT", "fundingRate": "-0.00010000", "fundingTime": stale_ms}
        ]
        with pytest.raises(DataFeedError, match="stale"):
            feed.get_funding_rate("BTCGBP")
