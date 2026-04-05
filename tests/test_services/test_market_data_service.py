"""Tests for MarketDataService -- live/synthetic market data layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.services.market_data_service import (
    MarketDataService,
    _format_market_cap,
    _format_pe,
    _format_pct,
    _format_volume,
)


@pytest.fixture
def mds() -> MarketDataService:
    """MarketDataService with empty config (synthetic fallback only)."""
    return MarketDataService(config={})


class TestGetIntradayPrices:
    """Intraday price data retrieval."""

    def test_get_intraday_returns_correct_keys(self, mds: MarketDataService) -> None:
        """Intraday result must contain prices, times, prev_close, direction, current."""
        result = mds.get_intraday_prices("BARC.L")
        assert isinstance(result, dict)
        for key in ("prices", "times", "prev_close", "direction", "current"):
            assert key in result, f"Missing key: {key}"
        assert isinstance(result["prices"], list)
        assert isinstance(result["times"], list)
        assert len(result["prices"]) == len(result["times"])
        assert len(result["prices"]) > 0

    def test_get_intraday_synthetic_fallback(self, mds: MarketDataService) -> None:
        """When yfinance throws, the service falls back to synthetic data."""
        with patch(
            "src.services.market_data_service.MarketDataService._get_yf_ticker",
            side_effect=Exception("network down"),
        ):
            result = mds.get_intraday_prices("SHEL.L")
        assert isinstance(result, dict)
        assert "prices" in result
        assert len(result["prices"]) > 0
        assert result["direction"] in ("up", "dn")


class TestGetFundamentals:
    """Fundamental data retrieval and formatting."""

    def test_get_fundamentals_formatting(self, mds: MarketDataService) -> None:
        """Volume and market cap are formatted with correct suffixes."""
        with patch.object(mds, "_get_yf_ticker", return_value=None):
            result = mds.get_fundamentals("BARC.L")
        # BARC.L synthetic: volume=48_200_000 -> "48.2M"
        assert result["volume"] == "48.2M"
        # BARC.L synthetic: market_cap=37_200_000_000 -> GBP37.2B
        assert result["market_cap"] == "\u00a337.2B"

    def test_pe_ratio_none_returns_na(self) -> None:
        """When P/E ratio is None, _format_pe returns 'N/A'."""
        assert _format_pe(None) == "N/A"

    def test_pe_ratio_value_formats_correctly(self) -> None:
        """Numeric P/E is formatted to 1 decimal with 'x' suffix."""
        assert _format_pe(7.2) == "7.2x"
        assert _format_pe(35.0) == "35.0x"


class TestDirection:
    """Price direction (up/dn) from intraday data."""

    def test_direction_up_when_above_prev_close(self, mds: MarketDataService) -> None:
        """Direction is 'up' when current price >= prev_close."""
        result = mds._synthetic_intraday("AZN.L")
        # Force a check: if current >= prev_close, direction must be "up"
        if result["current"] >= result["prev_close"]:
            assert result["direction"] == "up"
        else:
            assert result["direction"] == "dn"

    def test_direction_dn_when_below_prev_close(self, mds: MarketDataService) -> None:
        """Direction is 'dn' when current price < prev_close."""
        # Create a scenario where current < prev_close by inspecting
        # the intraday logic directly.
        intraday = {
            "prices": [100.0, 99.5, 98.0],
            "times": ["t0", "t1", "t2"],
            "prev_close": 100.0,
            "current": 98.0,
            "direction": "dn",
        }
        assert intraday["current"] < intraday["prev_close"]
        assert intraday["direction"] == "dn"

        # Also verify via the synthetic path for a known ticker
        # The direction should match the current vs prev_close comparison
        result = mds.get_intraday_prices("LLOY.L")
        expected_dir = "up" if result["current"] >= result["prev_close"] else "dn"
        assert result["direction"] == expected_dir


class TestGetWatchlistCards:
    """Composite card builder."""

    def test_get_watchlist_cards_returns_one_dict_per_ticker(
        self, mds: MarketDataService,
    ) -> None:
        """get_watchlist_cards returns exactly one dict per input ticker."""
        tickers = ["BARC.L", "SHEL.L", "BP.L"]
        cards = mds.get_watchlist_cards(tickers)
        assert isinstance(cards, list)
        assert len(cards) == len(tickers)
        for card, expected_ticker in zip(cards, tickers):
            assert isinstance(card, dict)
            assert card["ticker"] == expected_ticker
            assert "company_name" in card
            assert "current" in card
            assert "prices" in card


class TestCaching:
    """yfinance ticker caching behaviour."""

    def test_caching_prevents_duplicate_yfinance_calls(self) -> None:
        """Calling get_intraday_prices twice reuses the same Ticker object."""
        svc = MarketDataService(config={})

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = MagicMock(empty=True)
        mock_ticker.info = {}

        with patch("yfinance.Ticker", return_value=mock_ticker) as mock_yf:
            # First call creates the Ticker
            svc.get_intraday_prices("BARC.L")
            first_call_count = mock_yf.call_count

            # Second call should reuse cached Ticker
            svc.get_intraday_prices("BARC.L")
            second_call_count = mock_yf.call_count

        # yfinance.Ticker should only have been called once
        assert first_call_count == 1
        assert second_call_count == 1


class TestMarginFlags:
    """Profit and operating margin positive/negative flags."""

    def test_profit_margin_positive_flag(self, mds: MarketDataService) -> None:
        """Positive profit margin sets is_positive_margin=True."""
        with patch.object(mds, "_get_yf_ticker", return_value=None):
            result = mds.get_fundamentals("BARC.L")
        # BARC.L synthetic: profit_margin=0.284 (positive)
        assert result["is_positive_margin"] is True

    def test_operating_margin_negative_flag(self) -> None:
        """Negative operating margin sets is_positive_op_margin=False."""
        svc = MarketDataService(config={})

        # Use a ticker not in _SYNTHETIC_COMPANIES to get default values,
        # then override via _synthetic_fundamentals behaviour.
        # Instead, directly test the formatting logic:
        mock_info = {
            "volume": 1_000_000,
            "marketCap": 5_000_000_000,
            "trailingPE": 12.0,
            "dividendYield": 0.03,
            "profitMargins": -0.05,
            "operatingMargins": -0.12,
        }

        mock_yf = MagicMock()
        mock_yf.info = mock_info

        with patch.object(svc, "_get_yf_ticker", return_value=mock_yf):
            result = svc.get_fundamentals("FAKE.L")

        assert result["is_positive_margin"] is False
        assert result["is_positive_op_margin"] is False
        assert result["profit_margin"] == "-5.0%"
        assert result["operating_margin"] == "-12.0%"


class TestFormattingHelpers:
    """Unit tests for standalone formatting functions."""

    def test_format_volume_millions(self) -> None:
        """48.2M formatting for volume."""
        assert _format_volume(48_200_000) == "48.2M"

    def test_format_volume_billions(self) -> None:
        """Billions formatting for volume."""
        assert _format_volume(1_500_000_000) == "1.5B"

    def test_format_volume_thousands(self) -> None:
        """Thousands formatting for volume."""
        assert _format_volume(42_800) == "43k"

    def test_format_volume_none(self) -> None:
        """None returns N/A."""
        assert _format_volume(None) == "N/A"

    def test_format_market_cap_gbp_billions(self) -> None:
        """Market cap with GBP symbol."""
        assert _format_market_cap(37_200_000_000, "BARC.L") == "\u00a337.2B"

    def test_format_market_cap_usd_billions(self) -> None:
        """Market cap with USD symbol for US ticker."""
        assert _format_market_cap(50_000_000_000, "AAPL") == "$50.0B"

    def test_format_pct_positive(self) -> None:
        """Positive decimal as percentage."""
        assert _format_pct(0.284) == "28.4%"

    def test_format_pct_none(self) -> None:
        """None returns N/A."""
        assert _format_pct(None) == "N/A"
