"""Tests for FundingRateSignal."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.signals.funding_signal import FundingRateSignal


@pytest.fixture
def config(sample_settings: dict) -> dict:
    return sample_settings


@pytest.fixture
def mock_funding_feed() -> MagicMock:
    return MagicMock()


@pytest.fixture
def signal(
    config: dict, mock_funding_feed: MagicMock
) -> FundingRateSignal:
    return FundingRateSignal(config, mock_funding_feed)


class TestFundingRateSignal:
    def test_negative_funding_above_threshold(
        self, signal: FundingRateSignal, mock_funding_feed: MagicMock
    ) -> None:
        """Rate -0.02 with threshold -0.01 should produce score 1.0."""
        mock_funding_feed.get_funding_rate.return_value = -0.02
        result = signal.compute("BTCGBP")
        assert result.score == pytest.approx(1.0)  # 0.02/0.01 clamped
        assert result.name == "funding_rate"

    def test_negative_funding_below_threshold(
        self, signal: FundingRateSignal, mock_funding_feed: MagicMock
    ) -> None:
        """Rate -0.005 with threshold -0.01 should produce score 0.5."""
        mock_funding_feed.get_funding_rate.return_value = -0.005
        result = signal.compute("BTCGBP")
        assert result.score == pytest.approx(0.5)

    def test_positive_funding_returns_zero(
        self, signal: FundingRateSignal, mock_funding_feed: MagicMock
    ) -> None:
        """Positive funding rate should produce score 0.0."""
        mock_funding_feed.get_funding_rate.return_value = 0.01
        result = signal.compute("BTCGBP")
        assert result.score == 0.0

    def test_xrp_zero_rate_returns_zero(
        self, signal: FundingRateSignal, mock_funding_feed: MagicMock
    ) -> None:
        """Zero funding rate (e.g. XRP with no proxy) returns score 0.0."""
        mock_funding_feed.get_funding_rate.return_value = 0.0
        result = signal.compute("XRPGBP")
        assert result.score == 0.0
