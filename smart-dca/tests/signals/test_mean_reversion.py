"""Tests for MeanReversionSignal."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.signals.mean_reversion import MeanReversionSignal
from src.utils.exceptions import DataFeedError


@pytest.fixture
def config(sample_settings: dict) -> dict:
    return sample_settings


@pytest.fixture
def mock_price_feed() -> MagicMock:
    return MagicMock()


@pytest.fixture
def signal(
    config: dict, mock_price_feed: MagicMock
) -> MeanReversionSignal:
    return MeanReversionSignal(config, mock_price_feed)


class TestMeanReversionSignal:
    def test_5pct_dip_above_threshold(
        self, signal: MeanReversionSignal, mock_price_feed: MagicMock
    ) -> None:
        """5% dip with 4% threshold should produce score > 0.0."""
        mock_price_feed.get_price_change_pct.return_value = -5.0
        result = signal.compute("BTCGBP")
        assert result.score > 0.0
        assert result.score == pytest.approx(1.0)  # 5/4 clamped to 1.0
        assert result.name == "mean_reversion"

    def test_2pct_dip_below_threshold(
        self, signal: MeanReversionSignal, mock_price_feed: MagicMock
    ) -> None:
        """2% dip with 4% threshold should produce score 0.5."""
        mock_price_feed.get_price_change_pct.return_value = -2.0
        result = signal.compute("BTCGBP")
        assert result.score == pytest.approx(0.5)

    def test_price_rise_returns_zero(
        self, signal: MeanReversionSignal, mock_price_feed: MagicMock
    ) -> None:
        """Price rise of 3% should produce score 0.0."""
        mock_price_feed.get_price_change_pct.return_value = 3.0
        result = signal.compute("BTCGBP")
        assert result.score == 0.0

    def test_exception_in_feed_returns_zero_score(
        self, signal: MeanReversionSignal, mock_price_feed: MagicMock
    ) -> None:
        """If price feed raises, score should be 0.0 (never propagate)."""
        mock_price_feed.get_price_change_pct.side_effect = DataFeedError(
            "timeout"
        )
        result = signal.compute("BTCGBP")
        assert result.score == 0.0
        assert "failed" in result.reason.lower()
