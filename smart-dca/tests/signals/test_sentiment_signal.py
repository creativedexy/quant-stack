"""Tests for SentimentSignal."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.signals.sentiment_signal import SentimentSignal
from src.utils.exceptions import DataFeedError


@pytest.fixture
def config(sample_settings: dict) -> dict:
    return sample_settings


@pytest.fixture
def mock_sentiment_feed() -> MagicMock:
    feed = MagicMock()
    feed.get_current_index = AsyncMock(return_value=50)
    return feed


@pytest.fixture
def signal(
    config: dict, mock_sentiment_feed: MagicMock
) -> SentimentSignal:
    return SentimentSignal(config, mock_sentiment_feed)


class TestSentimentSignal:
    def test_extreme_fear_high_score(
        self,
        signal: SentimentSignal,
        mock_sentiment_feed: MagicMock,
    ) -> None:
        """Index 10 with threshold 30 should produce score ~0.67."""
        mock_sentiment_feed.get_current_index = AsyncMock(
            return_value=10
        )
        result = signal.compute("BTCGBP")
        assert result.score == pytest.approx(
            (30 - 10) / 30, abs=0.01
        )
        assert result.name == "sentiment"

    def test_fear_at_threshold_zero_score(
        self,
        signal: SentimentSignal,
        mock_sentiment_feed: MagicMock,
    ) -> None:
        """Index at threshold (30) should produce score 0.0."""
        mock_sentiment_feed.get_current_index = AsyncMock(
            return_value=30
        )
        result = signal.compute("BTCGBP")
        assert result.score == 0.0

    def test_greed_returns_zero(
        self,
        signal: SentimentSignal,
        mock_sentiment_feed: MagicMock,
    ) -> None:
        """Index 75 (greed) should produce score 0.0."""
        mock_sentiment_feed.get_current_index = AsyncMock(
            return_value=75
        )
        result = signal.compute("BTCGBP")
        assert result.score == 0.0

    def test_exception_returns_zero_score(
        self,
        signal: SentimentSignal,
        mock_sentiment_feed: MagicMock,
    ) -> None:
        """If feed raises, score should be 0.0 (never propagate)."""
        mock_sentiment_feed.get_current_index = AsyncMock(
            side_effect=DataFeedError("API down")
        )
        result = signal.compute("BTCGBP")
        assert result.score == 0.0
        assert "failed" in result.reason.lower()
