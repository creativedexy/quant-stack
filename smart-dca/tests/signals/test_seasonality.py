"""Tests for SeasonalitySignal."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.signals.seasonality import SeasonalitySignal


@pytest.fixture
def config(sample_settings: dict) -> dict:
    return sample_settings


@pytest.fixture
def signal(config: dict) -> SeasonalitySignal:
    return SeasonalitySignal(config)


class TestSeasonalitySignal:
    def test_saturday_early_morning_high_score(
        self, signal: SeasonalitySignal
    ) -> None:
        """Saturday 03:00 UTC should produce a high score (> 0.7)."""
        saturday_3am = datetime(2024, 1, 6, 3, 0, tzinfo=timezone.utc)
        result = signal.compute("BTCGBP", now=saturday_3am)
        assert result.score > 0.7
        assert result.name == "seasonality"

    def test_monday_midday_neutral_score(
        self, signal: SeasonalitySignal
    ) -> None:
        """Monday 12:00 UTC should produce a neutral score."""
        monday_noon = datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc)
        result = signal.compute("BTCGBP", now=monday_noon)
        assert 0.4 < result.score < 0.7

    def test_score_clamped_0_to_1(
        self, signal: SeasonalitySignal
    ) -> None:
        """All hour/day combinations must produce scores in [0, 1]."""
        for weekday_offset in range(7):
            for hour in range(24):
                dt = datetime(
                    2024, 1, 1 + weekday_offset, hour, 0,
                    tzinfo=timezone.utc,
                )
                result = signal.compute("BTCGBP", now=dt)
                assert 0.0 <= result.score <= 1.0, (
                    f"Score {result.score} out of range for "
                    f"weekday={dt.weekday()}, hour={hour}"
                )

    def test_never_raises(self) -> None:
        """Even with empty config, must return SignalResult not exception."""
        signal = SeasonalitySignal({})
        result = signal.compute("BTCGBP")
        assert result.score >= 0.0
        assert result.name == "seasonality"
