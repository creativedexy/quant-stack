"""Tests for SignalScorer."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.signals.base import BaseSignal, SignalResult
from src.signals.scorer import SignalScorer
from src.utils.exceptions import ConfigError


def _make_signal(
    name: str, score: float, weight: float
) -> MagicMock:
    """Create a mock signal that returns a fixed score."""
    sig = MagicMock(spec=BaseSignal)
    sig.name = name
    sig.compute.return_value = SignalResult(
        name=name,
        score=score,
        weight=weight,
        reason=f"{name} test reason",
        timestamp=datetime.now(timezone.utc),
    )
    return sig


@pytest.fixture
def config() -> dict:
    return {
        "scoring": {
            "buy_threshold": 65,
            "signal_weights": {
                "seasonality": 0.20,
                "mean_reversion": 0.35,
                "funding_rate": 0.25,
                "sentiment": 0.20,
            },
        },
    }


class TestSignalScorer:
    def test_weighted_composite_correct(self, config: dict) -> None:
        """Verify weighted arithmetic is exact."""
        signals = [
            _make_signal("seasonality", 0.50, 0.20),
            _make_signal("mean_reversion", 0.80, 0.35),
            _make_signal("funding_rate", 0.60, 0.25),
            _make_signal("sentiment", 0.90, 0.20),
        ]
        scorer = SignalScorer(signals, config)
        result = scorer.compute("BTCGBP")

        expected = (
            0.50 * 0.20 + 0.80 * 0.35
            + 0.60 * 0.25 + 0.90 * 0.20
        ) * 100
        assert result.score == pytest.approx(
            round(expected, 2), abs=0.01
        )

    def test_should_buy_true_above_threshold(
        self, config: dict
    ) -> None:
        """Composite 66 with threshold 65 -> should_buy True."""
        # Scores designed to produce composite = 66.0
        # 0.66 * 1.0 * 100 = 66.0
        signals = [_make_signal("only", 0.66, 1.0)]
        cfg = {
            "scoring": {
                "buy_threshold": 65,
                "signal_weights": {"only": 1.0},
            }
        }
        scorer = SignalScorer(signals, cfg)
        result = scorer.compute("BTCGBP")
        assert result.should_buy is True

    def test_should_buy_false_below_threshold(
        self, config: dict
    ) -> None:
        """Composite 64 with threshold 65 -> should_buy False."""
        signals = [_make_signal("only", 0.64, 1.0)]
        cfg = {
            "scoring": {
                "buy_threshold": 65,
                "signal_weights": {"only": 1.0},
            }
        }
        scorer = SignalScorer(signals, cfg)
        result = scorer.compute("BTCGBP")
        assert result.should_buy is False

    def test_weights_not_summing_to_one_raises(self) -> None:
        """Weights [0.5, 0.5, 0.5] should raise ConfigError."""
        signals = [
            _make_signal("a", 0.5, 0.5),
            _make_signal("b", 0.5, 0.5),
            _make_signal("c", 0.5, 0.5),
        ]
        cfg = {
            "scoring": {
                "buy_threshold": 65,
                "signal_weights": {"a": 0.5, "b": 0.5, "c": 0.5},
            }
        }
        with pytest.raises(ConfigError, match="sum to 1.0"):
            SignalScorer(signals, cfg)

    def test_all_zero_signals_returns_zero(
        self, config: dict
    ) -> None:
        """All signals scoring 0.0 produces composite 0.0."""
        signals = [
            _make_signal("seasonality", 0.0, 0.20),
            _make_signal("mean_reversion", 0.0, 0.35),
            _make_signal("funding_rate", 0.0, 0.25),
            _make_signal("sentiment", 0.0, 0.20),
        ]
        scorer = SignalScorer(signals, config)
        result = scorer.compute("BTCGBP")
        assert result.score == 0.0
        assert result.should_buy is False

    def test_explain_output_contains_all_signal_names(
        self, config: dict
    ) -> None:
        """Each signal name appears in explain() string."""
        signals = [
            _make_signal("seasonality", 0.50, 0.20),
            _make_signal("mean_reversion", 0.80, 0.35),
            _make_signal("funding_rate", 0.60, 0.25),
            _make_signal("sentiment", 0.90, 0.20),
        ]
        scorer = SignalScorer(signals, config)
        result = scorer.compute("BTCGBP")
        explanation = scorer.explain(result)

        for name in [
            "seasonality", "mean_reversion",
            "funding_rate", "sentiment",
        ]:
            assert name in explanation

    def test_explain_shows_correct_total(
        self, config: dict
    ) -> None:
        """Score in explain() matches composite.score."""
        signals = [
            _make_signal("seasonality", 0.50, 0.20),
            _make_signal("mean_reversion", 0.80, 0.35),
            _make_signal("funding_rate", 0.60, 0.25),
            _make_signal("sentiment", 0.90, 0.20),
        ]
        scorer = SignalScorer(signals, config)
        result = scorer.compute("BTCGBP")
        explanation = scorer.explain(result)
        assert str(result.score) in explanation

    def test_single_signal_full_weight(self) -> None:
        """One signal, weight 1.0, score 0.8 produces 80.0."""
        signals = [_make_signal("only", 0.8, 1.0)]
        cfg = {
            "scoring": {
                "buy_threshold": 65,
                "signal_weights": {"only": 1.0},
            }
        }
        scorer = SignalScorer(signals, cfg)
        result = scorer.compute("BTCGBP")
        assert result.score == pytest.approx(80.0)
        assert result.should_buy is True
