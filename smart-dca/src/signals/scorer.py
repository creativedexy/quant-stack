"""Signal scorer that produces weighted composite scores.

Combines individual signal results into a single 0-100 composite
score and determines whether the buy threshold has been met.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.signals.base import BaseSignal, SignalResult
from src.utils.exceptions import ConfigError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CompositeScore:
    """Result of scoring all signals for a symbol.

    Attributes:
        score: Weighted composite score from 0 to 100.
        signals: Individual SignalResult objects from each signal.
        should_buy: True if score >= buy_threshold.
        timestamp: UTC-aware datetime of computation.
    """

    score: float
    signals: list[SignalResult] = field(default_factory=list)
    should_buy: bool = False
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SignalScorer:
    """Computes weighted composite score from individual signals.

    Stateless: each compute() call is independent. All data fetching
    is handled by the signal objects, not the scorer.

    Args:
        signals: List of instantiated BaseSignal objects.
        config: Full settings dict from config_loader.

    Raises:
        ConfigError: If signal weights do not sum to 1.0 (+-0.001).
    """

    def __init__(
        self, signals: list[BaseSignal], config: dict
    ) -> None:
        self._signals = signals
        self._config = config
        self._buy_threshold = config.get("scoring", {}).get(
            "buy_threshold", 65
        )
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Check that configured weights sum to 1.0."""
        weights = self._config.get("scoring", {}).get(
            "signal_weights", {}
        )
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            raise ConfigError(
                f"Signal weights must sum to 1.0, got {total:.4f}"
            )

    def compute(
        self, symbol: str, **feed_kwargs: object
    ) -> CompositeScore:
        """Run all signals and compute weighted composite score.

        Args:
            symbol: Trading pair (e.g. 'BTCGBP').
            **feed_kwargs: Extra arguments passed to each signal.

        Returns:
            CompositeScore with weighted result and buy decision.
        """
        now = datetime.now(timezone.utc)
        results: list[SignalResult] = []

        for signal in self._signals:
            result = signal.compute(symbol, **feed_kwargs)
            results.append(result)

        composite = sum(r.score * r.weight for r in results) * 100
        composite = round(composite, 2)
        should_buy = composite >= self._buy_threshold

        logger.info(
            "composite_score_computed",
            symbol=symbol,
            score=composite,
            should_buy=should_buy,
        )

        return CompositeScore(
            score=composite,
            signals=results,
            should_buy=should_buy,
            timestamp=now,
        )

    def explain(self, score: CompositeScore) -> str:
        """Return human-readable breakdown of a composite score.

        Args:
            score: CompositeScore from a previous compute() call.

        Returns:
            Multi-line string showing each signal's contribution.
        """
        action = "BUY" if score.should_buy else "HOLD"
        lines = [f"Score: {score.score}/100 -> {action}"]

        for i, sig in enumerate(score.signals):
            prefix = "└" if i == len(score.signals) - 1 else "├"
            contribution = sig.score * sig.weight * 100
            lines.append(
                f"  {prefix} {sig.name:<16s} "
                f"{sig.score:.2f} x {sig.weight:.2f} = "
                f"{contribution:6.2f}"
            )

        return "\n".join(lines)
