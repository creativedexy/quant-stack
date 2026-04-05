"""Sentiment signal based on Fear & Greed index.

Uses the contrarian approach: extreme fear is a buy signal,
greed is not (crowd is already bullish).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from src.data.sentiment_feed import FearGreedFeed
from src.signals.base import BaseSignal, SignalResult, clamp
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentSignal(BaseSignal):
    """Scores Fear & Greed index as a contrarian buy indicator.

    Low fear values (below threshold) produce high scores.
    Neutral/greed values produce score 0.0.

    Args:
        config: Full settings dict from config_loader.
        sentiment_feed: FearGreedFeed instance for F&G data.
    """

    def __init__(
        self, config: dict, sentiment_feed: FearGreedFeed
    ) -> None:
        super().__init__(config)
        self._sentiment_feed = sentiment_feed

    @property
    def name(self) -> str:
        return "sentiment"

    def compute(
        self, symbol: str, **kwargs: object
    ) -> SignalResult:
        """Compute sentiment score from Fear & Greed index.

        Args:
            symbol: Trading pair (e.g. 'BTCGBP').

        Returns:
            SignalResult with fear-based contrarian score.
        """
        now = datetime.now(timezone.utc)
        weight = self.config.get("scoring", {}).get(
            "signal_weights", {}
        ).get("sentiment", 0.20)

        try:
            threshold = self.config.get("signals", {}).get(
                "sentiment_fear_threshold", 30
            )

            # Run async feed in sync context
            index = self._get_index()

            if index >= threshold:
                return SignalResult(
                    name=self.name,
                    score=0.0,
                    weight=weight,
                    reason=(
                        f"Fear & Greed index at {index} "
                        f"-> at/above {threshold} threshold, no signal"
                    ),
                    timestamp=now,
                )

            score = clamp((threshold - index) / threshold)

            qualifier = "extreme fear" if index < 20 else "fear"

            return SignalResult(
                name=self.name,
                score=score,
                weight=weight,
                reason=(
                    f"Fear & Greed index at {index} -> "
                    f"{qualifier}, strong contrarian buy signal"
                ),
                timestamp=now,
            )
        except Exception as exc:
            logger.error(
                "sentiment_signal_failed",
                symbol=symbol,
                error=str(exc),
            )
            return SignalResult(
                name=self.name,
                score=0.0,
                weight=weight,
                reason=f"SentimentSignal failed: {exc}",
                timestamp=now,
            )

    def _get_index(self) -> int:
        """Fetch Fear & Greed index, handling async context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    self._sentiment_feed.get_current_index(),
                ).result()
        else:
            return asyncio.run(
                self._sentiment_feed.get_current_index()
            )
