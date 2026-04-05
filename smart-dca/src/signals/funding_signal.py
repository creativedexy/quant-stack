"""Funding rate signal for perpetual futures sentiment.

Negative funding rates indicate bearish perp sentiment, which
is a bullish signal for spot purchases (contrarian indicator).
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.funding_feed import BinanceFundingFeed
from src.signals.base import BaseSignal, SignalResult, clamp
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FundingRateSignal(BaseSignal):
    """Scores negative funding rates as spot buy opportunities.

    Negative funding means shorts are paying longs — bearish perp
    sentiment that historically favours spot entry.

    Args:
        config: Full settings dict from config_loader.
        funding_feed: BinanceFundingFeed instance for funding data.
    """

    def __init__(
        self, config: dict, funding_feed: BinanceFundingFeed
    ) -> None:
        super().__init__(config)
        self._funding_feed = funding_feed

    @property
    def name(self) -> str:
        return "funding_rate"

    def compute(
        self, symbol: str, **kwargs: object
    ) -> SignalResult:
        """Compute funding rate signal.

        Args:
            symbol: Trading pair (e.g. 'BTCGBP').

        Returns:
            SignalResult with funding-rate-based score.
        """
        now = datetime.now(timezone.utc)
        weight = self.config.get("scoring", {}).get(
            "signal_weights", {}
        ).get("funding_rate", 0.25)

        try:
            threshold = self.config.get("signals", {}).get(
                "funding_rate_negative_threshold", -0.01
            )

            rate = self._funding_feed.get_funding_rate(symbol)

            if rate >= 0:
                return SignalResult(
                    name=self.name,
                    score=0.0,
                    weight=weight,
                    reason=(
                        f"Funding rate {rate:.5f} is positive/zero "
                        f"-> no spot buy signal"
                    ),
                    timestamp=now,
                )

            score = clamp(abs(rate) / abs(threshold))

            return SignalResult(
                name=self.name,
                score=score,
                weight=weight,
                reason=(
                    f"Funding rate {rate:.5f} -> "
                    f"negative funding favours spot entry"
                ),
                timestamp=now,
            )
        except Exception as exc:
            logger.error(
                "funding_rate_signal_failed",
                symbol=symbol,
                error=str(exc),
            )
            return SignalResult(
                name=self.name,
                score=0.0,
                weight=weight,
                reason=f"FundingRateSignal failed: {exc}",
                timestamp=now,
            )
