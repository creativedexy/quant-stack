"""Mean reversion signal based on 24h price change.

Scores based on how far a price has dropped relative to a threshold.
Larger dips produce stronger buy signals (contrarian mean reversion).
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.price_feed import BinancePriceFeed
from src.signals.base import BaseSignal, SignalResult, clamp
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MeanReversionSignal(BaseSignal):
    """Scores recent price drops as mean reversion opportunities.

    A 24h price drop of 4% (configurable) produces score 1.0.
    Price rises produce score 0.0 (no mean reversion signal).

    Args:
        config: Full settings dict from config_loader.
        price_feed: BinancePriceFeed instance for price data.
    """

    def __init__(
        self, config: dict, price_feed: BinancePriceFeed
    ) -> None:
        super().__init__(config)
        self._price_feed = price_feed

    @property
    def name(self) -> str:
        return "mean_reversion"

    def compute(
        self, symbol: str, **kwargs: object
    ) -> SignalResult:
        """Compute mean reversion score from 24h price change.

        Args:
            symbol: Trading pair (e.g. 'BTCGBP').

        Returns:
            SignalResult with dip-based score.
        """
        now = datetime.now(timezone.utc)
        weight = self.config.get("scoring", {}).get(
            "signal_weights", {}
        ).get("mean_reversion", 0.35)

        try:
            signals_cfg = self.config.get("signals", {})
            dip_threshold = signals_cfg.get(
                "mean_reversion_dip_pct", 4.0
            )
            window_hours = signals_cfg.get(
                "mean_reversion_window_hours", 24
            )

            change_pct = self._price_feed.get_price_change_pct(
                symbol, window_hours
            )

            if change_pct >= 0:
                return SignalResult(
                    name=self.name,
                    score=0.0,
                    weight=weight,
                    reason=(
                        f"{symbol} rose {change_pct:.1f}% in "
                        f"{window_hours}h -> no mean reversion signal"
                    ),
                    timestamp=now,
                )

            score = clamp(abs(change_pct) / dip_threshold)

            return SignalResult(
                name=self.name,
                score=score,
                weight=weight,
                reason=(
                    f"{symbol} dropped {abs(change_pct):.1f}% in "
                    f"{window_hours}h -> "
                    f"{'above' if abs(change_pct) >= dip_threshold else 'below'} "
                    f"{dip_threshold}% reversion threshold"
                ),
                timestamp=now,
            )
        except Exception as exc:
            logger.error(
                "mean_reversion_signal_failed",
                symbol=symbol,
                error=str(exc),
            )
            return SignalResult(
                name=self.name,
                score=0.0,
                weight=weight,
                reason=f"MeanReversionSignal failed: {exc}",
                timestamp=now,
            )
