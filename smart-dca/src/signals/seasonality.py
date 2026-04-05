"""Seasonality signal based on UTC hour and day of week.

Scores time windows by historical volatility and spread patterns.
No external data needed — uses current UTC time only.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.signals.base import BaseSignal, SignalResult, clamp
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SeasonalitySignal(BaseSignal):
    """Scores the current UTC hour/day for favourable entry timing.

    Peak hours (e.g. 14-18 UTC) get higher scores due to historically
    lower volatility and better spreads. Weekend early mornings also
    score highly. Low-activity hours (0-5 UTC) score lower.

    Args:
        config: Full settings dict from config_loader.
    """

    @property
    def name(self) -> str:
        return "seasonality"

    def compute(
        self, symbol: str, **kwargs: object
    ) -> SignalResult:
        """Compute seasonality score based on current UTC time.

        Args:
            symbol: Trading pair (unused but required by interface).
            **kwargs: Accepts 'now' as datetime override for testing.

        Returns:
            SignalResult with time-based score.
        """
        try:
            now: datetime = kwargs.get("now", datetime.now(timezone.utc))
            hour = now.hour
            weekday = now.weekday()  # 0=Monday, 5=Saturday, 6=Sunday

            signals_cfg = self.config.get("signals", {})
            seasonality_cfg = signals_cfg.get("seasonality_scores", {})
            peak_hours = seasonality_cfg.get(
                "peak_hours", [14, 15, 16, 17, 18]
            )
            low_hours = seasonality_cfg.get(
                "low_hours", [0, 1, 2, 3, 4, 5]
            )

            score = self._compute_score(hour, weekday, peak_hours, low_hours)
            weight = self.config.get("scoring", {}).get(
                "signal_weights", {}
            ).get("seasonality", 0.20)

            reason = self._build_reason(hour, weekday, score)

            return SignalResult(
                name=self.name,
                score=clamp(score),
                weight=weight,
                reason=reason,
                timestamp=now,
            )
        except Exception as exc:
            logger.error("seasonality_signal_failed", error=str(exc))
            return SignalResult(
                name=self.name,
                score=0.0,
                weight=0.20,
                reason=f"SeasonalitySignal failed: {exc}",
                timestamp=datetime.now(timezone.utc),
            )

    def _compute_score(
        self,
        hour: int,
        weekday: int,
        peak_hours: list[int],
        low_hours: list[int],
    ) -> float:
        """Calculate raw score from hour and weekday."""
        # Weekend early morning is historically favourable
        is_weekend = weekday in (5, 6)
        is_early_morning = hour < 8

        if is_weekend and is_early_morning:
            return 0.85

        if hour in peak_hours:
            return 0.75

        if hour in low_hours:
            return 0.30

        # Default neutral score
        return 0.50

    def _build_reason(
        self, hour: int, weekday: int, score: float
    ) -> str:
        """Build human-readable reason string."""
        day_names = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        day_name = day_names[weekday]

        if score >= 0.75:
            qualifier = "historically favourable entry window"
        elif score <= 0.35:
            qualifier = "low-activity period"
        else:
            qualifier = "neutral timing"

        return f"Hour {hour} UTC, {day_name} -> {qualifier}"


class BacktestSeasonalitySignal(SeasonalitySignal):
    """Seasonality signal that uses a caller-supplied backtest timestamp.

    In live mode SeasonalitySignal reads datetime.now(timezone.utc).
    During backtesting we need to evaluate against the historical
    timestamp being replayed, not the wall clock.

    Args:
        config: Full settings dict from config_loader.
        backtest_time: UTC-aware datetime to use instead of now().
    """

    def __init__(self, config: dict, backtest_time: datetime) -> None:
        super().__init__(config)
        self._backtest_time = backtest_time

    @property
    def backtest_time(self) -> datetime:
        """Return the currently configured backtest timestamp."""
        return self._backtest_time

    @backtest_time.setter
    def backtest_time(self, value: datetime) -> None:
        """Update the backtest timestamp for the next compute() call."""
        self._backtest_time = value

    def compute(
        self, symbol: str, **kwargs: object
    ) -> SignalResult:
        """Compute seasonality score using the backtest timestamp.

        Delegates to the parent class with the backtest_time injected
        as the ``now`` keyword argument, ensuring no wall-clock leakage.
        Any ``now`` value passed by the caller is discarded in favour
        of the stored backtest_time.

        Args:
            symbol: Trading pair (unused but required by interface).
            **kwargs: Forwarded to parent; ``now`` is always overridden.

        Returns:
            SignalResult with time-based score at the backtest timestamp.
        """
        # Remove caller-supplied 'now' to avoid duplicate keyword argument
        kwargs.pop("now", None)
        return super().compute(symbol, now=self._backtest_time, **kwargs)
