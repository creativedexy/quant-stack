"""Event detectors -- condition monitoring for event-triggered personas.

Each detector watches market data for specific conditions (drawdowns,
VIX spikes, etc.) and returns a :class:`DetectionResult` indicating
whether the persona should be activated.

Uses the same registry pattern as :mod:`src.backtest.strategy`.

Usage::

    from src.personas.detectors import detector_registry
    detector = detector_registry.create("drawdown", params={"drawdown_threshold": 0.10})
    result = detector.check(market_data)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Result type
# ------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Outcome of an event detector check.

    Attributes:
        triggered: Whether the condition was met.
        reason: Human-readable explanation of the trigger (or why not).
        metrics: Supporting numerical data for the brief.
    """

    triggered: bool
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------

class EventDetector(ABC):
    """Base class for event detectors.

    Args:
        params: Configuration parameters from YAML.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def check(self, market_data: dict[str, pd.DataFrame]) -> DetectionResult:
        """Evaluate whether the event condition is met.

        Args:
            market_data: Dictionary mapping ticker -> OHLCV DataFrame
                with DatetimeIndex.

        Returns:
            A :class:`DetectionResult` with triggered status and context.
        """


# ------------------------------------------------------------------
# Built-in detectors
# ------------------------------------------------------------------

class DrawdownDetector(EventDetector):
    """Detects market drawdowns exceeding a configurable threshold.

    Watches the persona's universe for rolling peak-to-trough
    drawdown over a lookback window.

    Params:
        drawdown_threshold: Trigger when drawdown exceeds this (e.g. 0.10).
        lookback_days: Number of trading days to look back (default 21).
    """

    def check(self, market_data: dict[str, pd.DataFrame]) -> DetectionResult:
        threshold = self.params.get("drawdown_threshold", 0.10)
        lookback = self.params.get("lookback_days", 21)

        worst_dd = 0.0
        worst_ticker = ""

        for ticker, df in market_data.items():
            if df.empty or "Close" not in df.columns:
                continue

            close = df["Close"].dropna()
            if len(close) < 2:
                continue

            # Use the lookback window
            recent = close.iloc[-lookback:] if len(close) >= lookback else close
            peak = recent.cummax()
            drawdown = (recent - peak) / peak
            current_dd = abs(float(drawdown.iloc[-1]))

            if current_dd > worst_dd:
                worst_dd = current_dd
                worst_ticker = ticker

        triggered = worst_dd >= threshold
        reason = (
            f"Market drawdown {worst_dd:.1%} on {worst_ticker} "
            f"over {lookback}d (threshold: {threshold:.0%})"
            if triggered
            else f"No drawdown exceeds {threshold:.0%} (worst: {worst_dd:.1%})"
        )

        return DetectionResult(
            triggered=triggered,
            reason=reason,
            metrics={
                "worst_drawdown": round(worst_dd, 4),
                "worst_ticker": worst_ticker,
                "threshold": threshold,
                "lookback_days": lookback,
            },
        )


class CryptoDrawdownDetector(EventDetector):
    """Detects extended crypto bear markets by watching a reference ticker.

    Params:
        index_ticker: Reference ticker to monitor (default ``"BTC-USD"``).
        drawdown_threshold: Trigger when peak-to-trough drop exceeds this.
        lookback_days: Number of days to look back (default 60).
    """

    def check(self, market_data: dict[str, pd.DataFrame]) -> DetectionResult:
        index_ticker = self.params.get("index_ticker", "BTC-USD")
        threshold = self.params.get("drawdown_threshold", 0.30)
        lookback = self.params.get("lookback_days", 60)

        df = market_data.get(index_ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return DetectionResult(
                triggered=False,
                reason=f"No data available for {index_ticker}",
                metrics={"index_ticker": index_ticker},
            )

        close = df["Close"].dropna()
        if len(close) < 2:
            return DetectionResult(
                triggered=False,
                reason=f"Insufficient data for {index_ticker}",
            )

        recent = close.iloc[-lookback:] if len(close) >= lookback else close
        peak = recent.cummax()
        drawdown = (recent - peak) / peak
        current_dd = abs(float(drawdown.iloc[-1]))

        triggered = current_dd >= threshold
        reason = (
            f"{index_ticker} drawdown {current_dd:.1%} "
            f"over {lookback}d (threshold: {threshold:.0%})"
            if triggered
            else (
                f"{index_ticker} drawdown {current_dd:.1%} "
                f"below {threshold:.0%} threshold"
            )
        )

        return DetectionResult(
            triggered=triggered,
            reason=reason,
            metrics={
                "index_ticker": index_ticker,
                "current_drawdown": round(current_dd, 4),
                "threshold": threshold,
                "lookback_days": lookback,
                "peak_price": round(float(peak.iloc[-1]), 2),
                "current_price": round(float(close.iloc[-1]), 2),
            },
        )


class VIXSpikeDetector(EventDetector):
    """Detects sharp market sell-offs via VIX spike + market drop.

    Requires both conditions to be met simultaneously:
    1. VIX above a threshold
    2. A market ticker has dropped by a percentage over lookback days

    Params:
        vix_ticker: VIX ticker (default ``"^VIX"``).
        vix_threshold: VIX level to trigger (default 30).
        market_drop_pct: Minimum market decline (default 0.05 = 5%).
        lookback_days: Days to measure market drop (default 5).
    """

    def check(self, market_data: dict[str, pd.DataFrame]) -> DetectionResult:
        vix_ticker = self.params.get("vix_ticker", "^VIX")
        vix_threshold = self.params.get("vix_threshold", 30)
        market_drop_pct = self.params.get("market_drop_pct", 0.05)
        lookback = self.params.get("lookback_days", 5)

        # Check VIX level
        vix_df = market_data.get(vix_ticker)
        if vix_df is None or vix_df.empty or "Close" not in vix_df.columns:
            return DetectionResult(
                triggered=False,
                reason=f"No VIX data available ({vix_ticker})",
                metrics={"vix_ticker": vix_ticker},
            )

        vix_level = float(vix_df["Close"].dropna().iloc[-1])
        vix_triggered = vix_level >= vix_threshold

        # Check market drop across non-VIX tickers
        worst_drop = 0.0
        worst_ticker = ""
        for ticker, df in market_data.items():
            if ticker == vix_ticker or df.empty or "Close" not in df.columns:
                continue
            close = df["Close"].dropna()
            if len(close) < lookback + 1:
                continue
            start_price = float(close.iloc[-(lookback + 1)])
            end_price = float(close.iloc[-1])
            if start_price > 0:
                drop = (start_price - end_price) / start_price
                if drop > worst_drop:
                    worst_drop = drop
                    worst_ticker = ticker

        market_triggered = worst_drop >= market_drop_pct
        triggered = vix_triggered and market_triggered

        if triggered:
            reason = (
                f"VIX at {vix_level:.1f} (>{vix_threshold}) and "
                f"{worst_ticker} down {worst_drop:.1%} over {lookback}d"
            )
        elif vix_triggered:
            reason = (
                f"VIX at {vix_level:.1f} (>{vix_threshold}) but no market "
                f"drop >{market_drop_pct:.0%} detected"
            )
        else:
            reason = (
                f"VIX at {vix_level:.1f} (<{vix_threshold}), "
                f"no sell-off conditions"
            )

        return DetectionResult(
            triggered=triggered,
            reason=reason,
            metrics={
                "vix_level": round(vix_level, 2),
                "vix_threshold": vix_threshold,
                "worst_market_drop": round(worst_drop, 4),
                "worst_ticker": worst_ticker,
                "market_drop_threshold": market_drop_pct,
                "lookback_days": lookback,
            },
        )


class ThresholdDetector(EventDetector):
    """Generic threshold detector for user-defined personas.

    Watches a configurable ticker's drawdown against a threshold.
    Provides extensibility without requiring new detector code.

    Params:
        watch_ticker: Ticker to monitor.
        threshold: Value to trigger at.
        lookback_days: Days to look back (default 21).
    """

    def check(self, market_data: dict[str, pd.DataFrame]) -> DetectionResult:
        watch_ticker = self.params.get("watch_ticker", "")
        threshold = self.params.get("threshold", 0.10)
        lookback = self.params.get("lookback_days", 21)

        if not watch_ticker:
            return DetectionResult(
                triggered=False,
                reason="No watch_ticker configured for threshold detector",
            )

        df = market_data.get(watch_ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return DetectionResult(
                triggered=False,
                reason=f"No data for {watch_ticker}",
            )

        close = df["Close"].dropna()
        if len(close) < 2:
            return DetectionResult(
                triggered=False,
                reason=f"Insufficient data for {watch_ticker}",
            )

        recent = close.iloc[-lookback:] if len(close) >= lookback else close
        peak = recent.cummax()
        drawdown = (recent - peak) / peak
        current_dd = abs(float(drawdown.iloc[-1]))

        triggered = current_dd >= threshold
        return DetectionResult(
            triggered=triggered,
            reason=(
                f"{watch_ticker} drawdown {current_dd:.1%} "
                f"{'exceeds' if triggered else 'below'} {threshold:.0%} threshold"
            ),
            metrics={
                "watch_ticker": watch_ticker,
                "current_drawdown": round(current_dd, 4),
                "threshold": threshold,
                "lookback_days": lookback,
            },
        )


# ------------------------------------------------------------------
# Detector Registry
# ------------------------------------------------------------------

class DetectorRegistry:
    """Central catalogue of available event detector types.

    Usage::

        registry = DetectorRegistry()
        registry.register("drawdown", DrawdownDetector)
        detector = registry.create("drawdown", params={"drawdown_threshold": 0.10})
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[EventDetector]] = {}

    def register(self, name: str, detector_class: type[EventDetector]) -> None:
        """Add a detector class to the registry.

        Args:
            name: Short lookup name (e.g. ``"drawdown"``).
            detector_class: Must be an :class:`EventDetector` subclass.
        """
        if not (
            isinstance(detector_class, type)
            and issubclass(detector_class, EventDetector)
        ):
            raise TypeError(f"{detector_class} is not an EventDetector subclass")
        self._registry[name] = detector_class

    def create(
        self, name: str, params: dict[str, Any] | None = None
    ) -> EventDetector:
        """Instantiate a registered detector.

        Args:
            name: Registered detector name.
            params: Configuration parameters passed to the constructor.

        Returns:
            A fresh EventDetector instance.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._registry:
            raise KeyError(
                f"Unknown detector '{name}'. "
                f"Available: {self.list_detectors()}"
            )
        return self._registry[name](params=params)

    def list_detectors(self) -> list[str]:
        """Return sorted list of registered detector names."""
        return sorted(self._registry.keys())


# ------------------------------------------------------------------
# Module-level registry
# ------------------------------------------------------------------

detector_registry = DetectorRegistry()
detector_registry.register("drawdown", DrawdownDetector)
detector_registry.register("crypto_drawdown", CryptoDrawdownDetector)
detector_registry.register("vix_spike", VIXSpikeDetector)
detector_registry.register("threshold", ThresholdDetector)
