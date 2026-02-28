"""Trading strategy base class and built-in implementations.

Defines an abstract :class:`Strategy` interface and three concrete strategies:

* :class:`MeanReversionStrategy` — RSI-based mean reversion.
* :class:`MomentumStrategy` — Price vs SMA trend following.
* :class:`MACDCrossoverStrategy` — MACD histogram crossover signals.

A :class:`StrategyRegistry` catalogues all available strategies and provides
factory-style instantiation.

Usage::

    from src.backtest.strategy import strategy_registry
    strat = strategy_registry.create("mean_reversion")
    signals = strat.generate_signals(features_df)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------

class Strategy(ABC):
    """Base class for trading strategies.

    Args:
        name: Human-readable strategy name.
        config: Optional configuration overrides.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}

    @abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        """Produce trading signals from features.

        Returns a DataFrame with the same DatetimeIndex as *features*.
        Each column corresponds to a ticker (or a single ``signal`` column
        for single-asset strategies).  Values: 1 = long, -1 = short,
        0 = flat.

        If all feature values are NaN the strategy must return all zeros.

        Args:
            features: Feature DataFrame (DatetimeIndex, indicator columns).

        Returns:
            Signal DataFrame with integer values in {-1, 0, 1}.
        """

    def describe(self) -> str:
        """Human-readable description of the strategy logic."""
        return f"Strategy: {self.name}"


# ------------------------------------------------------------------
# 1. Mean Reversion (RSI-based)
# ------------------------------------------------------------------

class MeanReversionStrategy(Strategy):
    """Buy when RSI < oversold threshold, sell when RSI > overbought threshold.

    Requires ``rsi_14`` (or whatever RSI column exists) in features.

    Config overrides:
        oversold_threshold: RSI level below which to go long (default 30).
        overbought_threshold: RSI level above which to go short (default 70).
    """

    def __init__(
        self,
        name: str = "mean_reversion",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, config)
        self._oversold = self.config.get("oversold_threshold", 30)
        self._overbought = self.config.get("overbought_threshold", 70)

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        # Find the RSI column (e.g. rsi_14)
        rsi_cols = [c for c in features.columns if c.startswith("rsi_")]
        if not rsi_cols:
            logger.warning("No RSI column found — returning flat signals")
            return pd.DataFrame(
                np.zeros((len(features), 1), dtype=int),
                index=features.index,
                columns=["signal"],
            )

        rsi_col = rsi_cols[0]
        rsi = features[rsi_col]

        signal = pd.Series(
            np.zeros(len(features), dtype=int),
            index=features.index,
            name="signal",
        )
        signal[rsi < self._oversold] = 1
        signal[rsi > self._overbought] = -1
        # NaN RSI → flat
        signal[rsi.isna()] = 0

        logger.info(
            "MeanReversion signals: %d long, %d short, %d flat",
            (signal == 1).sum(),
            (signal == -1).sum(),
            (signal == 0).sum(),
        )
        return signal.to_frame()

    def describe(self) -> str:
        return (
            f"Mean Reversion: buy when RSI < {self._oversold}, "
            f"sell when RSI > {self._overbought}"
        )


# ------------------------------------------------------------------
# 2. Momentum (Close > SMA)
# ------------------------------------------------------------------

class MomentumStrategy(Strategy):
    """Buy when Close > SMA_50, sell when Close < SMA_50.

    Requires ``sma_50`` in features and ``Close`` in the data.

    Config overrides:
        sma_window: SMA window to use (default 50).
    """

    def __init__(
        self,
        name: str = "momentum",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, config)
        self._sma_window = self.config.get("sma_window", 50)

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        sma_col = f"sma_{self._sma_window}"

        if sma_col not in features.columns:
            logger.warning("Column %s not found — returning flat signals", sma_col)
            return pd.DataFrame(
                np.zeros((len(features), 1), dtype=int),
                index=features.index,
                columns=["signal"],
            )

        if "Close" not in features.columns:
            logger.warning("Close column not found — returning flat signals")
            return pd.DataFrame(
                np.zeros((len(features), 1), dtype=int),
                index=features.index,
                columns=["signal"],
            )

        close = features["Close"]
        sma = features[sma_col]

        signal = pd.Series(
            np.zeros(len(features), dtype=int),
            index=features.index,
            name="signal",
        )
        signal[close > sma] = 1
        signal[close < sma] = -1
        # NaN → flat
        signal[close.isna() | sma.isna()] = 0

        logger.info(
            "Momentum signals: %d long, %d short, %d flat",
            (signal == 1).sum(),
            (signal == -1).sum(),
            (signal == 0).sum(),
        )
        return signal.to_frame()

    def describe(self) -> str:
        return f"Momentum: buy when Close > SMA_{self._sma_window}"


# ------------------------------------------------------------------
# 3. MACD Crossover
# ------------------------------------------------------------------

class MACDCrossoverStrategy(Strategy):
    """Buy on MACD histogram crossing from negative to positive; sell on
    the reverse crossover.  Hold (0) otherwise.

    Requires ``macd_histogram`` in features.
    """

    def __init__(
        self,
        name: str = "macd_crossover",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, config)

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        hist_col = "macd_histogram"
        if hist_col not in features.columns:
            logger.warning("Column %s not found — returning flat signals", hist_col)
            return pd.DataFrame(
                np.zeros((len(features), 1), dtype=int),
                index=features.index,
                columns=["signal"],
            )

        hist = features[hist_col]
        prev_hist = hist.shift(1)

        signal = pd.Series(
            np.zeros(len(features), dtype=int),
            index=features.index,
            name="signal",
        )

        # Bullish crossover: previous bar negative (or zero), current bar positive
        bullish = (prev_hist <= 0) & (hist > 0)
        # Bearish crossover: previous bar positive (or zero), current bar negative
        bearish = (prev_hist >= 0) & (hist < 0)

        signal[bullish] = 1
        signal[bearish] = -1
        # Everything else stays 0 (hold / flat)

        # NaN → flat
        signal[hist.isna() | prev_hist.isna()] = 0

        logger.info(
            "MACDCrossover signals: %d long, %d short, %d flat",
            (signal == 1).sum(),
            (signal == -1).sum(),
            (signal == 0).sum(),
        )
        return signal.to_frame()

    def describe(self) -> str:
        return "MACD Crossover: buy/sell on histogram zero-line crossover"


# ------------------------------------------------------------------
# Strategy Registry
# ------------------------------------------------------------------

class StrategyRegistry:
    """Central catalogue of available strategy types.

    Usage::

        registry = StrategyRegistry()
        registry.register("mean_reversion", MeanReversionStrategy)
        strat = registry.create("mean_reversion", config=cfg)
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[Strategy]] = {}

    def register(self, name: str, strategy_class: type[Strategy]) -> None:
        """Add a strategy class to the registry.

        Args:
            name: Short lookup name (e.g. ``"mean_reversion"``).
            strategy_class: Must be a :class:`Strategy` subclass.
        """
        if not (isinstance(strategy_class, type) and issubclass(strategy_class, Strategy)):
            raise TypeError(f"{strategy_class} is not a Strategy subclass")
        self._registry[name] = strategy_class

    def create(self, name: str, config: dict[str, Any] | None = None) -> Strategy:
        """Instantiate a registered strategy.

        Args:
            name: Registered strategy name.
            config: Optional configuration passed to the strategy constructor.

        Returns:
            A fresh Strategy instance.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._registry:
            raise KeyError(
                f"Unknown strategy '{name}'. Available: {self.list_strategies()}"
            )
        return self._registry[name](name=name, config=config)

    def list_strategies(self) -> list[str]:
        """Return sorted list of registered strategy names."""
        return sorted(self._registry.keys())


# ------------------------------------------------------------------
# Module-level registry — populated at import time
# ------------------------------------------------------------------

strategy_registry = StrategyRegistry()
strategy_registry.register("mean_reversion", MeanReversionStrategy)
strategy_registry.register("momentum", MomentumStrategy)
strategy_registry.register("macd_crossover", MACDCrossoverStrategy)
