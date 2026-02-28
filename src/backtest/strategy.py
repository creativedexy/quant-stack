"""Strategy base class and example implementations.

Every strategy takes a feature DataFrame (OHLCV + indicators) and produces a
signal Series with values in {-1, 0, 1} (short, flat, long) or continuous
weights.

Usage:
    from src.backtest.strategy import MomentumStrategy
    strat = MomentumStrategy()
    signals = strat.generate_signals(features_df)
"""

from __future__ import annotations

import abc
from typing import Any

import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ===================================================================
# Abstract base
# ===================================================================

class Strategy(abc.ABC):
    """Abstract base class for all trading strategies."""

    @abc.abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Produce a signal Series from a feature DataFrame.

        Args:
            features: DataFrame containing OHLCV columns and any technical
                indicators / model scores.  Must have a DatetimeIndex.

        Returns:
            Series with the same DatetimeIndex and values representing
            the desired position: -1 (short), 0 (flat), or 1 (long).
            Continuous weights are also acceptable.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "Strategy":
        """Construct a strategy instance from a config dict.

        Subclasses may override this to pull strategy-specific parameters
        from the config.  The default implementation simply calls the
        no-arg constructor.

        Args:
            config: Full project configuration dict.  If ``None`` the
                default settings.yaml is loaded.

        Returns:
            Configured Strategy instance.
        """
        return cls()


# ===================================================================
# Mean-reversion strategy
# ===================================================================

class MeanReversionStrategy(Strategy):
    """Buy when RSI is oversold, sell when overbought.

    Args:
        rsi_column: Name of the RSI column in the features DataFrame.
        rsi_lower: RSI threshold below which a buy signal is generated.
        rsi_upper: RSI threshold above which a sell signal is generated.
    """

    def __init__(
        self,
        rsi_column: str = "rsi_14",
        rsi_lower: int = 30,
        rsi_upper: int = 70,
    ) -> None:
        self.rsi_column = rsi_column
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper

    @property
    def name(self) -> str:
        return "mean_reversion"

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "MeanReversionStrategy":
        if config is None:
            config = load_config()
        bt_cfg = config.get("backtest", {}).get("strategies", {}).get("mean_reversion", {})
        return cls(
            rsi_lower=bt_cfg.get("rsi_lower", 30),
            rsi_upper=bt_cfg.get("rsi_upper", 70),
        )

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate mean-reversion signals based on RSI thresholds.

        Args:
            features: DataFrame containing an RSI column.

        Returns:
            Signal Series: 1 when RSI < lower, -1 when RSI > upper, 0 otherwise.
        """
        if self.rsi_column not in features.columns:
            raise KeyError(
                f"RSI column '{self.rsi_column}' not found in features. "
                f"Available columns: {list(features.columns)}"
            )

        rsi = features[self.rsi_column]
        signals = pd.Series(0, index=features.index, name="signal", dtype=int)
        signals[rsi < self.rsi_lower] = 1
        signals[rsi > self.rsi_upper] = -1

        logger.info(
            "MeanReversion signals: %d long, %d short, %d flat",
            (signals == 1).sum(),
            (signals == -1).sum(),
            (signals == 0).sum(),
        )
        return signals


# ===================================================================
# Momentum strategy
# ===================================================================

class MomentumStrategy(Strategy):
    """Buy when price is above SMA, sell when below.

    Args:
        sma_column: Name of the SMA column in the features DataFrame.
        price_column: Column to compare against the SMA.
    """

    def __init__(
        self,
        sma_column: str = "sma_50",
        price_column: str = "Close",
    ) -> None:
        self.sma_column = sma_column
        self.price_column = price_column

    @property
    def name(self) -> str:
        return "momentum"

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "MomentumStrategy":
        if config is None:
            config = load_config()
        bt_cfg = config.get("backtest", {}).get("strategies", {}).get("momentum", {})
        sma_window = bt_cfg.get("sma_window", 50)
        return cls(sma_column=f"sma_{sma_window}")

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate momentum signals based on price vs SMA.

        Args:
            features: DataFrame containing a price column and an SMA column.

        Returns:
            Signal Series: 1 when price > SMA, -1 when price < SMA.
        """
        if self.sma_column not in features.columns:
            raise KeyError(
                f"SMA column '{self.sma_column}' not found in features. "
                f"Available columns: {list(features.columns)}"
            )

        price = features[self.price_column]
        sma = features[self.sma_column]
        signals = pd.Series(0, index=features.index, name="signal", dtype=int)
        signals[price > sma] = 1
        signals[price < sma] = -1

        logger.info(
            "Momentum signals: %d long, %d short, %d flat",
            (signals == 1).sum(),
            (signals == -1).sum(),
            (signals == 0).sum(),
        )
        return signals


# ===================================================================
# Registry
# ===================================================================

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
}


def get_strategy(name: str, config: dict[str, Any] | None = None) -> Strategy:
    """Look up a strategy by name and instantiate from config.

    Args:
        name: Strategy name (must match a key in ``STRATEGY_REGISTRY``).
        config: Optional full config dict.

    Returns:
        Configured Strategy instance.

    Raises:
        ValueError: If the strategy name is not recognised.
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name].from_config(config)
