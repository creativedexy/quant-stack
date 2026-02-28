"""Backtesting engines and strategy definitions.

Public API::

    from src.backtest import (
        Strategy,
        MeanReversionStrategy,
        MomentumStrategy,
        MACDCrossoverStrategy,
        StrategyRegistry,
        strategy_registry,
        BacktestEngine,
        BacktestResult,
    )
"""

from src.backtest.strategy import (
    MACDCrossoverStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    Strategy,
    StrategyRegistry,
    strategy_registry,
)
from src.backtest.engine import BacktestEngine, BacktestResult

__all__ = [
    "Strategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "MACDCrossoverStrategy",
    "StrategyRegistry",
    "strategy_registry",
    "BacktestEngine",
    "BacktestResult",
]
