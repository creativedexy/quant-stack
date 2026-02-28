"""Backtesting engine wrapping VectorBT.

Runs vectorised backtests with realistic transaction costs and produces
a structured result dataclass.

Usage:
    from src.backtest.engine import run_backtest, compare_strategies
    result = run_backtest(strategy, features_df)
    comparison = compare_strategies([strat_a, strat_b], features_df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt

    _HAS_VBT = True
except ImportError:  # pragma: no cover
    _HAS_VBT = False

from src.backtest.strategy import Strategy
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ===================================================================
# Result container
# ===================================================================

@dataclass
class BacktestResult:
    """Container for backtest outputs.

    Attributes:
        strategy_name: Name of the strategy that was tested.
        equity_curve: Portfolio value over time (starts at initial_capital).
        returns: Daily portfolio returns.
        trades: DataFrame of executed trades.
        metrics: Summary performance metrics dict.
        signals: Raw signal series from the strategy.
    """

    strategy_name: str
    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    signals: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# ===================================================================
# Core engine
# ===================================================================

def run_backtest(
    strategy: Strategy,
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> BacktestResult:
    """Run a single-asset backtest using VectorBT.

    Args:
        strategy: A Strategy instance that generates signals.
        data: Features DataFrame with OHLCV columns and indicators.
            Must have a DatetimeIndex.
        config: Full project config.  If ``None`` loads the default.

    Returns:
        BacktestResult with equity curve, returns, trades, and metrics.

    Raises:
        ImportError: If VectorBT is not installed.
    """
    if not _HAS_VBT:  # pragma: no cover
        raise ImportError(
            "vectorbt is required for backtesting. "
            "Install with: pip install 'quant-stack[backtest]'"
        )

    if config is None:
        config = load_config()

    bt_cfg = config.get("backtest", {})
    initial_capital = float(bt_cfg.get("initial_capital", 100_000))
    commission = float(bt_cfg.get("commission_pct", 0.001))
    slippage = float(bt_cfg.get("slippage_pct", 0.0005))

    logger.info(
        "Running backtest for '%s': capital=%.0f, commission=%.4f, slippage=%.4f",
        strategy.name,
        initial_capital,
        commission,
        slippage,
    )

    # Generate signals
    signals = strategy.generate_signals(data)

    # Derive entry / exit arrays from signals
    close = data["Close"]
    entries = (signals == 1) & (signals.shift(1) != 1)
    exits = (signals != 1) & (signals.shift(1) == 1)
    short_entries = (signals == -1) & (signals.shift(1) != -1)
    short_exits = (signals != -1) & (signals.shift(1) == -1)

    # Run VectorBT portfolio simulation
    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=initial_capital,
        fees=commission,
        slippage=slippage,
    )

    equity_curve = pf.value()
    equity_curve.name = "equity"
    returns = pf.returns()
    returns.name = "returns"

    # Extract trades
    trades = _extract_trades(pf)

    # Compute metrics
    metrics = _compute_metrics(equity_curve, returns, initial_capital, len(trades))

    logger.info(
        "Backtest '%s' complete: total_return=%.4f, sharpe=%.2f, max_dd=%.4f",
        strategy.name,
        metrics["total_return"],
        metrics.get("sharpe_ratio", float("nan")),
        metrics["max_drawdown"],
    )

    return BacktestResult(
        strategy_name=strategy.name,
        equity_curve=equity_curve,
        returns=returns,
        trades=trades,
        metrics=metrics,
        signals=signals,
    )


# ===================================================================
# Strategy comparison
# ===================================================================

def compare_strategies(
    strategies: list[Strategy],
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run multiple strategies on the same data and compare results.

    Args:
        strategies: List of Strategy instances.
        data: Features DataFrame with OHLCV + indicators.
        config: Full project config.

    Returns:
        DataFrame with one row per strategy and columns for key metrics.
    """
    results: list[BacktestResult] = []
    for strat in strategies:
        result = run_backtest(strat, data, config)
        results.append(result)

    rows = []
    for r in results:
        row = {"strategy": r.strategy_name}
        row.update(r.metrics)
        rows.append(row)

    comparison = pd.DataFrame(rows).set_index("strategy")

    logger.info(
        "Comparison of %d strategies complete",
        len(strategies),
    )

    return comparison


# ===================================================================
# Internal helpers
# ===================================================================

def _extract_trades(pf: Any) -> pd.DataFrame:
    """Pull trade records from a VectorBT portfolio."""
    try:
        trades = pf.trades.records_readable.copy()
    except Exception:
        trades = pd.DataFrame()
    return trades


def _compute_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    initial_capital: float,
    total_trades: int = 0,
) -> dict[str, float]:
    """Derive standard performance metrics from equity and returns."""
    total_return = (equity_curve.iloc[-1] / initial_capital) - 1.0

    # Annualised return (assume 252 trading days)
    n_days = len(returns)
    if n_days > 1:
        ann_return = (1 + total_return) ** (252 / n_days) - 1
    else:
        ann_return = 0.0

    # Volatility
    ann_vol = float(returns.std(ddof=1) * np.sqrt(252)) if n_days > 1 else 0.0

    # Sharpe (excess of 0 for simplicity; risk-free adjustment done elsewhere)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    max_dd = float(drawdown.min())

    # Calmar
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    return {
        "initial_capital": initial_capital,
        "final_equity": float(equity_curve.iloc[-1]),
        "total_return": total_return,
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "total_trades": total_trades,
    }
