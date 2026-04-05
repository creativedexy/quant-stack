"""Backtesting engine using plain NumPy / Pandas.

Runs vectorised backtests with realistic transaction costs (commission and
slippage) and produces a structured :class:`BacktestResult`.

Future: replace inner loop with VectorBT for 10-100x speedup on large
universes.

Usage::

    from src.backtest.engine import BacktestEngine
    engine = BacktestEngine()
    result = engine.run(strategy, prices, features)
    comparison = engine.compare([strat_a, strat_b], prices, features)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.strategy import Strategy
from src.portfolio.risk import risk_summary
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for backtest outputs.

    Attributes:
        strategy_name: Name of the strategy that was tested.
        equity_curve: Daily portfolio value (starts at *initial_capital*).
        returns: Daily portfolio returns.
        positions: Daily position per ticker (+1 / -1 / 0).
        trades: Individual trades with date, ticker, side, quantity,
            price, and cost columns.
        metrics: Summary metrics from :func:`risk_summary`.
        config: Backtest parameters used for this run.
    """

    strategy_name: str
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class BacktestEngine:
    """Run strategy backtests with realistic transaction costs.

    Reads parameters from ``config['backtest']``:
        initial_capital, commission_pct, slippage_pct, benchmark.

    Args:
        config: Full project config dict.  If ``None`` the default
            ``config/settings.yaml`` is loaded.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()
        self._config = config

        bt = config.get("backtest", {})
        self.initial_capital: float = float(bt.get("initial_capital", 100_000))
        self.commission: float = float(bt.get("commission_pct", 0.001))
        self.slippage: float = float(bt.get("slippage_pct", 0.0005))
        self.benchmark: str = config.get("universe", {}).get("benchmark", "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        features: pd.DataFrame,
    ) -> BacktestResult:
        """Run a single strategy backtest.

        Args:
            strategy: Strategy instance to test.
            prices: OHLCV DataFrame with DatetimeIndex (must contain
                ``Close``).  For multi-ticker backtests pass a dict of
                ``{ticker: DataFrame}`` — each is processed independently
                with equal capital allocation (1/N).
            features: Feature DataFrame from :class:`FeaturePipeline`.

        Returns:
            :class:`BacktestResult` with equity curve, returns, positions,
            trades, metrics, and config.
        """
        logger.info(
            "Running backtest for '%s': capital=%.0f, commission=%.4f, "
            "slippage=%.4f",
            strategy.name,
            self.initial_capital,
            self.commission,
            self.slippage,
        )

        # 1. Generate signals
        signals_df = strategy.generate_signals(features)

        # Flatten to a single signal Series if the DataFrame has one column
        if isinstance(signals_df, pd.DataFrame) and signals_df.shape[1] == 1:
            signals = signals_df.iloc[:, 0]
        else:
            signals = signals_df

        # 2. Align prices and signals to a common index
        close = prices["Close"] if isinstance(prices, pd.DataFrame) else prices
        common_idx = close.index.intersection(signals.index)
        close = close.loc[common_idx]
        signals = signals.loc[common_idx]

        # 3. Convert signals to positions (carry forward — the signal
        #    on day t determines the position held from close t to close t+1)
        positions = signals.copy().astype(int)

        # 4. Detect trade days (position changes)
        position_changes = positions.diff().fillna(positions)
        trade_mask = position_changes != 0

        # 5. Build trade log
        trades = self._build_trades(
            close, positions, position_changes, trade_mask,
        )

        # 6. Compute daily returns with costs
        daily_returns = self._compute_returns(
            close, positions, position_changes, trade_mask,
        )

        # 7. Build equity curve
        equity = self.initial_capital * (1 + daily_returns).cumprod()
        equity.name = "equity"

        # 8. Risk metrics via portfolio risk module
        metrics = risk_summary(daily_returns)

        # Add extra backtest-specific metrics
        if len(trades) > 0:
            wins = trades.loc[trades["pnl"] > 0]
            metrics["total_trades"] = len(trades)
            metrics["win_rate"] = len(wins) / len(trades) if len(trades) > 0 else 0.0
        else:
            metrics["total_trades"] = 0
            metrics["win_rate"] = 0.0

        bt_config = {
            "initial_capital": self.initial_capital,
            "commission_pct": self.commission,
            "slippage_pct": self.slippage,
            "strategy": strategy.name,
        }

        logger.info(
            "Backtest '%s' complete: ann_return=%.4f, sharpe=%.2f, "
            "max_dd=%.4f, trades=%d",
            strategy.name,
            metrics["annualised_return"],
            metrics["sharpe"],
            metrics["max_drawdown"],
            metrics["total_trades"],
        )

        return BacktestResult(
            strategy_name=strategy.name,
            equity_curve=equity,
            returns=daily_returns,
            positions=positions.to_frame("position"),
            trades=trades,
            metrics=metrics,
            config=bt_config,
        )

    def compare(
        self,
        strategies: list[Strategy],
        prices: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run multiple strategies and produce a comparison table.

        Columns: strategy_name, annualised_return, sharpe, max_drawdown,
        total_trades, win_rate.  Sorted by Sharpe descending.

        Args:
            strategies: List of Strategy instances.
            prices: OHLCV DataFrame.
            features: Feature DataFrame.

        Returns:
            Comparison DataFrame with one row per strategy.
        """
        rows: list[dict[str, Any]] = []
        for strat in strategies:
            result = self.run(strat, prices, features)
            rows.append({
                "strategy_name": result.strategy_name,
                "annualised_return": result.metrics.get("annualised_return", 0.0),
                "sharpe": result.metrics.get("sharpe", 0.0),
                "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                "total_trades": result.metrics.get("total_trades", 0),
                "win_rate": result.metrics.get("win_rate", 0.0),
            })

        comparison = pd.DataFrame(rows)
        comparison = comparison.sort_values("sharpe", ascending=False)
        comparison = comparison.reset_index(drop=True)

        logger.info("Compared %d strategies", len(strategies))
        return comparison

    def plot_results(
        self,
        result: BacktestResult,
        benchmark_prices: pd.Series | None = None,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """Generate standard backtest visualisation.

        Three subplots with shared x-axis:
            - Top panel: equity curve vs benchmark (if provided).
            - Middle panel: drawdown chart.
            - Bottom panel: daily position / signal chart.

        Args:
            result: BacktestResult from :meth:`run`.
            benchmark_prices: Optional benchmark price series for overlay.
            save_path: If provided, save the figure to this path.

        Returns:
            The matplotlib Figure object.
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Backtest: {result.strategy_name}", fontsize=14)

        # --- Top: Equity curve ---
        ax_eq = axes[0]
        ax_eq.plot(result.equity_curve.index, result.equity_curve.values,
                   label="Portfolio", linewidth=1.2)
        if benchmark_prices is not None:
            # Normalise benchmark to start at same capital
            bm = benchmark_prices.reindex(result.equity_curve.index).dropna()
            if len(bm) > 0:
                bm_normalised = bm / bm.iloc[0] * self.initial_capital
                ax_eq.plot(bm_normalised.index, bm_normalised.values,
                           label="Benchmark", linewidth=1.0, alpha=0.7)
        ax_eq.set_ylabel("Portfolio Value")
        ax_eq.legend(loc="upper left")
        ax_eq.grid(True, alpha=0.3)

        # --- Middle: Drawdown ---
        ax_dd = axes[1]
        wealth = result.equity_curve
        running_max = wealth.cummax()
        drawdown = (wealth - running_max) / running_max
        ax_dd.fill_between(drawdown.index, drawdown.values, 0,
                           color="red", alpha=0.3)
        ax_dd.plot(drawdown.index, drawdown.values, color="red",
                   linewidth=0.8)
        ax_dd.set_ylabel("Drawdown")
        ax_dd.grid(True, alpha=0.3)

        # --- Bottom: Positions ---
        ax_pos = axes[2]
        pos = result.positions.iloc[:, 0] if isinstance(result.positions, pd.DataFrame) else result.positions
        ax_pos.fill_between(pos.index, pos.values, 0, alpha=0.4,
                            step="post")
        ax_pos.set_ylabel("Position")
        ax_pos.set_xlabel("Date")
        ax_pos.set_yticks([-1, 0, 1])
        ax_pos.set_yticklabels(["Short", "Flat", "Long"])
        ax_pos.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved backtest plot to %s", save_path)

        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_returns(
        self,
        close: pd.Series,
        positions: pd.Series,
        position_changes: pd.Series,
        trade_mask: pd.Series,
    ) -> pd.Series:
        """Compute daily returns accounting for commission and slippage."""
        # Raw asset returns
        asset_returns = close.pct_change().fillna(0.0)

        # Strategy return = position[t-1] * asset_return[t]
        # (position on day t-1 determines exposure for day t's return)
        strategy_returns = positions.shift(1).fillna(0) * asset_returns

        # Transaction costs — applied on trade days only
        # Cost = abs(position_change) * (commission + slippage)
        cost_pct = pd.Series(0.0, index=close.index)
        cost_pct[trade_mask] = (
            position_changes[trade_mask].abs()
            * (self.commission + self.slippage)
        )

        daily_returns = strategy_returns - cost_pct
        daily_returns.name = "returns"
        return daily_returns

    def _build_trades(
        self,
        close: pd.Series,
        positions: pd.Series,
        position_changes: pd.Series,
        trade_mask: pd.Series,
    ) -> pd.DataFrame:
        """Build a DataFrame of individual trades."""
        if not trade_mask.any():
            return pd.DataFrame(
                columns=["date", "ticker", "side", "quantity", "price", "cost", "pnl"],
            )

        trade_dates = close.index[trade_mask]
        records: list[dict[str, Any]] = []

        entry_price: float | None = None
        entry_side: int = 0

        for dt in trade_dates:
            change = int(position_changes.loc[dt])
            pos = int(positions.loc[dt])
            raw_price = float(close.loc[dt])

            # Determine side
            if change > 0:
                side = "buy"
                exec_price = raw_price * (1 + self.slippage)
            else:
                side = "sell"
                exec_price = raw_price * (1 - self.slippage)

            quantity = abs(change)
            cost = quantity * exec_price * self.commission

            # Simple PnL tracking: when closing a position, compute realised PnL
            pnl = 0.0
            if entry_price is not None and pos == 0:
                # Closing a position
                if entry_side == 1:
                    pnl = (exec_price - entry_price) * quantity - cost
                elif entry_side == -1:
                    pnl = (entry_price - exec_price) * quantity - cost
                entry_price = None
                entry_side = 0
            elif entry_price is None and pos != 0:
                # Opening a new position
                entry_price = exec_price
                entry_side = pos
            elif entry_price is not None and pos != 0 and pos != entry_side:
                # Reversing position — close old + open new
                if entry_side == 1:
                    pnl = (exec_price - entry_price) * quantity - cost
                elif entry_side == -1:
                    pnl = (entry_price - exec_price) * quantity - cost
                entry_price = exec_price
                entry_side = pos

            records.append({
                "date": dt,
                "ticker": "asset",
                "side": side,
                "quantity": quantity,
                "price": round(exec_price, 4),
                "cost": round(cost, 4),
                "pnl": round(pnl, 4),
            })

        return pd.DataFrame(records)
