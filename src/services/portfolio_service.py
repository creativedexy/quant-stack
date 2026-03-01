"""Portfolio service — portfolio state and analytics for the dashboard.

Provides current weights, risk metrics, equity curves, and allocation
data without exposing underlying file I/O or computation details.

Usage:
    from src.services.data_service import DataService
    from src.services.portfolio_service import PortfolioService
    svc = PortfolioService(DataService())
    weights = svc.get_current_weights()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.services.data_service import DataService
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PortfolioService:
    """Portfolio state and analytics for the dashboard."""

    def __init__(
        self,
        data_service: DataService,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the portfolio service.

        Args:
            data_service: DataService instance for price data access.
            config: Optional configuration dict.
        """
        self.data = data_service
        self._config = config or {}

        data_dir_name = (
            self._config.get("general", {}).get("data_dir", "data")
        )
        self._data_dir = _PROJECT_ROOT / data_dir_name
        self._processed_dir = self._data_dir / "processed"

    def get_current_weights(self) -> pd.Series:
        """Load last saved target weights.

        Reads from data/processed/weights.parquet if available.
        Falls back to equal weights for the configured universe.

        Returns:
            Series indexed by ticker with portfolio weights summing to 1.
        """
        weights_path = self._processed_dir / "weights.parquet"

        if weights_path.exists():
            try:
                df = pd.read_parquet(weights_path)
                # Expect a single-column DataFrame or Series stored as parquet
                if isinstance(df, pd.DataFrame) and len(df.columns) >= 1:
                    weights = df.iloc[:, -1]  # Last column = most recent
                    weights.name = "weight"
                    return weights
            except Exception as exc:
                logger.error(f"Failed to load weights: {exc}")

        # Fall back to equal weights
        tickers = self.data._tickers
        if not tickers:
            return pd.Series(dtype=float, name="weight")

        n = len(tickers)
        return pd.Series(
            [1.0 / n] * n, index=tickers, name="weight"
        )

    def get_risk_metrics(self) -> dict[str, float]:
        """Compute current risk metrics for the portfolio.

        Returns:
            Dict with keys: annual_return, annual_volatility, sharpe_ratio,
            max_drawdown, var_95, cvar_95. Values are NaN if computation
            is not possible.
        """
        returns = self.data.get_returns()
        weights = self.get_current_weights()

        defaults: dict[str, float] = {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "var_95": np.nan,
            "cvar_95": np.nan,
        }

        if returns.empty or weights.empty:
            return defaults

        # Align tickers
        common = returns.columns.intersection(weights.index)
        if len(common) == 0:
            return defaults

        returns = returns[common].dropna()
        w = weights[common]
        w = w / w.sum()  # Re-normalise

        portfolio_returns = returns @ w

        if len(portfolio_returns) < 2:
            return defaults

        annual_return = float(portfolio_returns.mean() * 252)
        annual_vol = float(portfolio_returns.std() * np.sqrt(252))
        sharpe = (
            annual_return / annual_vol if annual_vol > 0 else np.nan
        )

        # Max drawdown
        cum = (1 + portfolio_returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = float(drawdown.min())

        # VaR and CVaR (historical, 95%)
        sorted_returns = portfolio_returns.sort_values()
        var_idx = int(len(sorted_returns) * 0.05)
        var_95 = float(sorted_returns.iloc[var_idx]) if var_idx > 0 else np.nan
        cvar_95 = (
            float(sorted_returns.iloc[:var_idx].mean())
            if var_idx > 0
            else np.nan
        )

        return {
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }

    def get_equity_curve(
        self, strategy_name: str | None = None
    ) -> pd.Series:
        """Load or compute an equity curve.

        If a strategy name is given, attempts to load cached backtest
        results. Otherwise computes from current weights and prices.

        Args:
            strategy_name: Optional strategy to load results for.

        Returns:
            Series representing cumulative portfolio value over time.
        """
        # Try loading cached backtest results
        if strategy_name is not None:
            backtest_path = (
                self._processed_dir
                / "backtests"
                / f"{strategy_name}_equity.parquet"
            )
            if backtest_path.exists():
                try:
                    df = pd.read_parquet(backtest_path)
                    if isinstance(df, pd.DataFrame) and len(df.columns) >= 1:
                        return df.iloc[:, 0]
                except Exception as exc:
                    logger.error(
                        f"Failed to load equity curve for "
                        f"{strategy_name}: {exc}"
                    )

        # Compute from current weights + prices
        returns = self.data.get_returns()
        weights = self.get_current_weights()

        if returns.empty or weights.empty:
            return pd.Series(dtype=float, name="equity")

        common = returns.columns.intersection(weights.index)
        if len(common) == 0:
            return pd.Series(dtype=float, name="equity")

        w = weights[common]
        w = w / w.sum()
        portfolio_returns = returns[common] @ w
        equity = (1 + portfolio_returns).cumprod()
        equity.name = "equity"
        return equity

    def get_allocation_chart_data(self) -> dict[str, Any]:
        """Return data formatted for a pie/bar chart of current allocation.

        Returns:
            Dict with 'labels' (list[str]) and 'values' (list[float])
            suitable for charting libraries.
        """
        weights = self.get_current_weights()

        if weights.empty:
            return {"labels": [], "values": []}

        # Filter out zero-weight positions
        weights = weights[weights > 0]

        return {
            "labels": weights.index.tolist(),
            "values": weights.values.tolist(),
        }
