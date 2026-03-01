"""Strategy service — strategy execution and comparison for the dashboard.

Exposes strategy listings, backtest results, comparisons, and recent
signals without requiring the dashboard to know about file layouts.

Usage:
    from src.services.data_service import DataService
    from src.services.strategy_service import StrategyService
    svc = StrategyService(DataService())
    strategies = svc.get_available_strategies()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.services.data_service import DataService
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Built-in strategy names that are always listed
_DEFAULT_STRATEGIES = [
    "mean_reversion",
    "momentum",
    "trend_following",
]


class StrategyService:
    """Strategy execution and comparison for the dashboard."""

    def __init__(
        self,
        data_service: DataService,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the strategy service.

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
        self._backtests_dir = self._data_dir / "processed" / "backtests"
        self._signals_dir = self._data_dir / "processed" / "signals"

    def get_available_strategies(self) -> list[str]:
        """List registered strategy names.

        Combines built-in defaults with any strategies that have
        cached backtest results on disk.

        Returns:
            Sorted list of unique strategy names.
        """
        strategies = set(_DEFAULT_STRATEGIES)

        # Discover strategies from cached backtest files
        if self._backtests_dir.exists():
            for path in self._backtests_dir.glob("*_results.parquet"):
                name = path.stem.replace("_results", "")
                strategies.add(name)

        return sorted(strategies)

    def get_backtest_results(
        self, strategy_name: str
    ) -> dict[str, Any] | None:
        """Load cached backtest results for a strategy.

        Args:
            strategy_name: Name of the strategy.

        Returns:
            Dict with backtest metrics and data, or None if no cached
            results exist.
        """
        results_path = self._backtests_dir / f"{strategy_name}_results.parquet"

        if not results_path.exists():
            return None

        try:
            df = pd.read_parquet(results_path)
            return {
                "strategy": strategy_name,
                "data": df,
                "rows": len(df),
            }
        except Exception as exc:
            logger.error(
                f"Failed to load backtest results for "
                f"{strategy_name}: {exc}"
            )
            return None

    def get_strategy_comparison(self) -> pd.DataFrame | None:
        """Load cached comparison table if available.

        Looks for a pre-computed comparison file at
        data/processed/backtests/comparison.parquet.

        Returns:
            DataFrame comparing strategy metrics, or None if not available.
        """
        comparison_path = self._backtests_dir / "comparison.parquet"

        if not comparison_path.exists():
            return None

        try:
            return pd.read_parquet(comparison_path)
        except Exception as exc:
            logger.error(f"Failed to load strategy comparison: {exc}")
            return None

    def get_signals(
        self, strategy_name: str, n_days: int = 30
    ) -> pd.DataFrame:
        """Load most recent signals for a strategy.

        Args:
            strategy_name: Name of the strategy.
            n_days: Number of recent trading days to return.

        Returns:
            DataFrame of signals with DatetimeIndex, or empty DataFrame.
        """
        signals_path = self._signals_dir / f"{strategy_name}_signals.parquet"

        if not signals_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(signals_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df.iloc[-n_days:] if len(df) > n_days else df
        except Exception as exc:
            logger.error(
                f"Failed to load signals for {strategy_name}: {exc}"
            )
            return pd.DataFrame()
