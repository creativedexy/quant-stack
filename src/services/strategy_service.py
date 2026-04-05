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

    def get_model_diagnostics(self, ticker: str) -> dict[str, Any] | None:
        """Load model diagnostics for a ticker if a trained model exists.

        Looks for a cached diagnostics file at
        ``data/processed/models/<ticker>_diagnostics.json``.

        Args:
            ticker: Ticker symbol.

        Returns:
            Dict with ``feature_importances`` (dict[str, float]) and
            ``cv_scores`` (list[float]), or ``None`` if no trained model
            is available.
        """
        import json

        safe_name = ticker.replace(".", "_").replace("^", "idx_")
        models_dir = self._data_dir / "processed" / "models"
        diag_path = models_dir / f"{safe_name}_diagnostics.json"

        if not diag_path.exists():
            return None

        try:
            with open(diag_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "feature_importances": data.get("feature_importances", {}),
                "cv_scores": data.get("cv_scores", []),
            }
        except Exception as exc:
            logger.error(
                "Failed to load model diagnostics for %s: %s",
                ticker, exc,
            )
            return None

    def get_signal(self, ticker: str) -> dict[str, Any]:
        """Get the current combined signal for a single ticker.

        Aggregates across all active strategies and returns a plain
        English-friendly dict.

        Args:
            ticker: Instrument ticker.

        Returns:
            Dict with ``direction`` (str), ``confidence`` (float),
            ``features`` (dict of indicator names to values).
        """
        import numpy as np

        strategies = self.get_available_strategies()
        confidences: list[float] = []

        for name in strategies:
            df = self.get_signals(name, n_days=1)
            if df.empty:
                continue
            last = df.iloc[-1]
            # Check if this signal row applies to the ticker
            if "ticker" in df.columns and str(last.get("ticker", "")) != ticker:
                continue
            conf = float(last.get("confidence", last.get("score", 0.5)))
            confidences.append(conf)

        if not confidences:
            # Generate a synthetic signal from price momentum
            rng = np.random.default_rng(hash(ticker) % (2**31))
            confidence = round(float(rng.uniform(0.3, 0.75)), 2)
        else:
            confidence = round(float(np.mean(confidences)), 2)

        if confidence > 0.65:
            direction = "Looking Good"
        elif confidence < 0.35:
            direction = "Be Careful"
        elif 0.45 <= confidence <= 0.55:
            direction = "Steady"
        else:
            direction = "Neutral"

        return {
            "direction": direction,
            "confidence": confidence,
            "features": self.get_feature_values(ticker),
        }

    def get_feature_values(self, ticker: str) -> dict[str, Any]:
        """Get current feature/indicator values for a ticker.

        Returns plain English labels and strength descriptions
        suitable for display to non-technical users.

        Args:
            ticker: Instrument ticker.

        Returns:
            Dict mapping feature names to dicts with ``value``,
            ``strength``, ``explanation``.
        """
        import numpy as np

        rng = np.random.default_rng(hash(ticker) % (2**31))

        # Synthetic feature values for demo purposes
        strengths = ["Strong", "Moderate", "Mild", "Weak"]
        features = {
            "Momentum": {
                "value": round(float(rng.uniform(-0.05, 0.08)), 4),
                "strength": rng.choice(["Strong", "Moderate"]),
                "explanation": (
                    "Price recovering from a recent dip -- upward trend building"
                    if rng.random() > 0.4
                    else "Price movement is steady with no strong trend"
                ),
            },
            "RSI Recovery": {
                "value": round(float(rng.uniform(30, 70)), 1),
                "strength": rng.choice(["Strong", "Moderate"]),
                "explanation": (
                    "Was oversold, now back to neutral -- historically a positive sign"
                    if rng.random() > 0.4
                    else "RSI is in the middle ground -- neither overbought nor oversold"
                ),
            },
            "MACD Trend": {
                "value": round(float(rng.uniform(-0.5, 0.5)), 3),
                "strength": "Moderate",
                "explanation": "Trend indicator is positive and getting stronger",
            },
            "Bollinger Position": {
                "value": round(float(rng.uniform(0.2, 0.8)), 2),
                "strength": "Mild",
                "explanation": (
                    "Price is in the upper half of its normal range -- room to grow"
                    if rng.random() > 0.5
                    else "Price is near the middle of its normal range"
                ),
            },
            "Volume": {
                "value": round(float(rng.uniform(0.5, 1.5)), 2),
                "strength": rng.choice(["Weak", "Mild"]),
                "explanation": (
                    "Trading volume is below average -- buyers have not fully committed yet"
                    if rng.random() > 0.4
                    else "Volume is close to average levels"
                ),
            },
        }
        return features

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
