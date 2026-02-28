"""Tests for the backtesting layer — strategy, engine, and CLI wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestResult, compare_strategies, run_backtest
from src.backtest.strategy import (
    MeanReversionStrategy,
    MomentumStrategy,
    Strategy,
    get_strategy,
)
from src.data.synthetic import generate_synthetic_ohlcv
from src.features.pipeline import FeaturePipeline


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    """Synthetic OHLCV with all indicators, ready for backtesting."""
    ohlcv = generate_synthetic_ohlcv("TEST", days=504, seed=42)
    pipeline = FeaturePipeline()
    return pipeline.run(ohlcv)


@pytest.fixture
def backtest_config() -> dict:
    """Minimal config for backtest tests."""
    return {
        "backtest": {
            "initial_capital": 100000,
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
            "strategies": {
                "mean_reversion": {"rsi_lower": 30, "rsi_upper": 70},
                "momentum": {"sma_window": 50},
            },
        },
    }


@pytest.fixture
def zero_cost_config() -> dict:
    """Config with zero transaction costs for gross-return comparisons."""
    return {
        "backtest": {
            "initial_capital": 100000,
            "commission_pct": 0.0,
            "slippage_pct": 0.0,
            "strategies": {
                "mean_reversion": {"rsi_lower": 30, "rsi_upper": 70},
                "momentum": {"sma_window": 50},
            },
        },
    }


# ===================================================================
# Strategy tests
# ===================================================================

class TestStrategyBase:
    """Tests for the Strategy base class and registry."""

    def test_get_strategy_mean_reversion(self, backtest_config: dict) -> None:
        strat = get_strategy("mean_reversion", backtest_config)
        assert isinstance(strat, MeanReversionStrategy)
        assert strat.name == "mean_reversion"

    def test_get_strategy_momentum(self, backtest_config: dict) -> None:
        strat = get_strategy("momentum", backtest_config)
        assert isinstance(strat, MomentumStrategy)
        assert strat.name == "momentum"

    def test_get_strategy_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("magic_strategy")

    def test_strategy_is_abstract(self) -> None:
        """Cannot instantiate Strategy directly."""
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]


class TestMeanReversionSignals:
    """Tests for MeanReversionStrategy signal generation."""

    def test_produces_non_trivial_signals(self, features_df: pd.DataFrame) -> None:
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(features_df)
        unique = set(signals.unique())
        # Must produce at least two distinct signal values
        assert len(unique) >= 2

    def test_signal_values_in_expected_range(self, features_df: pd.DataFrame) -> None:
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(features_df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signal_length_matches_data(self, features_df: pd.DataFrame) -> None:
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(features_df)
        assert len(signals) == len(features_df)

    def test_missing_rsi_column_raises(self) -> None:
        df = pd.DataFrame({"Close": [100, 101]}, index=pd.bdate_range("2024-01-02", periods=2))
        strat = MeanReversionStrategy()
        with pytest.raises(KeyError, match="rsi_14"):
            strat.generate_signals(df)

    def test_from_config_picks_up_thresholds(self) -> None:
        config = {
            "backtest": {
                "strategies": {
                    "mean_reversion": {"rsi_lower": 25, "rsi_upper": 75},
                },
            },
        }
        strat = MeanReversionStrategy.from_config(config)
        assert strat.rsi_lower == 25
        assert strat.rsi_upper == 75


class TestMomentumSignals:
    """Tests for MomentumStrategy signal generation."""

    def test_produces_non_trivial_signals(self, features_df: pd.DataFrame) -> None:
        strat = MomentumStrategy()
        signals = strat.generate_signals(features_df)
        unique = set(signals.unique())
        assert len(unique) >= 2

    def test_signal_values_in_expected_range(self, features_df: pd.DataFrame) -> None:
        strat = MomentumStrategy()
        signals = strat.generate_signals(features_df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_known_price_above_sma(self) -> None:
        """When price is always above SMA, signal should be all 1."""
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        df = pd.DataFrame(
            {"Close": [200, 201, 202, 203, 204], "sma_50": [100, 100, 100, 100, 100]},
            index=dates,
        )
        strat = MomentumStrategy()
        signals = strat.generate_signals(df)
        assert (signals == 1).all()

    def test_known_price_below_sma(self) -> None:
        """When price is always below SMA, signal should be all -1."""
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        df = pd.DataFrame(
            {"Close": [50, 49, 48, 47, 46], "sma_50": [100, 100, 100, 100, 100]},
            index=dates,
        )
        strat = MomentumStrategy()
        signals = strat.generate_signals(df)
        assert (signals == -1).all()


# ===================================================================
# Engine tests
# ===================================================================

class TestRunBacktest:
    """Tests for the core run_backtest function."""

    def test_returns_backtest_result(
        self, features_df: pd.DataFrame, backtest_config: dict
    ) -> None:
        strat = MomentumStrategy()
        result = run_backtest(strat, features_df, backtest_config)
        assert isinstance(result, BacktestResult)

    def test_equity_curve_starts_near_initial_capital(
        self, features_df: pd.DataFrame, backtest_config: dict
    ) -> None:
        strat = MomentumStrategy()
        result = run_backtest(strat, features_df, backtest_config)
        initial = backtest_config["backtest"]["initial_capital"]
        # First value should be very close to initial capital
        # (may differ slightly due to first-day entry fees)
        assert abs(result.equity_curve.iloc[0] - initial) / initial < 0.01

    def test_metrics_dict_has_required_keys(
        self, features_df: pd.DataFrame, backtest_config: dict
    ) -> None:
        strat = MomentumStrategy()
        result = run_backtest(strat, features_df, backtest_config)
        required = {
            "initial_capital", "final_equity", "total_return",
            "annualised_return", "annualised_volatility", "sharpe_ratio",
            "max_drawdown", "calmar_ratio", "total_trades",
        }
        assert required.issubset(result.metrics.keys())

    def test_max_drawdown_is_non_positive(
        self, features_df: pd.DataFrame, backtest_config: dict
    ) -> None:
        strat = MomentumStrategy()
        result = run_backtest(strat, features_df, backtest_config)
        assert result.metrics["max_drawdown"] <= 0.0

    def test_commission_drag(
        self, features_df: pd.DataFrame, backtest_config: dict, zero_cost_config: dict
    ) -> None:
        """Total return with costs should be less than without costs."""
        strat = MomentumStrategy()
        result_with_cost = run_backtest(strat, features_df, backtest_config)
        result_no_cost = run_backtest(strat, features_df, zero_cost_config)
        assert result_with_cost.metrics["total_return"] < result_no_cost.metrics["total_return"]

    def test_equity_curve_length_matches_data(
        self, features_df: pd.DataFrame, backtest_config: dict
    ) -> None:
        strat = MeanReversionStrategy()
        result = run_backtest(strat, features_df, backtest_config)
        assert len(result.equity_curve) == len(features_df)

    def test_returns_length_matches_data(
        self, features_df: pd.DataFrame, backtest_config: dict
    ) -> None:
        strat = MeanReversionStrategy()
        result = run_backtest(strat, features_df, backtest_config)
        assert len(result.returns) == len(features_df)

    def test_signals_stored_in_result(
        self, features_df: pd.DataFrame, backtest_config: dict
    ) -> None:
        strat = MeanReversionStrategy()
        result = run_backtest(strat, features_df, backtest_config)
        assert len(result.signals) == len(features_df)
        assert set(result.signals.unique()).issubset({-1, 0, 1})


# ===================================================================
# Strategy comparison tests
# ===================================================================

class TestCompareStrategies:
    """Tests for the compare_strategies function."""

    def test_returns_dataframe(
        self, features_df: pd.DataFrame, backtest_config: dict
    ) -> None:
        strats = [MomentumStrategy(), MeanReversionStrategy()]
        comparison = compare_strategies(strats, features_df, backtest_config)
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2

    def test_index_is_strategy_names(
        self, features_df: pd.DataFrame, backtest_config: dict
    ) -> None:
        strats = [MomentumStrategy(), MeanReversionStrategy()]
        comparison = compare_strategies(strats, features_df, backtest_config)
        assert "momentum" in comparison.index
        assert "mean_reversion" in comparison.index

    def test_consistent_metric_columns(
        self, features_df: pd.DataFrame, backtest_config: dict
    ) -> None:
        strats = [MomentumStrategy(), MeanReversionStrategy()]
        comparison = compare_strategies(strats, features_df, backtest_config)
        # Both rows should have exactly the same columns
        assert "total_return" in comparison.columns
        assert "sharpe_ratio" in comparison.columns
        assert comparison.iloc[0].index.tolist() == comparison.iloc[1].index.tolist()

    def test_same_initial_capital(
        self, features_df: pd.DataFrame, backtest_config: dict
    ) -> None:
        strats = [MomentumStrategy(), MeanReversionStrategy()]
        comparison = compare_strategies(strats, features_df, backtest_config)
        assert (comparison["initial_capital"] == 100000).all()
