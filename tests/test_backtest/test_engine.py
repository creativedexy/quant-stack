"""Tests for the BacktestEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.strategy import (
    MACDCrossoverStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    Strategy,
)
from src.data.synthetic import generate_synthetic_ohlcv
from src.features.pipeline import FeaturePipeline


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    """Synthetic OHLCV data."""
    return generate_synthetic_ohlcv("TEST", days=504, seed=42)


@pytest.fixture(scope="module")
def features_df(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Synthetic OHLCV with all indicators (via compat run())."""
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
        },
        "universe": {"benchmark": "^TEST"},
    }


@pytest.fixture
def zero_cost_config() -> dict:
    """Config with zero transaction costs."""
    return {
        "backtest": {
            "initial_capital": 100000,
            "commission_pct": 0.0,
            "slippage_pct": 0.0,
        },
        "universe": {"benchmark": "^TEST"},
    }


@pytest.fixture
def uptrend_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic data with a strong uptrend for buy-and-hold testing."""
    ohlcv = generate_synthetic_ohlcv(
        "UP",
        days=504,
        seed=99,
        annual_return=0.30,
        annual_volatility=0.10,
    )
    pipeline = FeaturePipeline()
    features = pipeline.run(ohlcv)
    return ohlcv.loc[features.index], features


# ===================================================================
# BacktestResult structure
# ===================================================================

class TestBacktestResultFields:
    """BacktestResult contains all expected fields."""

    def test_result_has_strategy_name(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = engine.run(MomentumStrategy(), features_df, features_df)
        assert isinstance(result.strategy_name, str)
        assert result.strategy_name == "momentum"

    def test_result_has_equity_curve(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = engine.run(MomentumStrategy(), features_df, features_df)
        assert isinstance(result.equity_curve, pd.Series)

    def test_result_has_returns(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = engine.run(MomentumStrategy(), features_df, features_df)
        assert isinstance(result.returns, pd.Series)

    def test_result_has_positions(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = engine.run(MomentumStrategy(), features_df, features_df)
        assert isinstance(result.positions, pd.DataFrame)

    def test_result_has_trades(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = engine.run(MomentumStrategy(), features_df, features_df)
        assert isinstance(result.trades, pd.DataFrame)

    def test_result_has_metrics_dict(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = engine.run(MomentumStrategy(), features_df, features_df)
        assert isinstance(result.metrics, dict)

    def test_result_has_config_dict(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = engine.run(MomentumStrategy(), features_df, features_df)
        assert isinstance(result.config, dict)
        assert "initial_capital" in result.config


# ===================================================================
# Equity curve properties
# ===================================================================

class TestEquityCurve:
    """Equity curve starts at initial_capital and has correct length."""

    def test_starts_at_initial_capital(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = engine.run(MeanReversionStrategy(), features_df, features_df)
        initial = backtest_config["backtest"]["initial_capital"]
        # First value should be very close to initial capital
        assert abs(result.equity_curve.iloc[0] - initial) / initial < 0.02

    def test_length_matches_input(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = engine.run(MeanReversionStrategy(), features_df, features_df)
        assert len(result.equity_curve) == len(features_df)

    def test_returns_length_matches_input(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = engine.run(MeanReversionStrategy(), features_df, features_df)
        assert len(result.returns) == len(features_df)


# ===================================================================
# Commission drag
# ===================================================================

class TestCommissionDrag:
    """Commission reduces total return."""

    def test_zero_commission_equals_gross(
        self, features_df: pd.DataFrame, zero_cost_config: dict,
    ) -> None:
        engine = BacktestEngine(zero_cost_config)
        result = engine.run(MomentumStrategy(), features_df, features_df)
        # With zero costs, the return is purely from the strategy
        assert isinstance(result.metrics["annualised_return"], float)

    def test_commission_drag(
        self,
        features_df: pd.DataFrame,
        backtest_config: dict,
        zero_cost_config: dict,
    ) -> None:
        """Total return WITH commission is LESS than gross return."""
        engine_cost = BacktestEngine(backtest_config)
        engine_free = BacktestEngine(zero_cost_config)
        strat = MomentumStrategy()

        result_cost = engine_cost.run(strat, features_df, features_df)
        result_free = engine_free.run(strat, features_df, features_df)

        final_cost = result_cost.equity_curve.iloc[-1]
        final_free = result_free.equity_curve.iloc[-1]
        assert final_cost < final_free


# ===================================================================
# Buy-and-hold on uptrend
# ===================================================================

class TestBuyAndHold:
    """A buy-and-hold strategy on an uptrending asset produces positive return."""

    def test_uptrend_positive_return(
        self,
        uptrend_data: tuple[pd.DataFrame, pd.DataFrame],
        zero_cost_config: dict,
    ) -> None:
        prices, features = uptrend_data
        engine = BacktestEngine(zero_cost_config)

        # Create a strategy that always goes long
        class AlwaysLong(Strategy):
            def generate_signals(self, feat: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(
                    np.ones((len(feat), 1), dtype=int),
                    index=feat.index,
                    columns=["signal"],
                )

        strat = AlwaysLong(name="always_long")
        result = engine.run(strat, prices, features)
        assert result.equity_curve.iloc[-1] > zero_cost_config["backtest"]["initial_capital"]


# ===================================================================
# compare()
# ===================================================================

class TestCompare:
    """compare() returns DataFrame with correct number of rows."""

    def test_compare_returns_dataframe(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        strats = [MomentumStrategy(), MeanReversionStrategy()]
        comparison = engine.compare(strats, features_df, features_df)
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2

    def test_compare_has_expected_columns(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        strats = [MomentumStrategy(), MeanReversionStrategy()]
        comparison = engine.compare(strats, features_df, features_df)
        expected_cols = {
            "strategy_name", "annualised_return", "sharpe",
            "max_drawdown", "total_trades", "win_rate",
        }
        assert expected_cols.issubset(set(comparison.columns))

    def test_compare_sorted_by_sharpe(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        engine = BacktestEngine(backtest_config)
        strats = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            MACDCrossoverStrategy(),
        ]
        comparison = engine.compare(strats, features_df, features_df)
        sharpe_vals = comparison["sharpe"].values
        # Should be sorted descending
        for i in range(len(sharpe_vals) - 1):
            assert sharpe_vals[i] >= sharpe_vals[i + 1]


# ===================================================================
# plot_results
# ===================================================================

class TestPlotResults:
    """plot_results runs without error and produces a figure."""

    def test_plot_produces_figure(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        import matplotlib.pyplot as plt

        engine = BacktestEngine(backtest_config)
        result = engine.run(MomentumStrategy(), features_df, features_df)
        fig = engine.plot_results(result)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_plot_with_benchmark(
        self, features_df: pd.DataFrame, backtest_config: dict,
    ) -> None:
        import matplotlib.pyplot as plt

        engine = BacktestEngine(backtest_config)
        result = engine.run(MomentumStrategy(), features_df, features_df)
        benchmark = features_df["Close"]
        fig = engine.plot_results(result, benchmark_prices=benchmark)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_save(
        self, features_df: pd.DataFrame, backtest_config: dict, tmp_path,
    ) -> None:
        import matplotlib.pyplot as plt

        engine = BacktestEngine(backtest_config)
        result = engine.run(MomentumStrategy(), features_df, features_df)
        save_path = tmp_path / "test_plot.png"
        fig = engine.plot_results(result, save_path=save_path)
        assert save_path.exists()
        plt.close(fig)
