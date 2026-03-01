"""Integration-style tests: full pipeline from synthetic data to backtest result.

Verifies the complete chain works end-to-end:
    synthetic data → clean → features → strategy → backtest → result
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.strategy import (
    MACDCrossoverStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    strategy_registry,
)
from src.data.cleaner import DataCleaner
from src.data.synthetic import generate_synthetic_ohlcv
from src.features.pipeline import FeaturePipeline


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_pipeline_result() -> BacktestResult:
    """Run the full pipeline once and cache the result for the module."""
    # 1. Generate synthetic data
    raw = generate_synthetic_ohlcv("PIPE_TEST", days=756, seed=123)

    # 2. Clean
    cleaner = DataCleaner()
    clean = cleaner.clean(raw, ticker="PIPE_TEST")

    # 3. Features (compat mode — OHLCV + indicators in one DataFrame)
    pipeline = FeaturePipeline()
    features = pipeline.run(clean)

    # 4. Strategy
    strat = MeanReversionStrategy()

    # 5. Backtest
    config = {
        "backtest": {
            "initial_capital": 50000,
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
        },
        "universe": {"benchmark": ""},
    }
    engine = BacktestEngine(config)
    return engine.run(strat, features, features)


@pytest.fixture(scope="module")
def momentum_pipeline_result() -> BacktestResult:
    """Full pipeline using Momentum strategy."""
    raw = generate_synthetic_ohlcv("MOM_TEST", days=756, seed=456)
    cleaner = DataCleaner()
    clean = cleaner.clean(raw, ticker="MOM_TEST")
    pipeline = FeaturePipeline()
    features = pipeline.run(clean)
    strat = MomentumStrategy()
    config = {
        "backtest": {
            "initial_capital": 100000,
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
        },
        "universe": {"benchmark": ""},
    }
    engine = BacktestEngine(config)
    return engine.run(strat, features, features)


@pytest.fixture(scope="module")
def macd_pipeline_result() -> BacktestResult:
    """Full pipeline using MACD Crossover strategy."""
    raw = generate_synthetic_ohlcv("MACD_TEST", days=756, seed=789)
    cleaner = DataCleaner()
    clean = cleaner.clean(raw, ticker="MACD_TEST")
    pipeline = FeaturePipeline()
    features = pipeline.run(clean)
    strat = MACDCrossoverStrategy()
    config = {
        "backtest": {
            "initial_capital": 100000,
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
        },
        "universe": {"benchmark": ""},
    }
    engine = BacktestEngine(config)
    return engine.run(strat, features, features)


# ===================================================================
# End-to-end chain
# ===================================================================

class TestFullPipeline:
    """Verify the complete chain works end-to-end."""

    def test_result_type(self, full_pipeline_result: BacktestResult) -> None:
        assert isinstance(full_pipeline_result, BacktestResult)

    def test_equity_curve_not_empty(self, full_pipeline_result: BacktestResult) -> None:
        assert len(full_pipeline_result.equity_curve) > 0

    def test_returns_not_empty(self, full_pipeline_result: BacktestResult) -> None:
        assert len(full_pipeline_result.returns) > 0

    def test_positions_not_empty(self, full_pipeline_result: BacktestResult) -> None:
        assert len(full_pipeline_result.positions) > 0

    def test_momentum_pipeline(self, momentum_pipeline_result: BacktestResult) -> None:
        assert isinstance(momentum_pipeline_result, BacktestResult)
        assert len(momentum_pipeline_result.equity_curve) > 0

    def test_macd_pipeline(self, macd_pipeline_result: BacktestResult) -> None:
        assert isinstance(macd_pipeline_result, BacktestResult)
        assert len(macd_pipeline_result.equity_curve) > 0


# ===================================================================
# Metrics from risk_summary
# ===================================================================

class TestMetricsKeys:
    """Verify metrics dict contains all expected keys from risk_summary."""

    def test_sharpe_present(self, full_pipeline_result: BacktestResult) -> None:
        assert "sharpe" in full_pipeline_result.metrics

    def test_sortino_present(self, full_pipeline_result: BacktestResult) -> None:
        assert "sortino" in full_pipeline_result.metrics

    def test_max_drawdown_present(self, full_pipeline_result: BacktestResult) -> None:
        assert "max_drawdown" in full_pipeline_result.metrics

    def test_var_95_present(self, full_pipeline_result: BacktestResult) -> None:
        assert "var_95" in full_pipeline_result.metrics

    def test_cvar_95_present(self, full_pipeline_result: BacktestResult) -> None:
        assert "cvar_95" in full_pipeline_result.metrics

    def test_annualised_return_present(self, full_pipeline_result: BacktestResult) -> None:
        assert "annualised_return" in full_pipeline_result.metrics

    def test_annualised_volatility_present(self, full_pipeline_result: BacktestResult) -> None:
        assert "annualised_volatility" in full_pipeline_result.metrics

    def test_calmar_ratio_present(self, full_pipeline_result: BacktestResult) -> None:
        assert "calmar_ratio" in full_pipeline_result.metrics

    def test_total_trades_present(self, full_pipeline_result: BacktestResult) -> None:
        assert "total_trades" in full_pipeline_result.metrics

    def test_win_rate_present(self, full_pipeline_result: BacktestResult) -> None:
        assert "win_rate" in full_pipeline_result.metrics

    def test_metrics_values_are_finite(self, full_pipeline_result: BacktestResult) -> None:
        for key in ("sharpe", "sortino", "max_drawdown", "var_95", "cvar_95",
                     "annualised_return", "annualised_volatility"):
            val = full_pipeline_result.metrics[key]
            assert np.isfinite(val), f"Metric {key} is not finite: {val}"


# ===================================================================
# Trades DataFrame validation
# ===================================================================

class TestTradesDataFrame:
    """Verify trades DataFrame has sensible values."""

    def test_trades_has_expected_columns(self, full_pipeline_result: BacktestResult) -> None:
        if len(full_pipeline_result.trades) == 0:
            pytest.skip("No trades generated — skip column check")
        expected = {"date", "ticker", "side", "quantity", "price", "cost"}
        assert expected.issubset(set(full_pipeline_result.trades.columns))

    def test_no_negative_quantities(self, full_pipeline_result: BacktestResult) -> None:
        if len(full_pipeline_result.trades) == 0:
            pytest.skip("No trades generated")
        assert (full_pipeline_result.trades["quantity"] >= 0).all()

    def test_prices_positive(self, full_pipeline_result: BacktestResult) -> None:
        if len(full_pipeline_result.trades) == 0:
            pytest.skip("No trades generated")
        assert (full_pipeline_result.trades["price"] > 0).all()

    def test_costs_non_negative(self, full_pipeline_result: BacktestResult) -> None:
        if len(full_pipeline_result.trades) == 0:
            pytest.skip("No trades generated")
        assert (full_pipeline_result.trades["cost"] >= 0).all()

    def test_sides_are_buy_or_sell(self, full_pipeline_result: BacktestResult) -> None:
        if len(full_pipeline_result.trades) == 0:
            pytest.skip("No trades generated")
        assert set(full_pipeline_result.trades["side"].unique()).issubset({"buy", "sell"})

    def test_momentum_has_trades(self, momentum_pipeline_result: BacktestResult) -> None:
        """Momentum on synthetic data should generate at least some trades."""
        assert len(momentum_pipeline_result.trades) > 0

    def test_macd_has_trades(self, macd_pipeline_result: BacktestResult) -> None:
        """MACD crossover on synthetic data should generate at least some trades."""
        assert len(macd_pipeline_result.trades) > 0


# ===================================================================
# Multi-strategy comparison via registry
# ===================================================================

class TestRegistryIntegration:
    """Test creating strategies from the registry and running them."""

    def test_all_registered_strategies_run(self) -> None:
        """Every strategy in the registry can be created and run."""
        raw = generate_synthetic_ohlcv("REG_TEST", days=504, seed=42)
        pipeline = FeaturePipeline()
        features = pipeline.run(raw)

        config = {
            "backtest": {
                "initial_capital": 100000,
                "commission_pct": 0.001,
                "slippage_pct": 0.0005,
            },
            "universe": {"benchmark": ""},
        }
        engine = BacktestEngine(config)

        for name in strategy_registry.list_strategies():
            strat = strategy_registry.create(name)
            result = engine.run(strat, features, features)
            assert isinstance(result, BacktestResult)
            assert len(result.equity_curve) > 0
