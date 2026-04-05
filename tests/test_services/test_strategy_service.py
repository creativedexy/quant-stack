"""Tests for StrategyService — strategy execution and comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.services.data_service import DataService
from src.services.strategy_service import StrategyService


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def strategy_service(
    data_service: DataService, service_config: dict, tmp_data_dir: Path
) -> StrategyService:
    """StrategyService wired to temporary data."""
    svc = StrategyService(data_service, config=service_config)
    svc._data_dir = tmp_data_dir / "data"
    svc._backtests_dir = tmp_data_dir / "data" / "processed" / "backtests"
    svc._signals_dir = tmp_data_dir / "data" / "processed" / "signals"
    return svc


# ─────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────

class TestStrategyServiceInit:
    """StrategyService instantiation."""

    def test_instantiates_with_data_service(
        self, data_service: DataService
    ) -> None:
        svc = StrategyService(data_service)
        assert svc.data is data_service

    def test_defaults_without_config(
        self, data_service: DataService
    ) -> None:
        svc = StrategyService(data_service, config=None)
        assert svc._config == {}


# ─────────────────────────────────────────────────────────────
# get_available_strategies
# ─────────────────────────────────────────────────────────────

class TestGetAvailableStrategies:
    """Strategy listing."""

    def test_returns_default_strategies(
        self, strategy_service: StrategyService
    ) -> None:
        """Should always include the three default strategies."""
        strategies = strategy_service.get_available_strategies()
        assert "mean_reversion" in strategies
        assert "momentum" in strategies
        assert "trend_following" in strategies

    def test_returns_sorted_list(
        self, strategy_service: StrategyService
    ) -> None:
        strategies = strategy_service.get_available_strategies()
        assert strategies == sorted(strategies)

    def test_discovers_cached_backtests(
        self, strategy_service: StrategyService, tmp_data_dir: Path
    ) -> None:
        """Should discover strategies from cached backtest files."""
        bt_dir = tmp_data_dir / "data" / "processed" / "backtests"
        bt_dir.mkdir(parents=True, exist_ok=True)
        # Create a fake results file
        df = pd.DataFrame({"value": [1.0, 1.01, 1.02]})
        df.to_parquet(bt_dir / "custom_strategy_results.parquet")

        strategies = strategy_service.get_available_strategies()
        assert "custom_strategy" in strategies


# ─────────────────────────────────────────────────────────────
# get_backtest_results
# ─────────────────────────────────────────────────────────────

class TestGetBacktestResults:
    """Loading cached backtest results."""

    def test_returns_none_when_no_cache(
        self, strategy_service: StrategyService
    ) -> None:
        result = strategy_service.get_backtest_results("nonexistent")
        assert result is None

    def test_loads_cached_results(
        self, strategy_service: StrategyService, tmp_data_dir: Path
    ) -> None:
        bt_dir = tmp_data_dir / "data" / "processed" / "backtests"
        bt_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"return": [0.01, -0.02, 0.015]})
        df.to_parquet(bt_dir / "momentum_results.parquet")

        result = strategy_service.get_backtest_results("momentum")
        assert result is not None
        assert result["strategy"] == "momentum"
        assert result["rows"] == 3
        assert isinstance(result["data"], pd.DataFrame)


# ─────────────────────────────────────────────────────────────
# get_strategy_comparison
# ─────────────────────────────────────────────────────────────

class TestGetStrategyComparison:
    """Strategy comparison loading."""

    def test_returns_none_when_no_file(
        self, strategy_service: StrategyService
    ) -> None:
        assert strategy_service.get_strategy_comparison() is None

    def test_loads_comparison_dataframe(
        self, strategy_service: StrategyService, tmp_data_dir: Path
    ) -> None:
        bt_dir = tmp_data_dir / "data" / "processed" / "backtests"
        bt_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "strategy": ["momentum", "mean_reversion"],
            "sharpe": [0.8, 0.5],
        })
        df.to_parquet(bt_dir / "comparison.parquet")

        result = strategy_service.get_strategy_comparison()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────
# get_model_diagnostics
# ─────────────────────────────────────────────────────────────

class TestGetModelDiagnostics:
    """Model diagnostics loading."""

    def test_returns_none_when_no_file(
        self, strategy_service: StrategyService
    ) -> None:
        assert strategy_service.get_model_diagnostics("NONEXIST") is None

    def test_loads_diagnostics_json(
        self, strategy_service: StrategyService, tmp_data_dir: Path
    ) -> None:
        models_dir = tmp_data_dir / "data" / "processed" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        diag = {
            "feature_importances": {"sma_5": 0.3, "rsi_14": 0.7},
            "cv_scores": [0.52, 0.48, 0.55],
        }
        with open(models_dir / "TEST_A_diagnostics.json", "w") as f:
            json.dump(diag, f)

        result = strategy_service.get_model_diagnostics("TEST_A")
        assert result is not None
        assert "feature_importances" in result
        assert "cv_scores" in result
        assert result["feature_importances"]["sma_5"] == 0.3

    def test_handles_dotted_ticker(
        self, strategy_service: StrategyService, tmp_data_dir: Path
    ) -> None:
        """Tickers like BP.L should be sanitised to BP_L."""
        models_dir = tmp_data_dir / "data" / "processed" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        diag = {"feature_importances": {}, "cv_scores": []}
        with open(models_dir / "BP_L_diagnostics.json", "w") as f:
            json.dump(diag, f)

        result = strategy_service.get_model_diagnostics("BP.L")
        assert result is not None


# ─────────────────────────────────────────────────────────────
# get_signals
# ─────────────────────────────────────────────────────────────

class TestGetSignals:
    """Signal loading."""

    def test_returns_empty_when_no_file(
        self, strategy_service: StrategyService
    ) -> None:
        signals = strategy_service.get_signals("nonexistent")
        assert isinstance(signals, pd.DataFrame)
        assert signals.empty

    def test_loads_and_limits_rows(
        self, strategy_service: StrategyService, tmp_data_dir: Path
    ) -> None:
        sig_dir = tmp_data_dir / "data" / "processed" / "signals"
        sig_dir.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        df = pd.DataFrame(
            {"signal": range(60)}, index=dates
        )
        df.to_parquet(sig_dir / "momentum_signals.parquet")

        signals = strategy_service.get_signals("momentum", n_days=10)
        assert len(signals) == 10
        # Should return the most recent 10
        assert signals.index[-1] == dates[-1]

    def test_returns_all_when_fewer_than_n_days(
        self, strategy_service: StrategyService, tmp_data_dir: Path
    ) -> None:
        sig_dir = tmp_data_dir / "data" / "processed" / "signals"
        sig_dir.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({"signal": range(5)}, index=dates)
        df.to_parquet(sig_dir / "test_strat_signals.parquet")

        signals = strategy_service.get_signals("test_strat", n_days=30)
        assert len(signals) == 5
