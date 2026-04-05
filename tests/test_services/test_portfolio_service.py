"""Tests for PortfolioService — portfolio state and analytics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.services.data_service import DataService
from src.services.portfolio_service import PortfolioService


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def portfolio_service(
    data_service: DataService, service_config: dict, tmp_data_dir: Path
) -> PortfolioService:
    """PortfolioService wired to temporary data."""
    svc = PortfolioService(data_service, config=service_config)
    svc._data_dir = tmp_data_dir / "data"
    svc._processed_dir = tmp_data_dir / "data" / "processed"
    return svc


# ─────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────

class TestPortfolioServiceInit:
    """PortfolioService instantiation."""

    def test_instantiates_with_data_service(
        self, data_service: DataService
    ) -> None:
        svc = PortfolioService(data_service)
        assert svc.data is data_service

    def test_uses_config_data_dir(
        self, data_service: DataService, service_config: dict
    ) -> None:
        svc = PortfolioService(data_service, config=service_config)
        assert "data" in str(svc._data_dir)

    def test_defaults_without_config(
        self, data_service: DataService
    ) -> None:
        svc = PortfolioService(data_service, config=None)
        assert svc._config == {}


# ─────────────────────────────────────────────────────────────
# get_current_weights
# ─────────────────────────────────────────────────────────────

class TestGetCurrentWeights:
    """Loading current portfolio weights."""

    def test_equal_weights_when_no_file(
        self, portfolio_service: PortfolioService
    ) -> None:
        """Without a weights file, should fall back to equal weights."""
        weights = portfolio_service.get_current_weights()
        assert isinstance(weights, pd.Series)
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 1e-6
        assert weights.name == "weight"

    def test_loads_from_parquet(
        self, portfolio_service: PortfolioService, tmp_data_dir: Path
    ) -> None:
        """Should load weights from parquet if file exists."""
        processed = tmp_data_dir / "data" / "processed"
        weights_df = pd.DataFrame(
            {"weight": [0.5, 0.3, 0.2]},
            index=["TEST_A", "TEST_B", "TEST_C"],
        )
        weights_df.to_parquet(processed / "weights.parquet")

        weights = portfolio_service.get_current_weights()
        assert abs(weights["TEST_A"] - 0.5) < 1e-6
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_empty_when_no_tickers(
        self, service_config: dict
    ) -> None:
        """Should return empty Series if DataService has no tickers."""
        svc = DataService(config={})
        svc._tickers = []
        ps = PortfolioService(svc)
        weights = ps.get_current_weights()
        assert isinstance(weights, pd.Series)
        assert weights.empty


# ─────────────────────────────────────────────────────────────
# get_risk_metrics
# ─────────────────────────────────────────────────────────────

class TestGetRiskMetrics:
    """Risk metrics computation."""

    def test_returns_dict_with_expected_keys(
        self, portfolio_service: PortfolioService
    ) -> None:
        metrics = portfolio_service.get_risk_metrics()
        expected = {
            "annual_return", "annual_volatility", "sharpe_ratio",
            "max_drawdown", "var_95", "cvar_95",
        }
        assert set(metrics.keys()) == expected

    def test_values_are_numeric(
        self, portfolio_service: PortfolioService
    ) -> None:
        metrics = portfolio_service.get_risk_metrics()
        for key, val in metrics.items():
            assert isinstance(val, (int, float))

    def test_max_drawdown_is_negative_or_zero(
        self, portfolio_service: PortfolioService
    ) -> None:
        metrics = portfolio_service.get_risk_metrics()
        assert metrics["max_drawdown"] <= 0

    def test_defaults_when_no_data(self) -> None:
        """Should return NaN defaults when no data available."""
        svc = DataService(config={})
        svc._tickers = []
        from pathlib import Path
        svc._processed_dir = Path("/tmp/nonexistent_portfolio_test")
        svc._raw_dir = Path("/tmp/nonexistent_portfolio_test")
        ps = PortfolioService(svc)
        metrics = ps.get_risk_metrics()
        assert np.isnan(metrics["annual_return"])


# ─────────────────────────────────────────────────────────────
# get_target_weights
# ─────────────────────────────────────────────────────────────

class TestGetTargetWeights:
    """Loading target weights."""

    def test_falls_back_to_current_weights(
        self, portfolio_service: PortfolioService
    ) -> None:
        """Without a target file, should return current weights."""
        target = portfolio_service.get_target_weights()
        current = portfolio_service.get_current_weights()
        assert len(target) == len(current)
        assert target.name == "target_weight"

    def test_loads_from_parquet(
        self, portfolio_service: PortfolioService, tmp_data_dir: Path
    ) -> None:
        """Should load target weights from parquet if file exists."""
        processed = tmp_data_dir / "data" / "processed"
        target_df = pd.DataFrame(
            {"target": [0.6, 0.3, 0.1]},
            index=["TEST_A", "TEST_B", "TEST_C"],
        )
        target_df.to_parquet(processed / "target_weights.parquet")

        target = portfolio_service.get_target_weights()
        assert abs(target["TEST_A"] - 0.6) < 1e-6


# ─────────────────────────────────────────────────────────────
# get_rebalance_orders
# ─────────────────────────────────────────────────────────────

class TestGetRebalanceOrders:
    """Rebalance order computation."""

    def test_returns_dataframe_with_expected_columns(
        self, portfolio_service: PortfolioService
    ) -> None:
        orders = portfolio_service.get_rebalance_orders()
        expected_cols = {
            "ticker", "current_weight", "target_weight",
            "delta", "estimated_value_gbp",
        }
        assert set(orders.columns) == expected_cols

    def test_no_rebalance_when_equal(
        self, portfolio_service: PortfolioService
    ) -> None:
        """When current == target, deltas should be zero."""
        orders = portfolio_service.get_rebalance_orders()
        # Without separate files, current == target (fallback)
        assert (orders["delta"].abs() < 1e-6).all()

    def test_custom_portfolio_value(
        self, portfolio_service: PortfolioService, tmp_data_dir: Path
    ) -> None:
        """estimated_value_gbp should scale with portfolio_value."""
        processed = tmp_data_dir / "data" / "processed"
        target_df = pd.DataFrame(
            {"target": [0.8, 0.1, 0.1]},
            index=["TEST_A", "TEST_B", "TEST_C"],
        )
        target_df.to_parquet(processed / "target_weights.parquet")

        orders = portfolio_service.get_rebalance_orders(portfolio_value=200_000.0)
        # At least one non-zero delta expected
        assert (orders["estimated_value_gbp"] > 0).any()


# ─────────────────────────────────────────────────────────────
# get_equity_curve
# ─────────────────────────────────────────────────────────────

class TestGetEquityCurve:
    """Equity curve computation."""

    def test_returns_series(
        self, portfolio_service: PortfolioService
    ) -> None:
        equity = portfolio_service.get_equity_curve()
        assert isinstance(equity, pd.Series)
        assert len(equity) > 0

    def test_equity_starts_near_one(
        self, portfolio_service: PortfolioService
    ) -> None:
        """Cumulative product should start near 1.0."""
        equity = portfolio_service.get_equity_curve()
        assert abs(equity.iloc[0] - 1.0) < 0.1

    def test_empty_when_no_data(self) -> None:
        svc = DataService(config={})
        svc._tickers = []
        from pathlib import Path
        svc._processed_dir = Path("/tmp/nonexistent_equity_test")
        svc._raw_dir = Path("/tmp/nonexistent_equity_test")
        ps = PortfolioService(svc)
        equity = ps.get_equity_curve()
        assert equity.empty


# ─────────────────────────────────────────────────────────────
# get_allocation_chart_data
# ─────────────────────────────────────────────────────────────

class TestGetAllocationChartData:
    """Allocation chart data formatting."""

    def test_returns_labels_and_values(
        self, portfolio_service: PortfolioService
    ) -> None:
        data = portfolio_service.get_allocation_chart_data()
        assert "labels" in data
        assert "values" in data
        assert len(data["labels"]) == len(data["values"])
        assert len(data["labels"]) == 3

    def test_values_sum_to_one(
        self, portfolio_service: PortfolioService
    ) -> None:
        data = portfolio_service.get_allocation_chart_data()
        assert abs(sum(data["values"]) - 1.0) < 1e-6

    def test_empty_when_no_weights(self) -> None:
        svc = DataService(config={})
        svc._tickers = []
        ps = PortfolioService(svc)
        data = ps.get_allocation_chart_data()
        assert data == {"labels": [], "values": []}
