"""Tests for the execution service layer.

Covers:
- ExecutionService instantiation
- Paper broker connection
- Rebalance plan generation
- Plan execution and position mutation
- Execution history persistence and retrieval
- Reconciliation drift detection
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.execution_service import ExecutionService


@pytest.fixture
def exec_dir(tmp_path: Path) -> Path:
    """Temporary directory for execution reports."""
    d = tmp_path / "executions"
    d.mkdir()
    return d


@pytest.fixture
def service(exec_dir: Path) -> ExecutionService:
    """ExecutionService wired to a temporary execution directory."""
    return ExecutionService(
        config={
            "initial_cash": 100_000.0,
            "base_currency": "GBP",
            "commission_rate": 0.001,
            "slippage_bps": 5.0,
            "execution_dir": str(exec_dir),
        }
    )


@pytest.fixture
def connected_service(service: ExecutionService) -> ExecutionService:
    """ExecutionService with a connected paper broker and sample prices."""
    service.connect_paper_broker()
    service.set_prices({
        "AAPL": 150.0,
        "GOOG": 120.0,
        "MSFT": 200.0,
        "AMZN": 100.0,
    })
    return service


# ------------------------------------------------------------------
# 1. Instantiation
# ------------------------------------------------------------------

class TestInstantiation:
    """ExecutionService instantiates correctly with various arguments."""

    def test_default_instantiation(self) -> None:
        svc = ExecutionService()
        assert svc.broker is None
        assert svc.oms is None

    def test_with_config(self, exec_dir: Path) -> None:
        svc = ExecutionService(config={"execution_dir": str(exec_dir)})
        assert svc.broker is None
        assert svc.config["execution_dir"] == str(exec_dir)

    def test_with_services(self) -> None:
        svc = ExecutionService(data_service="mock_data", portfolio_service="mock_port")
        assert svc.data_service == "mock_data"
        assert svc.portfolio_service == "mock_port"


# ------------------------------------------------------------------
# 2. Paper broker connection
# ------------------------------------------------------------------

class TestConnection:
    """Paper broker connects successfully."""

    def test_connect_paper_broker_returns_true(self, service: ExecutionService) -> None:
        assert service.connect_paper_broker() is True

    def test_broker_status_after_connect(self, service: ExecutionService) -> None:
        service.connect_paper_broker()
        status = service.get_broker_status()
        assert status["connected"] is True
        assert status["mode"] == "paper"
        assert status["account_value"] == 100_000.0
        assert status["cash"] == 100_000.0

    def test_broker_status_before_connect(self, service: ExecutionService) -> None:
        status = service.get_broker_status()
        assert status["connected"] is False


# ------------------------------------------------------------------
# 3. Rebalance plan generation
# ------------------------------------------------------------------

class TestRebalancePlan:
    """Rebalance plan returns the correct structure."""

    def test_plan_structure(self, connected_service: ExecutionService) -> None:
        connected_service.set_target_weights({"AAPL": 0.4, "GOOG": 0.6})
        plan = connected_service.generate_rebalance_plan()

        assert "plan_id" in plan
        assert "orders" in plan
        assert "total_cost_estimate" in plan
        assert "turnover" in plan
        assert "timestamp" in plan
        assert isinstance(plan["orders"], list)

    def test_plan_orders_have_required_fields(
        self, connected_service: ExecutionService
    ) -> None:
        connected_service.set_target_weights({"AAPL": 0.5, "GOOG": 0.5})
        plan = connected_service.generate_rebalance_plan()

        for order in plan["orders"]:
            assert "ticker" in order
            assert "side" in order
            assert "quantity" in order
            assert "est_price" in order
            assert "est_cost" in order
            assert "reason" in order

    def test_plan_with_explicit_weights(
        self, connected_service: ExecutionService
    ) -> None:
        plan = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 0.5, "MSFT": 0.5}
        )
        tickers = {o["ticker"] for o in plan["orders"]}
        assert tickers == {"AAPL", "MSFT"}

    def test_plan_all_buys_from_cash(
        self, connected_service: ExecutionService
    ) -> None:
        plan = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 0.5, "GOOG": 0.5}
        )
        for order in plan["orders"]:
            assert order["side"] == "BUY"

    def test_plan_raises_without_weights(
        self, connected_service: ExecutionService
    ) -> None:
        with pytest.raises(ValueError, match="No target weights"):
            connected_service.generate_rebalance_plan()

    def test_plan_raises_without_broker(self, service: ExecutionService) -> None:
        with pytest.raises(ValueError, match="not connected"):
            service.generate_rebalance_plan(target_weights={"AAPL": 1.0})


# ------------------------------------------------------------------
# 4. Plan execution
# ------------------------------------------------------------------

class TestPlanExecution:
    """Executing a plan modifies positions on the paper broker."""

    def test_execute_plan_modifies_positions(
        self, connected_service: ExecutionService
    ) -> None:
        plan = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 0.5, "GOOG": 0.5}
        )

        result = connected_service.execute_plan(plan["plan_id"])

        assert result["status"] == "completed"
        assert result["orders_filled"] > 0

        # Positions should now be non-empty
        positions = connected_service.broker.get_positions()
        assert len(positions) > 0
        assert "AAPL" in positions
        assert "GOOG" in positions

    def test_execute_plan_reduces_cash(
        self, connected_service: ExecutionService
    ) -> None:
        initial_cash = connected_service.broker.cash

        plan = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 0.5, "GOOG": 0.5}
        )
        connected_service.execute_plan(plan["plan_id"])

        assert connected_service.broker.cash < initial_cash

    def test_execute_plan_returns_report_dict(
        self, connected_service: ExecutionService
    ) -> None:
        plan = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 1.0}
        )
        result = connected_service.execute_plan(plan["plan_id"])

        assert isinstance(result, dict)
        assert "plan_id" in result
        assert "fills" in result
        assert "total_commission" in result
        assert "status" in result

    def test_execute_unknown_plan_raises(
        self, connected_service: ExecutionService
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            connected_service.execute_plan("nonexistent_plan_id")

    def test_execute_plan_creates_sell_orders(
        self, connected_service: ExecutionService
    ) -> None:
        """Buy first, then rebalance to new weights to trigger sells."""
        plan1 = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 0.5, "GOOG": 0.5}
        )
        connected_service.execute_plan(plan1["plan_id"])

        # Now rebalance to only AAPL
        plan2 = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 1.0}
        )

        has_sell = any(o["side"] == "SELL" for o in plan2["orders"])
        assert has_sell, "Expected SELL orders when reducing GOOG position"


# ------------------------------------------------------------------
# 5. Execution history
# ------------------------------------------------------------------

class TestExecutionHistory:
    """Execution reports are saved and retrievable."""

    def test_history_returns_list_of_dicts(
        self, connected_service: ExecutionService
    ) -> None:
        plan = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 0.5, "GOOG": 0.5}
        )
        connected_service.execute_plan(plan["plan_id"])

        history = connected_service.get_execution_history()
        assert isinstance(history, list)
        assert len(history) >= 1
        assert isinstance(history[0], dict)

    def test_history_contains_report_fields(
        self, connected_service: ExecutionService
    ) -> None:
        plan = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 0.5, "GOOG": 0.5}
        )
        connected_service.execute_plan(plan["plan_id"])

        history = connected_service.get_execution_history()
        report = history[0]
        assert "plan_id" in report
        assert "fills" in report
        assert "status" in report
        assert "mode" in report

    def test_history_empty_when_no_executions(
        self, service: ExecutionService
    ) -> None:
        assert service.get_execution_history() == []

    def test_execution_reports_saved_as_json(
        self, connected_service: ExecutionService, exec_dir: Path
    ) -> None:
        plan = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 1.0}
        )
        connected_service.execute_plan(plan["plan_id"])

        json_files = list(exec_dir.glob("execution_*.json"))
        assert len(json_files) >= 1

        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "plan_id" in data
        assert "fills" in data

    def test_history_respects_limit(
        self, connected_service: ExecutionService
    ) -> None:
        # Execute multiple plans
        for i in range(3):
            weights = {"AAPL": 0.3 + i * 0.1, "GOOG": 0.7 - i * 0.1}
            plan = connected_service.generate_rebalance_plan(target_weights=weights)
            connected_service.execute_plan(plan["plan_id"])

        history = connected_service.get_execution_history(n=2)
        assert len(history) == 2


# ------------------------------------------------------------------
# 6. Reconciliation
# ------------------------------------------------------------------

class TestReconciliation:
    """Reconciliation detects drift between target and actual weights."""

    def test_reconciliation_detects_drift(
        self, connected_service: ExecutionService
    ) -> None:
        # Set target but don't execute — all positions are zero
        connected_service.set_target_weights({"AAPL": 0.5, "GOOG": 0.5})
        recon = connected_service.get_reconciliation()

        assert recon["total_drift"] > 0
        assert recon["aligned"] is False
        assert len(recon["tickers"]) == 2

    def test_reconciliation_aligned_after_execution(
        self, connected_service: ExecutionService
    ) -> None:
        plan = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 0.5, "GOOG": 0.5}
        )
        connected_service.execute_plan(plan["plan_id"])

        recon = connected_service.get_reconciliation()
        # Drift should be small (due to integer rounding, commissions, slippage)
        assert recon["total_drift"] < 0.10

    def test_reconciliation_ticker_fields(
        self, connected_service: ExecutionService
    ) -> None:
        connected_service.set_target_weights({"AAPL": 0.5, "GOOG": 0.5})
        recon = connected_service.get_reconciliation()

        for row in recon["tickers"]:
            assert "ticker" in row
            assert "target_weight" in row
            assert "actual_weight" in row
            assert "drift" in row

    def test_reconciliation_empty_when_disconnected(
        self, service: ExecutionService
    ) -> None:
        recon = service.get_reconciliation()
        assert recon["tickers"] == []
        assert recon["total_drift"] == 0.0
        assert recon["aligned"] is True

    def test_reconciliation_with_cash_position(
        self, connected_service: ExecutionService
    ) -> None:
        """Target 50% invested, 50% cash — reconciliation reflects this."""
        plan = connected_service.generate_rebalance_plan(
            target_weights={"AAPL": 0.5}
        )
        connected_service.execute_plan(plan["plan_id"])

        recon = connected_service.get_reconciliation()
        aapl = next(r for r in recon["tickers"] if r["ticker"] == "AAPL")
        assert aapl["target_weight"] == 0.5
        # Actual weight should be close to 0.5
        assert 0.3 < aapl["actual_weight"] < 0.7
