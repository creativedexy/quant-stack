"""Tests for the OrderManagementSystem."""

from __future__ import annotations

import pandas as pd
import pytest

from src.execution.broker import PaperBroker
from src.execution.oms import ExecutionReport, Order, OrderManagementSystem


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def paper_config() -> dict:
    """Config locked to paper mode with generous risk limits."""
    return {
        "execution": {
            "mode": "paper",
            "broker": {"host": "127.0.0.1", "port": 7497, "client_id": 1},
            "risk": {
                "max_trade_pct": 1.0,
                "max_daily_trades": 20,
                "min_trade_value": 50.0,
            },
        },
        "backtest": {"initial_capital": 100_000},
    }


@pytest.fixture
def paper_broker(paper_config: dict) -> PaperBroker:
    """Connected paper broker."""
    broker = PaperBroker(config=paper_config)
    broker.connect()
    return broker


@pytest.fixture
def oms(paper_broker: PaperBroker, paper_config: dict) -> OrderManagementSystem:
    """OMS wired to a paper broker."""
    return OrderManagementSystem(paper_broker, config=paper_config)


# ===================================================================
# compute_rebalance_orders — known scenario
# ===================================================================

class TestComputeRebalanceOrders:
    """Compute correct trades for known scenarios."""

    def test_from_empty_to_equal_weight(self, oms: OrderManagementSystem) -> None:
        """Current: {A:0, B:0}, Target: {A:0.5, B:0.5}
        Account: 10000, Prices: {A:100, B:50}
        Expected: Buy 50 of A, buy 100 of B."""
        target = pd.Series({"A": 0.5, "B": 0.5})
        orders = oms.compute_rebalance_orders(
            target_weights=target,
            current_positions={},
            account_value=10_000,
            current_prices={"A": 100.0, "B": 50.0},
        )
        assert len(orders) == 2
        a_order = next(o for o in orders if o.ticker == "A")
        b_order = next(o for o in orders if o.ticker == "B")
        assert a_order.side == "buy"
        assert a_order.quantity == 50
        assert b_order.side == "buy"
        assert b_order.quantity == 100

    def test_all_buys_from_empty(self, oms: OrderManagementSystem) -> None:
        target = pd.Series({"X": 0.5, "Y": 0.5})
        orders = oms.compute_rebalance_orders(
            target, {}, 100_000, {"X": 10.0, "Y": 10.0},
        )
        assert all(o.side == "buy" for o in orders)

    def test_generates_sells_and_buys(self, oms: OrderManagementSystem) -> None:
        """Overweight in A, need to sell A and buy B."""
        target = pd.Series({"A": 0.50, "B": 0.50})
        current = {"A": 10_000.0, "B": 0.0}  # All in A
        orders = oms.compute_rebalance_orders(
            target, current, 100_000, {"A": 10.0, "B": 10.0},
        )
        sells = [o for o in orders if o.side == "sell"]
        buys = [o for o in orders if o.side == "buy"]
        assert len(sells) >= 1
        assert len(buys) >= 1


# ===================================================================
# Order sorting
# ===================================================================

class TestOrderSorting:
    """Sells come before buys in order list."""

    def test_sells_before_buys(self, oms: OrderManagementSystem) -> None:
        target = pd.Series({"A": 0.30, "B": 0.70})
        current = {"A": 5_000.0, "B": 0.0}
        orders = oms.compute_rebalance_orders(
            target, current, 100_000, {"A": 10.0, "B": 10.0},
        )
        sides = [o.side for o in orders]
        seen_buy = False
        for s in sides:
            if s == "buy":
                seen_buy = True
            if seen_buy:
                assert s != "sell", "SELL found after BUY — sorting is wrong"


# ===================================================================
# Minimum trade filtering
# ===================================================================

class TestMinTradeFiltering:
    """Tiny trades below minimum are filtered out."""

    def test_tiny_trade_filtered(self) -> None:
        config = {
            "execution": {
                "mode": "paper",
                "risk": {
                    "max_trade_pct": 1.0,
                    "max_daily_trades": 20,
                    "min_trade_value": 200.0,  # High threshold
                },
            },
            "backtest": {"initial_capital": 10_000},
        }
        broker = PaperBroker(config=config)
        broker.connect()
        oms = OrderManagementSystem(broker, config=config)

        # Delta is 10000*0.01 = 100, which is < 200 min
        target = pd.Series({"TINY": 0.01})
        orders = oms.compute_rebalance_orders(
            target, {}, 10_000, {"TINY": 10.0},
        )
        assert len(orders) == 0

    def test_large_trade_not_filtered(self, oms: OrderManagementSystem) -> None:
        target = pd.Series({"BIG": 0.50})
        orders = oms.compute_rebalance_orders(
            target, {}, 100_000, {"BIG": 10.0},
        )
        assert len(orders) == 1


# ===================================================================
# Position limit
# ===================================================================

class TestPositionLimit:
    """max_trade_pct is respected."""

    def test_trade_capped_by_max_trade_pct(self) -> None:
        config = {
            "execution": {
                "mode": "paper",
                "risk": {
                    "max_trade_pct": 0.10,
                    "max_daily_trades": 50,
                    "min_trade_value": 0.0,
                },
            },
            "backtest": {"initial_capital": 100_000},
        }
        broker = PaperBroker(config=config)
        broker.connect()
        oms = OrderManagementSystem(broker, config=config)

        # Target 100% in one ticker from zero — should be capped to 10%
        target = pd.Series({"BIG": 1.0})
        orders = oms.compute_rebalance_orders(
            target, {}, 100_000, {"BIG": 10.0},
        )
        assert len(orders) == 1
        # 10% of 100k = 10k; at £10/share → max 1000 shares
        assert orders[0].quantity <= 1000


# ===================================================================
# execute_plan — dry run
# ===================================================================

class TestExecuteDryRun:
    """execute_plan with dry_run=True logs but doesn't modify broker state."""

    def test_dry_run_does_not_modify_positions(
        self, oms: OrderManagementSystem, paper_broker: PaperBroker,
    ) -> None:
        orders = [Order("A.L", "buy", 100, reason="test")]
        oms.execute_plan(orders, dry_run=True)
        assert paper_broker.get_positions() == {}

    def test_dry_run_returns_report(self, oms: OrderManagementSystem) -> None:
        orders = [Order("A.L", "buy", 100, reason="test")]
        report = oms.execute_plan(orders, dry_run=True)
        assert isinstance(report, ExecutionReport)
        assert len(report.orders_planned) == 1
        assert len(report.orders_executed) == 0

    def test_dry_run_no_failures(self, oms: OrderManagementSystem) -> None:
        orders = [Order("A.L", "buy", 100, reason="test")]
        report = oms.execute_plan(orders, dry_run=True)
        assert len(report.orders_failed) == 0


# ===================================================================
# execute_plan — live execution
# ===================================================================

class TestExecuteLive:
    """execute_plan with dry_run=False modifies broker state."""

    def test_execution_updates_positions(
        self, oms: OrderManagementSystem, paper_broker: PaperBroker,
    ) -> None:
        orders = [
            Order("A.L", "buy", 100, reason="buy A"),
            Order("B.L", "buy", 200, reason="buy B"),
        ]
        oms.execute_plan(orders, dry_run=False)
        positions = paper_broker.get_positions()
        assert positions["A.L"] == 100
        assert positions["B.L"] == 200

    def test_execution_returns_receipts(
        self, oms: OrderManagementSystem,
    ) -> None:
        orders = [
            Order("A.L", "buy", 100, reason="buy A"),
            Order("B.L", "buy", 50, reason="buy B"),
        ]
        report = oms.execute_plan(orders, dry_run=False)
        assert len(report.orders_executed) == 2
        assert all(r["status"] == "filled" for r in report.orders_executed)


# ===================================================================
# Reconciliation
# ===================================================================

class TestReconcile:
    """reconcile detects discrepancies correctly."""

    def test_aligned_when_matching(
        self,
        paper_config: dict,
    ) -> None:
        broker = PaperBroker(config=paper_config)
        broker.connect()
        broker.place_order("A", 500, "buy", price=100.0)
        broker.place_order("B", 500, "buy", price=100.0)

        oms = OrderManagementSystem(broker, config=paper_config)
        result = oms.reconcile(
            target_weights=pd.Series({"A": 0.50, "B": 0.50}),
            account_value=100_000,
            current_prices={"A": 100.0, "B": 100.0},
        )
        assert result["aligned"] is True
        assert len(result["discrepancies"]) == 0

    def test_detects_discrepancy(self, paper_config: dict) -> None:
        broker = PaperBroker(config=paper_config)
        broker.connect()
        broker.place_order("A", 600, "buy", price=100.0)
        broker.place_order("B", 400, "buy", price=100.0)

        oms = OrderManagementSystem(broker, config=paper_config)
        result = oms.reconcile(
            target_weights=pd.Series({"A": 0.50, "B": 0.50}),
            account_value=100_000,
            current_prices={"A": 100.0, "B": 100.0},
        )
        assert result["aligned"] is False
        assert len(result["discrepancies"]) == 2
        assert result["total_drift"] > 0

    def test_missing_position_detected(self, paper_config: dict) -> None:
        broker = PaperBroker(config=paper_config)
        broker.connect()
        broker.place_order("A", 500, "buy", price=100.0)
        # B missing entirely

        oms = OrderManagementSystem(broker, config=paper_config)
        result = oms.reconcile(
            target_weights=pd.Series({"A": 0.50, "B": 0.50}),
            account_value=100_000,
            current_prices={"A": 100.0, "B": 100.0},
        )
        assert result["aligned"] is False
        b_disc = next(d for d in result["discrepancies"] if d["ticker"] == "B")
        assert b_disc["actual_weight"] == 0.0
        assert b_disc["diff"] < 0


# ===================================================================
# No order when already at target
# ===================================================================

class TestNoOrderWhenAtTarget:
    """If positions exactly match targets, no orders needed."""

    def test_no_orders(self, oms: OrderManagementSystem) -> None:
        target = pd.Series({"X": 0.50, "Y": 0.50})
        current = {"X": 500.0, "Y": 500.0}
        orders = oms.compute_rebalance_orders(
            target, current, 100_000, {"X": 100.0, "Y": 100.0},
        )
        assert len(orders) == 0


# ===================================================================
# Order has reason
# ===================================================================

class TestOrderReason:
    """Every order has a non-empty reason."""

    def test_order_reason(self, oms: OrderManagementSystem) -> None:
        target = pd.Series({"A": 0.50, "B": 0.50})
        orders = oms.compute_rebalance_orders(
            target, {}, 100_000, {"A": 10.0, "B": 10.0},
        )
        for order in orders:
            assert len(order.reason) > 0
            assert "rebalance" in order.reason


# ===================================================================
# Max daily trades cap
# ===================================================================

class TestMaxDailyTrades:
    """Plan truncated to max_daily_trades."""

    def test_capped(self) -> None:
        config = {
            "execution": {
                "mode": "paper",
                "risk": {
                    "max_trade_pct": 1.0,
                    "max_daily_trades": 2,
                    "min_trade_value": 0.0,
                },
            },
            "backtest": {"initial_capital": 100_000},
        }
        broker = PaperBroker(config=config)
        broker.connect()
        oms = OrderManagementSystem(broker, config=config)

        target = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        orders = oms.compute_rebalance_orders(
            target, {}, 100_000, {"A": 10, "B": 10, "C": 10, "D": 10},
        )
        assert len(orders) <= 2
