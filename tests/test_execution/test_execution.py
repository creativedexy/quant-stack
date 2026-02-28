"""Tests for the execution layer — broker, OMS, and safety gates."""

from __future__ import annotations

import pandas as pd
import pytest

from src.execution.broker import IBBroker
from src.execution.oms import ExecutionResult, Order, OrderManager


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def paper_config() -> dict:
    """Config locked to paper mode."""
    return {
        "execution": {
            "mode": "paper",
            "broker": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1,
                "timeout": 10,
                "max_retries": 3,
                "retry_delay": 2,
            },
            "risk": {
                "max_trade_pct": 0.10,
                "max_daily_trades": 20,
            },
        },
    }


@pytest.fixture
def live_config() -> dict:
    """Config set to live mode (for testing the mode flag only)."""
    return {
        "execution": {
            "mode": "live",
            "broker": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1,
                "timeout": 10,
                "max_retries": 3,
                "retry_delay": 2,
            },
            "risk": {
                "max_trade_pct": 0.10,
                "max_daily_trades": 20,
            },
        },
    }


@pytest.fixture
def paper_broker(paper_config: dict) -> IBBroker:
    """Connected paper broker ready for testing."""
    broker = IBBroker(config=paper_config)
    broker.connect()
    return broker


@pytest.fixture
def target_weights() -> pd.Series:
    """Equal-weight target for 5 tickers."""
    tickers = ["SHEL.L", "BP.L", "HSBA.L", "AZN.L", "ULVR.L"]
    return pd.Series(0.20, index=tickers, name="weights")


@pytest.fixture
def prices() -> dict[str, float]:
    """Mock current prices."""
    return {
        "SHEL.L": 25.0,
        "BP.L": 5.0,
        "HSBA.L": 6.5,
        "AZN.L": 110.0,
        "ULVR.L": 40.0,
    }


# ===================================================================
# IBBroker tests
# ===================================================================

class TestIBBrokerPaper:
    """Tests for the paper-mode broker (no real IB connection)."""

    def test_default_mode_is_paper(self) -> None:
        broker = IBBroker(config={"execution": {}})
        assert broker.mode == "paper"

    def test_paper_connect_always_succeeds(self, paper_config: dict) -> None:
        broker = IBBroker(config=paper_config)
        assert broker.connect() is True
        assert broker.is_connected() is True

    def test_paper_disconnect(self, paper_broker: IBBroker) -> None:
        paper_broker.disconnect()
        assert paper_broker.is_connected() is False

    def test_paper_order_not_rejected(self, paper_broker: IBBroker) -> None:
        receipt = paper_broker.place_order("TEST.L", 100, "MKT")
        assert receipt["status"] == "paper_filled"
        assert receipt["mode"] == "paper"

    def test_paper_order_logged(self, paper_broker: IBBroker) -> None:
        paper_broker.place_order("TEST.L", 100, "MKT")
        paper_broker.place_order("OTHER.L", -50, "MKT")
        assert len(paper_broker.order_log) == 2

    def test_paper_positions_track_orders(self, paper_broker: IBBroker) -> None:
        paper_broker.place_order("A.L", 100, "MKT")
        paper_broker.place_order("B.L", 200, "MKT")
        paper_broker.place_order("A.L", -50, "MKT")
        positions = paper_broker.get_positions()
        assert positions["A.L"] == 50
        assert positions["B.L"] == 200

    def test_paper_position_removed_when_flat(self, paper_broker: IBBroker) -> None:
        paper_broker.place_order("X.L", 100, "MKT")
        paper_broker.place_order("X.L", -100, "MKT")
        assert "X.L" not in paper_broker.get_positions()

    def test_account_summary_in_paper(self, paper_broker: IBBroker) -> None:
        summary = paper_broker.get_account_summary()
        assert "net_liquidation" in summary
        assert "available_funds" in summary
        assert summary["net_liquidation"] > 0

    def test_order_without_connection_raises(self, paper_config: dict) -> None:
        broker = IBBroker(config=paper_config)
        # Not connected yet
        with pytest.raises(ConnectionError, match="not connected"):
            broker.place_order("TEST.L", 100, "MKT")

    def test_limit_order_without_price_raises(self, paper_broker: IBBroker) -> None:
        with pytest.raises(ValueError, match="limit_price"):
            paper_broker.place_order("TEST.L", 100, "LMT")

    def test_limit_order_with_price_ok(self, paper_broker: IBBroker) -> None:
        receipt = paper_broker.place_order("TEST.L", 100, "LMT", limit_price=50.0)
        assert receipt["status"] == "paper_filled"
        assert receipt["limit_price"] == 50.0

    def test_sell_order_side_is_sell(self, paper_broker: IBBroker) -> None:
        receipt = paper_broker.place_order("TEST.L", -100, "MKT")
        assert receipt["side"] == "SELL"

    def test_order_id_increments(self, paper_broker: IBBroker) -> None:
        r1 = paper_broker.place_order("A.L", 10, "MKT")
        r2 = paper_broker.place_order("B.L", 20, "MKT")
        assert r2["order_id"] == r1["order_id"] + 1


class TestIBBrokerModeFlag:
    """Tests to verify the live/paper mode flag is respected."""

    def test_live_config_sets_live_mode(self, live_config: dict) -> None:
        broker = IBBroker(config=live_config)
        assert broker.mode == "live"

    def test_explicit_paper_config(self, paper_config: dict) -> None:
        broker = IBBroker(config=paper_config)
        assert broker.mode == "paper"

    def test_missing_mode_defaults_to_paper(self) -> None:
        broker = IBBroker(config={})
        assert broker.mode == "paper"


# ===================================================================
# OMS tests
# ===================================================================

class TestOrderManager:
    """Tests for the Order Management System."""

    def test_generate_plan_creates_orders(
        self, paper_config: dict, target_weights: pd.Series, prices: dict
    ) -> None:
        oms = OrderManager(config=paper_config)
        plan = oms.generate_plan(
            target_weights, current_positions={}, portfolio_value=100_000, prices=prices,
        )
        assert len(plan) > 0
        assert all(isinstance(o, Order) for o in plan)

    def test_all_orders_are_buys_from_empty(
        self, paper_config: dict, target_weights: pd.Series, prices: dict
    ) -> None:
        oms = OrderManager(config=paper_config)
        plan = oms.generate_plan(
            target_weights, current_positions={}, portfolio_value=100_000, prices=prices,
        )
        assert all(o.side == "BUY" for o in plan)

    def test_sell_orders_when_overweight(
        self, paper_config: dict, prices: dict
    ) -> None:
        target = pd.Series({"A.L": 0.50, "B.L": 0.50})
        current = {"A.L": 10_000.0, "B.L": 0.0}  # All in A
        pv = 100_000
        p = {"A.L": 10.0, "B.L": 10.0}
        oms = OrderManager(config=paper_config)
        plan = oms.generate_plan(target, current, pv, p)
        sells = [o for o in plan if o.side == "SELL"]
        buys = [o for o in plan if o.side == "BUY"]
        assert len(sells) >= 1
        assert len(buys) >= 1

    def test_sells_come_before_buys(
        self, paper_config: dict
    ) -> None:
        target = pd.Series({"A.L": 0.30, "B.L": 0.70})
        current = {"A.L": 5_000.0, "B.L": 0.0}
        pv = 100_000
        p = {"A.L": 10.0, "B.L": 10.0}
        oms = OrderManager(config=paper_config)
        plan = oms.generate_plan(target, current, pv, p)
        sides = [o.side for o in plan]
        # Once we see a BUY, there should be no more SELLs
        seen_buy = False
        for s in sides:
            if s == "BUY":
                seen_buy = True
            if seen_buy:
                assert s != "SELL", "SELL found after BUY — order sorting is wrong"

    def test_trade_capped_by_max_trade_pct(self) -> None:
        """A single order should not exceed max_trade_pct of portfolio."""
        config = {
            "execution": {"risk": {"max_trade_pct": 0.10, "max_daily_trades": 50}},
        }
        oms = OrderManager(config=config)
        # Target 100% in one ticker from zero — should be capped to 10%
        target = pd.Series({"BIG.L": 1.0})
        plan = oms.generate_plan(
            target, current_positions={}, portfolio_value=100_000,
            prices={"BIG.L": 10.0},
        )
        assert len(plan) == 1
        # 10% of 100k = 10k; at £10/share → max 1000 shares
        assert plan[0].quantity <= 1000

    def test_no_order_when_already_at_target(
        self, paper_config: dict, prices: dict
    ) -> None:
        """If positions exactly match targets, no orders needed."""
        target = pd.Series({"X.L": 0.50, "Y.L": 0.50})
        current = {"X.L": 500.0, "Y.L": 500.0}
        p = {"X.L": 100.0, "Y.L": 100.0}
        pv = 100_000
        oms = OrderManager(config=paper_config)
        plan = oms.generate_plan(target, current, pv, p)
        assert len(plan) == 0

    def test_max_daily_trades_cap(self) -> None:
        """Plan should be truncated to max_daily_trades."""
        config = {
            "execution": {"risk": {"max_trade_pct": 1.0, "max_daily_trades": 2}},
        }
        oms = OrderManager(config=config)
        target = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        plan = oms.generate_plan(
            target, current_positions={}, portfolio_value=100_000,
            prices={"A": 10, "B": 10, "C": 10, "D": 10},
        )
        assert len(plan) <= 2

    def test_order_has_reason(
        self, paper_config: dict, target_weights: pd.Series, prices: dict
    ) -> None:
        oms = OrderManager(config=paper_config)
        plan = oms.generate_plan(
            target_weights, current_positions={}, portfolio_value=100_000, prices=prices,
        )
        for order in plan:
            assert len(order.reason) > 0
            assert "rebalance" in order.reason


class TestExecutePlan:
    """Tests for plan execution via the broker."""

    def test_execute_returns_execution_result(
        self, paper_broker: IBBroker, paper_config: dict
    ) -> None:
        oms = OrderManager(config=paper_config)
        plan = [Order("A.L", "BUY", 100, "test")]
        result = oms.execute_plan(plan, paper_broker)
        assert isinstance(result, ExecutionResult)
        assert result.orders_sent == 1
        assert result.orders_filled == 1

    def test_execute_multiple_orders(
        self, paper_broker: IBBroker, paper_config: dict
    ) -> None:
        oms = OrderManager(config=paper_config)
        plan = [
            Order("A.L", "BUY", 100, "buy A"),
            Order("B.L", "SELL", 50, "sell B"),
            Order("C.L", "BUY", 200, "buy C"),
        ]
        result = oms.execute_plan(plan, paper_broker)
        assert result.orders_sent == 3
        assert result.orders_filled == 3
        assert len(result.receipts) == 3

    def test_execute_on_disconnected_broker_raises(
        self, paper_config: dict
    ) -> None:
        broker = IBBroker(config=paper_config)
        # Not connected
        oms = OrderManager(config=paper_config)
        plan = [Order("A.L", "BUY", 100, "test")]
        with pytest.raises(ConnectionError):
            oms.execute_plan(plan, broker)

    def test_positions_updated_after_execution(
        self, paper_broker: IBBroker, paper_config: dict
    ) -> None:
        oms = OrderManager(config=paper_config)
        plan = [
            Order("A.L", "BUY", 100, "buy A"),
            Order("B.L", "BUY", 200, "buy B"),
        ]
        oms.execute_plan(plan, paper_broker)
        positions = paper_broker.get_positions()
        assert positions["A.L"] == 100
        assert positions["B.L"] == 200


# ===================================================================
# Reconciliation tests
# ===================================================================

class TestReconciliation:
    """Tests for post-execution reconciliation."""

    def test_perfect_reconciliation(self) -> None:
        target = pd.Series({"A": 0.50, "B": 0.50})
        actual = {"A": 500.0, "B": 500.0}
        prices = {"A": 100.0, "B": 100.0}
        report = OrderManager.reconcile(target, actual, prices, 100_000)
        assert (report["weight_diff"].abs() <= 0.001).all()

    def test_discrepancy_detected(self) -> None:
        target = pd.Series({"A": 0.50, "B": 0.50})
        actual = {"A": 600.0, "B": 400.0}  # A overweight
        prices = {"A": 100.0, "B": 100.0}
        report = OrderManager.reconcile(target, actual, prices, 100_000)
        assert report.loc["A", "weight_diff"] > 0
        assert report.loc["B", "weight_diff"] < 0

    def test_missing_position_shows_shortfall(self) -> None:
        target = pd.Series({"A": 0.50, "B": 0.50})
        actual = {"A": 500.0}  # B missing entirely
        prices = {"A": 100.0, "B": 100.0}
        report = OrderManager.reconcile(target, actual, prices, 100_000)
        assert report.loc["B", "actual_weight"] == 0.0
        assert report.loc["B", "weight_diff"] < 0

    def test_report_has_expected_columns(self) -> None:
        target = pd.Series({"A": 1.0})
        actual = {"A": 100.0}
        prices = {"A": 100.0}
        report = OrderManager.reconcile(target, actual, prices, 10_000)
        expected_cols = {"target_weight", "actual_weight", "weight_diff", "value_diff"}
        assert expected_cols == set(report.columns)


# ===================================================================
# End-to-end paper flow
# ===================================================================

class TestEndToEndPaper:
    """Full generate → execute → reconcile cycle in paper mode."""

    def test_full_paper_cycle(self) -> None:
        # Use a permissive trade limit so the full target can be reached
        e2e_config = {
            "execution": {
                "mode": "paper",
                "broker": {"host": "127.0.0.1", "port": 7497, "client_id": 1},
                "risk": {"max_trade_pct": 1.0, "max_daily_trades": 20},
            },
        }

        broker = IBBroker(config=e2e_config)
        broker.connect()

        target = pd.Series({"A.L": 0.60, "B.L": 0.40})
        prices = {"A.L": 50.0, "B.L": 25.0}
        pv = 100_000.0

        oms = OrderManager(config=e2e_config)
        plan = oms.generate_plan(target, {}, pv, prices)
        assert len(plan) > 0

        result = oms.execute_plan(plan, broker)
        assert result.orders_filled == result.orders_sent

        actual = broker.get_positions()
        report = oms.reconcile(target, actual, prices, pv)
        # Integer rounding means small deviations are expected
        assert (report["weight_diff"].abs() < 0.05).all()

        broker.disconnect()
        assert not broker.is_connected()
