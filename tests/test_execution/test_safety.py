"""Safety gate tests for the execution layer.

Verifies that the execution layer's fail-safe mechanisms work correctly:
- IBBroker requires ibapi
- create_broker returns PaperBroker by default
- ExecutionReport always has timestamp and mode
- execute_plan halts on first failure
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.execution.broker import Broker, IBBroker, PaperBroker, create_broker
from src.execution.oms import ExecutionReport, Order, OrderManagementSystem


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def paper_config() -> dict:
    return {
        "execution": {
            "mode": "paper",
            "broker": {"host": "127.0.0.1", "port": 7497, "client_id": 1},
            "risk": {
                "max_trade_pct": 1.0,
                "max_daily_trades": 20,
                "min_trade_value": 0.0,
            },
        },
        "backtest": {"initial_capital": 100_000},
    }


@pytest.fixture
def live_config() -> dict:
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
        "backtest": {"initial_capital": 100_000},
    }


# ===================================================================
# IBBroker safety
# ===================================================================

class TestIBBrokerSafety:
    """IBBroker requires ibapi to connect."""

    def test_ibbroker_connect_raises_without_ibapi(
        self, paper_config: dict,
    ) -> None:
        """IBBroker.connect() raises ImportError if ibapi not installed."""
        broker = IBBroker(config=paper_config)
        with pytest.raises(ImportError, match="ibapi"):
            broker.connect()

    def test_ibbroker_paper_mode(self, paper_config: dict) -> None:
        broker = IBBroker(config=paper_config)
        assert broker.mode == "paper"

    def test_ibbroker_live_mode(self, live_config: dict) -> None:
        broker = IBBroker(config=live_config)
        assert broker.mode == "live"

    def test_ibbroker_missing_mode_defaults_paper(self) -> None:
        broker = IBBroker(config={"execution": {}})
        assert broker.mode == "paper"

    def test_ibbroker_live_uses_port_7496(self, live_config: dict) -> None:
        broker = IBBroker(config=live_config)
        assert broker.port == 7496

    def test_ibbroker_paper_uses_port_7497(self, paper_config: dict) -> None:
        broker = IBBroker(config=paper_config)
        assert broker.port == 7497


# ===================================================================
# create_broker factory
# ===================================================================

class TestCreateBroker:
    """create_broker returns the correct broker type."""

    def test_returns_paper_broker_by_default(self, paper_config: dict) -> None:
        broker = create_broker(paper_config)
        assert isinstance(broker, PaperBroker)

    def test_returns_paper_broker_when_mode_paper(
        self, paper_config: dict,
    ) -> None:
        broker = create_broker(paper_config)
        assert isinstance(broker, PaperBroker)

    def test_returns_ib_broker_when_mode_live(
        self, live_config: dict,
    ) -> None:
        broker = create_broker(live_config)
        assert isinstance(broker, IBBroker)

    def test_paper_broker_is_a_broker(self, paper_config: dict) -> None:
        broker = create_broker(paper_config)
        assert isinstance(broker, Broker)

    def test_empty_config_returns_paper(self) -> None:
        broker = create_broker({"execution": {}})
        assert isinstance(broker, PaperBroker)


# ===================================================================
# ExecutionReport metadata
# ===================================================================

class TestExecutionReportMetadata:
    """ExecutionReport always includes timestamp and mode."""

    def test_has_timestamp(self) -> None:
        report = ExecutionReport(orders_planned=[], mode="paper")
        assert isinstance(report.timestamp, datetime)

    def test_has_mode(self) -> None:
        report = ExecutionReport(orders_planned=[], mode="paper")
        assert report.mode == "paper"

    def test_mode_from_oms(self, paper_config: dict) -> None:
        broker = PaperBroker(config=paper_config)
        broker.connect()
        oms = OrderManagementSystem(broker, config=paper_config)
        report = oms.execute_plan([], dry_run=True)
        assert report.mode == "paper"

    def test_timestamp_is_recent(self, paper_config: dict) -> None:
        from datetime import timezone

        broker = PaperBroker(config=paper_config)
        broker.connect()
        oms = OrderManagementSystem(broker, config=paper_config)
        before = datetime.now(timezone.utc)
        report = oms.execute_plan([], dry_run=True)
        after = datetime.now(timezone.utc)
        assert before <= report.timestamp <= after


# ===================================================================
# Fail-closed: halt on first failure
# ===================================================================

class _FailingBroker(Broker):
    """A broker that fails on specific tickers for testing."""

    def __init__(self, fail_on: set[str]) -> None:
        self._fail_on = fail_on
        self._connected = True
        self._positions: dict[str, float] = {}

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_account_value(self) -> float:
        return 100_000.0

    def place_order(
        self,
        ticker: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
        price: float | None = None,
    ) -> dict:
        if ticker in self._fail_on:
            raise RuntimeError(f"Simulated failure for {ticker}")
        self._positions[ticker] = self._positions.get(ticker, 0.0) + (
            quantity if side == "buy" else -quantity
        )
        return {
            "order_id": "test",
            "ticker": ticker,
            "status": "filled",
            "mode": "paper",
        }

    def get_order_status(self, order_id: str) -> dict:
        return {"order_id": order_id, "status": "unknown"}


class TestHaltOnFailure:
    """execute_plan halts on first failed order."""

    def test_halts_after_failure(self) -> None:
        """Orders after the failing one are not executed."""
        broker = _FailingBroker(fail_on={"B.L"})
        config = {
            "execution": {
                "mode": "paper",
                "risk": {
                    "max_trade_pct": 1.0,
                    "max_daily_trades": 20,
                    "min_trade_value": 0.0,
                },
            },
        }
        oms = OrderManagementSystem(broker, config=config)

        orders = [
            Order("A.L", "buy", 100, reason="first"),
            Order("B.L", "buy", 200, reason="will fail"),
            Order("C.L", "buy", 300, reason="should not execute"),
        ]
        report = oms.execute_plan(orders, dry_run=False)

        # A.L should have executed
        assert len(report.orders_executed) == 1
        assert report.orders_executed[0]["ticker"] == "A.L"

        # B.L failed
        assert len(report.orders_failed) == 1
        assert "B.L" in str(report.orders_failed[0]["error"])

        # C.L was never attempted — only 1 executed + 1 failed = 2 total
        # out of 3 planned
        assert len(report.orders_planned) == 3

    def test_no_position_for_skipped_orders(self) -> None:
        """The broker should not have positions for orders after failure."""
        broker = _FailingBroker(fail_on={"B.L"})
        config = {
            "execution": {
                "mode": "paper",
                "risk": {
                    "max_trade_pct": 1.0,
                    "max_daily_trades": 20,
                    "min_trade_value": 0.0,
                },
            },
        }
        oms = OrderManagementSystem(broker, config=config)

        orders = [
            Order("A.L", "buy", 100, reason="first"),
            Order("B.L", "buy", 200, reason="will fail"),
            Order("C.L", "buy", 300, reason="should not execute"),
        ]
        oms.execute_plan(orders, dry_run=False)

        positions = broker.get_positions()
        assert "A.L" in positions
        assert "B.L" not in positions
        assert "C.L" not in positions

    def test_all_succeed_no_halt(self) -> None:
        """When all orders succeed, all are executed."""
        broker = _FailingBroker(fail_on=set())
        config = {
            "execution": {
                "mode": "paper",
                "risk": {
                    "max_trade_pct": 1.0,
                    "max_daily_trades": 20,
                    "min_trade_value": 0.0,
                },
            },
        }
        oms = OrderManagementSystem(broker, config=config)

        orders = [
            Order("A.L", "buy", 100, reason="first"),
            Order("B.L", "buy", 200, reason="second"),
            Order("C.L", "buy", 300, reason="third"),
        ]
        report = oms.execute_plan(orders, dry_run=False)

        assert len(report.orders_executed) == 3
        assert len(report.orders_failed) == 0


# ===================================================================
# End-to-end paper flow
# ===================================================================

class TestEndToEndPaper:
    """Full generate -> execute -> reconcile cycle in paper mode."""

    def test_full_paper_cycle(self, paper_config: dict) -> None:
        broker = PaperBroker(config=paper_config)
        broker.connect()

        oms = OrderManagementSystem(broker, config=paper_config)

        target = pd.Series({"A.L": 0.60, "B.L": 0.40})
        prices = {"A.L": 50.0, "B.L": 25.0}
        pv = 100_000.0

        # Compute orders
        orders = oms.compute_rebalance_orders(target, {}, pv, prices)
        assert len(orders) > 0

        # Execute
        report = oms.execute_plan(orders, dry_run=False)
        assert len(report.orders_executed) == len(orders)
        assert len(report.orders_failed) == 0

        # Reconcile
        recon = oms.reconcile(target, pv, prices)
        # Integer rounding means small deviations expected
        assert recon["total_drift"] < 0.05

        broker.disconnect()
        assert not broker.is_connected()
