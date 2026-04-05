"""Tests for PaperBroker and the Broker interface."""

from __future__ import annotations

import pytest

from src.execution.broker import Broker, PaperBroker, create_broker


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
        "backtest": {
            "initial_capital": 100_000,
        },
    }


@pytest.fixture
def paper_broker(paper_config: dict) -> PaperBroker:
    """Connected paper broker ready for testing."""
    broker = PaperBroker(config=paper_config)
    broker.connect()
    return broker


# ===================================================================
# Instantiation
# ===================================================================

class TestPaperBrokerInstantiation:
    """PaperBroker can be instantiated with config."""

    def test_instantiates(self, paper_config: dict) -> None:
        broker = PaperBroker(config=paper_config)
        assert isinstance(broker, Broker)

    def test_starts_with_correct_capital(self, paper_config: dict) -> None:
        broker = PaperBroker(config=paper_config)
        assert broker.cash == 100_000

    def test_starts_with_no_positions(self, paper_config: dict) -> None:
        broker = PaperBroker(config=paper_config)
        assert broker.get_positions() == {}

    def test_custom_capital(self) -> None:
        config = {"backtest": {"initial_capital": 50_000}, "execution": {}}
        broker = PaperBroker(config=config)
        assert broker.cash == 50_000


# ===================================================================
# Connection
# ===================================================================

class TestPaperBrokerConnection:
    """Paper broker connection lifecycle."""

    def test_connect_succeeds(self, paper_config: dict) -> None:
        broker = PaperBroker(config=paper_config)
        assert broker.connect() is True
        assert broker.is_connected() is True

    def test_disconnect(self, paper_broker: PaperBroker) -> None:
        paper_broker.disconnect()
        assert paper_broker.is_connected() is False

    def test_order_without_connection_raises(self, paper_config: dict) -> None:
        broker = PaperBroker(config=paper_config)
        with pytest.raises(ConnectionError, match="not connected"):
            broker.place_order("TEST.L", 100, "buy")


# ===================================================================
# Order placement
# ===================================================================

class TestPlaceOrder:
    """place_order updates positions and cash correctly."""

    def test_buy_updates_position(self, paper_broker: PaperBroker) -> None:
        paper_broker.place_order("A.L", 100, "buy", price=10.0)
        assert paper_broker.get_positions()["A.L"] == 100

    def test_buy_reduces_cash(self, paper_broker: PaperBroker) -> None:
        paper_broker.place_order("A.L", 100, "buy", price=10.0)
        assert paper_broker.cash == pytest.approx(100_000 - 100 * 10.0)

    def test_sell_updates_position(self, paper_broker: PaperBroker) -> None:
        paper_broker.place_order("A.L", 100, "buy", price=10.0)
        paper_broker.place_order("A.L", 30, "sell", price=10.0)
        assert paper_broker.get_positions()["A.L"] == 70

    def test_sell_increases_cash(self, paper_broker: PaperBroker) -> None:
        paper_broker.place_order("A.L", 100, "buy", price=10.0)
        paper_broker.place_order("A.L", 50, "sell", price=12.0)
        expected = 100_000 - 100 * 10.0 + 50 * 12.0
        assert paper_broker.cash == pytest.approx(expected)

    def test_multiple_positions_tracked(self, paper_broker: PaperBroker) -> None:
        paper_broker.place_order("A.L", 100, "buy", price=10.0)
        paper_broker.place_order("B.L", 200, "buy", price=5.0)
        paper_broker.place_order("A.L", 50, "sell", price=10.0)
        positions = paper_broker.get_positions()
        assert positions["A.L"] == 50
        assert positions["B.L"] == 200

    def test_buy_then_sell_same_qty_returns_to_zero(
        self, paper_broker: PaperBroker,
    ) -> None:
        paper_broker.place_order("X.L", 100, "buy", price=10.0)
        paper_broker.place_order("X.L", 100, "sell", price=10.0)
        assert "X.L" not in paper_broker.get_positions()

    def test_order_receipt_has_fields(self, paper_broker: PaperBroker) -> None:
        receipt = paper_broker.place_order("A.L", 100, "buy", price=10.0)
        assert "order_id" in receipt
        assert receipt["status"] == "filled"
        assert receipt["mode"] == "paper"
        assert receipt["side"] == "buy"

    def test_order_id_increments(self, paper_broker: PaperBroker) -> None:
        r1 = paper_broker.place_order("A.L", 10, "buy", price=10.0)
        r2 = paper_broker.place_order("B.L", 20, "buy", price=10.0)
        assert int(r2["order_id"]) == int(r1["order_id"]) + 1

    def test_order_log_grows(self, paper_broker: PaperBroker) -> None:
        paper_broker.place_order("A.L", 10, "buy", price=10.0)
        paper_broker.place_order("B.L", 20, "sell", price=10.0)
        assert len(paper_broker.order_log) == 2


# ===================================================================
# Limit orders
# ===================================================================

class TestLimitOrders:
    """Limit order validation."""

    def test_limit_order_without_price_raises(
        self, paper_broker: PaperBroker,
    ) -> None:
        with pytest.raises(ValueError, match="limit_price"):
            paper_broker.place_order("A.L", 100, "buy", order_type="limit")

    def test_limit_order_with_price_ok(self, paper_broker: PaperBroker) -> None:
        receipt = paper_broker.place_order(
            "A.L", 100, "buy", order_type="limit", limit_price=50.0,
        )
        assert receipt["status"] == "filled"
        assert receipt["fill_price"] == 50.0


# ===================================================================
# Invalid side
# ===================================================================

class TestInvalidSide:
    """Invalid side parameter is rejected."""

    def test_invalid_side_raises(self, paper_broker: PaperBroker) -> None:
        with pytest.raises(ValueError, match="side"):
            paper_broker.place_order("A.L", 100, "hold")


# ===================================================================
# Short positions
# ===================================================================

class TestShortPositions:
    """Selling more than held creates a short position."""

    def test_sell_without_holding_creates_short(
        self, paper_broker: PaperBroker,
    ) -> None:
        paper_broker.place_order("A.L", 50, "sell", price=10.0)
        assert paper_broker.get_positions()["A.L"] == -50

    def test_sell_more_than_held_creates_short(
        self, paper_broker: PaperBroker,
    ) -> None:
        paper_broker.place_order("A.L", 100, "buy", price=10.0)
        paper_broker.place_order("A.L", 150, "sell", price=10.0)
        assert paper_broker.get_positions()["A.L"] == -50


# ===================================================================
# get_order_status
# ===================================================================

class TestOrderStatus:
    """get_order_status returns correct status."""

    def test_known_order(self, paper_broker: PaperBroker) -> None:
        receipt = paper_broker.place_order("A.L", 10, "buy", price=10.0)
        status = paper_broker.get_order_status(receipt["order_id"])
        assert status["status"] == "filled"

    def test_unknown_order(self, paper_broker: PaperBroker) -> None:
        status = paper_broker.get_order_status("999")
        assert status["status"] == "unknown"
