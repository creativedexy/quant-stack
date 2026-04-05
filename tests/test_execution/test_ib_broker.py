"""Unit tests for IBBroker with mocked IB connection.

All tests run without IB Gateway -- the _IBConnection is replaced
with a lightweight mock that simulates callbacks immediately.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from src.execution.broker import IBBroker, _HAS_IBAPI


# -------------------------------------------------------------------
# Skip marker for classes that require ibapi
# -------------------------------------------------------------------
_skip_no_ibapi = pytest.mark.skipif(
    not _HAS_IBAPI,
    reason="ibapi not installed -- IBBroker connection tests skipped",
)


# -------------------------------------------------------------------
# Mock IB connection
# -------------------------------------------------------------------

class MockIBConnection:
    """Simulates _IBConnection without a socket.

    Callbacks fire synchronously on the calling thread so that
    Event.wait() returns immediately.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_order_id: int = 100
        self._positions: dict[str, float] = {}
        self._account_values: dict[str, str] = {}
        self._order_statuses: dict[int, dict[str, Any]] = {}
        self._last_message_time: float = time.monotonic()
        self._order_events: dict[int, threading.Event] = {}
        self._last_error: str | None = None
        self._connection_lost = threading.Event()

        self._next_id_ready = threading.Event()
        self._positions_ready = threading.Event()
        self._account_ready = threading.Event()

        # Simulate successful connection
        self._next_id_ready.set()
        self._connected = True

        self._contract_ready = threading.Event()
        self._contract_details: dict[str, Any] = {}

        # Test data to populate on request callbacks
        self._test_positions: dict[str, float] = {
            "SHEL": 500.0, "AAPL": 100.0,
        }
        self._test_account_values: dict[str, str] = {
            "NetLiquidation": "150000.0",
            "TotalCashValue": "50000.0",
            "GrossPositionValue": "100000.0",
        }

    def connect(self, host: str, port: int, clientId: int) -> None:  # noqa: N803
        pass

    def disconnect(self) -> None:
        self._connected = False

    def isConnected(self) -> bool:  # noqa: N802
        return self._connected

    def run(self) -> None:
        """Simulate the reader loop (exits immediately)."""
        pass

    def reqPositions(self) -> None:  # noqa: N802
        """Simulate positions callback — populate from test data."""
        with self._lock:
            self._positions.update(self._test_positions)
        self._positions_ready.set()

    def reqAccountSummary(  # noqa: N802
        self, reqId: int, group: str, tags: str,  # noqa: N803
    ) -> None:
        """Simulate account summary callback — populate from test data."""
        with self._lock:
            self._account_values.update(self._test_account_values)
        self._account_ready.set()

    def cancelAccountSummary(self, reqId: int) -> None:  # noqa: N802, N803
        """Simulate cancel account summary."""
        pass

    def reqContractDetails(  # noqa: N802
        self, reqId: int, contract: Any,  # noqa: N803
    ) -> None:
        """Simulate contract details callback."""
        with self._lock:
            self._contract_details = {
                "conId": 12345,
                "longName": f"Mock {contract.symbol}",
                "secType": contract.secType,
                "exchange": contract.exchange,
                "currency": contract.currency,
            }
        self._contract_ready.set()

    def placeOrder(  # noqa: N802
        self, orderId: int, contract: Any, order: Any,  # noqa: N803
    ) -> None:
        """Simulate order fill callback."""
        with self._lock:
            self._order_statuses[orderId] = {
                "status": "filled",
                "filled": order.totalQuantity,
                "remaining": 0.0,
                "avg_fill_price": 150.0,
            }
            if orderId in self._order_events:
                self._order_events[orderId].set()

    def get_next_order_id(self) -> int:
        with self._lock:
            oid = self._next_order_id
            self._next_order_id += 1
            return oid


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def paper_config() -> dict:
    return {
        "execution": {
            "mode": "paper",
            "broker": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1,
                "timeout": 5,
                "max_retries": 1,
                "retry_delay": 0,
                "heartbeat_interval": 60,
                "heartbeat_timeout": 120,
                "ticker_map": {
                    "default_exchange": "SMART",
                    "default_currency": "USD",
                    "suffix_rules": {
                        ".L": {"exchange": "LSE", "currency": "GBP"},
                    },
                },
            },
        },
    }


@pytest.fixture
def live_config() -> dict:
    return {
        "execution": {
            "mode": "live",
            "broker": {
                "host": "127.0.0.1",
                "port": 7497,
                "live_port": 7496,
                "client_id": 1,
                "timeout": 5,
                "max_retries": 1,
                "retry_delay": 0,
                "heartbeat_interval": 60,
                "heartbeat_timeout": 120,
            },
        },
    }


@pytest.fixture
def mock_broker(paper_config: dict) -> IBBroker:
    """IBBroker with _IBConnection replaced by MockIBConnection."""
    broker = IBBroker(config=paper_config)
    mock_conn = MockIBConnection()
    broker._conn = mock_conn
    broker._connected = True
    broker._thread = threading.Thread(target=lambda: None, daemon=True)
    return broker


# -------------------------------------------------------------------
# Connection tests
# -------------------------------------------------------------------

@_skip_no_ibapi
class TestIBBrokerConnect:
    """IBBroker.connect() behaviour."""

    def test_connect_success_with_mock(self, paper_config: dict) -> None:
        """Connection succeeds when _IBConnection fires nextValidId."""
        broker = IBBroker(config=paper_config)

        with patch(
            "src.execution.broker._IBConnection",
            return_value=MockIBConnection(),
        ):
            result = broker.connect()

        assert result is True
        assert broker.is_connected()

    def test_connect_timeout_returns_false(self, paper_config: dict) -> None:
        """Connection returns False when nextValidId never fires."""
        broker = IBBroker(config=paper_config)

        mock_conn = MockIBConnection()
        mock_conn._next_id_ready.clear()  # Never fire

        with patch(
            "src.execution.broker._IBConnection",
            return_value=mock_conn,
        ):
            result = broker.connect()

        assert result is False
        assert not broker.is_connected()


@_skip_no_ibapi
class TestIBBrokerDisconnect:
    """IBBroker.disconnect() behaviour."""

    def test_disconnect_clears_state(self, mock_broker: IBBroker) -> None:
        """Disconnect clears connection state."""
        assert mock_broker.is_connected()
        mock_broker.disconnect()
        assert not mock_broker.is_connected()
        assert mock_broker._conn is None
        assert mock_broker._thread is None


# -------------------------------------------------------------------
# Position / account tests
# -------------------------------------------------------------------

@_skip_no_ibapi
class TestIBBrokerPositions:
    """IBBroker.get_positions() behaviour."""

    def test_returns_positions(self, mock_broker: IBBroker) -> None:
        """get_positions returns the position dict from IB."""
        positions = mock_broker.get_positions()
        assert "SHEL" in positions
        assert positions["SHEL"] == 500.0
        assert "AAPL" in positions

    def test_returns_empty_when_disconnected(
        self, paper_config: dict,
    ) -> None:
        """get_positions returns empty dict when not connected."""
        broker = IBBroker(config=paper_config)
        assert broker.get_positions() == {}


@_skip_no_ibapi
class TestIBBrokerAccountValue:
    """IBBroker.get_account_value() behaviour."""

    def test_returns_net_liquidation(self, mock_broker: IBBroker) -> None:
        """get_account_value returns NetLiquidation from IB."""
        value = mock_broker.get_account_value()
        assert value == 150_000.0

    def test_returns_zero_when_disconnected(
        self, paper_config: dict,
    ) -> None:
        """get_account_value returns 0.0 when not connected."""
        broker = IBBroker(config=paper_config)
        assert broker.get_account_value() == 0.0


# -------------------------------------------------------------------
# Order tests
# -------------------------------------------------------------------

@_skip_no_ibapi
class TestIBBrokerPlaceOrder:
    """IBBroker.place_order() behaviour."""

    def test_market_buy_returns_filled_receipt(
        self, mock_broker: IBBroker,
    ) -> None:
        """Market buy returns a receipt with status filled."""
        receipt = mock_broker.place_order(
            "SHEL.L", 100, "buy", order_type="market",
        )
        assert receipt["status"] == "filled"
        assert receipt["ticker"] == "SHEL.L"
        assert receipt["side"] == "buy"
        assert receipt["quantity"] == 100
        assert receipt["fill_price"] == 150.0
        assert receipt["mode"] == "paper"

    def test_limit_sell_returns_receipt(
        self, mock_broker: IBBroker,
    ) -> None:
        """Limit sell order returns a receipt."""
        receipt = mock_broker.place_order(
            "AAPL", 50, "sell", order_type="limit", limit_price=175.0,
        )
        assert receipt["status"] == "filled"
        assert receipt["limit_price"] == 175.0

    def test_invalid_side_raises(self, mock_broker: IBBroker) -> None:
        """Invalid side raises ValueError."""
        with pytest.raises(ValueError, match="side must be"):
            mock_broker.place_order("AAPL", 10, "short")

    def test_limit_without_price_raises(
        self, mock_broker: IBBroker,
    ) -> None:
        """Limit order without limit_price raises ValueError."""
        with pytest.raises(ValueError, match="limit_price"):
            mock_broker.place_order(
                "AAPL", 10, "buy", order_type="limit",
            )

    def test_not_connected_raises(self, paper_config: dict) -> None:
        """Placing an order when disconnected raises ConnectionError."""
        broker = IBBroker(config=paper_config)
        with pytest.raises(ConnectionError, match="not connected"):
            broker.place_order("AAPL", 10, "buy")


@_skip_no_ibapi
class TestIBBrokerOrderStatus:
    """IBBroker.get_order_status() behaviour."""

    def test_known_order_returns_status(
        self, mock_broker: IBBroker,
    ) -> None:
        """get_order_status returns status for a known order."""
        receipt = mock_broker.place_order("AAPL", 10, "buy")
        status = mock_broker.get_order_status(receipt["order_id"])
        assert status["status"] == "filled"

    def test_unknown_order_returns_unknown(
        self, mock_broker: IBBroker,
    ) -> None:
        """get_order_status returns 'unknown' for unrecognised ID."""
        status = mock_broker.get_order_status("99999")
        assert status["status"] == "unknown"


# -------------------------------------------------------------------
# Contract mapping tests
# -------------------------------------------------------------------

@_skip_no_ibapi
class TestContractMapping:
    """IBBroker._make_contract() ticker-to-contract mapping."""

    def test_lse_ticker(self, mock_broker: IBBroker) -> None:
        """Tickers ending in .L map to LSE/GBP."""
        contract = mock_broker._make_contract("SHEL.L")
        assert contract.symbol == "SHEL"
        assert contract.exchange == "LSE"
        assert contract.currency == "GBP"
        assert contract.secType == "STK"

    def test_us_ticker(self, mock_broker: IBBroker) -> None:
        """US tickers map to SMART/USD."""
        contract = mock_broker._make_contract("AAPL")
        assert contract.symbol == "AAPL"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"
        assert contract.secType == "STK"

    def test_another_lse_ticker(self, mock_broker: IBBroker) -> None:
        """Another .L ticker confirms the suffix rule."""
        contract = mock_broker._make_contract("VUSA.L")
        assert contract.symbol == "VUSA"
        assert contract.exchange == "LSE"
        assert contract.currency == "GBP"


# -------------------------------------------------------------------
# Error handling tests
# -------------------------------------------------------------------

@_skip_no_ibapi
class TestIBBrokerErrorHandling:
    """IBBroker error handling scenarios."""

    def test_rejected_order(self, paper_config: dict) -> None:
        """Rejected order returns status 'error'."""
        broker = IBBroker(config=paper_config)

        mock_conn = MockIBConnection()

        # Override placeOrder to simulate rejection
        def reject_order(
            orderId: int, contract: Any, order: Any,  # noqa: N803
        ) -> None:
            with mock_conn._lock:
                mock_conn._order_statuses[orderId] = {
                    "status": "error",
                    "error_code": 201,
                    "error_msg": "Order rejected",
                }
                if orderId in mock_conn._order_events:
                    mock_conn._order_events[orderId].set()

        mock_conn.placeOrder = reject_order
        broker._conn = mock_conn
        broker._connected = True
        broker._thread = threading.Thread(target=lambda: None, daemon=True)

        receipt = broker.place_order("AAPL", 10, "buy")
        assert receipt["status"] == "error"

    def test_timeout_order(self, paper_config: dict) -> None:
        """Timed-out order returns status 'timeout'."""
        broker = IBBroker(config=paper_config)
        # Override timeout to be very short
        broker.timeout = 0.1

        mock_conn = MockIBConnection()

        # Override placeOrder to never signal the event
        def slow_order(
            orderId: int, contract: Any, order: Any,  # noqa: N803
        ) -> None:
            pass  # Never set the event

        mock_conn.placeOrder = slow_order
        broker._conn = mock_conn
        broker._connected = True
        broker._thread = threading.Thread(target=lambda: None, daemon=True)

        receipt = broker.place_order("AAPL", 10, "buy")
        assert receipt["status"] == "timeout"


# -------------------------------------------------------------------
# Mode tests
# -------------------------------------------------------------------

@_skip_no_ibapi
class TestIBBrokerMode:
    """IBBroker mode configuration."""

    def test_paper_mode(self, paper_config: dict) -> None:
        """Paper config results in paper mode."""
        broker = IBBroker(config=paper_config)
        assert broker.mode == "paper"
        assert broker.get_mode() == "paper"

    def test_live_mode(self, live_config: dict) -> None:
        """Live config results in live mode and port 7496."""
        broker = IBBroker(config=live_config)
        assert broker.mode == "live"
        assert broker.port == 7496
        assert broker.get_mode() == "live"

    def test_default_mode_is_paper(self) -> None:
        """Empty config defaults to paper mode."""
        broker = IBBroker(config={"execution": {}})
        assert broker.mode == "paper"

    def test_legacy_string_broker_config(self) -> None:
        """Legacy string broker config is handled gracefully."""
        config = {
            "execution": {
                "mode": "paper",
                "broker": "ibkr",
            },
        }
        broker = IBBroker(config=config)
        assert broker.mode == "paper"
        assert broker.host == "127.0.0.1"
        assert broker.port == 7497


# -------------------------------------------------------------------
# Crypto CFD contract mapping tests
# -------------------------------------------------------------------

@_skip_no_ibapi
class TestCryptoCFDContractMapping:
    """IBBroker._make_contract() crypto CFD ticker mapping."""

    def test_crypto_cfd_ticker(self, mock_broker: IBBroker) -> None:
        """BTC.CFD maps to symbol BTC, secType CFD, SMART/USD."""
        mock_broker._suffix_rules[".CFD"] = {
            "sec_type": "CFD",
            "exchange": "SMART",
            "currency": "USD",
        }
        contract = mock_broker._make_contract("BTC.CFD")
        assert contract.symbol == "BTC"
        assert contract.secType == "CFD"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"

    def test_eth_cfd_ticker(self, mock_broker: IBBroker) -> None:
        """ETH.CFD maps to symbol ETH, secType CFD, SMART/USD."""
        mock_broker._suffix_rules[".CFD"] = {
            "sec_type": "CFD",
            "exchange": "SMART",
            "currency": "USD",
        }
        contract = mock_broker._make_contract("ETH.CFD")
        assert contract.symbol == "ETH"
        assert contract.secType == "CFD"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"


# -------------------------------------------------------------------
# Ticker translation tests (no ibapi dependency)
# -------------------------------------------------------------------

class TestTickerTranslation:
    """translate_ticker() conversion between ticker formats."""

    def test_yahoo_crypto_to_ib(self) -> None:
        """Yahoo BTC-USD translates to IB BTC.CFD."""
        from src.execution.broker import translate_ticker

        assert translate_ticker("BTC-USD") == "BTC.CFD"

    def test_yahoo_eth_to_ib(self) -> None:
        """Yahoo ETH-USD translates to IB ETH.CFD."""
        from src.execution.broker import translate_ticker

        assert translate_ticker("ETH-USD") == "ETH.CFD"

    def test_equity_passthrough(self) -> None:
        """US equity ticker passes through unchanged."""
        from src.execution.broker import translate_ticker

        assert translate_ticker("AAPL") == "AAPL"

    def test_lse_passthrough(self) -> None:
        """LSE ticker passes through unchanged."""
        from src.execution.broker import translate_ticker

        assert translate_ticker("SHEL.L") == "SHEL.L"

    def test_ib_to_yahoo_crypto(self) -> None:
        """IB BTC.CFD translates back to Yahoo BTC-USD."""
        from src.execution.broker import translate_ticker

        assert translate_ticker("BTC.CFD", "ib", "yahoo") == "BTC-USD"

    def test_identity_unknown_format(self) -> None:
        """Unknown format pair returns ticker unchanged."""
        from src.execution.broker import translate_ticker

        assert translate_ticker("AAPL", "foo", "bar") == "AAPL"


# -------------------------------------------------------------------
# create_broker factory tests (no ibapi dependency)
# -------------------------------------------------------------------

@_skip_no_ibapi
class TestIBBrokerContractValidation:
    """Contract validation method."""

    def test_validate_contract_returns_details(
        self, mock_broker: IBBroker,
    ) -> None:
        """validate_contract returns contract details dict."""
        result = mock_broker.validate_contract("AAPL")
        assert isinstance(result, dict)
        assert "conId" in result
        assert result["secType"] == "STK"

    def test_validate_contract_lse(
        self, mock_broker: IBBroker,
    ) -> None:
        """validate_contract returns LSE details for .L ticker."""
        result = mock_broker.validate_contract("SHEL.L")
        assert result["exchange"] == "LSE"
        assert result["currency"] == "GBP"

    def test_validate_contract_disconnected(
        self, paper_config: dict,
    ) -> None:
        """validate_contract returns error when disconnected."""
        broker = IBBroker(config=paper_config)
        result = broker.validate_contract("AAPL")
        assert "error" in result


@_skip_no_ibapi
class TestIBBrokerAccountSummary:
    """get_account_summary returns real values, not placeholders."""

    def test_account_summary_has_real_cash_value(
        self, mock_broker: IBBroker,
    ) -> None:
        """Cash and invested should not be hardcoded zeros."""
        summary = mock_broker.get_account_summary()
        assert summary["cash"] == 50000.0
        assert summary["invested"] == 100000.0
        assert summary["account_value"] == 150000.0
        assert summary["positions_count"] == 2

    def test_account_summary_disconnected(
        self, paper_config: dict,
    ) -> None:
        """Disconnected broker returns zeros."""
        broker = IBBroker(config=paper_config)
        summary = broker.get_account_summary()
        assert summary["account_value"] == 0.0
        assert summary["cash"] == 0.0
        assert summary["invested"] == 0.0
        assert summary["positions_count"] == 0


# -------------------------------------------------------------------
# create_broker factory tests (no ibapi dependency)
# -------------------------------------------------------------------

class TestCreateBrokerFactory:
    """create_broker() factory function."""

    def test_default_returns_paper_broker(self) -> None:
        """Empty config returns a PaperBroker."""
        from src.execution.broker import PaperBroker, create_broker

        broker = create_broker({})
        assert isinstance(broker, PaperBroker)

    def test_paper_type_returns_paper_broker(self) -> None:
        """Explicit paper type config returns a PaperBroker."""
        from src.execution.broker import PaperBroker, create_broker

        config = {
            "execution": {
                "broker": {"type": "paper"},
            },
        }
        broker = create_broker(config)
        assert isinstance(broker, PaperBroker)
