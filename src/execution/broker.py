"""Broker abstraction with paper and Interactive Brokers implementations.

Every design decision defaults to safety:
- Default mode is ALWAYS paper trading.
- Live trading requires explicit config AND command-line gate.
- All orders are logged before execution.
- Any error during execution halts further orders (fail-closed).

Usage::

    from src.execution.broker import create_broker, PaperBroker
    broker = create_broker(config)   # returns PaperBroker by default
    broker.connect()
    broker.place_order("SHEL.L", 100, "buy", price=25.50)
"""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_EXEC_DIR = _PROJECT_ROOT / "data" / "processed" / "executions"

# Guard ibapi imports -- module must work without ibapi installed.
try:
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.order import Order as IBOrder
    from ibapi.wrapper import EWrapper

    _HAS_IBAPI = True
except ImportError:
    _HAS_IBAPI = False


# ------------------------------------------------------------------
# Abstract Broker
# ------------------------------------------------------------------

class Broker(ABC):
    """Abstract broker interface.

    All concrete brokers must implement these methods.  The interface
    is intentionally minimal so that both paper and live implementations
    remain simple to reason about.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish a connection to the broker."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the broker connection gracefully."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return whether the broker is currently connected."""

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """Return current holdings as ``{ticker: quantity}``."""

    @abstractmethod
    def get_account_value(self) -> float:
        """Return total portfolio / net liquidation value."""

    @abstractmethod
    def place_order(
        self,
        ticker: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
        price: float | None = None,
    ) -> dict:
        """Place a single order.

        Args:
            ticker: Instrument symbol.
            quantity: Unsigned number of shares.
            side: ``"buy"`` or ``"sell"``.
            order_type: ``"market"`` or ``"limit"``.
            limit_price: Required when *order_type* is ``"limit"``.
            price: Current market price used for paper fills.

        Returns:
            Order receipt dict with at least ``order_id`` and ``status``.
        """

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        """Query the status of an existing order.

        Args:
            order_id: Identifier returned from :meth:`place_order`.

        Returns:
            Status dict with at least ``order_id`` and ``status``.
        """


# ------------------------------------------------------------------
# PaperBroker
# ------------------------------------------------------------------

class PaperBroker(Broker):
    """Simulated broker for paper trading.

    Tracks positions and fills in memory.  Always available -- no
    external dependencies required.  All actions are logged via
    structured logging.

    Can be instantiated with a config dict **or** with explicit keyword
    arguments (used by the service layer).

    Args:
        config: Full project config dict.  If ``None`` and no keyword
            arguments are supplied, defaults are used.
        initial_cash: Starting cash balance in base currency.
        base_currency: ISO currency code (default ``'GBP'``).
        commission_rate: Proportional commission per trade (default 0.001).
        slippage_bps: Slippage in basis points (default 5).
        execution_dir: Directory for saving execution reports.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        initial_cash: float | None = None,
        base_currency: str = "GBP",
        commission_rate: float = 0.001,
        slippage_bps: float = 5.0,
        execution_dir: Path | str | None = None,
    ) -> None:
        if config is not None:
            bt_cfg = config.get("backtest", {})
            exec_cfg = config.get("execution", {})
            self._initial_capital = float(
                bt_cfg.get(
                    "initial_capital",
                    exec_cfg.get("initial_capital", 100_000),
                )
            )
            self.base_currency = "GBP"
            self.commission_rate = 0.001
            self.slippage_bps = 5.0
            self.execution_dir = _DEFAULT_EXEC_DIR
        elif initial_cash is not None:
            self._initial_capital = initial_cash
            self.base_currency = base_currency
            self.commission_rate = commission_rate
            self.slippage_bps = slippage_bps
            self.execution_dir = (
                Path(execution_dir) if execution_dir else _DEFAULT_EXEC_DIR
            )
        else:
            self._initial_capital = 100_000.0
            self.base_currency = base_currency
            self.commission_rate = commission_rate
            self.slippage_bps = slippage_bps
            self.execution_dir = _DEFAULT_EXEC_DIR

        self._cash: float = self._initial_capital
        self._positions: dict[str, float] = {}
        self._prices: dict[str, float] = {}
        self._connected: bool = False
        self._order_log: list[dict[str, Any]] = []
        self._next_order_id: int = 1

        logger.info(
            "PaperBroker initialised: capital=%.0f",
            self._initial_capital,
        )

    # -- Connection --------------------------------------------------

    def connect(self) -> bool:
        self._connected = True
        logger.info("PaperBroker connected (simulated)")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("PaperBroker disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def get_mode(self) -> str:
        """Return ``'paper'``."""
        return "paper"

    # -- Positions / account -----------------------------------------

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_account_value(self) -> float:
        invested = sum(
            abs(qty) * self._prices.get(ticker, 0.0)
            for ticker, qty in self._positions.items()
        )
        return self._cash + invested

    def get_account_summary(self) -> dict[str, Any]:
        """Return account-level information.

        Returns:
            Dict with ``account_value``, ``cash``, ``invested``,
            ``positions_count``.
        """
        invested = sum(
            abs(qty) * self._prices.get(ticker, 0.0)
            for ticker, qty in self._positions.items()
        )
        return {
            "account_value": self._cash + invested,
            "cash": self._cash,
            "invested": invested,
            "positions_count": len(self._positions),
        }

    def set_prices(self, prices: dict[str, float]) -> None:
        """Update market prices used for valuation.

        Args:
            prices: Mapping of ticker to latest price.
        """
        self._prices.update(prices)

    def set_price(self, ticker: str, price: float) -> None:
        """Update the market price for a single ticker.

        Args:
            ticker: Asset ticker symbol.
            price: Latest price.
        """
        self.set_prices({ticker: price})

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def order_log(self) -> list[dict[str, Any]]:
        """Full audit trail of orders placed."""
        return list(self._order_log)

    # -- Orders ------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
        price: float | None = None,
    ) -> dict:
        """Place a simulated order.

        The *price* parameter is used to compute the cash impact.  If
        not supplied the order is still recorded but no cash adjustment
        is made (useful for signal-only backtests).

        Raises:
            ConnectionError: If the broker is not connected.
            ValueError: If a limit order is placed without *limit_price*,
                or if *side* is not ``"buy"`` / ``"sell"``.
        """
        if not self._connected:
            raise ConnectionError(
                "Broker is not connected. Call connect() first."
            )

        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")

        if order_type == "limit" and limit_price is None:
            raise ValueError("limit_price is required for limit orders.")

        order_id = str(self._next_order_id)
        self._next_order_id += 1

        fill_price = limit_price if order_type == "limit" else price

        # Update positions
        if side == "buy":
            self._positions[ticker] = (
                self._positions.get(ticker, 0.0) + quantity
            )
            if fill_price is not None:
                self._cash -= quantity * fill_price
        else:
            current = self._positions.get(ticker, 0.0)
            new_qty = current - quantity
            if abs(new_qty) < 1e-10:
                self._positions.pop(ticker, None)
            else:
                self._positions[ticker] = new_qty
            if fill_price is not None:
                self._cash += quantity * fill_price

        receipt: dict[str, Any] = {
            "order_id": order_id,
            "ticker": ticker,
            "quantity": quantity,
            "side": side,
            "order_type": order_type,
            "limit_price": limit_price,
            "fill_price": fill_price,
            "status": "filled",
            "mode": "paper",
        }
        self._order_log.append(receipt)

        logger.info(
            "PAPER ORDER: %s %g %s @ %s (id=%s)",
            side.upper(),
            quantity,
            ticker,
            f"{fill_price:.4f}" if fill_price is not None else order_type,
            order_id,
        )
        return receipt

    def submit_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Submit an order using a dict interface (for service layer).

        Applies slippage and commission, unlike :meth:`place_order` which
        is a raw simulated fill.

        Args:
            order: Dict with ``ticker``, ``side``, ``quantity``, and
                optionally ``est_price``.

        Returns:
            Fill report dict.
        """
        ticker = order["ticker"]
        side = order["side"].lower()
        quantity = int(order["quantity"])
        est_price = order.get(
            "est_price", self._prices.get(ticker, 100.0)
        )

        # Apply slippage
        slip = self.slippage_bps / 10_000
        if side == "buy":
            fill_price = est_price * (1 + slip)
        else:
            fill_price = est_price * (1 - slip)

        trade_value = fill_price * quantity
        commission = trade_value * self.commission_rate

        if side == "buy":
            self._positions[ticker] = (
                self._positions.get(ticker, 0.0) + quantity
            )
            self._cash -= trade_value + commission
        else:
            current = self._positions.get(ticker, 0.0)
            new_qty = current - quantity
            if abs(new_qty) < 1e-10:
                self._positions.pop(ticker, None)
            else:
                self._positions[ticker] = new_qty
            self._cash += trade_value - commission

        return {
            "ticker": ticker,
            "side": side.upper(),
            "quantity": quantity,
            "fill_price": round(fill_price, 4),
            "commission": round(commission, 4),
            "trade_value": round(trade_value, 4),
            "status": "filled",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_order_status(self, order_id: str) -> dict:
        for order in self._order_log:
            if order["order_id"] == order_id:
                return {"order_id": order_id, "status": order["status"]}
        return {"order_id": order_id, "status": "unknown"}

    # -- Report persistence ------------------------------------------

    def save_execution_report(self, report: dict[str, Any]) -> Path:
        """Save an execution report as a timestamped JSON file.

        Args:
            report: Execution report dictionary.

        Returns:
            Path to the saved JSON file.
        """
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filepath = self.execution_dir / f"execution_{ts}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Execution report saved to %s", filepath)
        return filepath


# ------------------------------------------------------------------
# _IBConnection -- low-level IB API bridge (daemon thread)
# ------------------------------------------------------------------

if _HAS_IBAPI:
    class _IBConnection(EWrapper, EClient):
        """Low-level IB API connection running on a daemon thread.

        Inherits from both ``EWrapper`` (callbacks) and ``EClient``
        (requests).  All shared state is protected by a
        :class:`threading.Lock`.  Synchronous wait-for-callback uses
        :class:`threading.Event` objects.
        """

        def __init__(self) -> None:
            EWrapper.__init__(self)
            EClient.__init__(self, wrapper=self)

            self._lock = threading.Lock()
            self._next_order_id: int = 0

            # Events for synchronous wait patterns
            self._next_id_ready = threading.Event()
            self._positions_ready = threading.Event()
            self._account_ready = threading.Event()
            self._contract_ready = threading.Event()
            self._order_events: dict[int, threading.Event] = {}

            # Shared state
            self._positions: dict[str, float] = {}
            self._contract_details: dict[str, Any] = {}
            self._account_values: dict[str, str] = {}
            self._order_statuses: dict[int, dict[str, Any]] = {}

            # Heartbeat tracking
            self._last_message_time: float = 0.0

            # Error tracking
            self._last_error: str | None = None
            self._connection_lost = threading.Event()

        def _touch(self) -> None:
            """Update last message timestamp (called from every callback)."""
            self._last_message_time = time.monotonic()

        # -- EWrapper callbacks ----------------------------------------

        def connectAck(self) -> None:  # noqa: N802
            """Called when the connection is acknowledged."""
            self._touch()
            logger.info("IB gateway connection acknowledged")

        def nextValidId(self, orderId: int) -> None:  # noqa: N802, N803
            """Receive the next valid order ID from IB gateway."""
            self._touch()
            with self._lock:
                self._next_order_id = orderId
            self._next_id_ready.set()
            logger.info("Next valid order ID: %d", orderId)

        def error(  # type: ignore[override]
            self,
            reqId: int,  # noqa: N803
            errorCode: int,  # noqa: N803
            errorString: str,  # noqa: N803
            advancedOrderRejectJson: str = "",  # noqa: N803
        ) -> None:
            """Handle error callbacks from IB gateway.

            Error codes 1100-1102 relate to connectivity:
            - 1100: Connectivity lost
            - 1101: Connectivity restored (data lost)
            - 1102: Connectivity restored (data maintained)
            """
            self._touch()

            if errorCode == 1100:
                logger.error(
                    "IB connectivity lost: %s", errorString,
                )
                self._connection_lost.set()
            elif errorCode in (1101, 1102):
                logger.info(
                    "IB connectivity restored (code %d): %s",
                    errorCode,
                    errorString,
                )
                self._connection_lost.clear()
            elif errorCode in (2104, 2106, 2158):
                # Informational messages (data farm connections)
                logger.debug(
                    "IB info (code %d, reqId %d): %s",
                    errorCode,
                    reqId,
                    errorString,
                )
            else:
                logger.warning(
                    "IB error (code %d, reqId %d): %s",
                    errorCode,
                    reqId,
                    errorString,
                )
                with self._lock:
                    self._last_error = (
                        f"code={errorCode}: {errorString}"
                    )
                    # If the error is for a pending order, wake it up
                    if reqId in self._order_events:
                        self._order_statuses[reqId] = {
                            "status": "error",
                            "error_code": errorCode,
                            "error_msg": errorString,
                        }
                        self._order_events[reqId].set()

        def position(  # type: ignore[override]
            self,
            account: str,
            contract: Any,
            pos: float,
            avgCost: float,  # noqa: N803
        ) -> None:
            """Receive a single position row."""
            self._touch()
            symbol = getattr(contract, "symbol", "UNKNOWN")
            with self._lock:
                if abs(pos) > 1e-10:
                    self._positions[symbol] = pos
                else:
                    self._positions.pop(symbol, None)

        def positionEnd(self) -> None:  # noqa: N802
            """All positions have been received."""
            self._touch()
            self._positions_ready.set()

        def accountSummary(  # noqa: N802
            self,
            reqId: int,  # noqa: N803
            account: str,
            tag: str,
            value: str,
            currency: str,
        ) -> None:
            """Receive a single account summary tag."""
            self._touch()
            with self._lock:
                self._account_values[tag] = value

        def accountSummaryEnd(  # noqa: N802
            self,
            reqId: int,  # noqa: N803
        ) -> None:
            """All account summary tags have been received."""
            self._touch()
            self._account_ready.set()

        def contractDetails(  # noqa: N802
            self,
            reqId: int,  # noqa: N803
            contractDetails: Any,  # noqa: N803
        ) -> None:
            """Receive contract details from IB."""
            self._touch()
            with self._lock:
                self._contract_details = {
                    "conId": contractDetails.contract.conId,
                    "longName": contractDetails.longName,
                    "secType": contractDetails.contract.secType,
                    "exchange": contractDetails.contract.exchange,
                    "currency": contractDetails.contract.currency,
                }

        def contractDetailsEnd(  # noqa: N802
            self,
            reqId: int,  # noqa: N803
        ) -> None:
            """All contract details received."""
            self._touch()
            self._contract_ready.set()

        def orderStatus(  # noqa: N802
            self,
            orderId: int,  # noqa: N803
            status: str,
            filled: float,
            remaining: float,
            avgFillPrice: float,  # noqa: N803
            permId: int,  # noqa: N803
            parentId: int,  # noqa: N803
            lastFillPrice: float,  # noqa: N803
            clientId: int,  # noqa: N803
            whyHeld: str,  # noqa: N803
            mktCapPrice: float = 0.0,  # noqa: N803
        ) -> None:
            """Receive an order status update."""
            self._touch()
            with self._lock:
                self._order_statuses[orderId] = {
                    "status": status.lower(),
                    "filled": filled,
                    "remaining": remaining,
                    "avg_fill_price": avgFillPrice,
                }
                if orderId in self._order_events:
                    # Wake up if terminal state
                    if status.lower() in (
                        "filled",
                        "cancelled",
                        "inactive",
                        "apicancelled",
                    ):
                        self._order_events[orderId].set()

        def openOrder(  # noqa: N802
            self,
            orderId: int,  # noqa: N803
            contract: Any,
            order: Any,
            orderState: Any,  # noqa: N803
        ) -> None:
            """Receive open order details."""
            self._touch()
            status = getattr(orderState, "status", "unknown").lower()
            with self._lock:
                if orderId not in self._order_statuses:
                    self._order_statuses[orderId] = {"status": status}
                else:
                    self._order_statuses[orderId]["status"] = status

        # -- Thread-safe helpers ---------------------------------------

        def get_next_order_id(self) -> int:
            """Return the next order ID and increment the counter."""
            with self._lock:
                oid = self._next_order_id
                self._next_order_id += 1
                return oid


# ------------------------------------------------------------------
# IBBroker
# ------------------------------------------------------------------

class IBBroker(Broker):
    """Interactive Brokers API wrapper.

    Only instantiable when ``ibapi`` is installed AND the config
    ``execution.mode`` allows it.  In paper mode the IB gateway paper
    port (7497) is used; in live mode the live port (7496) is used.

    Delegates to an internal ``_IBConnection(EWrapper, EClient)`` that
    runs on a daemon thread.  All public methods are synchronous: they
    set up a ``threading.Event``, send a request, then wait with a
    configurable timeout.

    Args:
        config: Full project config dict.

    Raises:
        ImportError: If ``ibapi`` is not installed (on connect).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = {}

        exec_cfg = config.get("execution", {})
        self.mode: str = exec_cfg.get("mode", "paper")
        broker_cfg = exec_cfg.get("broker", {})
        if isinstance(broker_cfg, str):
            # Handle legacy config where broker is just a string like "ibkr"
            broker_cfg = {}

        self.host: str = broker_cfg.get("host", "127.0.0.1")
        if self.mode == "live":
            self.port: int = int(broker_cfg.get("live_port", 7496))
        else:
            self.port = int(broker_cfg.get("port", 7497))
        self.client_id: int = int(broker_cfg.get("client_id", 1))
        self.timeout: int = int(broker_cfg.get("timeout", 10))
        self.max_retries: int = int(broker_cfg.get("max_retries", 3))
        self.retry_delay: int = int(broker_cfg.get("retry_delay", 2))

        # Heartbeat configuration
        self.heartbeat_interval: int = int(
            broker_cfg.get("heartbeat_interval", 10)
        )
        self.heartbeat_timeout: int = int(
            broker_cfg.get("heartbeat_timeout", 30)
        )

        # Ticker mapping configuration
        ticker_map_cfg = broker_cfg.get("ticker_map", {})
        self._default_exchange: str = ticker_map_cfg.get(
            "default_exchange", "SMART"
        )
        self._default_currency: str = ticker_map_cfg.get(
            "default_currency", "USD"
        )
        self._suffix_rules: dict[str, dict[str, str]] = ticker_map_cfg.get(
            "suffix_rules", {
                ".L": {"exchange": "LSE", "currency": "GBP"},
            }
        )

        self._connected: bool = False
        self._conn: Any | None = None  # _IBConnection when connected
        self._thread: threading.Thread | None = None
        self._heartbeat_timer: threading.Timer | None = None
        self._order_log: list[dict[str, Any]] = []

        if self.mode == "live":
            logger.warning(
                "IBBroker initialised in LIVE mode -- orders WILL be sent "
                "to IB gateway at %s:%d",
                self.host,
                self.port,
            )
        else:
            logger.info(
                "IBBroker initialised in PAPER mode (port %d)", self.port,
            )

    # -- Connection --------------------------------------------------

    def connect(self) -> bool:
        """Connect to the IB gateway with retry logic.

        Creates an ``_IBConnection``, starts a daemon thread for the
        event loop, and waits for ``nextValidId`` to confirm the
        gateway is alive.

        Raises:
            ImportError: If ``ibapi`` is not installed.

        Returns:
            ``True`` if connected successfully, ``False`` otherwise.
        """
        if not _HAS_IBAPI:
            raise ImportError(
                "ibapi is required for IB broker connections. "
                "Install with: pip install ibapi"
            )

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Connecting to IB gateway %s:%d (attempt %d/%d)",
                    self.host, self.port, attempt, self.max_retries,
                )

                conn = _IBConnection()
                conn.connect(self.host, self.port, self.client_id)

                # Start the message reader on a daemon thread
                thread = threading.Thread(
                    target=conn.run, daemon=True, name="ib-reader",
                )
                thread.start()

                # Wait for nextValidId -- proves gateway is alive
                if not conn._next_id_ready.wait(timeout=self.timeout):
                    logger.warning(
                        "Timeout waiting for nextValidId on attempt %d",
                        attempt,
                    )
                    try:
                        conn.disconnect()
                    except OSError:
                        pass
                    thread.join(timeout=2)
                    if attempt < self.max_retries:
                        backoff = self.retry_delay * (2 ** (attempt - 1))
                        time.sleep(backoff)
                    continue

                self._conn = conn
                self._thread = thread
                self._connected = True
                self._start_heartbeat()
                logger.info("Connected to IB gateway")
                return True

            except Exception as exc:
                logger.warning(
                    "Connection attempt %d failed: %s", attempt, exc,
                )
                if attempt < self.max_retries:
                    backoff = self.retry_delay * (2 ** (attempt - 1))
                    time.sleep(backoff)

        logger.error(
            "Failed to connect after %d attempts", self.max_retries,
        )
        return False

    def disconnect(self) -> None:
        """Disconnect from the IB gateway gracefully."""
        self._stop_heartbeat()

        if self._conn is not None:
            try:
                logger.info("Disconnecting from IB gateway")
                self._conn.disconnect()
            except OSError:
                # Socket may already be closed on Windows
                pass

        if self._thread is not None:
            if self._thread.is_alive():
                self._thread.join(timeout=5)
            self._thread = None

        self._conn = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_mode(self) -> str:
        """Return the current trading mode (``'paper'`` or ``'live'``)."""
        return self.mode

    def get_account_summary(self) -> dict[str, Any]:
        """Return account-level information matching PaperBroker interface.

        Requests NetLiquidation, TotalCashValue, and GrossPositionValue
        from IB to compute account overview.

        Returns:
            Dict with ``account_value``, ``cash``, ``invested``,
            ``positions_count``.
        """
        if not self._connected or self._conn is None:
            return {
                "account_value": 0.0,
                "cash": 0.0,
                "invested": 0.0,
                "positions_count": 0,
            }

        with self._conn._lock:
            self._conn._account_values.clear()
        self._conn._account_ready.clear()

        self._conn.reqAccountSummary(
            9002, "All",
            "NetLiquidation,TotalCashValue,GrossPositionValue",
        )

        if not self._conn._account_ready.wait(timeout=self.timeout):
            logger.warning(
                "Timeout waiting for account summary (%.0fs)",
                self.timeout,
            )

        try:
            self._conn.cancelAccountSummary(9002)
        except Exception:
            pass

        with self._conn._lock:
            vals = self._conn._account_values

        def _parse(key: str) -> float:
            try:
                return float(vals.get(key, "0"))
            except ValueError:
                return 0.0

        positions = self.get_positions()
        return {
            "account_value": _parse("NetLiquidation"),
            "cash": _parse("TotalCashValue"),
            "invested": _parse("GrossPositionValue"),
            "positions_count": len(positions),
        }

    # -- Heartbeat ---------------------------------------------------

    def _start_heartbeat(self) -> None:
        """Start a periodic heartbeat check for connection liveness."""
        self._stop_heartbeat()
        self._heartbeat_timer = threading.Timer(
            self.heartbeat_interval, self._check_heartbeat,
        )
        self._heartbeat_timer.daemon = True
        self._heartbeat_timer.start()

    def _stop_heartbeat(self) -> None:
        """Cancel the heartbeat timer if running."""
        if self._heartbeat_timer is not None:
            self._heartbeat_timer.cancel()
            self._heartbeat_timer = None

    def _check_heartbeat(self) -> None:
        """Check whether we have received messages recently.

        If the gateway has been silent for longer than
        ``heartbeat_timeout`` seconds, mark as disconnected and attempt
        reconnection with exponential backoff.
        """
        if not self._connected or self._conn is None:
            return

        elapsed = time.monotonic() - self._conn._last_message_time
        if elapsed > self.heartbeat_timeout:
            logger.warning(
                "IB heartbeat stale (%.0fs) -> disconnecting", elapsed,
            )
            self._connected = False
            self._attempt_reconnect()
        else:
            # Schedule next check
            self._start_heartbeat()

    def _attempt_reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        self.disconnect()

        for attempt in range(1, self.max_retries + 1):
            backoff = self.retry_delay * (2 ** (attempt - 1))
            logger.info(
                "Reconnection attempt %d/%d after %.0fs backoff",
                attempt, self.max_retries, backoff,
            )
            time.sleep(backoff)
            if self.connect():
                logger.info("Reconnected to IB gateway")
                return

        logger.error(
            "Failed to reconnect after %d attempts", self.max_retries,
        )

    # -- Contract building -------------------------------------------

    def _make_contract(self, ticker: str) -> Any:
        """Build an IB ``Contract`` object from a ticker symbol.

        Suffix rules (configurable via ``settings.yaml``):
        - ``.L`` -> exchange ``LSE``, currency ``GBP``, secType ``STK``
        - ``.CFD`` -> exchange ``SMART``, currency ``USD``, secType ``CFD``
        - Default -> exchange ``SMART``, currency ``USD``, secType ``STK``

        Each suffix rule may include a ``sec_type`` key to override the
        default ``"STK"`` security type.

        Args:
            ticker: System ticker symbol (e.g. ``"SHEL.L"``, ``"AAPL"``,
                ``"BTC.CFD"``).

        Returns:
            Populated ``Contract`` instance.
        """
        contract = Contract()

        # Check suffix rules (longest suffix first for correct matching)
        for suffix, mapping in sorted(
            self._suffix_rules.items(), key=lambda x: len(x[0]), reverse=True,
        ):
            if ticker.endswith(suffix):
                contract.symbol = ticker[: -len(suffix)]
                contract.secType = mapping.get("sec_type", "STK")
                contract.exchange = mapping.get(
                    "exchange", self._default_exchange,
                )
                contract.currency = mapping.get(
                    "currency", self._default_currency,
                )
                return contract

        # No suffix matched -- use defaults (equity)
        contract.secType = "STK"
        contract.symbol = ticker
        contract.exchange = self._default_exchange
        contract.currency = self._default_currency
        return contract

    def validate_contract(self, ticker: str) -> dict[str, Any]:
        """Validate a ticker resolves to a real IB contract.

        Uses ``reqContractDetails`` to check whether IB recognises
        the contract built by ``_make_contract``.

        Args:
            ticker: System ticker symbol (e.g. ``"SHEL.L"``).

        Returns:
            Dict with contract details if valid, or ``{"error": "..."}``
            if not found or not connected.
        """
        if not self._connected or self._conn is None:
            return {"error": "Not connected to IB gateway"}

        contract = self._make_contract(ticker)

        with self._conn._lock:
            self._conn._contract_details.clear()
        self._conn._contract_ready.clear()

        self._conn.reqContractDetails(9003, contract)

        if not self._conn._contract_ready.wait(timeout=self.timeout):
            return {"error": f"Timeout resolving contract for {ticker}"}

        with self._conn._lock:
            details = dict(self._conn._contract_details)

        if not details:
            return {"error": f"No contract found for {ticker}"}

        return details

    # -- Positions / account -----------------------------------------

    def get_positions(self) -> dict[str, float]:
        """Request positions from IB and wait for the response.

        Returns:
            Dict of ``{ticker: quantity}``, empty if not connected.
        """
        if not self._connected or self._conn is None:
            logger.warning("Not connected -- returning empty positions")
            return {}

        with self._conn._lock:
            self._conn._positions.clear()
        self._conn._positions_ready.clear()

        self._conn.reqPositions()

        if not self._conn._positions_ready.wait(timeout=self.timeout):
            logger.warning(
                "Timeout waiting for positions (%.0fs)", self.timeout,
            )
            return {}

        with self._conn._lock:
            return dict(self._conn._positions)

    def get_account_value(self) -> float:
        """Request the net liquidation value from IB.

        Returns:
            Net liquidation value as a float, or ``0.0`` if not
            connected or the request times out.
        """
        if not self._connected or self._conn is None:
            logger.warning("Not connected -- returning 0")
            return 0.0

        with self._conn._lock:
            self._conn._account_values.clear()
        self._conn._account_ready.clear()

        # reqId=9001 is an arbitrary tag for this summary request
        self._conn.reqAccountSummary(
            9001, "All", "NetLiquidation",
        )

        if not self._conn._account_ready.wait(timeout=self.timeout):
            logger.warning(
                "Timeout waiting for account value (%.0fs)", self.timeout,
            )
            return 0.0

        # Cancel the subscription to avoid repeated callbacks
        try:
            self._conn.cancelAccountSummary(9001)
        except Exception:
            pass

        with self._conn._lock:
            nlv_str = self._conn._account_values.get(
                "NetLiquidation", "0",
            )
            try:
                return float(nlv_str)
            except ValueError:
                logger.warning(
                    "Could not parse NetLiquidation value: %s", nlv_str,
                )
                return 0.0

    # -- Orders ------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
        price: float | None = None,
    ) -> dict:
        """Place an order via the IB API.

        Builds a ``Contract`` and ``Order``, acquires a thread-safe
        order ID, sends the order, and waits for a terminal status
        callback (filled, cancelled, or timeout).

        Args:
            ticker: Instrument symbol.
            quantity: Unsigned number of shares.
            side: ``"buy"`` or ``"sell"``.
            order_type: ``"market"`` or ``"limit"``.
            limit_price: Required when *order_type* is ``"limit"``.
            price: Ignored for IB orders (used by PaperBroker only).

        Returns:
            Order receipt dict with ``order_id``, ``status``, ``ticker``,
            ``quantity``, ``side``, ``order_type``, ``mode``.

        Raises:
            ConnectionError: If the broker is not connected.
            ValueError: If *side* is invalid or *limit_price* is missing
                for limit orders.
        """
        if not self._connected or self._conn is None:
            raise ConnectionError(
                "Broker is not connected. Call connect() first."
            )

        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")

        if order_type == "limit" and limit_price is None:
            raise ValueError("limit_price is required for limit orders.")

        # Build contract and order
        contract = self._make_contract(ticker)

        ib_order = IBOrder()
        ib_order.action = "BUY" if side == "buy" else "SELL"
        ib_order.totalQuantity = quantity
        if order_type == "limit":
            ib_order.orderType = "LMT"
            ib_order.lmtPrice = limit_price
        else:
            ib_order.orderType = "MKT"

        # Acquire thread-safe order ID
        oid = self._conn.get_next_order_id()

        # Register an event for this order
        event = threading.Event()
        with self._conn._lock:
            self._conn._order_events[oid] = event

        logger.info(
            "%s ORDER: %s %g %s @ %s (oid=%d)",
            "LIVE" if self.mode == "live" else "IB-PAPER",
            side.upper(),
            quantity,
            ticker,
            order_type if order_type == "market" else f"limit {limit_price}",
            oid,
        )

        # Send the order
        self._conn.placeOrder(oid, contract, ib_order)

        # Wait for terminal status
        event.wait(timeout=self.timeout)

        # Read the result
        with self._conn._lock:
            status_info = self._conn._order_statuses.get(oid, {})
            status = status_info.get("status", "timeout")
            fill_price_val = status_info.get("avg_fill_price")
            # Clean up
            self._conn._order_events.pop(oid, None)

        order_id = str(oid)
        receipt: dict[str, Any] = {
            "order_id": order_id,
            "ticker": ticker,
            "quantity": quantity,
            "side": side,
            "order_type": order_type,
            "limit_price": limit_price,
            "fill_price": fill_price_val,
            "status": status,
            "mode": self.mode,
        }
        self._order_log.append(receipt)
        return receipt

    def get_order_status(self, order_id: str) -> dict:
        """Query the status of an existing order.

        Checks the local order log first, then the connection's
        status cache for live updates from IB callbacks.
        """
        # Check local order log
        for order in self._order_log:
            if order["order_id"] == order_id:
                # If connected, try to get live status from IB cache
                if self._conn is not None:
                    with self._conn._lock:
                        try:
                            oid = int(order_id)
                            if oid in self._conn._order_statuses:
                                live = self._conn._order_statuses[oid]
                                return {
                                    "order_id": order_id,
                                    "status": live.get(
                                        "status", order["status"],
                                    ),
                                }
                        except (ValueError, TypeError):
                            pass
                return {"order_id": order_id, "status": order["status"]}
        return {"order_id": order_id, "status": "unknown"}


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def translate_ticker(
    ticker: str,
    from_format: str = "yahoo",
    to_format: str = "ib",
) -> str:
    """Convert between ticker formats.

    Currently supports Yahoo Finance to IB translation:
    - ``BTC-USD`` -> ``BTC.CFD`` (crypto)
    - ``ETH-USD`` -> ``ETH.CFD`` (crypto)
    - ``SHEL.L`` -> ``SHEL.L`` (no change, already IB-compatible)
    - ``AAPL`` -> ``AAPL`` (no change)

    Args:
        ticker: Source ticker string.
        from_format: Source format (``"yahoo"`` supported).
        to_format: Target format (``"ib"`` supported).

    Returns:
        Translated ticker string.
    """
    if from_format == "yahoo" and to_format == "ib":
        # Yahoo crypto tickers use -USD suffix (e.g. BTC-USD, ETH-USD)
        if ticker.endswith("-USD") and "." not in ticker:
            base = ticker[: -len("-USD")]
            return f"{base}.CFD"
        return ticker

    if from_format == "ib" and to_format == "yahoo":
        if ticker.endswith(".CFD"):
            base = ticker[: -len(".CFD")]
            return f"{base}-USD"
        return ticker

    return ticker


def create_broker(config: dict[str, Any] | None = None) -> Broker:
    """Create the appropriate broker based on config.

    Uses two config keys under ``execution``:

    - ``broker.type``: ``"paper"`` (default) or ``"ibkr"``
    - ``mode``: ``"paper"`` (default) or ``"live"``

    Combinations:

    - ``type: paper`` -> :class:`PaperBroker` (simulation, no IB)
    - ``type: ibkr, mode: paper`` -> :class:`IBBroker` on port 7497
    - ``type: ibkr, mode: live`` -> :class:`IBBroker` on port 7496

    Args:
        config: Full project config dict.

    Returns:
        Broker instance.
    """
    if config is None:
        config = {}

    exec_cfg = config.get("execution", {})
    mode = exec_cfg.get("mode", "paper")
    broker_cfg = exec_cfg.get("broker", {})

    # Support both new "type" key and legacy "mode: live" behaviour
    broker_type = broker_cfg.get("type", "paper")

    if broker_type == "ibkr" or mode == "live":
        if mode == "live":
            logger.warning("LIVE TRADING MODE -- real money at risk")
        else:
            logger.info("IB paper trading mode")
        return IBBroker(config)

    return PaperBroker(config)
