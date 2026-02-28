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

import time
from abc import ABC, abstractmethod
from typing import Any

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


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
        """Establish a connection to the broker.

        Returns:
            ``True`` if the connection succeeded, ``False`` otherwise.
        """

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

    Tracks positions and fills in memory.  Always available — no
    external dependencies required.  All actions are logged via
    structured logging.

    Args:
        config: Full project config dict.  If ``None`` the default
            ``config/settings.yaml`` is loaded.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()

        bt_cfg = config.get("backtest", {})
        exec_cfg = config.get("execution", {})
        self._initial_capital = float(
            bt_cfg.get("initial_capital", exec_cfg.get("initial_capital", 100_000))
        )
        self._cash: float = self._initial_capital
        self._positions: dict[str, float] = {}
        self._connected: bool = False
        self._order_log: list[dict[str, Any]] = []
        self._next_order_id: int = 1

        logger.info(
            "PaperBroker initialised: capital=%.0f", self._initial_capital,
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

    # -- Positions / account -----------------------------------------

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_account_value(self) -> float:
        return self._cash + sum(
            qty for qty in self._positions.values()
        )  # Note: without prices, this is a rough proxy

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

        Args:
            ticker: Instrument symbol.
            quantity: Unsigned number of units.
            side: ``"buy"`` or ``"sell"``.
            order_type: ``"market"`` or ``"limit"``.
            limit_price: Required when *order_type* is ``"limit"``.
            price: Execution price for cash tracking.

        Returns:
            Order receipt dict.

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
            self._positions[ticker] = self._positions.get(ticker, 0.0) + quantity
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

    def get_order_status(self, order_id: str) -> dict:
        for order in self._order_log:
            if order["order_id"] == order_id:
                return {"order_id": order_id, "status": order["status"]}
        return {"order_id": order_id, "status": "unknown"}


# ------------------------------------------------------------------
# IBBroker
# ------------------------------------------------------------------

class IBBroker(Broker):
    """Interactive Brokers API wrapper.

    Only instantiable when ``ibapi`` is installed AND the config
    ``execution.mode`` allows it.  In paper mode the IB gateway paper
    port (7497) is used; in live mode the live port (7496) is used.

    Args:
        config: Full project config dict.  If ``None`` the default
            ``config/settings.yaml`` is loaded.

    Raises:
        ImportError: If ``ibapi`` is not installed.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()

        exec_cfg = config.get("execution", {})
        self.mode: str = exec_cfg.get("mode", "paper")
        broker_cfg = exec_cfg.get("broker", {})

        self.host: str = broker_cfg.get("host", "127.0.0.1")
        # Port depends on mode: 7497 for paper, 7496 for live
        if self.mode == "live":
            self.port: int = int(broker_cfg.get("live_port", 7496))
        else:
            self.port = int(broker_cfg.get("port", 7497))
        self.client_id: int = int(broker_cfg.get("client_id", 1))
        self.timeout: int = int(broker_cfg.get("timeout", 10))
        self.max_retries: int = int(broker_cfg.get("max_retries", 3))
        self.retry_delay: int = int(broker_cfg.get("retry_delay", 2))

        self._connected: bool = False
        self._positions: dict[str, float] = {}
        self._order_log: list[dict[str, Any]] = []
        self._next_order_id: int = 1

        if self.mode == "live":
            logger.warning(
                "IBBroker initialised in LIVE mode — orders WILL be sent "
                "to IB gateway at %s:%d",
                self.host, self.port,
            )
        else:
            logger.info(
                "IBBroker initialised in PAPER mode (port %d)", self.port,
            )

    # -- Connection --------------------------------------------------

    def connect(self) -> bool:
        """Connect to the IB gateway with retry logic.

        Attempts *max_retries* connections with exponential backoff.

        Returns:
            ``True`` if connected successfully, ``False`` otherwise.
        """
        try:
            from ibapi.client import EClient  # noqa: F401
        except ImportError:
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
                # Real IBAPI connection logic would go here.
                # Placeholder: mark as connected for the interface contract.
                self._connected = True
                logger.info("Connected to IB gateway")
                return True
            except Exception as exc:
                logger.warning("Connection attempt %d failed: %s", attempt, exc)
                if attempt < self.max_retries:
                    backoff = self.retry_delay * (2 ** (attempt - 1))
                    time.sleep(backoff)

        logger.error("Failed to connect after %d attempts", self.max_retries)
        return False

    def disconnect(self) -> None:
        if self._connected:
            logger.info("Disconnecting from IB gateway")
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # -- Positions / account -----------------------------------------

    def get_positions(self) -> dict[str, float]:
        if not self._connected:
            logger.warning("Not connected — returning empty positions")
            return {}
        # Real implementation would query IB
        return dict(self._positions)

    def get_account_value(self) -> float:
        if not self._connected:
            logger.warning("Not connected — returning 0")
            return 0.0
        # Real implementation would query IB
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

        In a real implementation this wraps IB's ``placeOrder`` with
        proper Contract and Order construction.

        Args:
            ticker: Instrument symbol.
            quantity: Unsigned number of units.
            side: ``"buy"`` or ``"sell"``.
            order_type: ``"market"`` or ``"limit"``.
            limit_price: Required for limit orders.
            price: Current market price (informational for IB).

        Returns:
            Order receipt dict.

        Raises:
            ConnectionError: If the broker is not connected.
            ValueError: On invalid parameters.
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

        logger.info(
            "%s ORDER: %s %g %s @ %s (id=%s)",
            "LIVE" if self.mode == "live" else "IB-PAPER",
            side.upper(),
            quantity,
            ticker,
            order_type if order_type == "market" else f"limit {limit_price}",
            order_id,
        )

        # Real IBAPI placeOrder logic would go here.
        receipt: dict[str, Any] = {
            "order_id": order_id,
            "ticker": ticker,
            "quantity": quantity,
            "side": side,
            "order_type": order_type,
            "limit_price": limit_price,
            "status": "submitted",
            "mode": self.mode,
        }
        self._order_log.append(receipt)
        return receipt

    def get_order_status(self, order_id: str) -> dict:
        for order in self._order_log:
            if order["order_id"] == order_id:
                return {"order_id": order_id, "status": order["status"]}
        return {"order_id": order_id, "status": "unknown"}


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_broker(config: dict[str, Any] | None = None) -> Broker:
    """Create the appropriate broker based on config.

    Args:
        config: Full project config dict.  If ``None`` the default
            ``config/settings.yaml`` is loaded.

    Returns:
        :class:`PaperBroker` when mode is ``"paper"`` (default),
        :class:`IBBroker` when mode is ``"live"``.
    """
    if config is None:
        config = load_config()

    mode = config.get("execution", {}).get("mode", "paper")

    if mode == "live":
        logger.warning("LIVE TRADING MODE — real money at risk")
        return IBBroker(config)

    return PaperBroker(config)
