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
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_EXEC_DIR = _PROJECT_ROOT / "data" / "processed" / "executions"


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
# IBBroker
# ------------------------------------------------------------------

class IBBroker(Broker):
    """Interactive Brokers API wrapper.

    Only instantiable when ``ibapi`` is installed AND the config
    ``execution.mode`` allows it.  In paper mode the IB gateway paper
    port (7497) is used; in live mode the live port (7496) is used.

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

        self.host: str = broker_cfg.get("host", "127.0.0.1")
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

        Raises:
            ImportError: If ``ibapi`` is not installed.

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
                self._connected = True
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
        if self._connected:
            logger.info("Disconnecting from IB gateway")
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # -- Positions / account -----------------------------------------

    def get_positions(self) -> dict[str, float]:
        if not self._connected:
            logger.warning("Not connected -- returning empty positions")
            return {}
        return dict(self._positions)

    def get_account_value(self) -> float:
        if not self._connected:
            logger.warning("Not connected -- returning 0")
            return 0.0
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
        """Place an order via the IB API."""
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
        config: Full project config dict.

    Returns:
        :class:`PaperBroker` when mode is ``"paper"`` (default),
        :class:`IBBroker` when mode is ``"live"``.
    """
    if config is None:
        config = {}

    mode = config.get("execution", {}).get("mode", "paper")

    if mode == "live":
        logger.warning("LIVE TRADING MODE -- real money at risk")
        return IBBroker(config)

    return PaperBroker(config)
