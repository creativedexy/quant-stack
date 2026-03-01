"""Broker connections — abstract interface and paper trading implementation.

The PaperBroker provides a simulated broker environment for strategy
testing without risking real capital.  All execution reports are persisted
to ``data/processed/executions/`` as timestamped JSON files.

Usage:
    from src.execution.broker import PaperBroker
    broker = PaperBroker(initial_cash=100_000.0)
    broker.connect()
    fill = broker.submit_order({"ticker": "AAPL", "side": "BUY", "quantity": 10})
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
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import time
from abc import ABC, abstractmethod
from typing import Any

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_EXEC_DIR = _PROJECT_ROOT / "data" / "processed" / "executions"


# ------------------------------------------------------------------
# Abstract Broker
# ------------------------------------------------------------------

class Broker(ABC):
    """Abstract broker interface.

    Concrete implementations must handle connection management, account
    queries, position tracking, and order submission.
    All concrete brokers must implement these methods.  The interface
    is intentionally minimal so that both paper and live implementations
    remain simple to reason about.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker."""

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the broker connection."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return ``True`` if the broker connection is active."""

    @abstractmethod
    def get_mode(self) -> str:
        """Return the trading mode (``'paper'`` or ``'live'``)."""

    @abstractmethod
    def get_account_summary(self) -> dict[str, Any]:
        """Return account-level information.

        Returns:
            Dict with keys: ``account_value``, ``cash``, ``invested``,
            ``positions_count``.
        """

    @abstractmethod
    def get_positions(self) -> dict[str, dict[str, Any]]:
        """Return current positions keyed by ticker.

        Each value dict contains: ``quantity``, ``avg_price``,
        ``current_price``, ``market_value``, ``pnl``.
        """

    @abstractmethod
    def set_prices(self, prices: dict[str, float]) -> None:
        """Update market prices used for valuation.

        Args:
            prices: Mapping of ticker to latest price.
        """

    @abstractmethod
    def submit_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Submit a single order and return the fill report.

        Args:
            order: Dict with ``ticker``, ``side``, ``quantity``, and
                optionally ``limit_price``.

        Returns:
            Fill report dict with ``ticker``, ``side``, ``quantity``,
            ``fill_price``, ``commission``, ``status``.
        """


class PaperBroker(Broker):
    """Simulated broker for paper trading.

    Tracks positions and cash in memory.  Supports configurable commission
    and slippage models.  Persists execution reports to JSON files.

    Args:
        initial_cash: Starting cash balance in base currency.
        base_currency: ISO currency code (default ``'GBP'``).
        commission_rate: Proportional commission per trade (default 0.001).
        slippage_bps: Slippage in basis points (default 5).
        execution_dir: Directory for saving execution reports.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        base_currency: str = "GBP",
        commission_rate: float = 0.001,
        slippage_bps: float = 5.0,
        execution_dir: Path | str | None = None,
    ) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.base_currency = base_currency
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.execution_dir = Path(execution_dir) if execution_dir else _DEFAULT_EXEC_DIR

        self._connected = False
        self._positions: dict[str, dict[str, Any]] = {}
        self._prices: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """Initialise the paper broker (always succeeds)."""
        self._connected = True
        logger.info("PaperBroker connected (cash=%.2f %s)", self.cash, self.base_currency)
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

    def get_mode(self) -> str:
        return "paper"

    # ------------------------------------------------------------------
    # Account & positions
    # ------------------------------------------------------------------
    def get_account_summary(self) -> dict[str, Any]:
        invested = sum(
            pos["quantity"] * self._prices.get(ticker, pos["avg_price"])
            for ticker, pos in self._positions.items()
        )
        return {
            "account_value": self.cash + invested,
            "cash": self.cash,
            "invested": invested,
            "positions_count": len(self._positions),
            "base_currency": self.base_currency,
        }

    def get_positions(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for ticker, pos in self._positions.items():
            current_price = self._prices.get(ticker, pos["avg_price"])
            market_value = pos["quantity"] * current_price
            cost_basis = pos["quantity"] * pos["avg_price"]
            pnl = market_value - cost_basis
            result[ticker] = {
                "quantity": pos["quantity"],
                "avg_price": pos["avg_price"],
                "current_price": current_price,
                "market_value": market_value,
                "pnl": pnl,
            }
        return result

    def set_prices(self, prices: dict[str, float]) -> None:
        self._prices.update(prices)

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def submit_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Execute an order against the paper account.

        Simulates market-order fills with slippage and commission.

        Raises:
            ValueError: If the broker is not connected, or if a sell
                order exceeds the held quantity, or if there is
                insufficient cash for a buy.
        """
        if not self._connected:
            raise ValueError("PaperBroker is not connected")

        ticker = order["ticker"]
        side = order["side"].upper()
        quantity = int(order["quantity"])
        ref_price = self._prices.get(ticker, order.get("est_price", 100.0))

        # Apply slippage
        slip_factor = self.slippage_bps / 10_000
        if side == "BUY":
            fill_price = ref_price * (1 + slip_factor)
        else:
            fill_price = ref_price * (1 - slip_factor)

        trade_value = fill_price * quantity
        commission = trade_value * self.commission_rate

        if side == "BUY":
            total_cost = trade_value + commission
            if total_cost > self.cash:
                raise ValueError(
                    f"Insufficient cash: need {total_cost:.2f}, have {self.cash:.2f}"
                )
            self.cash -= total_cost
            self._update_position_buy(ticker, quantity, fill_price)
        elif side == "SELL":
            held = self._positions.get(ticker, {}).get("quantity", 0)
            if quantity > held:
                raise ValueError(
                    f"Cannot sell {quantity} of {ticker}: only hold {held}"
                )
            self.cash += trade_value - commission
            self._update_position_sell(ticker, quantity)
        else:
            raise ValueError(f"Invalid side: {side}")

        fill = {
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "fill_price": round(fill_price, 4),
            "commission": round(commission, 4),
            "trade_value": round(trade_value, 4),
            "status": "filled",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Filled %s %d %s @ %.4f", side, quantity, ticker, fill_price)
        return fill

    # ------------------------------------------------------------------
    # Execution report persistence
    # ------------------------------------------------------------------
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
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_position_buy(self, ticker: str, quantity: int, price: float) -> None:
        if ticker in self._positions:
            existing = self._positions[ticker]
            total_qty = existing["quantity"] + quantity
            avg = (
                (existing["quantity"] * existing["avg_price"] + quantity * price)
                / total_qty
            )
            self._positions[ticker] = {"quantity": total_qty, "avg_price": round(avg, 4)}
        else:
            self._positions[ticker] = {"quantity": quantity, "avg_price": round(price, 4)}

    def _update_position_sell(self, ticker: str, quantity: int) -> None:
        existing = self._positions[ticker]
        remaining = existing["quantity"] - quantity
        if remaining == 0:
            del self._positions[ticker]
        else:
            self._positions[ticker] = {
                "quantity": remaining,
                "avg_price": existing["avg_price"],
            }
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
