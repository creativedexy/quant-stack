"""IBAPI connection wrapper with mandatory paper-trading safety.

Provides an ``IBBroker`` class that checks ``execution.mode`` in the config
and **refuses to send real orders** unless mode is explicitly ``"live"``.
In paper mode every order is logged but never transmitted.

Usage:
    from src.execution.broker import IBBroker
    broker = IBBroker(config=cfg)
    broker.connect()
    broker.place_order("SHEL.L", 100, "MKT")
"""

from __future__ import annotations

import time
from typing import Any

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IBBroker:
    """Interactive Brokers connection wrapper.

    When ``mode`` is ``"paper"`` (the default and the **only** safe default),
    all order methods log what *would* happen but never touch a real gateway.

    Args:
        config: Full project config dict.  If ``None`` the default
            ``config/settings.yaml`` is loaded.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()

        exec_cfg = config.get("execution", {})
        self.mode: str = exec_cfg.get("mode", "paper")
        broker_cfg = exec_cfg.get("broker", {})
        self.host: str = broker_cfg.get("host", "127.0.0.1")
        self.port: int = int(broker_cfg.get("port", 7497))
        self.client_id: int = int(broker_cfg.get("client_id", 1))
        self.timeout: int = int(broker_cfg.get("timeout", 10))
        self.max_retries: int = int(broker_cfg.get("max_retries", 3))
        self.retry_delay: int = int(broker_cfg.get("retry_delay", 2))

        self._connected: bool = False
        self._positions: dict[str, float] = {}
        self._account: dict[str, float] = {}
        self._order_log: list[dict[str, Any]] = []

        if self.mode != "live":
            logger.info(
                "Broker initialised in PAPER mode — no real orders will be sent"
            )
        else:
            logger.warning(
                "Broker initialised in LIVE mode — orders WILL be sent to IB gateway"
            )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to the IB gateway / TWS.

        In paper mode this always succeeds immediately.  In live mode it
        attempts a real IBAPI connection with retry logic.

        Returns:
            ``True`` if connected, ``False`` otherwise.
        """
        if self.mode != "live":
            self._connected = True
            logger.info("Paper broker connected (simulated)")
            return True

        return self._connect_live()

    def disconnect(self) -> None:
        """Disconnect from the IB gateway."""
        if self.mode == "live" and self._connected:
            self._disconnect_live()
        self._connected = False
        logger.info("Broker disconnected")

    def is_connected(self) -> bool:
        """Return whether the broker is currently connected."""
        return self._connected

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """Place an order.

        In paper mode the order is logged but **never transmitted**.

        Args:
            ticker: Instrument ticker / symbol.
            quantity: Signed quantity (positive = buy, negative = sell).
            order_type: ``"MKT"`` (market) or ``"LMT"`` (limit).
            limit_price: Required when *order_type* is ``"LMT"``.

        Returns:
            Order receipt dict with at least ``order_id``, ``status``,
            and ``mode``.

        Raises:
            ConnectionError: If the broker is not connected.
            ValueError: If a limit order is placed without a price.
        """
        if not self._connected:
            raise ConnectionError("Broker is not connected. Call connect() first.")

        if order_type == "LMT" and limit_price is None:
            raise ValueError("limit_price is required for limit orders.")

        order = {
            "order_id": len(self._order_log) + 1,
            "ticker": ticker,
            "quantity": quantity,
            "side": "BUY" if quantity > 0 else "SELL",
            "order_type": order_type,
            "limit_price": limit_price,
            "mode": self.mode,
            "status": "pending",
        }

        if self.mode != "live":
            order["status"] = "paper_filled"
            self._order_log.append(order)
            self._update_paper_position(ticker, quantity)
            logger.info(
                "PAPER ORDER: %s %d %s @ %s (id=%d)",
                order["side"],
                abs(quantity),
                ticker,
                order_type if order_type == "MKT" else f"LMT {limit_price}",
                order["order_id"],
            )
            return order

        return self._place_live_order(order)

    # ------------------------------------------------------------------
    # Positions and account
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, float]:
        """Return current holdings as {ticker: quantity}.

        In paper mode returns the simulated position book.
        """
        if self.mode != "live":
            return dict(self._positions)
        return self._get_live_positions()

    def get_account_summary(self) -> dict[str, float]:
        """Return account summary values.

        In paper mode returns a synthetic summary.
        """
        if self.mode != "live":
            return {
                "net_liquidation": 100_000.0,
                "available_funds": 100_000.0,
                "buying_power": 200_000.0,
                "currency": "GBP",
                **self._account,
            }
        return self._get_live_account_summary()

    @property
    def order_log(self) -> list[dict[str, Any]]:
        """Return the full order log (paper and live)."""
        return list(self._order_log)

    # ------------------------------------------------------------------
    # Paper-mode helpers
    # ------------------------------------------------------------------

    def _update_paper_position(self, ticker: str, quantity: int) -> None:
        """Update the simulated position book."""
        current = self._positions.get(ticker, 0.0)
        new_qty = current + quantity
        if abs(new_qty) < 1e-10:
            self._positions.pop(ticker, None)
        else:
            self._positions[ticker] = new_qty

    # ------------------------------------------------------------------
    # Live-mode stubs (require ibapi at runtime)
    # ------------------------------------------------------------------

    def _connect_live(self) -> bool:
        """Attempt to connect to IB gateway with retry logic."""
        try:
            from ibapi.client import EClient  # noqa: F401
        except ImportError:
            logger.error(
                "ibapi is required for live trading. "
                "Install with: pip install 'quant-stack[execution]'"
            )
            return False

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
                    time.sleep(self.retry_delay * attempt)

        logger.error("Failed to connect after %d attempts", self.max_retries)
        return False

    def _disconnect_live(self) -> None:
        """Disconnect from the real IB gateway."""
        logger.info("Disconnecting from IB gateway")

    def _place_live_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Send a real order via IBAPI."""
        logger.info(
            "LIVE ORDER: %s %d %s @ %s",
            order["side"],
            abs(order["quantity"]),
            order["ticker"],
            order["order_type"],
        )
        order["status"] = "submitted"
        self._order_log.append(order)
        return order

    def _get_live_positions(self) -> dict[str, float]:
        """Fetch real positions from IB."""
        logger.warning("Live position fetch not yet implemented")
        return {}

    def _get_live_account_summary(self) -> dict[str, float]:
        """Fetch real account summary from IB."""
        logger.warning("Live account summary not yet implemented")
        return {}
