"""Broker connections — abstract interface and paper trading implementation.

The PaperBroker provides a simulated broker environment for strategy
testing without risking real capital.  All execution reports are persisted
to ``data/processed/executions/`` as timestamped JSON files.

Usage:
    from src.execution.broker import PaperBroker
    broker = PaperBroker(initial_cash=100_000.0)
    broker.connect()
    fill = broker.submit_order({"ticker": "AAPL", "side": "BUY", "quantity": 10})
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_EXEC_DIR = _PROJECT_ROOT / "data" / "processed" / "executions"


class Broker(ABC):
    """Abstract broker interface.

    Concrete implementations must handle connection management, account
    queries, position tracking, and order submission.
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
