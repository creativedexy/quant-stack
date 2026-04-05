"""Binance order manager for limit buy orders.

Handles order placement, status checking, and cancellation via the
Binance REST API. All orders are GTC limit orders -- market orders
are never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor

from src.utils.exceptions import OrderError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OrderResult:
    """Result of an order operation.

    Attributes:
        order_id: Binance order ID.
        asset: Trading pair symbol.
        status: FILLED, PARTIALLY_FILLED, CANCELLED, REJECTED, NEW.
        fill_price: Execution price (0.0 if not filled).
        quantity: Asset units purchased.
        amount_gbp: Quote currency amount.
        timestamp: UTC-aware datetime of the operation.
        raw_response: Full Binance API response for audit.
    """

    order_id: str
    asset: str
    status: str
    fill_price: float
    quantity: float
    amount_gbp: float
    timestamp: datetime
    raw_response: dict


class BinanceOrderManager:
    """Manages limit buy orders on Binance.

    Pre-flight risk validation is performed before every order.
    Lot size rules are fetched from the exchange on first use per
    symbol and cached in memory.

    Args:
        client: An initialised python-binance Client instance.
        config: Full settings dict from config_loader.
    """

    def __init__(self, client: object, config: dict) -> None:
        self._client = client
        self._config = config
        risk = config.get("risk", {})
        self._max_single_order = risk.get("max_single_order_gbp", 50.0)
        self._min_balance_buffer = risk.get("min_balance_buffer_gbp", 10.0)
        self._lot_size_cache: dict[str, dict] = {}

    def _get_lot_size(self, symbol: str) -> dict:
        """Fetch and cache lot size filter for a symbol."""
        if symbol in self._lot_size_cache:
            return self._lot_size_cache[symbol]

        try:
            info = self._client.get_symbol_info(symbol)
        except Exception as exc:
            raise OrderError(
                f"Failed to fetch symbol info for {symbol}: {exc}"
            ) from exc

        if info is None:
            raise OrderError(f"Symbol {symbol} not found on exchange")

        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                lot = {
                    "min_qty": float(f["minQty"]),
                    "step_size": float(f["stepSize"]),
                }
                self._lot_size_cache[symbol] = lot
                return lot

        raise OrderError(f"No LOT_SIZE filter found for {symbol}")

    def _round_quantity(self, quantity: float, step_size: float) -> float:
        """Round quantity down to the nearest step size."""
        if step_size <= 0:
            return quantity
        precision = len(str(step_size).rstrip("0").split(".")[-1])
        return round(floor(quantity / step_size) * step_size, precision)

    def _validate_order(self, amount_gbp: float) -> None:
        """Pre-flight risk validation."""
        if amount_gbp > self._max_single_order:
            raise OrderError(
                f"Order amount {amount_gbp:.2f} GBP exceeds "
                f"max single order {self._max_single_order:.2f} GBP"
            )
        if amount_gbp <= 0:
            raise OrderError(
                f"Order amount must be positive, got {amount_gbp:.2f}"
            )

    def place_limit_buy(
        self,
        asset: str,
        amount_gbp: float,
        limit_price: float,
    ) -> OrderResult:
        """Place a GTC limit buy order.

        Quantity is computed from amount_gbp / limit_price, rounded to
        Binance lot size rules. Returns immediately -- does not wait
        for fill.

        Args:
            asset: Trading pair symbol (e.g. 'BTCGBP').
            amount_gbp: Amount to spend in GBP.
            limit_price: Limit price for the order.

        Returns:
            OrderResult with order details.

        Raises:
            OrderError: If validation fails or Binance API errors.
        """
        self._validate_order(amount_gbp)

        lot = self._get_lot_size(asset)
        raw_qty = amount_gbp / limit_price
        quantity = self._round_quantity(raw_qty, lot["step_size"])

        if quantity < lot["min_qty"]:
            raise OrderError(
                f"Computed quantity {quantity} below minimum {lot['min_qty']} "
                f"for {asset}"
            )

        try:
            response = self._client.create_order(
                symbol=asset,
                side="BUY",
                type="LIMIT",
                timeInForce="GTC",
                quantity=f"{quantity:.8f}",
                price=f"{limit_price:.2f}",
            )
        except Exception as exc:
            raise OrderError(
                f"Binance API error placing order for {asset}: {exc}"
            ) from exc

        result = self._parse_order_response(asset, response)
        logger.info(
            "order_placed",
            order_id=result.order_id,
            asset=asset,
            status=result.status,
            quantity=quantity,
            price=limit_price,
        )
        return result

    def get_order_status(self, asset: str, order_id: str) -> OrderResult:
        """Check the current status of an existing order.

        Args:
            asset: Trading pair symbol.
            order_id: Binance order ID.

        Returns:
            OrderResult with current status.

        Raises:
            OrderError: If the status check fails.
        """
        try:
            response = self._client.get_order(symbol=asset, orderId=order_id)
        except Exception as exc:
            raise OrderError(
                f"Failed to get order status for {order_id} on {asset}: {exc}"
            ) from exc

        return self._parse_order_response(asset, response)

    def cancel_order(self, asset: str, order_id: str) -> OrderResult:
        """Cancel an existing open order.

        Args:
            asset: Trading pair symbol.
            order_id: Binance order ID.

        Returns:
            OrderResult with CANCELLED status.

        Raises:
            OrderError: If cancellation fails.
        """
        try:
            response = self._client.cancel_order(symbol=asset, orderId=order_id)
        except Exception as exc:
            raise OrderError(
                f"Failed to cancel order {order_id} on {asset}: {exc}"
            ) from exc

        result = self._parse_cancel_response(asset, response)
        logger.info(
            "order_cancelled",
            order_id=order_id,
            asset=asset,
        )
        return result

    def get_account_balance(self, currency: str = "GBP") -> float:
        """Return available (free) balance for a currency.

        Args:
            currency: Currency code (default 'GBP').

        Returns:
            Available balance as float.

        Raises:
            OrderError: If balance check fails.
        """
        try:
            account = self._client.get_account()
        except Exception as exc:
            raise OrderError(f"Failed to get account balance: {exc}") from exc

        for balance in account.get("balances", []):
            if balance["asset"] == currency:
                return float(balance["free"])

        return 0.0

    def _parse_order_response(self, asset: str, response: dict) -> OrderResult:
        """Parse a Binance order response into an OrderResult."""
        # Handle both transactTime and time fields
        ts_ms = response.get("transactTime") or response.get("time", 0)
        return OrderResult(
            order_id=str(response.get("orderId", "")),
            asset=asset,
            status=response.get("status", "UNKNOWN"),
            fill_price=float(response.get("price", "0")),
            quantity=float(response.get("executedQty", "0")),
            amount_gbp=float(response.get("cummulativeQuoteQty", "0")),
            timestamp=(
                datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                if ts_ms
                else datetime.now(timezone.utc)
            ),
            raw_response=response,
        )

    def _parse_cancel_response(self, asset: str, response: dict) -> OrderResult:
        """Parse a Binance cancel response into an OrderResult."""
        return OrderResult(
            order_id=str(response.get("orderId", "")),
            asset=asset,
            status=response.get("status", "CANCELLED"),
            fill_price=0.0,
            quantity=0.0,
            amount_gbp=0.0,
            timestamp=datetime.now(timezone.utc),
            raw_response=response,
        )
