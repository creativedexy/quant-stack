"""Order Management System — order creation, execution, and reporting.

Bridges the gap between portfolio-level rebalance decisions and
broker-level order submission.  Handles order batching, execution
sequencing (sells before buys), and report generation.

Usage:
    from src.execution.oms import OrderManagementSystem
    from src.execution.broker import PaperBroker

    broker = PaperBroker()
    broker.connect()
    oms = OrderManagementSystem(broker)
    report = oms.execute_orders(orders)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from src.execution.broker import Broker
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Order:
    """Represents a single order to be submitted to a broker.

    Attributes:
        ticker: Instrument symbol.
        side: ``'BUY'`` or ``'SELL'``.
        quantity: Number of shares/units.
        order_type: ``'MARKET'`` or ``'LIMIT'``.
        limit_price: Required if ``order_type`` is ``'LIMIT'``.
        est_price: Estimated execution price (for cost estimation).
        reason: Human-readable reason for the order.
    """

    ticker: str
    side: str
    quantity: int
    order_type: str = "MARKET"
    limit_price: float | None = None
    est_price: float = 0.0
    reason: str = ""


@dataclass
class ExecutionReport:
    """Summary of a batch order execution.

    Attributes:
        plan_id: Unique identifier for the rebalance plan.
        timestamp: ISO-8601 UTC timestamp.
        mode: ``'paper'`` or ``'live'``.
        orders_submitted: Number of orders submitted.
        orders_filled: Number of orders successfully filled.
        fills: List of individual fill dicts from the broker.
        total_commission: Aggregate commission across all fills.
        total_trade_value: Aggregate trade notional.
        status: ``'completed'``, ``'partial'``, or ``'failed'``.
    """

    plan_id: str
    timestamp: str = ""
    mode: str = "paper"
    orders_submitted: int = 0
    orders_filled: int = 0
    fills: list[dict[str, Any]] = field(default_factory=list)
    total_commission: float = 0.0
    total_trade_value: float = 0.0
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dict."""
        return asdict(self)


class OrderManagementSystem:
    """Manages order lifecycle from creation through execution.

    Args:
        broker: A connected :class:`~src.execution.broker.Broker` instance.
    """

    def __init__(self, broker: Broker) -> None:
        self.broker = broker

    def create_rebalance_orders(
        self,
        current_positions: dict[str, dict[str, Any]],
        target_weights: dict[str, float],
        account_value: float,
        prices: dict[str, float],
    ) -> list[Order]:
        """Compute the orders needed to move from current positions to target weights.

        Sizes are adjusted downward slightly to reserve headroom for
        commissions and slippage so that sequential order submission does
        not run out of cash.

        Args:
            current_positions: Current positions from the broker.
            target_weights: Target portfolio weights keyed by ticker (0–1).
            account_value: Total account value for sizing.
            prices: Current market prices keyed by ticker.

        Returns:
            List of :class:`Order` objects, sells ordered before buys.
        """
        orders: list[Order] = []

        # Reserve a small buffer for commissions + slippage so that the
        # aggregate buy value never exceeds available cash.
        cost_buffer = getattr(self.broker, "commission_rate", 0.001) + (
            getattr(self.broker, "slippage_bps", 5.0) / 10_000
        )
        sizing_value = account_value * (1 - cost_buffer)

        all_tickers = set(current_positions.keys()) | set(target_weights.keys())

        for ticker in sorted(all_tickers):
            target_w = target_weights.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)
            if price <= 0:
                logger.warning("Skipping %s: no valid price", ticker)
                continue

            target_value = sizing_value * target_w
            target_qty = int(target_value / price)

            current_qty = current_positions.get(ticker, {}).get("quantity", 0)
            diff = target_qty - current_qty

            if diff == 0:
                continue

            side = "BUY" if diff > 0 else "SELL"
            abs_qty = abs(diff)
            reason = f"Rebalance: {current_qty} -> {target_qty}"

            orders.append(Order(
                ticker=ticker,
                side=side,
                quantity=abs_qty,
                est_price=price,
                reason=reason,
            ))

        # Execute sells before buys to free up cash
        orders.sort(key=lambda o: (0 if o.side == "SELL" else 1, o.ticker))
        return orders

    def execute_orders(self, orders: list[Order], plan_id: str | None = None) -> ExecutionReport:
        """Submit a batch of orders to the broker and produce an execution report.

        Sells are submitted before buys.  If an individual order fails the
        remaining orders are still attempted (best-effort).

        Args:
            orders: Orders to execute.
            plan_id: Optional plan identifier; auto-generated if omitted.

        Returns:
            :class:`ExecutionReport` summarising the execution.
        """
        if plan_id is None:
            plan_id = uuid.uuid4().hex[:12]

        report = ExecutionReport(
            plan_id=plan_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode=self.broker.get_mode(),
            orders_submitted=len(orders),
        )

        for order in orders:
            try:
                fill = self.broker.submit_order({
                    "ticker": order.ticker,
                    "side": order.side,
                    "quantity": order.quantity,
                    "est_price": order.est_price,
                })
                report.fills.append(fill)
                report.orders_filled += 1
                report.total_commission += fill.get("commission", 0.0)
                report.total_trade_value += fill.get("trade_value", 0.0)
            except Exception:
                logger.exception("Order failed: %s %d %s", order.side, order.quantity, order.ticker)
                report.fills.append({
                    "ticker": order.ticker,
                    "side": order.side,
                    "quantity": order.quantity,
                    "status": "failed",
                    "error": "Order rejected",
                })

        if report.orders_filled == 0 and report.orders_submitted > 0:
            report.status = "failed"
        elif report.orders_filled < report.orders_submitted:
            report.status = "partial"
        else:
            report.status = "completed"

        # Persist report via broker
        self.broker.save_execution_report(report.to_dict())

        logger.info(
            "Execution %s: %d/%d filled, status=%s",
            plan_id,
            report.orders_filled,
            report.orders_submitted,
            report.status,
        )
        return report
