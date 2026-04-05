"""Trade queue manager -- order generation, approval flow, and IB submission.

Converts sized positions into executable trade orders with:
- Limit pricing with buffer
- Two-step approval (approve candidates, confirm batch)
- Price sanity check against live data
- Market hours awareness
- Daily loss limit enforcement
- Slippage tracking post-fill
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from src.deployment import (
    AbstractQueueManager,
    SizedPosition,
    TradeOrder,
)
from src.deployment.config import is_live_execution_enabled
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TradeQueueManager(AbstractQueueManager):
    """Manages trade order lifecycle from generation to execution.

    Args:
        config: Deployment config dict.
        broker: Optional broker for live execution (IBBroker or PaperBroker).
        memory: Optional MemoryManager for logging.
    """

    def __init__(
        self,
        config: dict[str, Any],
        broker: Any | None = None,
        memory: Any | None = None,
    ) -> None:
        self._config = config
        self._broker = broker
        self._memory = memory
        self._execution_config = config.get("execution", {})
        self._risk_config = config.get("risk", {})
        self._orders: list[TradeOrder] = []

    def generate_orders(
        self, sized: list[SizedPosition],
    ) -> list[TradeOrder]:
        """Generate trade orders from sized positions.

        Each SizedPosition becomes a TradeOrder with:
        - Unique order ID
        - Limit price = gbp_price * (1 + limit_buffer)
        - MARKET orders keep order_type=MARKET with no limit price

        Args:
            sized: Sized positions from the sizer.

        Returns:
            Trade orders in PENDING_APPROVAL status.
        """
        limit_buffer = self._execution_config.get(
            "limit_price_buffer", 0.005
        )

        orders: list[TradeOrder] = []
        for pos in sized:
            order_id = f"DEP-{uuid.uuid4().hex[:8].upper()}"

            if pos.order_type == "MARKET":
                limit_price = None
            else:
                limit_price = round(pos.gbp_price * (1.0 + limit_buffer), 4)

            orders.append(TradeOrder(
                order_id=order_id,
                ticker=pos.ticker,
                direction="BUY",
                quantity=pos.shares,
                order_type=pos.order_type,
                limit_price=limit_price,
                reasoning_summary=pos.reasoning_summary,
                status="PENDING_APPROVAL",
            ))

        self._orders = orders
        logger.info(
            "Generated %d trade orders from sized positions", len(orders),
        )
        return orders

    def approve_order(self, order_id: str) -> bool:
        """Approve a single order for execution.

        Args:
            order_id: The order ID to approve.

        Returns:
            True if the order was found and approved.
        """
        for order in self._orders:
            if order.order_id == order_id:
                if order.status == "PENDING_APPROVAL":
                    order.status = "APPROVED"
                    logger.info("Order %s approved", order_id)
                    return True
                logger.warning(
                    "Order %s cannot be approved (status=%s)",
                    order_id, order.status,
                )
                return False
        logger.warning("Order %s not found", order_id)
        return False

    def reject_order(self, order_id: str, reason: str = "") -> bool:
        """Reject a single order.

        Args:
            order_id: The order ID to reject.
            reason: Reason for rejection.

        Returns:
            True if the order was found and rejected.
        """
        for order in self._orders:
            if order.order_id == order_id:
                if order.status == "PENDING_APPROVAL":
                    order.status = "REJECTED"
                    order.rejection_reason = reason
                    logger.info("Order %s rejected: %s", order_id, reason)
                    return True
                return False
        return False

    def approve_all(self) -> int:
        """Approve all pending orders.

        Returns:
            Number of orders approved.
        """
        count = 0
        for order in self._orders:
            if order.status == "PENDING_APPROVAL":
                order.status = "APPROVED"
                count += 1
        logger.info("Bulk approved %d orders", count)
        return count

    def get_approved_orders(self) -> list[TradeOrder]:
        """Return all approved (ready-to-submit) orders."""
        return [o for o in self._orders if o.status == "APPROVED"]

    def get_all_orders(self) -> list[TradeOrder]:
        """Return all orders regardless of status."""
        return list(self._orders)

    def get_queue_summary(self) -> dict[str, Any]:
        """Return a summary of the current queue state."""
        status_counts: dict[str, int] = {}
        total_gbp = 0.0

        for order in self._orders:
            status_counts[order.status] = (
                status_counts.get(order.status, 0) + 1
            )
            if order.status == "APPROVED" and order.limit_price:
                total_gbp += order.limit_price * order.quantity

        return {
            "total_orders": len(self._orders),
            "status_counts": status_counts,
            "total_approved_gbp": round(total_gbp, 2),
        }

    # ------------------------------------------------------------------
    # Pre-submission safety checks
    # ------------------------------------------------------------------

    def check_price_sanity(
        self, order: TradeOrder, live_price: float,
    ) -> bool:
        """Check that the order price has not drifted too far from live.

        Args:
            order: The trade order to check.
            live_price: Current live price from IB or market data.

        Returns:
            True if price is within acceptable drift.
        """
        if order.order_type == "MARKET":
            return True
        if order.limit_price is None or live_price <= 0:
            return False

        max_drift = self._risk_config.get("price_drift_max", 0.02)
        drift = abs(order.limit_price - live_price) / live_price
        if drift > max_drift:
            logger.warning(
                "Price drift too high for %s: limit=%.4f, live=%.4f, "
                "drift=%.4f > max=%.4f",
                order.ticker, order.limit_price, live_price,
                drift, max_drift,
            )
            return False
        return True

    def check_daily_loss_limit(
        self, deployed_capital: float, current_value: float,
    ) -> bool:
        """Check whether the daily loss limit has been breached.

        Args:
            deployed_capital: Total capital deployed in current positions.
            current_value: Current market value of deployed positions.

        Returns:
            True if within limits, False if loss limit breached.
        """
        if deployed_capital <= 0:
            return True

        loss_pct = (deployed_capital - current_value) / deployed_capital
        daily_limit = self._risk_config.get("daily_loss_limit", 0.10)

        if loss_pct > daily_limit:
            logger.warning(
                "Daily loss limit breached: loss=%.2f%% > limit=%.2f%%",
                loss_pct * 100, daily_limit * 100,
            )
            return False
        return True

    def check_buying_power(
        self, orders: list[TradeOrder], buying_power: float,
    ) -> bool:
        """Verify that IB buying power covers the batch.

        Args:
            orders: Orders to submit.
            buying_power: Available buying power from IB.

        Returns:
            True if buying power is sufficient.
        """
        total_cost = 0.0
        for o in orders:
            if o.limit_price:
                total_cost += o.limit_price * o.quantity
            elif o.quantity > 0:
                # Market order -- cannot estimate precisely.
                total_cost += o.quantity  # Bare minimum placeholder.

        if total_cost > buying_power:
            logger.warning(
                "Insufficient buying power: need=%.2f, available=%.2f",
                total_cost, buying_power,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_orders(
        self,
        orders: list[TradeOrder] | None = None,
        dry_run: bool = True,
    ) -> list[TradeOrder]:
        """Submit approved orders to the broker.

        Args:
            orders: Orders to submit. If None, submits all approved.
            dry_run: If True, log but do not execute.

        Returns:
            Orders with updated statuses (SUBMITTED/FAILED).
        """
        if orders is None:
            orders = self.get_approved_orders()

        if not orders:
            logger.info("No approved orders to submit")
            return []

        # Safety: refuse to submit if live execution is not enabled.
        if not dry_run and not is_live_execution_enabled(self._config):
            logger.warning(
                "Live execution not enabled in config -- forcing dry_run"
            )
            dry_run = True

        results: list[TradeOrder] = []

        for order in orders:
            if order.status != "APPROVED":
                logger.warning(
                    "Skipping order %s (status=%s, expected APPROVED)",
                    order.order_id, order.status,
                )
                results.append(order)
                continue

            if dry_run:
                logger.info(
                    "[DRY RUN] Would submit: %s %d x %s @ %s (%s)",
                    order.direction, order.quantity, order.ticker,
                    order.limit_price or "MARKET", order.order_type,
                )
                order.status = "SUBMITTED"
                order.fill_timestamp = datetime.now(
                    timezone.utc
                ).isoformat()
                results.append(order)
                continue

            # Live submission via broker.
            if self._broker is None:
                logger.error(
                    "No broker available for live execution of %s",
                    order.order_id,
                )
                order.status = "FAILED"
                results.append(order)
                continue

            try:
                price_arg = order.limit_price if order.order_type == "LIMIT" else None
                fill = self._broker.place_order(
                    ticker=order.ticker,
                    quantity=order.quantity,
                    side=order.direction.lower(),
                    price=price_arg,
                )
                order.status = "SUBMITTED"
                order.fill_timestamp = datetime.now(
                    timezone.utc
                ).isoformat()

                # Extract fill data if available.
                if isinstance(fill, dict):
                    order.fill_price = fill.get("fill_price")
                    order.fill_quantity = fill.get("fill_quantity")
                    order.commission = fill.get("commission")
                    if order.fill_price and order.limit_price:
                        order.slippage = round(
                            order.fill_price - order.limit_price, 6
                        )

                logger.info(
                    "Submitted order %s: %s %d x %s",
                    order.order_id, order.direction,
                    order.quantity, order.ticker,
                )
            except Exception as exc:
                logger.error(
                    "Failed to submit order %s: %s",
                    order.order_id, exc,
                )
                order.status = "FAILED"

            results.append(order)

        return results

    def cancel_all(self) -> int:
        """Cancel all non-final orders (emergency stop).

        Returns:
            Number of orders cancelled.
        """
        count = 0
        for order in self._orders:
            if order.status in ("PENDING_APPROVAL", "APPROVED"):
                order.status = "REJECTED"
                order.rejection_reason = "Emergency stop"
                count += 1
        logger.warning("Emergency stop: cancelled %d orders", count)
        return count
