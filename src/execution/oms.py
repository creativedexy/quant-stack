"""Order Management System -- translates target weights into executable orders.

Takes target portfolio weights (from the optimiser) and current positions
(from the broker), computes the required trades, applies risk limits,
and optionally executes via the broker.

Safety guarantees:
- ``execute_plan`` defaults to ``dry_run=True`` (log-only).
- When ``dry_run=False``, execution halts on the first failed order
  (fail-closed, not fail-open).
- All orders are logged before submission.

Usage::

    from src.execution.oms import OrderManagementSystem
    oms = OrderManagementSystem(broker, config=cfg)
    orders = oms.compute_rebalance_orders(
        target_weights, current_positions, account_value, current_prices,
    )
    report = oms.execute_plan(orders, dry_run=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.execution.broker import Broker
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class Order:
    """A single order in a rebalance plan.

    Attributes:
        ticker: Instrument identifier.
        side: ``"buy"`` or ``"sell"``.
        quantity: Unsigned number of shares.
        order_type: ``"market"`` or ``"limit"``.
        limit_price: Price for limit orders, ``None`` for market.
        est_price: Estimated fill price (for cost estimation).
        reason: Human-readable explanation of why this trade exists.
    """

    ticker: str
    side: str
    quantity: int
    order_type: str = "market"
    limit_price: float | None = None
    est_price: float = 0.0
    reason: str = ""


@dataclass
class ExecutionReport:
    """Full audit trail for an execution run.

    Attributes:
        orders_planned: The orders that were in the plan.
        orders_executed: Broker response dicts for successfully sent orders.
        orders_failed: Dicts with ``order`` and ``error`` for failures.
        timestamp: UTC time when execution started.
        mode: ``"paper"`` or ``"live"``.
    """

    orders_planned: list[Order]
    orders_executed: list[dict[str, Any]] = field(default_factory=list)
    orders_failed: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    mode: str = "paper"


# ------------------------------------------------------------------
# OMS
# ------------------------------------------------------------------


class OrderManagementSystem:
    """Compute rebalance orders and execute them through a broker.

    Args:
        broker: Connected :class:`Broker` instance.
        config: Full project config dict.  Risk parameters are read
            from ``config["execution"]["risk"]``.
    """

    def __init__(
        self,
        broker: Broker,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.broker = broker
        self._config = config or {}

        exec_cfg = self._config.get("execution", {})
        risk_cfg = exec_cfg.get("risk", {})
        self.max_trade_pct: float = float(
            risk_cfg.get("max_trade_pct", 1.0),
        )
        self.max_daily_trades: int = int(
            risk_cfg.get("max_daily_trades", 100),
        )
        self.min_trade_value: float = float(
            risk_cfg.get("min_trade_value", 0.0),
        )
        self.mode: str = exec_cfg.get("mode", "paper")

    # ------------------------------------------------------------------
    # Rebalance order computation
    # ------------------------------------------------------------------

    def compute_rebalance_orders(
        self,
        target_weights: pd.Series,
        current_positions: dict[str, float],
        account_value: float,
        current_prices: dict[str, float],
    ) -> list[Order]:
        """Compute orders needed to move from current to target allocation.

        Process:
            1. Compute target position quantities (weight * value / price).
            2. Compute deltas (target - current).
            3. Filter out tiny trades below ``min_trade_value``.
            4. Apply ``max_trade_pct`` position limit per trade.
            5. Generate :class:`Order` objects with reasons.
            6. Sort: sells first, then buys (free up cash before spending).

        Args:
            target_weights: Series mapping ticker to target weight
                (should sum to approximately 1.0).
            current_positions: Dict of ticker to shares currently held.
            account_value: Total portfolio value in base currency.
            current_prices: Dict of ticker to current share price.

        Returns:
            Sorted list of :class:`Order` objects.
        """
        orders: list[Order] = []

        all_tickers = set(target_weights.index) | set(current_positions.keys())

        for ticker in sorted(all_tickers):
            target_w = float(target_weights.get(ticker, 0.0))
            price = current_prices.get(ticker)
            if price is None or price <= 0:
                logger.warning("No valid price for %s -- skipping", ticker)
                continue

            target_value = account_value * target_w
            current_qty = current_positions.get(ticker, 0.0)
            current_value = current_qty * price
            delta_value = target_value - current_value

            # 3. Filter tiny trades
            if abs(delta_value) < self.min_trade_value:
                logger.debug(
                    "Skipping %s: trade value %.2f below minimum %.2f",
                    ticker,
                    abs(delta_value),
                    self.min_trade_value,
                )
                continue

            # 4. Apply per-trade risk limit
            max_trade_value = account_value * self.max_trade_pct
            if abs(delta_value) > max_trade_value:
                capped_value = (
                    max_trade_value if delta_value > 0 else -max_trade_value
                )
                logger.info(
                    "Capping %s trade from %.0f to %.0f "
                    "(max_trade_pct=%.0f%%)",
                    ticker,
                    delta_value,
                    capped_value,
                    self.max_trade_pct * 100,
                )
                delta_value = capped_value

            delta_shares = int(delta_value / price)

            if delta_shares == 0:
                continue

            side = "buy" if delta_shares > 0 else "sell"
            reason = (
                f"rebalance to target weight {target_w:.2%}: "
                f"current {current_value:,.0f} -> target {target_value:,.0f}"
            )

            orders.append(
                Order(
                    ticker=ticker,
                    side=side,
                    quantity=abs(delta_shares),
                    reason=reason,
                )
            )

        # 6. Sort: sells first (to free capital), then buys
        orders.sort(key=lambda o: (0 if o.side == "sell" else 1, o.ticker))

        # Cap total number of orders
        if len(orders) > self.max_daily_trades:
            logger.warning(
                "Order plan has %d orders, capping to %d",
                len(orders),
                self.max_daily_trades,
            )
            orders = orders[: self.max_daily_trades]

        logger.info(
            "Computed rebalance plan: %d orders (%d buys, %d sells)",
            len(orders),
            sum(1 for o in orders if o.side == "buy"),
            sum(1 for o in orders if o.side == "sell"),
        )

        return orders

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_plan(
        self,
        orders: list[Order],
        dry_run: bool = True,
    ) -> ExecutionReport:
        """Execute a list of orders through the broker.

        If ``dry_run=True`` (default): log all orders but do not send
        them to the broker.

        If ``dry_run=False``: send to broker, but **halt on first
        failure** (fail-closed).  Any remaining orders are not executed.

        Args:
            orders: List of :class:`Order` objects.
            dry_run: If ``True``, only log -- do not modify broker state.

        Returns:
            :class:`ExecutionReport` with full audit trail.
        """
        report = ExecutionReport(
            orders_planned=list(orders),
            mode=self.mode,
        )

        for order in orders:
            # Log every order before any attempt
            logger.info(
                "%s %s %g %s -- %s",
                "[DRY RUN]" if dry_run else "[EXECUTE]",
                order.side.upper(),
                order.quantity,
                order.ticker,
                order.reason,
            )

            if dry_run:
                continue

            try:
                receipt = self.broker.place_order(
                    ticker=order.ticker,
                    quantity=order.quantity,
                    side=order.side,
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                    price=order.est_price if order.est_price else None,
                )
                report.orders_executed.append(receipt)
            except Exception as exc:
                logger.error(
                    "Order FAILED for %s: %s -- HALTING execution",
                    order.ticker,
                    exc,
                )
                report.orders_failed.append({
                    "order": order,
                    "error": str(exc),
                })
                # Fail-closed: stop processing remaining orders
                break

        if not dry_run:
            logger.info(
                "Execution complete: %d executed, %d failed "
                "out of %d planned",
                len(report.orders_executed),
                len(report.orders_failed),
                len(report.orders_planned),
            )

        return report

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(
        self,
        target_weights: pd.Series,
        account_value: float,
        current_prices: dict[str, float],
    ) -> dict[str, Any]:
        """Compare target allocation vs actual broker positions.

        Args:
            target_weights: Intended weight per ticker.
            account_value: Total portfolio value.
            current_prices: Current share prices per ticker.

        Returns:
            Dictionary with:
            - ``aligned``: ``True`` if all positions match targets
              within tolerance (0.1% weight).
            - ``discrepancies``: list of dicts with ``ticker``,
              ``target_weight``, ``actual_weight``, ``diff``.
            - ``total_drift``: sum of absolute weight differences.
        """
        actual_positions = self.broker.get_positions()
        tolerance = 0.001  # 0.1% weight

        discrepancies: list[dict[str, Any]] = []
        total_drift = 0.0

        for ticker in target_weights.index:
            target_w = float(target_weights[ticker])
            qty = actual_positions.get(ticker, 0.0)
            price = current_prices.get(ticker, 0.0)
            actual_value = qty * price
            actual_w = (
                actual_value / account_value if account_value > 0 else 0.0
            )
            diff = actual_w - target_w
            total_drift += abs(diff)

            if abs(diff) > tolerance:
                discrepancies.append({
                    "ticker": ticker,
                    "target_weight": round(target_w, 4),
                    "actual_weight": round(actual_w, 4),
                    "diff": round(diff, 4),
                })

        aligned = len(discrepancies) == 0

        if aligned:
            logger.info(
                "Reconciliation passed -- all weights within tolerance",
            )
        else:
            logger.warning(
                "Reconciliation: %d discrepancies, total drift %.4f",
                len(discrepancies),
                total_drift,
            )

        return {
            "aligned": aligned,
            "discrepancies": discrepancies,
            "total_drift": round(total_drift, 6),
        }
