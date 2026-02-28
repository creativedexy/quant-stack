"""Order Management System — translates target weights into executable orders.

Takes target portfolio weights (from the optimiser) and current positions
(from the broker), computes the required trades, applies risk limits,
and optionally executes via the broker.

Usage:
    from src.execution.oms import OrderManager
    oms = OrderManager(config=cfg)
    plan = oms.generate_plan(target_weights, current_positions, portfolio_value)
    oms.execute_plan(plan, broker)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.execution.broker import IBBroker
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ===================================================================
# Order dataclass
# ===================================================================

@dataclass
class Order:
    """A single order in a rebalance plan.

    Attributes:
        ticker: Instrument identifier.
        side: ``"BUY"`` or ``"SELL"``.
        quantity: Unsigned number of shares.
        reason: Human-readable explanation (e.g. "rebalance to target 20.0%").
        order_type: ``"MKT"`` or ``"LMT"``.
        limit_price: Price for limit orders, ``None`` for market.
    """

    ticker: str
    side: str
    quantity: int
    reason: str
    order_type: str = "MKT"
    limit_price: float | None = None


# ===================================================================
# Execution result
# ===================================================================

@dataclass
class ExecutionResult:
    """Result of executing an order plan.

    Attributes:
        orders_sent: Number of orders transmitted.
        orders_filled: Number confirmed as filled.
        receipts: Raw order receipts from the broker.
    """

    orders_sent: int
    orders_filled: int
    receipts: list[dict[str, Any]]


# ===================================================================
# Order Manager
# ===================================================================

class OrderManager:
    """Generates and executes rebalance order plans.

    Args:
        config: Full project config dict.  Falls back to the default
            ``config/settings.yaml`` when ``None``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()

        exec_cfg = config.get("execution", {})
        risk_cfg = exec_cfg.get("risk", {})
        self.max_trade_pct: float = float(risk_cfg.get("max_trade_pct", 0.10))
        self.max_daily_trades: int = int(risk_cfg.get("max_daily_trades", 20))

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------

    def generate_plan(
        self,
        target_weights: pd.Series,
        current_positions: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float],
    ) -> list[Order]:
        """Compute the orders needed to move from current to target allocation.

        Args:
            target_weights: Series mapping ticker → target weight (summing
                to 1.0).
            current_positions: Dict of ticker → number of shares currently held.
            portfolio_value: Total portfolio value in base currency.
            prices: Dict of ticker → current share price, used to convert
                monetary amounts to share quantities.

        Returns:
            List of Order objects, sorted sells-first (to free capital).
        """
        orders: list[Order] = []

        for ticker, target_w in target_weights.items():
            price = prices.get(ticker)
            if price is None or price <= 0:
                logger.warning("No valid price for %s — skipping", ticker)
                continue

            target_value = portfolio_value * target_w
            current_qty = current_positions.get(ticker, 0.0)
            current_value = current_qty * price
            delta_value = target_value - current_value

            # Apply per-trade risk limit
            max_trade_value = portfolio_value * self.max_trade_pct
            if abs(delta_value) > max_trade_value:
                capped_value = max_trade_value if delta_value > 0 else -max_trade_value
                logger.info(
                    "Capping %s trade from £%.0f to £%.0f (max_trade_pct=%.0f%%)",
                    ticker, delta_value, capped_value, self.max_trade_pct * 100,
                )
                delta_value = capped_value

            delta_shares = int(delta_value / price)

            if delta_shares == 0:
                continue

            side = "BUY" if delta_shares > 0 else "SELL"
            reason = (
                f"rebalance: target {target_w:.1%}, "
                f"current £{current_value:,.0f} → target £{target_value:,.0f}"
            )

            orders.append(Order(
                ticker=ticker,
                side=side,
                quantity=abs(delta_shares),
                reason=reason,
            ))

        # Sort: sells first (to free capital), then buys
        orders.sort(key=lambda o: (0 if o.side == "SELL" else 1, o.ticker))

        # Cap total number of orders
        if len(orders) > self.max_daily_trades:
            logger.warning(
                "Order plan has %d orders, capping to %d",
                len(orders), self.max_daily_trades,
            )
            orders = orders[: self.max_daily_trades]

        logger.info(
            "Generated rebalance plan: %d orders (%d buys, %d sells)",
            len(orders),
            sum(1 for o in orders if o.side == "BUY"),
            sum(1 for o in orders if o.side == "SELL"),
        )

        return orders

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

    def execute_plan(
        self,
        plan: list[Order],
        broker: IBBroker,
    ) -> ExecutionResult:
        """Send every order in the plan to the broker sequentially.

        Args:
            plan: List of Order objects from :meth:`generate_plan`.
            broker: Connected IBBroker instance.

        Returns:
            ExecutionResult summarising what was sent and filled.

        Raises:
            ConnectionError: If the broker is not connected.
        """
        if not broker.is_connected():
            raise ConnectionError("Broker is not connected.")

        receipts: list[dict[str, Any]] = []
        filled = 0

        for order in plan:
            signed_qty = order.quantity if order.side == "BUY" else -order.quantity

            logger.info(
                "Executing: %s %d %s — %s",
                order.side, order.quantity, order.ticker, order.reason,
            )

            receipt = broker.place_order(
                ticker=order.ticker,
                quantity=signed_qty,
                order_type=order.order_type,
                limit_price=order.limit_price,
            )
            receipts.append(receipt)

            if receipt.get("status") in ("paper_filled", "filled"):
                filled += 1

        result = ExecutionResult(
            orders_sent=len(plan),
            orders_filled=filled,
            receipts=receipts,
        )

        logger.info(
            "Execution complete: %d sent, %d filled",
            result.orders_sent,
            result.orders_filled,
        )

        return result

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    @staticmethod
    def reconcile(
        target_weights: pd.Series,
        actual_positions: dict[str, float],
        prices: dict[str, float],
        portfolio_value: float,
    ) -> pd.DataFrame:
        """Compare intended allocation against actual post-execution positions.

        Args:
            target_weights: Intended weight per ticker (summing to 1.0).
            actual_positions: Post-execution holdings {ticker: quantity}.
            prices: Current prices {ticker: price}.
            portfolio_value: Total portfolio value.

        Returns:
            DataFrame with columns: target_weight, actual_weight,
            weight_diff, value_diff.
        """
        rows = []
        for ticker in target_weights.index:
            target_w = target_weights[ticker]
            qty = actual_positions.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)
            actual_value = qty * price
            actual_w = actual_value / portfolio_value if portfolio_value > 0 else 0.0
            target_value = target_w * portfolio_value

            rows.append({
                "ticker": ticker,
                "target_weight": round(target_w, 4),
                "actual_weight": round(actual_w, 4),
                "weight_diff": round(actual_w - target_w, 4),
                "value_diff": round(actual_value - target_value, 2),
            })

        report = pd.DataFrame(rows).set_index("ticker")

        discrepancies = report[report["weight_diff"].abs() > 0.001]
        if not discrepancies.empty:
            logger.warning(
                "Reconciliation discrepancies found:\n%s",
                discrepancies.to_string(),
            )
        else:
            logger.info("Reconciliation passed — all weights within tolerance")

        return report
