"""Execution service — manages execution state for the dashboard.

Provides a high-level interface over the broker and OMS, handling plan
generation, execution, history retrieval, and position reconciliation.

Usage:
    from src.services.execution_service import ExecutionService
    svc = ExecutionService()
    svc.connect_paper_broker()
    plan = svc.generate_rebalance_plan(target_weights)
    result = svc.execute_plan(plan["plan_id"])
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from src.execution.broker import PaperBroker
from src.execution.oms import OrderManagementSystem, Order
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_EXEC_DIR = _PROJECT_ROOT / "data" / "processed" / "executions"


class ExecutionService:
    """Manages execution state for the dashboard.

    Orchestrates the broker, OMS, and optional upstream services to
    provide plan generation, execution, reconciliation, and history.

    Args:
        data_service: Optional data service for fetching current prices.
        portfolio_service: Optional portfolio service for target weights.
        config: Optional configuration dict; keys ``initial_cash``,
            ``base_currency``, ``commission_rate``, ``slippage_bps``.
    """

    def __init__(
        self,
        data_service: Any | None = None,
        portfolio_service: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.data_service = data_service
        self.portfolio_service = portfolio_service
        self.config = config or {}
        self.broker: PaperBroker | None = None
        self.oms: OrderManagementSystem | None = None

        # In-memory plan cache (plan_id -> plan dict)
        self._plans: dict[str, dict[str, Any]] = {}
        self._target_weights: dict[str, float] = {}

        self._exec_dir = Path(
            self.config.get("execution_dir", str(_DEFAULT_EXEC_DIR))
        )

    # ------------------------------------------------------------------
    # Broker status
    # ------------------------------------------------------------------
    def get_broker_status(self) -> dict[str, Any]:
        """Return current broker connection and account status.

        Returns:
            Dict with ``connected``, ``mode``, ``account_value``, ``cash``,
            ``invested``, ``positions_count``.
        """
        if self.broker is None or not self.broker.is_connected():
            return {
                "connected": False,
                "mode": None,
                "account_value": 0.0,
                "cash": 0.0,
                "invested": 0.0,
                "positions_count": 0,
            }

        summary = self.broker.get_account_summary()
        return {
            "connected": True,
            "mode": self.broker.get_mode(),
            **summary,
        }

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect_paper_broker(self) -> bool:
        """Initialise and connect a PaperBroker.

        Reads ``initial_cash``, ``base_currency``, ``commission_rate``,
        and ``slippage_bps`` from the service config if present.

        Returns:
            ``True`` on successful connection.
        """
        self.broker = PaperBroker(
            initial_cash=self.config.get("initial_cash", 100_000.0),
            base_currency=self.config.get("base_currency", "GBP"),
            commission_rate=self.config.get("commission_rate", 0.001),
            slippage_bps=self.config.get("slippage_bps", 5.0),
            execution_dir=self._exec_dir,
        )
        connected = self.broker.connect()
        if connected:
            self.oms = OrderManagementSystem(self.broker)
        logger.info("Paper broker connected: %s", connected)
        return connected

    # ------------------------------------------------------------------
    # Rebalance planning
    # ------------------------------------------------------------------
    def generate_rebalance_plan(
        self,
        target_weights: pd.Series | dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Compute a rebalance plan from current positions to target weights.

        If *target_weights* is not provided the most recently saved weights
        (from ``set_target_weights`` or the portfolio service) are used.

        Args:
            target_weights: Target allocation keyed by ticker (values 0–1).

        Returns:
            Plan dict with ``plan_id``, ``orders``, ``total_cost_estimate``,
            ``turnover``, ``timestamp``.

        Raises:
            ValueError: If the broker is not connected or no weights are
                available.
        """
        if self.broker is None or not self.broker.is_connected():
            raise ValueError("Broker is not connected")

        # Resolve target weights
        if target_weights is not None:
            if isinstance(target_weights, pd.Series):
                weights = target_weights.to_dict()
            else:
                weights = dict(target_weights)
            self._target_weights = weights
        elif self._target_weights:
            weights = self._target_weights
        elif self.portfolio_service is not None and hasattr(self.portfolio_service, "get_weights"):
            weights = self.portfolio_service.get_weights()
            self._target_weights = weights
        else:
            raise ValueError("No target weights available")

        # Ensure broker has prices for all tickers in the target
        self._ensure_prices(list(weights.keys()))

        summary = self.broker.get_account_summary()
        positions = self.broker.get_positions()
        prices = {t: self.broker._prices.get(t, 100.0) for t in
                  set(list(weights.keys()) + list(positions.keys()))}

        orders = self.oms.create_rebalance_orders(
            current_positions=positions,
            target_weights=weights,
            account_value=summary["account_value"],
            prices=prices,
        )

        # Compute cost estimate and turnover
        total_cost = sum(o.est_price * o.quantity for o in orders)
        account_val = summary["account_value"]
        turnover = total_cost / account_val if account_val > 0 else 0.0

        plan_id = uuid.uuid4().hex[:12]
        plan = {
            "plan_id": plan_id,
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "target_weights": weights,
            "orders": [
                {
                    "ticker": o.ticker,
                    "side": o.side,
                    "quantity": o.quantity,
                    "est_price": o.est_price,
                    "est_cost": round(o.est_price * o.quantity, 2),
                    "reason": o.reason,
                }
                for o in orders
            ],
            "total_cost_estimate": round(total_cost, 2),
            "turnover": round(turnover, 4),
        }

        self._plans[plan_id] = plan
        logger.info("Rebalance plan %s generated: %d orders", plan_id, len(orders))
        return plan

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------
    def execute_plan(self, plan_id: str) -> dict[str, Any]:
        """Execute a previously generated rebalance plan.

        Only paper-mode execution is permitted from the dashboard.

        Args:
            plan_id: Identifier of the plan to execute.

        Returns:
            :class:`~src.execution.oms.ExecutionReport` as a dict.

        Raises:
            ValueError: If the broker is not connected, the plan is not
                found, or the broker is in live mode.
        """
        if self.broker is None or not self.broker.is_connected():
            raise ValueError("Broker is not connected")

        if self.broker.get_mode() != "paper":
            raise ValueError("Dashboard execution is restricted to paper mode")

        plan = self._plans.get(plan_id)
        if plan is None:
            raise ValueError(f"Plan '{plan_id}' not found")

        orders = [
            Order(
                ticker=o["ticker"],
                side=o["side"],
                quantity=o["quantity"],
                est_price=o["est_price"],
                reason=o.get("reason", ""),
            )
            for o in plan["orders"]
        ]

        report = self.oms.execute_orders(orders, plan_id=plan_id)
        return report.to_dict()

    # ------------------------------------------------------------------
    # History & reconciliation
    # ------------------------------------------------------------------
    def get_execution_history(self, n: int = 20) -> list[dict[str, Any]]:
        """Load the *n* most recent execution reports from disk.

        Reports are stored as JSON in ``data/processed/executions/``.

        Args:
            n: Maximum number of reports to return (most recent first).

        Returns:
            List of execution-report dicts, newest first.
        """
        if not self._exec_dir.exists():
            return []

        files = sorted(
            self._exec_dir.glob("execution_*.json"),
            key=lambda p: p.name,
            reverse=True,
        )

        reports: list[dict[str, Any]] = []
        for fp in files[:n]:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    reports.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to read execution report: %s", fp)
        return reports

    def get_reconciliation(self) -> dict[str, Any]:
        """Compare current positions against target weights.

        Returns:
            Dict with ``tickers`` (list of per-ticker dicts with
            ``ticker``, ``target_weight``, ``actual_weight``, ``drift``),
            ``total_drift`` (sum of absolute drifts), and ``aligned``
            (``True`` if total drift < 0.02).
        """
        if self.broker is None or not self.broker.is_connected():
            return {"tickers": [], "total_drift": 0.0, "aligned": True}

        positions = self.broker.get_positions()
        summary = self.broker.get_account_summary()
        account_value = summary["account_value"]

        all_tickers = sorted(
            set(list(self._target_weights.keys()) + list(positions.keys()))
        )

        rows: list[dict[str, Any]] = []
        total_drift = 0.0

        for ticker in all_tickers:
            target_w = self._target_weights.get(ticker, 0.0)
            market_value = positions.get(ticker, {}).get("market_value", 0.0)
            actual_w = market_value / account_value if account_value > 0 else 0.0
            drift = actual_w - target_w

            rows.append({
                "ticker": ticker,
                "target_weight": round(target_w, 4),
                "actual_weight": round(actual_w, 4),
                "drift": round(drift, 4),
            })
            total_drift += abs(drift)

        return {
            "tickers": rows,
            "total_drift": round(total_drift, 4),
            "aligned": total_drift < 0.02,
        }

    # ------------------------------------------------------------------
    # Target weight management
    # ------------------------------------------------------------------
    def set_target_weights(self, weights: dict[str, float]) -> None:
        """Store target weights for future rebalance plans.

        Args:
            weights: Mapping of ticker to target weight (0–1).
        """
        self._target_weights = dict(weights)

    def set_prices(self, prices: dict[str, float]) -> None:
        """Update market prices on the broker.

        Args:
            prices: Mapping of ticker to current price.
        """
        if self.broker is not None:
            self.broker.set_prices(prices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_prices(self, tickers: list[str]) -> None:
        """Set default prices for tickers that the broker doesn't yet know about."""
        if self.broker is None:
            return

        for ticker in tickers:
            if ticker not in self.broker._prices:
                # Try data service first
                if self.data_service is not None and hasattr(self.data_service, "get_latest_price"):
                    price = self.data_service.get_latest_price(ticker)
                    if price:
                        self.broker._prices[ticker] = price
                        continue
                # Fallback: use a nominal price so orders can still be generated
                self.broker._prices[ticker] = 100.0
                logger.debug(
                    "Using default price 100.0 for %s (no data service)", ticker
                )
