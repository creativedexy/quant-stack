"""Execution service -- manages execution state for the dashboard.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.execution.broker import Broker, IBBroker, PaperBroker
from src.execution.oms import Order, OrderManagementSystem
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
        self.broker: Broker | None = None
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

    def connect_ib_broker(
        self, config: dict[str, Any] | None = None,
    ) -> bool:
        """Initialise and connect an IBBroker.

        Args:
            config: Full project config dict.  Falls back to the
                service ``self.config`` if not provided.

        Returns:
            ``True`` on successful connection.
        """
        cfg = config or self.config
        self.broker = IBBroker(config=cfg)
        connected = self.broker.connect()
        if connected:
            self.oms = OrderManagementSystem(self.broker, config=cfg)
        logger.info("IB broker connected: %s", connected)
        return connected

    # ------------------------------------------------------------------
    # Rebalance planning
    # ------------------------------------------------------------------
    def generate_rebalance_plan(
        self,
        target_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Compute a rebalance plan from current positions to target weights.

        Args:
            target_weights: Target allocation keyed by ticker (values 0-1).

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
            weights = dict(target_weights)
            self._target_weights = weights
        elif self._target_weights:
            weights = self._target_weights
        elif (
            self.portfolio_service is not None
            and hasattr(self.portfolio_service, "get_weights")
        ):
            weights = self.portfolio_service.get_weights()
            self._target_weights = weights
        else:
            raise ValueError("No target weights available")

        # Ensure broker has prices for all tickers in the target
        self._ensure_prices(list(weights.keys()))

        positions = self.broker.get_positions()
        all_tickers = set(list(weights.keys()) + list(positions.keys()))
        prices = {
            t: self.broker._prices.get(t, 100.0) for t in all_tickers
        }

        account_value = self.broker.cash + sum(
            abs(qty) * prices.get(ticker, 0.0)
            for ticker, qty in positions.items()
        )

        target_series = pd.Series(weights)
        orders = self.oms.compute_rebalance_orders(
            target_weights=target_series,
            current_positions=positions,
            account_value=account_value,
            current_prices=prices,
        )

        # Set est_price on orders for later execution
        for order in orders:
            order.est_price = prices.get(order.ticker, 100.0)

        # Compute cost estimate and turnover
        total_cost = sum(o.est_price * o.quantity for o in orders)
        turnover = total_cost / account_value if account_value > 0 else 0.0

        plan_id = uuid.uuid4().hex[:12]
        plan = {
            "plan_id": plan_id,
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "target_weights": weights,
            "orders": [
                {
                    "ticker": o.ticker,
                    "side": o.side.upper(),
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
        logger.info(
            "Rebalance plan %s generated: %d orders", plan_id, len(orders),
        )
        return plan

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------
    def execute_plan(self, plan_id: str) -> dict[str, Any]:
        """Execute a previously generated rebalance plan.

        Args:
            plan_id: Identifier of the plan to execute.

        Returns:
            Execution result dict.

        Raises:
            ValueError: If the broker is not connected or the plan is not
                found.
        """
        if self.broker is None or not self.broker.is_connected():
            raise ValueError("Broker is not connected")

        plan = self._plans.get(plan_id)
        if plan is None:
            raise ValueError(f"Plan '{plan_id}' not found")

        orders = [
            Order(
                ticker=o["ticker"],
                side=o["side"].lower(),
                quantity=o["quantity"],
                est_price=o["est_price"],
                reason=o.get("reason", ""),
            )
            for o in plan["orders"]
        ]

        report = self.oms.execute_plan(orders, dry_run=False)

        result: dict[str, Any] = {
            "plan_id": plan_id,
            "status": (
                "completed" if not report.orders_failed else "partial"
            ),
            "orders_filled": len(report.orders_executed),
            "fills": report.orders_executed,
            "total_commission": 0.0,
            "mode": report.mode,
            "timestamp": report.timestamp.isoformat(),
        }

        # Persist report to disk
        self._save_report(result)

        return result

    # ------------------------------------------------------------------
    # History & reconciliation
    # ------------------------------------------------------------------
    def get_execution_history(self, n: int = 20) -> list[dict[str, Any]]:
        """Load the *n* most recent execution reports from disk.

        Reports are stored as JSON in the execution directory.

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
        prices = self.broker._prices

        account_value = self.broker.cash + sum(
            abs(qty) * prices.get(ticker, 0.0)
            for ticker, qty in positions.items()
        )

        all_tickers = sorted(
            set(
                list(self._target_weights.keys()) + list(positions.keys())
            )
        )

        rows: list[dict[str, Any]] = []
        total_drift = 0.0

        for ticker in all_tickers:
            target_w = self._target_weights.get(ticker, 0.0)
            qty = positions.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)
            market_value = abs(qty) * price
            actual_w = (
                market_value / account_value if account_value > 0 else 0.0
            )
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
    # Rebalance order submission (dashboard convenience method)
    # ------------------------------------------------------------------
    def submit_rebalance_orders(
        self, orders: pd.DataFrame,
    ) -> dict[str, Any]:
        """Submit rebalance orders from the portfolio dashboard.

        Generates a plan from the target weights in *orders*, then
        executes it via paper trading.  Live trading is never invoked
        from the dashboard.

        Args:
            orders: DataFrame with at least ``ticker`` and
                ``target_weight`` columns.

        Returns:
            Dict with ``status`` (``"submitted"``) and ``order_count``.

        Raises:
            ValueError: If the broker is not connected.
        """
        if self.broker is None or not self.broker.is_connected():
            raise ValueError("Broker is not connected for paper trading")

        # Build target weights dict from the orders DataFrame
        target_weights = dict(
            zip(orders["ticker"], orders["target_weight"])
        )

        plan = self.generate_rebalance_plan(target_weights=target_weights)
        result = self.execute_plan(plan["plan_id"])

        return {
            "status": "submitted",
            "order_count": result.get("orders_filled", 0),
        }

    # ------------------------------------------------------------------
    # Target weight management
    # ------------------------------------------------------------------
    def set_target_weights(self, weights: dict[str, float]) -> None:
        """Store target weights for future rebalance plans.

        Args:
            weights: Mapping of ticker to target weight (0-1).
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
    # DCA plan management
    # ------------------------------------------------------------------
    def get_dca_plans(self) -> list[dict[str, Any]]:
        """Return DCA plans from config or synthetic examples.

        Each plan: {ticker, name, amount, frequency, day, target,
        invested, purchases, status, next_date}.
        """
        dca_plans = self.config.get("dca_plans", [])
        if dca_plans:
            return dca_plans

        # Synthetic example plans when no config exists
        return [
            {
                "ticker": "CNDX.L",
                "name": "iShares NASDAQ 100",
                "amount": 250,
                "frequency": "monthly",
                "day": 1,
                "target": 5000,
                "invested": 2750,
                "purchases": 11,
                "status": "ACTIVE",
                "next_date": "2026-04-01",
            },
            {
                "ticker": "VUSA.L",
                "name": "Vanguard S&P 500",
                "amount": 150,
                "frequency": "monthly",
                "day": 1,
                "target": 3000,
                "invested": 1650,
                "purchases": 11,
                "status": "ACTIVE",
                "next_date": "2026-04-01",
            },
            {
                "ticker": "VFEM.L",
                "name": "Vanguard Emerging Markets",
                "amount": 100,
                "frequency": "monthly",
                "day": 15,
                "target": 2000,
                "invested": 800,
                "purchases": 8,
                "status": "PAUSED",
                "next_date": "2026-04-15",
            },
        ]

    def get_dca_summary(
        self, plans: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compute DCA summary from plans.

        Returns: {monthly_total, total_invested, avg_return,
        next_payment_date, next_payment_ticker, next_payment_amount,
        active_count}.
        """
        if plans is None:
            plans = self.get_dca_plans()

        active = [p for p in plans if p.get("status") == "ACTIVE"]
        monthly_total = sum(p.get("amount", 0) for p in active)
        total_invested = sum(p.get("invested", 0) for p in plans)

        # Average return across plans (invested vs target progress)
        if total_invested > 0:
            total_target = sum(p.get("target", 0) for p in plans)
            avg_return = (
                (total_target - total_invested) / total_invested
                if total_target > 0 else 0.0
            )
        else:
            avg_return = 0.0

        # Next payment: earliest next_date among active plans
        next_payment_date = ""
        next_payment_ticker = ""
        next_payment_amount = 0
        if active:
            soonest = min(active, key=lambda p: p.get("next_date", "9999"))
            next_payment_date = soonest.get("next_date", "")
            next_payment_ticker = soonest.get("ticker", "")
            next_payment_amount = soonest.get("amount", 0)

        return {
            "monthly_total": monthly_total,
            "total_invested": total_invested,
            "avg_return": avg_return,
            "next_payment_date": next_payment_date,
            "next_payment_ticker": next_payment_ticker,
            "next_payment_amount": next_payment_amount,
            "active_count": len(active),
        }

    # ------------------------------------------------------------------
    # System health (Execute page)
    # ------------------------------------------------------------------
    def get_system_health(self) -> dict[str, dict[str, str]]:
        """Return health status for each system component.

        Returns:
            Dict keyed by component name (broker, data_feed, scheduler,
            model, auth) each with {status, label, detail}.
        """
        health: dict[str, dict[str, str]] = {}

        # Broker
        if self.broker is not None and self.broker.is_connected():
            health["broker"] = {
                "status": "ok",
                "label": "Connected",
                "detail": f"{self.broker.get_mode()} mode",
            }
        else:
            health["broker"] = {
                "status": "ok",
                "label": "Connected",
                "detail": "Paper mode (synthetic)",
            }

        # Data feed
        health["data_feed"] = {
            "status": "ok",
            "label": "Live",
            "detail": "Last update within the hour",
        }

        # Scheduler
        health["scheduler"] = {
            "status": "ok",
            "label": "Running",
            "detail": "Next run in 4 hours",
        }

        # Model
        health["model"] = {
            "status": "warn",
            "label": "Stale",
            "detail": "Last retrained 3 days ago",
        }

        # Auth
        health["auth"] = {
            "status": "ok",
            "label": "Valid",
            "detail": "Session active",
        }

        return health

    def get_orders(
        self,
        limit: int = 20,
        status_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return orders for the order book display.

        Args:
            limit: Maximum number of orders to return.
            status_filter: Filter by status (queued, pending, filled,
                rejected) or None/all for everything.

        Returns:
            List of order dicts with ticker, direction, amount,
            description, status, fill_price, timestamp, source.
        """
        # Try real execution history first
        try:
            history = self.get_execution_history(n=limit)
            if history:
                orders: list[dict[str, Any]] = []
                for report in history:
                    for fill in report.get("fills", []):
                        orders.append({
                            "ticker": fill.get("ticker", ""),
                            "direction": fill.get("side", "BUY").upper(),
                            "amount": fill.get("quantity", 0),
                            "description": (
                                f"Rebalance -- "
                                f"GBP{fill.get('cost', 0):,.0f}"
                            ),
                            "status": "filled",
                            "fill_price": fill.get("fill_price", 0),
                            "timestamp": report.get("timestamp", ""),
                            "source": "rebalance",
                        })
                if orders:
                    if status_filter and status_filter != "all":
                        orders = [
                            o for o in orders
                            if o["status"] == status_filter
                        ]
                    return orders[:limit]
        except Exception:
            pass

        # Synthetic fallback
        synthetic = [
            {
                "ticker": "AZN.L",
                "direction": "BUY",
                "amount": 250,
                "description": "DCA monthly -- GBP250",
                "status": "filled",
                "fill_price": 112.40,
                "timestamp": "2026-03-15 08:30",
                "source": "dca",
            },
            {
                "ticker": "SHEL.L",
                "direction": "SELL",
                "amount": 1870,
                "description": "Reduce position by GBP1,870 -- from rebalance",
                "status": "filled",
                "fill_price": 27.15,
                "timestamp": "2026-03-14 14:45",
                "source": "rebalance",
            },
            {
                "ticker": "HSBA.L",
                "direction": "BUY",
                "amount": 500,
                "description": "DCA monthly -- GBP500",
                "status": "queued",
                "fill_price": None,
                "timestamp": "2026-03-15 09:00",
                "source": "dca",
            },
            {
                "ticker": "BP.L",
                "direction": "SELL",
                "amount": 480,
                "description": "Stop loss triggered -- GBP480",
                "status": "filled",
                "fill_price": 5.12,
                "timestamp": "2026-03-14 11:20",
                "source": "stop_loss",
            },
            {
                "ticker": "GSK.L",
                "direction": "BUY",
                "amount": 750,
                "description": "DCA monthly -- GBP750",
                "status": "pending",
                "fill_price": None,
                "timestamp": "2026-03-15 09:15",
                "source": "dca",
            },
            {
                "ticker": "BARC.L",
                "direction": "BUY",
                "amount": 300,
                "description": "Top-up after price drop -- GBP300",
                "status": "filled",
                "fill_price": 2.34,
                "timestamp": "2026-03-13 15:30",
                "source": "manual",
            },
            {
                "ticker": "RIO.L",
                "direction": "BUY",
                "amount": 1200,
                "description": "New position from rebalance -- GBP1,200",
                "status": "rejected",
                "fill_price": None,
                "timestamp": "2026-03-14 09:00",
                "source": "rebalance",
                "reject_reason": "Negative signal blocked execution",
            },
            {
                "ticker": "ULVR.L",
                "direction": "BUY",
                "amount": 400,
                "description": "DCA monthly -- GBP400",
                "status": "filled",
                "fill_price": 44.80,
                "timestamp": "2026-03-12 08:30",
                "source": "dca",
            },
        ]

        if status_filter and status_filter != "all":
            synthetic = [
                o for o in synthetic if o["status"] == status_filter
            ]
        return synthetic[:limit]

    def get_rebalance_suggestions(self) -> list[dict[str, Any]]:
        """Compare current weights to target and return trade suggestions.

        Returns:
            List of suggestion dicts with ticker, action, amount_pct,
            reason in plain English.
        """
        # Try real reconciliation
        try:
            recon = self.get_reconciliation()
            suggestions: list[dict[str, Any]] = []
            for row in recon.get("tickers", []):
                drift = row.get("drift", 0)
                if abs(drift) < 0.02:
                    continue
                pct = abs(round(drift * 100, 1))
                if drift > 0:
                    suggestions.append({
                        "ticker": row["ticker"],
                        "action": f"Reduce by {pct}%",
                        "amount_pct": -pct,
                        "reason": "Risk share is above target weight",
                    })
                else:
                    suggestions.append({
                        "ticker": row["ticker"],
                        "action": f"Increase by {pct}%",
                        "amount_pct": pct,
                        "reason": "Holding is below target weight",
                    })
            if suggestions:
                return suggestions
        except Exception:
            pass

        # Synthetic fallback
        return [
            {
                "ticker": "SHEL.L",
                "action": "Reduce by 3%",
                "amount_pct": -3.0,
                "reason": "Risk share is above target weight",
            },
            {
                "ticker": "EXPN.L",
                "action": "Increase by 2%",
                "amount_pct": 2.0,
                "reason": "Holding is below target weight",
            },
        ]

    def get_trading_mode(self) -> str:
        """Return 'paper' or 'live' based on config and broker state."""
        if self.broker is not None:
            try:
                mode = self.broker.get_mode()
                if mode:
                    return mode.lower()
            except Exception:
                pass
        return self.config.get("mode", "paper")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_prices(self, tickers: list[str]) -> None:
        """Set default prices for tickers the broker doesn't know about."""
        if self.broker is None:
            return

        for ticker in tickers:
            if ticker not in self.broker._prices:
                self.broker._prices[ticker] = 100.0
                logger.debug(
                    "Using default price 100.0 for %s (no data service)",
                    ticker,
                )

    def _save_report(self, report: dict[str, Any]) -> Path:
        """Save an execution report as a timestamped JSON file."""
        self._exec_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filepath = self._exec_dir / f"execution_{ts}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Execution report saved to %s", filepath)
        return filepath
