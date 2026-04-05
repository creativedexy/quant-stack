"""Deployment service -- UI-facing service layer for the Capital Deployment Engine.

Wraps all deployment module interactions so that the web UI never imports
from src/deployment/ directly.  Follows the existing service layer pattern
(lazy init, returns plain dicts).

Usage::

    from src.services.deployment_service import DeploymentService
    svc = DeploymentService()
    scored = svc.get_scored_candidates()
    svc.approve_order(order_id)
    svc.confirm_and_submit(dry_run=False)
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DeploymentService:
    """UI-facing service for the Capital Deployment Engine.

    Args:
        config: Deployment config dict.  If None, loads from default path.
        broker: Optional broker instance for live execution.
        portfolio_service: Optional portfolio service for held positions.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        broker: Any | None = None,
        portfolio_service: Any | None = None,
    ) -> None:
        self._config = config
        self._broker = broker
        self._portfolio_service = portfolio_service

        # Lazy-initialised components.
        self._memory = None
        self._scanner = None
        self._enricher = None
        self._analyst = None
        self._sizer = None
        self._queue = None
        self._pipeline = None

        # Cached pipeline state for UI.
        self._last_scored: list[dict[str, Any]] = []
        self._last_sized: list[dict[str, Any]] = []
        self._cycle_status: dict[str, Any] = {"state": "IDLE"}

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_config(self) -> dict[str, Any]:
        """Load config if not already loaded."""
        if self._config is None:
            from src.deployment.config import load_deployment_config
            self._config = load_deployment_config()
        return self._config

    def _ensure_memory(self) -> Any:
        """Initialise memory manager if needed."""
        if self._memory is None:
            from src.deployment.memory import MemoryManager
            cfg = self._ensure_config()
            self._memory = MemoryManager.from_config(cfg)
        return self._memory

    def _ensure_queue(self) -> Any:
        """Initialise trade queue if needed."""
        if self._queue is None:
            from src.deployment.queue import TradeQueueManager
            cfg = self._ensure_config()
            self._queue = TradeQueueManager(
                config=cfg, broker=self._broker, memory=self._memory,
            )
        return self._queue

    # ------------------------------------------------------------------
    # Scored candidates
    # ------------------------------------------------------------------

    def get_scored_candidates(self) -> list[dict[str, Any]]:
        """Return the last set of scored candidates as plain dicts."""
        return list(self._last_scored)

    def set_scored_candidates(self, scored: list[Any]) -> None:
        """Cache scored candidates from the pipeline (called by orchestrator)."""
        self._last_scored = [
            asdict(s) if hasattr(s, "__dataclass_fields__") else s
            for s in scored
        ]

    # ------------------------------------------------------------------
    # Trade queue
    # ------------------------------------------------------------------

    def get_trade_queue(self) -> list[dict[str, Any]]:
        """Return all orders in the current queue."""
        queue = self._ensure_queue()
        return [
            asdict(o) if hasattr(o, "__dataclass_fields__") else o
            for o in queue.get_all_orders()
        ]

    def get_queue_summary(self) -> dict[str, Any]:
        """Return queue summary (counts, total GBP)."""
        queue = self._ensure_queue()
        return queue.get_queue_summary()

    def approve_order(self, order_id: str) -> bool:
        """Approve a single order."""
        queue = self._ensure_queue()
        return queue.approve_order(order_id)

    def reject_order(self, order_id: str, reason: str = "") -> bool:
        """Reject a single order."""
        queue = self._ensure_queue()
        return queue.reject_order(order_id, reason)

    def approve_all_orders(self) -> int:
        """Approve all pending orders."""
        queue = self._ensure_queue()
        return queue.approve_all()

    def confirm_and_submit(self, dry_run: bool = True) -> list[dict[str, Any]]:
        """Submit approved orders to the broker.

        Args:
            dry_run: If True (default), log but do not execute.

        Returns:
            List of order dicts with updated statuses.
        """
        queue = self._ensure_queue()
        results = queue.submit_orders(dry_run=dry_run)
        return [
            asdict(o) if hasattr(o, "__dataclass_fields__") else o
            for o in results
        ]

    def emergency_stop(self) -> dict[str, Any]:
        """Cancel all pending/approved orders immediately."""
        queue = self._ensure_queue()
        cancelled = queue.cancel_all()
        self._cycle_status = {
            "state": "HALTED",
            "halted_at": datetime.now(timezone.utc).isoformat(),
            "orders_cancelled": cancelled,
        }
        logger.warning("Emergency stop: cancelled %d orders", cancelled)
        return self._cycle_status

    # ------------------------------------------------------------------
    # Cycle status
    # ------------------------------------------------------------------

    def get_cycle_status(self) -> dict[str, Any]:
        """Return the current pipeline cycle status."""
        return dict(self._cycle_status)

    def set_cycle_status(self, status: dict[str, Any]) -> None:
        """Update cycle status (called by orchestrator)."""
        self._cycle_status = status

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def get_memory_summary(self) -> dict[str, Any]:
        """Return memory state summary for the UI."""
        try:
            memory = self._ensure_memory()
            state = memory.load()
            cost = memory.get_cost_summary()
            learnings = memory.get_learnings()
            return {
                "total_cycles": state.get("total_cycles", 0),
                "last_cycle_id": state.get("last_cycle_id", ""),
                "watchlist": learnings.get("watchlist", []),
                "excluded_tickers": [
                    t["ticker"]
                    for t in state.get("rejected_tickers", [])
                    if not memory._is_cooldown_expired(t)
                ],
                "cost_summary": cost,
                "learnings_count": len(
                    learnings.get("sector_trends", {})
                ),
            }
        except Exception as exc:
            logger.error("Failed to load memory summary: %s", exc)
            return {
                "total_cycles": 0,
                "last_cycle_id": "",
                "watchlist": [],
                "excluded_tickers": [],
                "cost_summary": {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "estimated_gbp_total": 0.0,
                    "cost_per_cycle_avg": 0.0,
                },
                "learnings_count": 0,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # IB integration
    # ------------------------------------------------------------------

    def get_ib_buying_power(self) -> float | None:
        """Query IB for available buying power.

        Returns:
            Buying power in GBP, or None if broker unavailable.
        """
        if self._broker is None:
            return None
        try:
            return self._broker.get_account_value()
        except Exception as exc:
            logger.error("Failed to get buying power: %s", exc)
            return None

    def get_portfolio_positions(self) -> list[dict[str, Any]]:
        """Return current portfolio positions."""
        if self._portfolio_service is not None:
            try:
                return self._portfolio_service.get_positions()
            except Exception as exc:
                logger.error("Failed to get positions: %s", exc)
        return []
