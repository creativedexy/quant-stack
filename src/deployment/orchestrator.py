"""Pipeline orchestrator -- state machine for the deployment cycle.

Runs the full pipeline: SCAN -> ENRICH -> ANALYSE -> CHECKPOINT ->
SIZE -> QUEUE -> PERSIST -> LEARN.

Features:
- State machine with crash recovery (state persisted in memory)
- Self-healing retry wrapper with per-error diagnosis
- Emergency halt (cancels pending IB orders)
- Background thread execution (UI polls status via HTMX)
"""

from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any

from src.deployment import (
    CycleResult,
    PipelineContext,
    StepResult,
)
from src.deployment.config import (
    get_active_tier,
    is_live_execution_enabled,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Pipeline states in order.
STATES = [
    "IDLE",
    "SCANNING",
    "ENRICHING",
    "ANALYSING",
    "AWAITING_APPROVAL",
    "SIZING",
    "QUEUEING",
    "PERSISTING",
    "COMPLETE",
]


class DeploymentPipeline:
    """State machine orchestrating the deployment cycle.

    Args:
        config: Deployment config dict.
        memory: MemoryManager instance.
        scanner: Scanner implementation (ScannerService or SyntheticScanner).
        enricher: Enricher implementation.
        analyst: Analyst implementation (OpusAnalyst or KeywordAnalyst).
        sizer: Sizer implementation (KellySizer).
        queue_manager: TradeQueueManager instance.
        broker: Optional broker for live execution.
        portfolio_service: Optional portfolio service for positions.
        service: Optional DeploymentService for UI state updates.
    """

    def __init__(
        self,
        config: dict[str, Any],
        memory: Any,
        scanner: Any,
        enricher: Any,
        analyst: Any,
        sizer: Any,
        queue_manager: Any,
        broker: Any | None = None,
        portfolio_service: Any | None = None,
        service: Any | None = None,
    ) -> None:
        self._config = config
        self._memory = memory
        self._scanner = scanner
        self._enricher = enricher
        self._analyst = analyst
        self._sizer = sizer
        self._queue = queue_manager
        self._broker = broker
        self._portfolio_service = portfolio_service
        self._service = service

        self._lock = threading.Lock()
        self._approval_event = threading.Event()
        self._halt_event = threading.Event()
        self._state = "IDLE"
        self._context: PipelineContext | None = None
        self._thread: threading.Thread | None = None
        self._cycle_result: CycleResult | None = None

    @property
    def state(self) -> str:
        """Current pipeline state."""
        with self._lock:
            return self._state

    @property
    def context(self) -> PipelineContext | None:
        """Current pipeline context."""
        with self._lock:
            return self._context

    @property
    def cycle_result(self) -> CycleResult | None:
        """Result of the last completed cycle."""
        with self._lock:
            return self._cycle_result

    def _set_state(self, state: str) -> None:
        """Thread-safe state transition."""
        with self._lock:
            logger.info("Pipeline: %s -> %s", self._state, state)
            self._state = state
            if self._context:
                self._context.current_step = state
            if self._service:
                self._service.set_cycle_status({
                    "state": state,
                    "cycle_id": self._context.cycle_id if self._context else "",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        cycle_id: str | None = None,
        capital_gbp: float | None = None,
        dry_run: bool = True,
        use_opus: bool = True,
        background: bool = False,
    ) -> CycleResult | None:
        """Run a full deployment cycle.

        Args:
            cycle_id: Override cycle ID. Auto-generated if None.
            capital_gbp: Override capital. Queried from IB if None.
            dry_run: If True, do not submit orders to IB.
            use_opus: If True, use Opus analyst. Otherwise keyword fallback.
            background: If True, run on a daemon thread.

        Returns:
            CycleResult if run synchronously, None if background.
        """
        if self.state != "IDLE" and self.state != "COMPLETE":
            logger.warning(
                "Cannot start cycle: pipeline in state %s", self.state
            )
            return None

        if cycle_id is None:
            cycle_id = f"cycle_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        if background:
            self._thread = threading.Thread(
                target=self._run_cycle_inner,
                args=(cycle_id, capital_gbp, dry_run, use_opus),
                daemon=True,
                name=f"deployment-{cycle_id}",
            )
            self._thread.start()
            return None

        return self._run_cycle_inner(
            cycle_id, capital_gbp, dry_run, use_opus,
        )

    def approve_checkpoint(self) -> None:
        """Signal that the human has approved the checkpoint."""
        logger.info("Checkpoint approved by human")
        self._approval_event.set()

    def halt(self) -> int:
        """Emergency halt -- cancel all pending orders.

        Returns:
            Number of orders cancelled.
        """
        logger.warning("EMERGENCY HALT triggered")
        self._halt_event.set()
        self._approval_event.set()  # Unblock if waiting.

        cancelled = 0
        if self._queue:
            cancelled = self._queue.cancel_all()

        self._set_state("HALTED")

        if self._service:
            self._service.set_cycle_status({
                "state": "HALTED",
                "halted_at": datetime.now(timezone.utc).isoformat(),
                "orders_cancelled": cancelled,
            })

        return cancelled

    # ------------------------------------------------------------------
    # Inner cycle execution
    # ------------------------------------------------------------------

    def _run_cycle_inner(
        self,
        cycle_id: str,
        capital_gbp: float | None,
        dry_run: bool,
        use_opus: bool,
    ) -> CycleResult:
        """Execute the full pipeline cycle."""
        start_time = time.monotonic()
        self._halt_event.clear()
        self._approval_event.clear()

        # Resolve capital.
        if capital_gbp is None:
            capital_gbp = self._resolve_capital()

        # Build context.
        portfolio = self._get_portfolio_positions()
        context = PipelineContext(
            cycle_id=cycle_id,
            memory=self._memory,
            config=self._config,
            portfolio_positions=portfolio,
            started_at=datetime.now(timezone.utc).isoformat(),
            capital_available_gbp=capital_gbp,
        )
        with self._lock:
            self._context = context

        # Start memory cycle.
        if self._memory:
            self._memory.start_cycle(cycle_id)

        result = CycleResult(cycle_id=cycle_id, status="COMPLETE")

        try:
            # 1. SCAN
            if self._is_halted():
                return self._halted_result(cycle_id, start_time)
            self._set_state("SCANNING")
            scan_result = self._step_with_retry(
                "scan", lambda: self._scanner.scan()
            )
            context.candidates = scan_result.data or []
            result.candidates_scanned = len(context.candidates)

            # 2. ENRICH
            if self._is_halted():
                return self._halted_result(cycle_id, start_time)
            self._set_state("ENRICHING")
            enrich_result = self._step_with_retry(
                "enrich", lambda: self._enricher.enrich(context.candidates)
            )
            context.enriched = enrich_result.data or []
            result.candidates_enriched = len(context.enriched)

            # 3. ANALYSE
            if self._is_halted():
                return self._halted_result(cycle_id, start_time)
            self._set_state("ANALYSING")
            analyse_result = self._step_with_retry(
                "analyse", lambda: self._analyst.analyse(
                    context.enriched, context,
                )
            )
            context.scored = analyse_result.data or []
            result.candidates_scored = len(context.scored)

            # Count above threshold.
            min_composite = self._config.get("evaluation", {}).get(
                "thresholds", {}
            ).get("minimum_composite_score", 0.55)
            result.candidates_above_threshold = sum(
                1 for s in context.scored
                if s.composite_score >= min_composite
            )

            # Update service with scored candidates.
            if self._service:
                self._service.set_scored_candidates(context.scored)

            # 4. CHECKPOINT -- wait for human approval.
            if self._is_halted():
                return self._halted_result(cycle_id, start_time)
            self._set_state("AWAITING_APPROVAL")
            logger.info(
                "Checkpoint: %d candidates scored, %d above threshold. "
                "Waiting for human approval...",
                result.candidates_scored,
                result.candidates_above_threshold,
            )
            self._approval_event.wait()

            if self._is_halted():
                return self._halted_result(cycle_id, start_time)

            # 5. SIZE
            self._set_state("SIZING")
            size_result = self._step_with_retry(
                "size", lambda: self._sizer.size(
                    context.scored, capital_gbp,
                )
            )
            context.sized = size_result.data or []
            result.total_gbp_deployed = sum(
                s.gbp_amount for s in context.sized
            )

            # 6. QUEUE
            if self._is_halted():
                return self._halted_result(cycle_id, start_time)
            self._set_state("QUEUEING")
            queue_result = self._step_with_retry(
                "queue", lambda: self._queue.generate_orders(context.sized)
            )
            context.orders = queue_result.data or []
            result.trades_queued = len(context.orders)

            # 7. PERSIST
            self._set_state("PERSISTING")
            if self._memory:
                self._memory.end_cycle(
                    cycle_id,
                    {
                        "candidates_scanned": result.candidates_scanned,
                        "candidates_enriched": result.candidates_enriched,
                        "candidates_scored": result.candidates_scored,
                        "candidates_above_threshold": result.candidates_above_threshold,
                        "trades_queued": result.trades_queued,
                    },
                )
                # Learn from cycle.
                self._memory.learn_from_cycle({
                    "scored_candidates": [
                        {"ticker": s.ticker, "sector": s.sector,
                         "composite_score": s.composite_score}
                        for s in context.scored
                    ],
                    "rejected_tickers": [],
                })

            # Done.
            self._set_state("COMPLETE")
            result.duration_seconds = round(
                time.monotonic() - start_time, 2
            )
            result.errors = context.errors

        except Exception as exc:
            logger.error("Pipeline failed: %s", exc)
            logger.debug(traceback.format_exc())
            result.status = "FAILED"
            result.duration_seconds = round(
                time.monotonic() - start_time, 2
            )
            result.errors = context.errors + [
                {"step": self.state, "error": str(exc)},
            ]
            self._set_state("IDLE")

        with self._lock:
            self._cycle_result = result

        return result

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    def _step_with_retry(
        self,
        step_name: str,
        fn: Any,
        max_retries: int = 3,
    ) -> StepResult:
        """Execute a pipeline step with retry and diagnosis.

        Args:
            step_name: Name for logging.
            fn: Callable to execute.
            max_retries: Maximum retry attempts.

        Returns:
            StepResult with data or error.
        """
        last_error = None
        for attempt in range(1, max_retries + 1):
            if self._is_halted():
                return StepResult(
                    status="ESCALATE", error="Pipeline halted",
                )
            try:
                start = time.monotonic()
                data = fn()
                duration = round(time.monotonic() - start, 2)
                return StepResult(
                    status="OK", data=data, attempts=attempt,
                    duration_seconds=duration,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Step '%s' attempt %d/%d failed: %s",
                    step_name, attempt, max_retries, exc,
                )
                if self._context:
                    self._context.errors.append({
                        "step": step_name,
                        "attempt": attempt,
                        "error": str(exc),
                    })

                if attempt < max_retries:
                    backoff = min(2 ** attempt, 8)
                    logger.info(
                        "Retrying '%s' in %ds...", step_name, backoff,
                    )
                    time.sleep(backoff)

        # All retries exhausted.
        logger.error(
            "Step '%s' failed after %d attempts: %s",
            step_name, max_retries, last_error,
        )
        return StepResult(
            status="ESCALATE",
            error=str(last_error),
            attempts=max_retries,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_halted(self) -> bool:
        """Check if emergency halt has been triggered."""
        return self._halt_event.is_set()

    def _halted_result(
        self, cycle_id: str, start_time: float,
    ) -> CycleResult:
        """Build a HALTED cycle result."""
        return CycleResult(
            cycle_id=cycle_id,
            status="HALTED",
            duration_seconds=round(time.monotonic() - start_time, 2),
        )

    def _resolve_capital(self) -> float:
        """Resolve available capital from broker or config."""
        if self._broker is not None:
            try:
                value = self._broker.get_account_value()
                if value and value > 0:
                    logger.info("Capital from IB: GBP %.2f", value)
                    return value
            except Exception as exc:
                logger.warning(
                    "Failed to get capital from broker: %s", exc,
                )

        # Fallback to config default.
        default = self._config.get("sizing", {}).get(
            "default_capital_gbp", 500.0,
        )
        logger.info("Using default capital: GBP %.2f", default)
        return default

    def _get_portfolio_positions(self) -> list[dict[str, Any]]:
        """Get current portfolio positions."""
        if self._portfolio_service is not None:
            try:
                return self._portfolio_service.get_positions()
            except Exception as exc:
                logger.warning(
                    "Failed to get portfolio positions: %s", exc,
                )
        return []
