"""CLI entry point for the Capital Deployment Engine.

Usage::

    py -3 -m scripts.run_deployment
    py -3 -m scripts.run_deployment --dry-run
    py -3 -m scripts.run_deployment --no-opus
    py -3 -m scripts.run_deployment --cycle-id test_001
    py -3 -m scripts.run_deployment --capital 500
    py -3 -m scripts.run_deployment --halt

Exit codes:
    0  -- Cycle completed successfully.
    1  -- Cycle failed.
    42 -- Cycle paused at AWAITING_APPROVAL (human review needed).
"""

from __future__ import annotations

import argparse
import sys

from src.deployment.config import load_deployment_config
from src.deployment.memory import MemoryManager
from src.deployment.scanner import SyntheticScanner
from src.deployment.enricher import SyntheticEnricher
from src.deployment.analyst import KeywordAnalyst, OpusAnalyst
from src.deployment.sizer import KellySizer
from src.deployment.queue import TradeQueueManager
from src.deployment.orchestrator import DeploymentPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Run a deployment cycle from the command line."""
    parser = argparse.ArgumentParser(
        description="Capital Deployment Engine -- autonomous trading pipeline",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Log orders but do not submit to IB (default: True)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Submit orders to IB (overrides --dry-run)",
    )
    parser.add_argument(
        "--no-opus", action="store_true",
        help="Use keyword analyst instead of Opus",
    )
    parser.add_argument(
        "--cycle-id", type=str, default=None,
        help="Override cycle ID",
    )
    parser.add_argument(
        "--capital", type=float, default=None,
        help="Override capital in GBP (default: query IB or config)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic scanner and enricher (offline mode)",
    )
    parser.add_argument(
        "--auto-approve", action="store_true",
        help="Auto-approve checkpoint (skip human review)",
    )
    args = parser.parse_args()

    dry_run = not args.live
    use_opus = not args.no_opus

    # Load config.
    try:
        config = load_deployment_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    # Initialise components.
    memory = MemoryManager.from_config(config)

    if args.synthetic:
        scanner = SyntheticScanner()
        enricher = SyntheticEnricher()
    else:
        # Import real implementations.
        from src.deployment.scanner import ScannerService
        from src.deployment.enricher import EnrichmentService
        scanner = ScannerService(config=config, memory=memory)
        enricher = EnrichmentService(config=config)

    if use_opus:
        analyst = OpusAnalyst(config=config, memory=memory)
    else:
        analyst = KeywordAnalyst(config=config)

    sizer = KellySizer(config=config)
    queue = TradeQueueManager(config=config, memory=memory)

    pipeline = DeploymentPipeline(
        config=config,
        memory=memory,
        scanner=scanner,
        enricher=enricher,
        analyst=analyst,
        sizer=sizer,
        queue_manager=queue,
    )

    # Auto-approve if requested.
    if args.auto_approve:
        import threading
        def auto_approve():
            """Wait for AWAITING_APPROVAL then approve."""
            while pipeline.state != "AWAITING_APPROVAL":
                if pipeline.state in ("COMPLETE", "HALTED"):
                    return
                import time
                time.sleep(0.5)
            pipeline.approve_checkpoint()

        threading.Thread(target=auto_approve, daemon=True).start()

    # Run cycle.
    logger.info(
        "Starting deployment cycle (dry_run=%s, opus=%s, synthetic=%s)",
        dry_run, use_opus, args.synthetic,
    )

    result = pipeline.run_cycle(
        cycle_id=args.cycle_id,
        capital_gbp=args.capital,
        dry_run=dry_run,
        use_opus=use_opus,
    )

    if result is None:
        logger.error("Pipeline returned None")
        return 1

    # Report results.
    logger.info("Cycle %s: %s", result.cycle_id, result.status)
    logger.info(
        "  Scanned=%d, Enriched=%d, Scored=%d, Above threshold=%d, "
        "Queued=%d",
        result.candidates_scanned,
        result.candidates_enriched,
        result.candidates_scored,
        result.candidates_above_threshold,
        result.trades_queued,
    )
    logger.info("  Duration: %.1fs", result.duration_seconds)

    if result.errors:
        logger.warning("  Errors: %d", len(result.errors))
        for err in result.errors:
            logger.warning("    %s", err)

    if result.status == "AWAITING_APPROVAL":
        return 42
    elif result.status == "COMPLETE":
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
