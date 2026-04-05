#!/usr/bin/env python3
"""CLI for manual pipeline runs and scheduler management.

Usage:
    python -m scripts.run_pipeline                  # Run daily pipeline once
    python -m scripts.run_pipeline --schedule       # Start scheduler daemon
    python -m scripts.run_pipeline --retrain        # Retrain models
    python -m scripts.run_pipeline --rebalance      # Check and compute rebalance
    python -m scripts.run_pipeline --status         # Show scheduler status
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the quant-stack data pipeline",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--schedule",
        action="store_true",
        help="Start the scheduler daemon (runs continuously)",
    )
    group.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain models on latest data",
    )
    group.add_argument(
        "--rebalance",
        action="store_true",
        help="Check if rebalancing is needed and compute new weights",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Show last pipeline run status",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: config/settings.yaml)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.schedule:
        _run_scheduler(config)
    elif args.retrain:
        _run_retrain(config)
    elif args.rebalance:
        _run_rebalance(config)
    elif args.status:
        _show_status(config)
    else:
        _run_daily(config)


def _run_daily(config: dict) -> None:
    """Run the daily pipeline once and exit."""
    from src.scheduler.pipeline import PipelineRunner

    runner = PipelineRunner(config)
    result = runner.run_daily()

    logger.info(f"Pipeline finished: {result['status']}")
    logger.info(f"  Tickers updated: {len(result['tickers_updated'])}")
    logger.info(f"  Tickers failed:  {len(result['tickers_failed'])}")
    logger.info(f"  Duration:        {result['duration_seconds']}s")

    if result["errors"]:
        for err in result["errors"]:
            logger.error(f"  Error: {err}")

    sys.exit(0 if result["status"] != "failed" else 1)


def _run_scheduler(config: dict) -> None:
    """Start the scheduler daemon and run until interrupted."""
    from src.scheduler.scheduler import PipelineScheduler

    scheduler = PipelineScheduler(config)
    scheduler.start()

    logger.info("Scheduler running. Press Ctrl+C to stop.")

    # Handle graceful shutdown
    def _shutdown(signum: int, frame: object) -> None:
        logger.info("Shutdown signal received")
        scheduler.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.stop()


def _run_retrain(config: dict) -> None:
    """Retrain models and exit."""
    from src.scheduler.pipeline import PipelineRunner

    runner = PipelineRunner(config)
    result = runner.run_model_retrain()

    logger.info(f"Retrain result: {result['status']}")
    if result.get("reason"):
        logger.info(f"  Reason: {result['reason']}")


def _run_rebalance(config: dict) -> None:
    """Check rebalance and exit."""
    from src.scheduler.pipeline import PipelineRunner

    runner = PipelineRunner(config)
    result = runner.run_rebalance_check()

    logger.info(f"Rebalance needed: {result['rebalance_needed']}")
    logger.info(f"  Reason: {result['reason']}")
    if result.get("new_weights"):
        logger.info(f"  New weights: {result['new_weights']}")


def _show_status(config: dict) -> None:
    """Show the last pipeline run status from the status JSON."""
    base = Path(__file__).parent.parent
    data_dir = config.get("general", {}).get("data_dir", "data")
    status_path = base / data_dir / "processed" / "pipeline_status.json"

    if not status_path.exists():
        logger.info("No pipeline status found. Run the pipeline first.")
        return

    with open(status_path, "r", encoding="utf-8") as f:
        status = json.load(f)

    logger.info("Last pipeline run:")
    logger.info(f"  Status:    {status.get('status')}")
    logger.info(f"  Timestamp: {status.get('timestamp')}")
    logger.info(f"  Duration:  {status.get('duration_seconds')}s")
    logger.info(f"  Updated:   {status.get('tickers_updated', [])}")
    logger.info(f"  Failed:    {status.get('tickers_failed', [])}")

    if status.get("errors"):
        for err in status["errors"]:
            logger.error(f"  Error: {err}")


if __name__ == "__main__":
    main()
