"""APScheduler-based evaluation loop for smart-dca.

Instantiates all components from config, then runs evaluate_all() on
a fixed interval via APScheduler BackgroundScheduler. Alerts and
performance logging are handled after each evaluation cycle.

Usage:
    python -m scripts.run_scheduler
"""

from __future__ import annotations

import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.data.funding_feed import BinanceFundingFeed
from src.data.price_feed import BinancePriceFeed
from src.data.sentiment_feed import FearGreedFeed
from src.execution.budget_tracker import BudgetTracker
from src.execution.buy_engine import BuyEngine
from src.execution.order_manager import BinanceOrderManager
from src.notify.alerter import Alerter
from src.signals.funding_signal import FundingRateSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.scorer import SignalScorer
from src.signals.seasonality import SeasonalitySignal
from src.signals.sentiment_signal import SentimentSignal
from src.utils.config_loader import load_assets, load_settings
from src.utils.logger import configure_logging, get_logger

# PerformanceLog may be built in parallel -- graceful fallback.
try:
    from src.monitor.performance_log import FillRecord, PerformanceLog
except ImportError:  # pragma: no cover
    PerformanceLog = None  # type: ignore[assignment,misc]
    FillRecord = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)


def build_components(config: dict, assets_config: dict) -> dict[str, Any]:
    """Instantiate all runtime components in dependency order.

    The Binance client is created in testnet mode unless the
    BINANCE_TESTNET environment variable is explicitly set to "false".

    Args:
        config: Full settings dict (already loaded, not loaded here).
        assets_config: Assets dict from load_assets().

    Returns:
        Dict with keys: scorer, budget_tracker, order_manager,
        buy_engine, alerter, performance_log, price_feed.
    """
    # --- Binance client ---------------------------------------------------
    is_testnet = os.environ.get("BINANCE_TESTNET", "true").lower() == "true"
    binance_cfg = config.get("binance", {})

    from binance.client import Client as BinanceClient

    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    client = BinanceClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=is_testnet,
        requests_params={
            "timeout": binance_cfg.get("request_timeout_seconds", 10),
        },
    )

    # --- Data feeds --------------------------------------------------------
    price_feed = BinancePriceFeed(client, config)
    funding_feed = BinanceFundingFeed(client)
    sentiment_http = httpx.AsyncClient()
    sentiment_feed = FearGreedFeed(sentiment_http)

    # --- Signals -----------------------------------------------------------
    seasonality = SeasonalitySignal(config)
    mean_reversion = MeanReversionSignal(config, price_feed)
    funding_signal = FundingRateSignal(config, funding_feed)
    sentiment_signal = SentimentSignal(config, sentiment_feed)

    signals = [seasonality, mean_reversion, funding_signal, sentiment_signal]

    # --- Scorer ------------------------------------------------------------
    scorer = SignalScorer(signals, config)

    # --- Execution ---------------------------------------------------------
    state_file = Path(
        config.get("execution", {}).get("state_file", "data/budget_state.json")
    )
    budget_tracker = BudgetTracker(config, assets_config, state_file)
    order_manager = BinanceOrderManager(client, config)

    # --- Buy engine --------------------------------------------------------
    # Merge asset list into config so BuyEngine sees it
    engine_config = {**config, "assets": assets_config.get("assets", {})}
    buy_engine = BuyEngine(scorer, budget_tracker, order_manager, price_feed, engine_config)

    # --- Alerter -----------------------------------------------------------
    alerter = Alerter(config)

    # --- Performance log ---------------------------------------------------
    if PerformanceLog is not None:
        db_path = Path(config.get("monitor", {}).get("db_path", "data/performance.db"))
        performance_log = PerformanceLog(db_path)
    else:
        logger.warning("performance_log_unavailable", reason="module not yet built")
        performance_log = None

    return {
        "scorer": scorer,
        "budget_tracker": budget_tracker,
        "order_manager": order_manager,
        "buy_engine": buy_engine,
        "alerter": alerter,
        "performance_log": performance_log,
        "price_feed": price_feed,
    }


def run_evaluation_cycle(components: dict[str, Any]) -> None:
    """Execute one full evaluation cycle for all enabled assets.

    Called by the scheduler on each tick. Handles alerting and
    performance logging for every BuyDecision returned by the engine.

    Never raises -- all exceptions are caught and logged.

    Args:
        components: Dict returned by build_components().
    """
    buy_engine = components["buy_engine"]
    alerter: Alerter = components["alerter"]
    performance_log = components["performance_log"]

    scoring_cfg = buy_engine._config.get("scoring", {})
    buy_threshold = scoring_cfg.get("buy_threshold", 65)
    notify_threshold = scoring_cfg.get("notify_threshold", 55)

    try:
        decisions = buy_engine.evaluate_all()
    except Exception as exc:
        logger.error("evaluation_cycle_failed", error=str(exc))
        return

    for decision in decisions:
        score_value = decision.score.score

        if decision.executed:
            # Successful buy -- alert and record fill
            alerter.send_buy_executed(decision)

            if performance_log is not None and FillRecord is not None:
                try:
                    order = decision.order_result
                    fill = FillRecord(
                        asset=decision.asset,
                        fill_price=order.fill_price if order else 0.0,
                        amount_gbp=order.amount_gbp if order else 0.0,
                        quantity=order.quantity if order else 0.0,
                        naive_price=0.0,  # populated by weekly baseline
                        score_at_buy=score_value,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    performance_log.record_fill(fill)
                except Exception as exc:
                    logger.error(
                        "fill_record_failed",
                        asset=decision.asset,
                        error=str(exc),
                    )

        elif score_value >= buy_threshold:
            # Score was above buy threshold but order was not executed
            # (budget exhausted, order failed, etc.)
            alerter.send_buy_failed(decision)

        elif score_value >= notify_threshold:
            # Score is building but below buy threshold
            alerter.send_score_alert(decision.score, decision.asset)

    logger.info(
        "evaluation_cycle_complete",
        decisions=len(decisions),
        executed=sum(1 for d in decisions if d.executed),
    )


def main() -> None:
    """Configure logging, build components, start the scheduler, and block."""
    configure_logging()

    # Check live mode
    is_testnet = os.environ.get("BINANCE_TESTNET", "true").lower() == "true"
    if not is_testnet:
        logger.warning("LIVE MODE ACTIVE -- not using Binance testnet")

    config = load_settings()
    assets_config = load_assets()
    components = build_components(config, assets_config)

    interval_minutes = config.get("schedule", {}).get("evaluation_interval_minutes", 15)

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_evaluation_cycle,
        trigger=IntervalTrigger(minutes=interval_minutes),
        args=[components],
        id="evaluation_cycle",
        name="DCA evaluation cycle",
        max_instances=1,
    )
    scheduler.start()

    logger.info(
        "scheduler_started",
        interval_minutes=interval_minutes,
        testnet=is_testnet,
    )

    # Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(signum: int, frame: object) -> None:
        logger.info("shutdown_signal_received", signal=signum)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block the main thread while the scheduler runs in the background
    try:
        scheduler.print_jobs()
        import time

        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        logger.info("scheduler_stopped")


if __name__ == "__main__":
    main()
