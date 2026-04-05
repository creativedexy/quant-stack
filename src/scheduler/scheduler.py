"""Pipeline scheduler — automated job scheduling with APScheduler.

Manages scheduled execution of the data pipeline, model retraining,
and rebalance checks at configured times.

Usage:
    from src.scheduler.scheduler import PipelineScheduler
    scheduler = PipelineScheduler(config)
    scheduler.start()
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.scheduler.alerts import AlertService
from src.scheduler.news_job import NewsRefreshJob
from src.scheduler.pipeline import PipelineRunner
from src.services.news_service import NewsService
from src.utils.logging import get_logger

logger = get_logger(__name__)

_REBALANCE_FREQUENCY_MAP = {
    "daily": {"day": "*"},
    "weekly": {"day_of_week": "mon"},
    "monthly": {"day": "1"},
    "quarterly": {"month": "1,4,7,10", "day": "1"},
}

_RETRAIN_DAY_MAP = {
    "monday": "mon",
    "tuesday": "tue",
    "wednesday": "wed",
    "thursday": "thu",
    "friday": "fri",
    "saturday": "sat",
    "sunday": "sun",
}


class PipelineScheduler:
    """Schedules pipeline runs using APScheduler.

    Configures four recurring jobs:
    - Daily pipeline: fetch, clean, features, signals
    - Weekly model retrain
    - Periodic rebalance check (per config frequency)
    - News refresh (interval-based, default every 15 minutes)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialise the scheduler with project configuration.

        Args:
            config: Project configuration dict. If None, loads from
                    config/settings.yaml.
        """
        if config is None:
            from src.utils.config import load_config
            config = load_config()

        self.config = config
        self.runner = PipelineRunner(config)
        self.alerts = AlertService(config)
        self._news_job = NewsRefreshJob(NewsService(config), config)

        sched_cfg = config.get("scheduler", {})
        tz = sched_cfg.get("timezone", "Europe/London")
        self.scheduler = BackgroundScheduler(timezone=tz)
        self._last_results: dict[str, dict[str, Any]] = {}
        self._timezone = tz

    def start(self) -> None:
        """Start the scheduler with configured jobs.

        Jobs:
        - Daily at configured time (default 17:30 London): run_daily
        - Weekly on configured day (default Sunday): run_model_retrain
        - Per rebalance_frequency in config: run_rebalance_check
        """
        sched_cfg = self.config.get("scheduler", {})

        # ── Daily pipeline ─────────────────────────────────────────
        daily_time = sched_cfg.get("daily_run_time", "17:30")
        hour, minute = daily_time.split(":")
        self.scheduler.add_job(
            self._run_daily_with_alerts,
            CronTrigger(hour=int(hour), minute=int(minute), timezone=self._timezone),
            id="daily_pipeline",
            name="Daily data pipeline",
            replace_existing=True,
        )
        logger.info(f"Scheduled daily pipeline at {daily_time} {self._timezone}")

        # ── Weekly model retrain ───────────────────────────────────
        retrain_day = sched_cfg.get("retrain_day", "sunday").lower()
        retrain_time = sched_cfg.get("retrain_time", "08:00")
        rt_hour, rt_minute = retrain_time.split(":")
        day_abbr = _RETRAIN_DAY_MAP.get(retrain_day, "sun")
        self.scheduler.add_job(
            self._run_retrain_with_alerts,
            CronTrigger(
                day_of_week=day_abbr,
                hour=int(rt_hour),
                minute=int(rt_minute),
                timezone=self._timezone,
            ),
            id="model_retrain",
            name="Weekly model retrain",
            replace_existing=True,
        )
        logger.info(f"Scheduled model retrain: {retrain_day} at {retrain_time}")

        # ── Rebalance check ────────────────────────────────────────
        rebalance_freq = (
            self.config
            .get("portfolio", {})
            .get("rebalance", {})
            .get("frequency", "monthly")
        )
        cron_kwargs = _REBALANCE_FREQUENCY_MAP.get(rebalance_freq, {"day": "1"})
        self.scheduler.add_job(
            self._run_rebalance_with_alerts,
            CronTrigger(hour=9, minute=0, timezone=self._timezone, **cron_kwargs),
            id="rebalance_check",
            name=f"Rebalance check ({rebalance_freq})",
            replace_existing=True,
        )
        logger.info(f"Scheduled rebalance check: {rebalance_freq}")

        # ── News refresh (interval) ───────────────────────────────
        news_minutes = self.config.get("news", {}).get("refresh_minutes", 15)
        self.scheduler.add_job(
            self._run_news_refresh,
            IntervalTrigger(minutes=news_minutes, timezone=self._timezone),
            id="news_refresh",
            name=f"News refresh (every {news_minutes} min)",
            replace_existing=True,
        )
        logger.info(f"Scheduled news refresh every {news_minutes} minutes")

        # ── Persona event detection (interval) ──────────────────
        personas_cfg = self.config.get("personas", {})
        if personas_cfg.get("enabled", False):
            persona_interval = personas_cfg.get(
                "event_check_interval_minutes", 30
            )
            self.scheduler.add_job(
                self._run_persona_event_check,
                IntervalTrigger(
                    minutes=persona_interval, timezone=self._timezone
                ),
                id="persona_event_check",
                name=f"Persona event detection (every {persona_interval} min)",
                replace_existing=True,
            )
            logger.info(
                "Scheduled persona event check every %d minutes",
                persona_interval,
            )

            # ── Persona daily briefs ────────────────────────────
            brief_time = personas_cfg.get("brief_schedule", "18:00")
            b_hour, b_minute = brief_time.split(":")
            self.scheduler.add_job(
                self._run_persona_briefs,
                CronTrigger(
                    hour=int(b_hour),
                    minute=int(b_minute),
                    timezone=self._timezone,
                ),
                id="persona_daily_briefs",
                name=f"Persona daily briefs at {brief_time}",
                replace_existing=True,
            )
            logger.info("Scheduled persona daily briefs at %s", brief_time)

        self.scheduler.start()
        logger.info("Pipeline scheduler started")

    def stop(self) -> None:
        """Gracefully stop all scheduled jobs."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Pipeline scheduler stopped")

    def run_now(self, job_name: str = "daily") -> dict[str, Any]:
        """Manually trigger a pipeline run.

        Args:
            job_name: Which job to run — 'daily', 'retrain', or 'rebalance'.

        Returns:
            Result dictionary from the executed job.

        Raises:
            ValueError: If job_name is not recognised.
        """
        runners = {
            "daily": self._run_daily_with_alerts,
            "retrain": self._run_retrain_with_alerts,
            "rebalance": self._run_rebalance_with_alerts,
            "news": self._run_news_refresh,
        }

        if job_name not in runners:
            raise ValueError(
                f"Unknown job: '{job_name}'. Available: {list(runners.keys())}"
            )

        logger.info(f"Manual trigger: {job_name}")
        return runners[job_name]()

    def get_status(self) -> dict[str, Any]:
        """Return scheduler status including job info and last results.

        Returns:
            Dictionary with:
            - running: whether the scheduler is active
            - jobs: list of scheduled jobs with next run times
            - last_results: most recent result for each job
        """
        jobs = []
        if self.scheduler.running:
            for job in self.scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": (
                        job.next_run_time.isoformat()
                        if job.next_run_time else None
                    ),
                })

        return {
            "running": self.scheduler.running,
            "jobs": jobs,
            "last_results": self._last_results,
        }

    def _run_daily_with_alerts(self) -> dict[str, Any]:
        """Execute daily pipeline and check alert conditions."""
        result = self.runner.run_daily()
        self._last_results["daily"] = result
        self.alerts.check_and_alert(pipeline_result=result, risk_metrics={})
        return result

    def _run_retrain_with_alerts(self) -> dict[str, Any]:
        """Execute model retrain and record results."""
        result = self.runner.run_model_retrain()
        self._last_results["retrain"] = result
        return result

    def _run_news_refresh(self) -> dict[str, Any]:
        """Execute news refresh job."""
        self._news_job.run()
        result = {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
        self._last_results["news"] = result
        return result

    def _run_rebalance_with_alerts(self) -> dict[str, Any]:
        """Execute rebalance check and optionally auto-execute in paper mode.

        When ``execution.auto_execute_rebalance`` is True **and** the
        execution mode is ``paper``, the scheduler will automatically
        create a broker, compute orders, and execute the rebalance via
        the OMS.  In all other cases it only alerts.
        """
        result = self.runner.run_rebalance_check()
        self._last_results["rebalance"] = result

        if not result.get("rebalance_needed"):
            return result

        exec_cfg = self.config.get("execution", {})
        auto_exec = exec_cfg.get("auto_execute_rebalance", False)
        mode = exec_cfg.get("mode", "paper")

        if auto_exec and mode == "paper" and result.get("new_weights"):
            self._auto_execute_rebalance(result["new_weights"])
        else:
            self.alerts.check_and_alert(
                pipeline_result={},
                risk_metrics={"rebalance_needed_not_executed": True},
            )

        return result

    def _auto_execute_rebalance(
        self,
        new_weights: dict[str, float],
    ) -> None:
        """Auto-execute a rebalance in paper mode.

        Safety: only called when ``mode == "paper"`` and
        ``auto_execute_rebalance == True`` in config.

        Args:
            new_weights: Target weight dict from rebalance check.
        """
        try:
            import pandas as pd

            from src.execution.broker import create_broker
            from src.execution.oms import OrderManagementSystem

            broker = create_broker(self.config)
            broker.connect()

            oms = OrderManagementSystem(broker, config=self.config)
            positions = broker.get_positions()
            account_value = broker.get_account_value()

            if account_value <= 0:
                account_value = float(
                    self.config.get("backtest", {}).get("initial_capital", 100_000),
                )

            target = pd.Series(new_weights, name="weights")

            # Use last known prices (from processed data or default)
            prices: dict[str, float] = {}
            processed = Path(
                self.config.get("data_dir", "data"),
            ) / "processed"
            for ticker in new_weights:
                safe = ticker.replace(".", "_").replace("^", "idx_")
                parquet = processed / f"{safe}.parquet"
                if parquet.exists():
                    df = pd.read_parquet(parquet)
                    if "Close" in df.columns and len(df) > 0:
                        prices[ticker] = float(df["Close"].iloc[-1])

            if not prices:
                logger.warning("No prices available -- skipping auto-rebalance")
                broker.disconnect()
                return

            orders = oms.compute_rebalance_orders(
                target, positions, account_value, prices,
            )

            if orders:
                report = oms.execute_plan(orders, dry_run=False)
                logger.info(
                    "Auto-rebalance executed: %d/%d orders",
                    len(report.orders_executed),
                    len(report.orders_planned),
                )
            else:
                logger.info("Auto-rebalance: no trades needed")

            broker.disconnect()
        except Exception as exc:
            logger.error("Auto-rebalance failed: %s", exc)

    # ------------------------------------------------------------------
    # Persona jobs
    # ------------------------------------------------------------------

    def _run_persona_event_check(self) -> dict[str, Any]:
        """Run event detection for dormant personas and alert on pending."""
        try:
            from src.personas.engine import PersonaEngine

            engine = PersonaEngine(self.config)

            # Fetch market data for all persona universes
            market_data = self._fetch_persona_market_data(engine)

            newly_pending = engine.check_events(market_data)

            for entry in newly_pending:
                self.alerts.check_and_deliver(
                    "persona_activation_pending",
                    {
                        "severity": "warning",
                        "message": (
                            f"{entry['name']} conditions detected: "
                            f"{entry['reason']}. "
                            "Confirm activation via dashboard or CLI."
                        ),
                        "persona_id": entry["persona_id"],
                        "reason": entry["reason"],
                        "metrics": entry["metrics"],
                    },
                )

            result = {
                "status": "ok",
                "newly_pending": len(newly_pending),
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._last_results["persona_events"] = result
            return result

        except Exception as exc:
            logger.error("Persona event check failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _run_persona_briefs(self) -> dict[str, Any]:
        """Generate daily briefs for all active personas."""
        try:
            from src.personas.engine import PersonaEngine

            engine = PersonaEngine(self.config)
            market_data = self._fetch_persona_market_data(engine)

            capital = float(
                self.config.get("personas", {}).get(
                    "total_capital",
                    self.config.get("backtest", {}).get(
                        "initial_capital", 100_000
                    ),
                )
            )

            briefs = engine.generate_all_briefs(market_data, capital)

            result = {
                "status": "ok",
                "briefs_generated": len(briefs),
                "personas": list(briefs.keys()),
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._last_results["persona_briefs"] = result
            return result

        except Exception as exc:
            logger.error("Persona brief generation failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _fetch_persona_market_data(
        self, engine: Any
    ) -> dict[str, "pd.DataFrame"]:
        """Fetch market data for all persona universes.

        Args:
            engine: PersonaEngine instance to get universe tickers from.

        Returns:
            Dictionary of ticker -> OHLCV DataFrame.
        """
        import pandas as pd

        # Collect all tickers across all personas
        all_tickers: set[str] = set()
        status = engine.get_status()
        for persona_info in status.values():
            all_tickers.update(persona_info.get("universe", []))

            # Include detector-specific tickers
            activation = engine.get_persona(
                persona_info["persona_id"]
            )
            if activation:
                params = activation.activation_config.get("params", {})
                for key in ("index_ticker", "vix_ticker", "watch_ticker"):
                    val = params.get(key)
                    if val:
                        all_tickers.add(val)

        if not all_tickers:
            return {}

        # Try to load from processed data
        data_dir = Path(
            self.config.get("general", {}).get("data_dir", "data")
        )
        processed = data_dir / "processed"
        market_data: dict[str, pd.DataFrame] = {}

        for ticker in all_tickers:
            safe = ticker.replace(".", "_").replace("^", "idx_")
            parquet = processed / f"{safe}.parquet"
            if parquet.exists():
                try:
                    market_data[ticker] = pd.read_parquet(parquet)
                except Exception:
                    pass

        return market_data
