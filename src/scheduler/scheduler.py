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
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.scheduler.alerts import AlertService
from src.scheduler.pipeline import PipelineRunner
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

    Configures three recurring jobs:
    - Daily pipeline: fetch, clean, features, signals
    - Weekly model retrain
    - Periodic rebalance check (per config frequency)
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

    def _run_rebalance_with_alerts(self) -> dict[str, Any]:
        """Execute rebalance check and alert if needed but not executed."""
        result = self.runner.run_rebalance_check()
        self._last_results["rebalance"] = result

        if result.get("rebalance_needed") and not result.get("new_weights"):
            self.alerts.check_and_alert(
                pipeline_result={},
                risk_metrics={"rebalance_needed_not_executed": True},
            )

        return result
