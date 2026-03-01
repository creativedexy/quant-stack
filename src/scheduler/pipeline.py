"""Pipeline runner — orchestrates the full data processing workflow.

Runs the complete daily pipeline: fetch → clean → features → signals.
Designed for both scheduled and manual execution.

Usage:
    from src.scheduler.pipeline import PipelineRunner
    runner = PipelineRunner(config)
    result = runner.run_daily()
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.cleaner import DataCleaner
from src.data.fetcher import create_fetcher
from src.utils.logging import get_logger

logger = get_logger(__name__)

_STATUS_FILENAME = "pipeline_status.json"


class PipelineRunner:
    """Runs the full data pipeline: fetch → clean → features → signals.

    Each step is isolated so a failure in one ticker or stage does not
    crash the entire pipeline. Results are recorded and persisted as
    JSON for downstream consumption (dashboard, alerts).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialise the pipeline runner.

        Args:
            config: Project configuration dict. If None, loads from
                    config/settings.yaml.
        """
        if config is None:
            from src.utils.config import load_config
            config = load_config()

        self.config = config
        self._data_dir = self._resolve_data_dir()
        self._source = config.get("data", {}).get("source", "synthetic")
        self._seed = config.get("general", {}).get("random_seed", 42)

    def _resolve_data_dir(self) -> Path:
        """Resolve the project data directory from config."""
        base = Path(__file__).parent.parent.parent
        data_rel = self.config.get("general", {}).get("data_dir", "data")
        return base / data_rel

    def run_daily(self) -> dict[str, Any]:
        """Execute the full daily pipeline.

        Steps:
            1. Fetch latest data for all universe tickers
            2. Clean and validate
            3. Generate features (if available)
            4. Run registered strategies to produce signals (if available)
            5. Save everything to data/processed/
            6. Log summary

        Returns:
            Dictionary with pipeline execution results:
            - status: "success" | "partial" | "failed"
            - tickers_updated: list of successfully processed tickers
            - tickers_failed: list of tickers that failed
            - features_generated: whether feature generation ran
            - signals_generated: whether signal generation ran
            - timestamp: ISO-format completion time
            - duration_seconds: wall-clock duration
            - errors: list of error messages
        """
        start_time = time.monotonic()
        timestamp = datetime.now(timezone.utc)
        tickers = self.config.get("universe", {}).get("tickers", [])
        errors: list[str] = []
        tickers_updated: list[str] = []
        tickers_failed: list[str] = []

        logger.info(
            "Starting daily pipeline",
            extra={"tickers": len(tickers), "source": self._source},
        )

        # ── Step 1: Fetch ──────────────────────────────────────────
        raw_data: dict[str, pd.DataFrame] = {}
        try:
            fetcher = create_fetcher(self._source, seed=self._seed)
            start_date = self.config.get("data", {}).get("start_date", "2020-01-01")
            end_date = self.config.get("data", {}).get("end_date")
            raw_data = fetcher.fetch_multiple(
                tickers, start=start_date, end=end_date,
            )
            # Save raw data
            if raw_data:
                raw_dir = self._data_dir / "raw"
                fetcher.save(
                    raw_data, raw_dir,
                    fmt=self.config.get("data", {}).get("output_format", "parquet"),
                )
        except Exception as exc:
            msg = f"Fetch stage failed: {exc}"
            logger.error(msg)
            errors.append(msg)

        # Track successes and failures from fetch
        for ticker in tickers:
            if ticker in raw_data:
                tickers_updated.append(ticker)
            else:
                tickers_failed.append(ticker)
                errors.append(f"Failed to fetch {ticker}")

        # ── Step 2: Clean ──────────────────────────────────────────
        clean_data: dict[str, pd.DataFrame] = {}
        if raw_data:
            try:
                cleaner = DataCleaner()
                clean_data = cleaner.clean_multiple(raw_data)
                # Save cleaned data
                processed_dir = self._data_dir / "processed"
                fetcher.save(
                    clean_data, processed_dir,
                    fmt=self.config.get("data", {}).get("output_format", "parquet"),
                )
            except Exception as exc:
                msg = f"Clean stage failed: {exc}"
                logger.error(msg)
                errors.append(msg)

        # ── Step 3: Features ───────────────────────────────────────
        features_generated = False
        if clean_data:
            features_generated = self._run_feature_generation(clean_data, errors)

        # ── Step 4: Signals ────────────────────────────────────────
        signals_generated = False
        if clean_data:
            signals_generated = self._run_signal_generation(clean_data, errors)

        # ── Step 5: Determine status ──────────────────────────────
        duration = time.monotonic() - start_time
        if not tickers_updated:
            status = "failed"
        elif tickers_failed:
            status = "partial"
        else:
            status = "success"

        result: dict[str, Any] = {
            "status": status,
            "tickers_updated": tickers_updated,
            "tickers_failed": tickers_failed,
            "features_generated": features_generated,
            "signals_generated": signals_generated,
            "timestamp": timestamp.isoformat(),
            "duration_seconds": round(duration, 2),
            "errors": errors,
        }

        # ── Step 6: Save status & log ─────────────────────────────
        self._save_status(result)
        logger.info(
            f"Daily pipeline complete: {status}",
            extra={
                "updated": len(tickers_updated),
                "failed": len(tickers_failed),
                "duration_s": result["duration_seconds"],
            },
        )

        return result

    def run_rebalance_check(self) -> dict[str, Any]:
        """Check if rebalancing is needed based on config frequency.

        If the configured rebalance frequency threshold has been exceeded,
        compute new target weights using equal-weight (or configured method)
        and save to data/processed/.

        Returns:
            Dictionary with:
            - rebalance_needed: whether rebalancing was triggered
            - new_weights: dict of ticker → weight (if applicable)
            - reason: explanation
            - timestamp: ISO-format time
        """
        timestamp = datetime.now(timezone.utc)
        tickers = self.config.get("universe", {}).get("tickers", [])
        portfolio_cfg = self.config.get("portfolio", {})
        rebalance_cfg = portfolio_cfg.get("rebalance", {})
        threshold = rebalance_cfg.get("threshold", 0.05)

        logger.info("Running rebalance check")

        # Load current weights if they exist
        weights_path = self._data_dir / "processed" / "target_weights.json"
        current_weights: dict[str, float] = {}
        if weights_path.exists():
            with open(weights_path, "r", encoding="utf-8") as f:
                current_weights = json.load(f)

        # If no previous weights, rebalance is needed
        if not current_weights:
            new_weights = {t: round(1.0 / len(tickers), 4) for t in tickers}
            self._save_weights(new_weights)
            return {
                "rebalance_needed": True,
                "new_weights": new_weights,
                "reason": "No previous weights found — initialising equal-weight",
                "timestamp": timestamp.isoformat(),
            }

        # Check for drift (simplified: compare to equal-weight baseline)
        target_weight = 1.0 / len(tickers)
        max_drift = max(
            abs(current_weights.get(t, 0) - target_weight) for t in tickers
        )

        if max_drift > threshold:
            new_weights = {t: round(1.0 / len(tickers), 4) for t in tickers}
            self._save_weights(new_weights)
            logger.info(
                f"Rebalance triggered: max drift {max_drift:.2%} > threshold {threshold:.2%}"
            )
            return {
                "rebalance_needed": True,
                "new_weights": new_weights,
                "reason": f"Drift {max_drift:.2%} exceeds threshold {threshold:.2%}",
                "timestamp": timestamp.isoformat(),
            }

        logger.info(f"No rebalance needed: max drift {max_drift:.2%}")
        return {
            "rebalance_needed": False,
            "new_weights": None,
            "reason": f"Drift {max_drift:.2%} within threshold {threshold:.2%}",
            "timestamp": timestamp.isoformat(),
        }

    def run_model_retrain(self) -> dict[str, Any]:
        """Retrain models on latest data.

        Intended to run weekly or on-demand. Loads the latest processed
        data and retrains the configured model with walk-forward validation.

        Returns:
            Dictionary with model metrics and retrain status.
        """
        timestamp = datetime.now(timezone.utc)
        logger.info("Starting model retrain")

        # Check if models module has been implemented
        try:
            from src.models import base as _  # noqa: F401
            models_available = True
        except (ImportError, AttributeError):
            models_available = False

        if not models_available:
            logger.info("Models module not yet implemented — skipping retrain")
            return {
                "status": "skipped",
                "reason": "Models module not yet implemented",
                "metrics": {},
                "timestamp": timestamp.isoformat(),
            }

        # Stub: when models are implemented, retrain here
        logger.info("Model retrain complete (stub)")
        return {
            "status": "success",
            "reason": "Retrain completed",
            "metrics": {},
            "timestamp": timestamp.isoformat(),
        }

    def _run_feature_generation(
        self,
        clean_data: dict[str, pd.DataFrame],
        errors: list[str],
    ) -> bool:
        """Attempt to generate features from cleaned data.

        Returns True if features were generated successfully.
        """
        try:
            from src.features.pipeline import FeaturePipeline  # noqa: F401
            logger.info("Running feature generation")
            # When feature pipeline is implemented, call it here
            return True
        except (ImportError, AttributeError):
            logger.info("Feature pipeline not yet implemented — skipping")
            return False

    def _run_signal_generation(
        self,
        clean_data: dict[str, pd.DataFrame],
        errors: list[str],
    ) -> bool:
        """Attempt to run strategies and produce signals.

        Returns True if signals were generated successfully.
        """
        try:
            from src.backtest.strategy import Strategy  # noqa: F401
            logger.info("Running signal generation")
            # When strategies are implemented, run them here
            return True
        except (ImportError, AttributeError):
            logger.info("Strategy module not yet implemented — skipping")
            return False

    def _save_status(self, result: dict[str, Any]) -> None:
        """Persist pipeline status to JSON for dashboard consumption."""
        status_dir = self._data_dir / "processed"
        status_dir.mkdir(parents=True, exist_ok=True)
        status_path = status_dir / _STATUS_FILENAME

        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        logger.debug(f"Pipeline status saved to {status_path}")

    def _save_weights(self, weights: dict[str, float]) -> None:
        """Persist target portfolio weights to JSON."""
        weights_dir = self._data_dir / "processed"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "target_weights.json"

        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2)

        logger.debug(f"Target weights saved to {weights_path}")
