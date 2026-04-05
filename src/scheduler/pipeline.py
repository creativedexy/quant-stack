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
from src.services.model_service import ModelService
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
        portfolio_cfg = self.config.get("optimisation", {})
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

        Orchestrates the full retrain workflow:
        1. Generate features for all universe tickers
        2. Train models with walk-forward cross-validation
        3. Evaluate and log metrics

        Intended to run weekly or on-demand.

        Returns:
            Dictionary with model metrics and retrain status.
        """
        timestamp = datetime.now(timezone.utc)
        tickers = self.config.get("universe", {}).get("tickers", [])
        logger.info(
            "Starting model retrain",
            extra={"tickers": len(tickers)},
        )

        if not tickers:
            return {
                "status": "failed",
                "reason": "No tickers in universe",
                "metrics": {},
                "timestamp": timestamp.isoformat(),
            }

        # Step 1: Generate features
        features_dict = self._run_feature_pipeline(tickers)
        if not features_dict:
            return {
                "status": "failed",
                "reason": "Feature generation produced no data",
                "metrics": {},
                "timestamp": timestamp.isoformat(),
            }

        # Step 2: Train models
        train_results = self._run_model_training(features_dict)

        # Step 3: Evaluate models
        eval_results = self._run_model_evaluation(features_dict)

        # Determine overall status
        trained = [r for r in train_results if not r.get("skipped")]
        if not trained:
            status = "failed"
            reason = "No models were trained"
        elif len(trained) < len(tickers):
            status = "partial"
            reason = f"Trained {len(trained)}/{len(tickers)} tickers"
        else:
            status = "success"
            reason = f"All {len(trained)} models trained"

        logger.info("Model retrain complete: %s", reason)
        return {
            "status": status,
            "reason": reason,
            "metrics": {r["ticker"]: r for r in train_results},
            "evaluation": {r["ticker"]: r for r in eval_results},
            "timestamp": timestamp.isoformat(),
        }

    def _run_feature_generation(
        self,
        clean_data: dict[str, pd.DataFrame],
        errors: list[str],
    ) -> bool:
        """Generate features from cleaned data and save to disk.

        Delegates to :class:`FeaturePipeline` for each ticker, saving
        results as Parquet files under ``data/processed/features/``.

        Returns True if features were generated for at least one ticker.
        """
        try:
            from src.features.pipeline import FeaturePipeline

            fp = FeaturePipeline(self.config)
            features_dir = self._data_dir / "processed" / "features"
            features_dir.mkdir(parents=True, exist_ok=True)

            generated = 0
            for ticker, df in clean_data.items():
                try:
                    features = fp.generate(df)
                    safe = ticker.replace(".", "_").replace("^", "idx_")
                    features.to_parquet(features_dir / f"{safe}_features.parquet")
                    generated += 1
                except Exception as exc:
                    msg = f"Feature generation failed for {ticker}: {exc}"
                    logger.warning(msg)
                    errors.append(msg)

            logger.info("Features generated for %d/%d tickers", generated, len(clean_data))
            return generated > 0
        except ImportError:
            logger.info("Feature pipeline not available -- skipping")
            return False

    def _run_signal_generation(
        self,
        clean_data: dict[str, pd.DataFrame],
        errors: list[str],
    ) -> bool:
        """Run active strategies on cleaned data and save latest signals.

        Reads strategy names from ``config.strategies.active``, generates
        features, runs each strategy, and saves the latest signal per
        ticker to ``data/processed/signals/{strategy}_latest.json``.

        Returns True if signals were generated for at least one strategy.
        """
        try:
            from src.backtest.strategy import strategy_registry
            from src.features.pipeline import FeaturePipeline

            fp = FeaturePipeline(self.config)
            strategy_names = (
                self.config.get("strategies", {}).get("active", ["momentum"])
            )

            signals_dir = self._data_dir / "processed" / "signals"
            signals_dir.mkdir(parents=True, exist_ok=True)

            generated = 0
            for strat_name in strategy_names:
                try:
                    strategy = strategy_registry.create(strat_name)
                except KeyError:
                    logger.warning(
                        "Strategy '%s' not registered -- skipping", strat_name,
                    )
                    continue

                all_signals: dict[str, int] = {}
                for ticker, df in clean_data.items():
                    try:
                        features = fp.generate(df)
                        sig_df = strategy.generate_signals(features)
                        if not sig_df.empty and "signal" in sig_df.columns:
                            all_signals[ticker] = int(sig_df["signal"].iloc[-1])
                    except Exception as exc:
                        logger.warning(
                            "Signal generation failed for %s/%s: %s",
                            strat_name, ticker, exc,
                        )

                if all_signals:
                    sig_path = signals_dir / f"{strat_name}_latest.json"
                    sig_path.write_text(
                        json.dumps(all_signals, indent=2), encoding="utf-8",
                    )
                    generated += 1
                    logger.info(
                        "Strategy '%s': signals for %d tickers",
                        strat_name, len(all_signals),
                    )

            logger.info(
                "Signal generation complete: %d/%d strategies",
                generated, len(strategy_names),
            )
            return generated > 0
        except ImportError:
            logger.info("Strategy/feature modules not available -- skipping")
            return False

    # ------------------------------------------------------------------
    # Model retrain sub-steps
    # ------------------------------------------------------------------

    def _run_feature_pipeline(
        self,
        tickers: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Generate features with target column for all tickers.

        Loads cleaned data from ``data/processed/``, runs the feature
        pipeline, and appends a direction target column.

        Args:
            tickers: List of ticker symbols.

        Returns:
            ``dict[ticker, DataFrame]`` where each DataFrame contains
            feature columns plus the target column ready for training.
        """
        from src.features.pipeline import FeaturePipeline
        from src.models.targets import create_direction_target

        target_col = self.config.get("models", {}).get(
            "target_column", "forward_return_5d"
        )
        fp = FeaturePipeline(self.config)
        results: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            try:
                df = self._load_processed(ticker)
                if df is None or df.empty:
                    logger.warning(
                        "[%s] No processed data found -- skipping features", ticker
                    )
                    continue

                features = fp.generate(df)
                if features.empty:
                    logger.warning("[%s] Feature pipeline returned empty", ticker)
                    continue

                # Build direction target from Close prices
                if "Close" in df.columns:
                    target = create_direction_target(df["Close"], horizon=5)
                    target.name = target_col
                    features = features.join(target, how="left")

                results[ticker] = features
                logger.info(
                    "[%s] Features generated: %d cols, %d rows",
                    ticker, features.shape[1], len(features),
                )
            except Exception as exc:
                logger.error("[%s] Feature generation failed: %s", ticker, exc)

        return results

    def _run_model_training(
        self,
        features_dict: dict[str, pd.DataFrame],
    ) -> list[dict[str, Any]]:
        """Train and save models for each ticker.

        Delegates to :class:`ModelService` which uses
        :class:`sklearn.model_selection.TimeSeriesSplit` for walk-forward
        cross-validation.

        Args:
            features_dict: ``dict[ticker, DataFrame]`` from
                :meth:`_run_feature_pipeline`.

        Returns:
            List of result dicts, one per ticker.
        """
        service = ModelService(self.config)
        results: list[dict[str, Any]] = []

        for ticker, features in features_dict.items():
            try:
                result = service.train_and_save(ticker, features)
                results.append(result)
            except Exception as exc:
                logger.error("[%s] Model training failed: %s", ticker, exc)
                results.append({
                    "ticker": ticker,
                    "cv_mean": 0.0,
                    "cv_std": 0.0,
                    "model_path": None,
                    "feature_count": 0,
                    "skipped": True,
                    "reason": str(exc),
                })

        return results

    def _run_model_evaluation(
        self,
        features_dict: dict[str, pd.DataFrame],
    ) -> list[dict[str, Any]]:
        """Evaluate models for each ticker via walk-forward CV.

        Re-runs cross-validation without saving, to log evaluation
        metrics independently of the training step.

        Args:
            features_dict: ``dict[ticker, DataFrame]`` from
                :meth:`_run_feature_pipeline`.

        Returns:
            List of evaluation result dicts, one per ticker.
        """
        service = ModelService(self.config)
        results: list[dict[str, Any]] = []

        for ticker, features in features_dict.items():
            try:
                result = service.evaluate(ticker, features)
                results.append(result)
            except Exception as exc:
                logger.error("[%s] Model evaluation failed: %s", ticker, exc)
                results.append({
                    "ticker": ticker,
                    "cv_mean": 0.0,
                    "cv_std": 0.0,
                    "model_path": None,
                    "feature_count": 0,
                    "skipped": True,
                    "reason": str(exc),
                })

        return results

    def _load_processed(self, ticker: str) -> pd.DataFrame | None:
        """Load cleaned OHLCV data for a ticker from the processed directory."""
        safe_name = ticker.replace(".", "_").replace("^", "idx_")
        parquet_path = self._data_dir / "processed" / f"{safe_name}.parquet"
        csv_path = self._data_dir / "processed" / f"{safe_name}.csv"

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index.name = "Date"
            return df

        return None

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
