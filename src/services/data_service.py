"""Data service — read-only access to pipeline outputs and status.

Provides a clean interface for the dashboard and other consumers to
query pipeline status, processed data, and portfolio weights without
directly touching the filesystem.

Usage:
    from src.services.data_service import DataService
    service = DataService(config)
    status = service.get_pipeline_status()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataService:
    """Read-only service for accessing pipeline outputs and status."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialise the data service.

        Args:
            config: Project configuration dict. If None, loads from
                    config/settings.yaml.
        """
        if config is None:
            from src.utils.config import load_config
            config = load_config()

        self.config = config
        self._data_dir = self._resolve_data_dir()

    def _resolve_data_dir(self) -> Path:
        """Resolve the project data directory from config."""
        base = Path(__file__).parent.parent.parent
        data_rel = self.config.get("general", {}).get("data_dir", "data")
        return base / data_rel

    def get_pipeline_status(self) -> dict[str, Any] | None:
        """Read the last pipeline result from pipeline_status.json.

        Returns:
            Pipeline status dictionary, or None if no status file exists.
        """
        status_path = self._data_dir / "processed" / "pipeline_status.json"

        if not status_path.exists():
            logger.debug("No pipeline status file found")
            return None

        try:
            with open(status_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Failed to read pipeline status: {exc}")
            return None

    def get_target_weights(self) -> dict[str, float] | None:
        """Read current target portfolio weights.

        Returns:
            Dictionary of ticker → weight, or None if not available.
        """
        weights_path = self._data_dir / "processed" / "target_weights.json"

        if not weights_path.exists():
            return None

        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Failed to read target weights: {exc}")
            return None

    def get_processed_data(self, ticker: str) -> pd.DataFrame | None:
        """Load processed OHLCV data for a single ticker.

        Args:
            ticker: Ticker symbol to load.

        Returns:
            Cleaned OHLCV DataFrame, or None if not available.
        """
        safe_name = ticker.replace(".", "_").replace("^", "idx_")
        parquet_path = self._data_dir / "processed" / f"{safe_name}.parquet"
        csv_path = self._data_dir / "processed" / f"{safe_name}.csv"

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index.name = "Date"
            return df

        logger.debug(f"No processed data found for {ticker}")
        return None

    def list_available_tickers(self) -> list[str]:
        """List tickers that have processed data available.

        Returns:
            Sorted list of ticker symbols with processed data on disk.
        """
        processed_dir = self._data_dir / "processed"
        if not processed_dir.exists():
            return []

        tickers = []
        for path in processed_dir.iterdir():
            if path.suffix in (".parquet", ".csv") and path.stem != "pipeline_status":
                # Reverse the safe-name transformation
                name = path.stem.replace("idx_", "^").replace("_", ".")
                tickers.append(name)

        return sorted(tickers)
