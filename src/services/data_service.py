"""Data service — clean data access layer for the dashboard.

Handles caching, fallbacks, and formatting so dashboard code
never touches raw files or fetcher logic directly.

Usage:
    from src.services.data_service import DataService
    svc = DataService()
    prices = svc.get_prices(["SHEL.L", "HSBA.L"])
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Project root (three levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DataService:
    """Provides clean data access to the dashboard.

    Handles caching, fallbacks, and formatting.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise the data service.

        Args:
            config: Optional configuration dict. If not provided, attempts
                to load from the default settings.yaml.
        """
        if config is None:
            try:
                from src.utils.config import load_config
                config = load_config()
            except FileNotFoundError:
                logger.warning("No settings.yaml found; using defaults")
                config = {}

        self._config = config

        # Resolve data directories
        data_dir_name = (
            config.get("general", {}).get("data_dir", "data")
        )
        self._data_dir = _PROJECT_ROOT / data_dir_name
        self._processed_dir = self._data_dir / "processed"
        self._raw_dir = self._data_dir / "raw"

        # Universe tickers from config
        self._tickers: list[str] = (
            config.get("universe", {}).get("tickers", [])
        )

    # ── Public API ───────────────────────────────────────────

    def get_prices(
        self,
        tickers: list[str] | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Load price data from processed parquet files.

        Falls back to raw data if processed not available.
        Returns DataFrame with tickers as columns and DatetimeIndex.

        Args:
            tickers: Ticker symbols to load. Defaults to universe.
            start: Start date filter (inclusive).
            end: End date filter (inclusive).

        Returns:
            DataFrame with close prices, tickers as columns.
        """
        tickers = tickers or self._tickers
        if not tickers:
            return pd.DataFrame()

        frames: dict[str, pd.Series] = {}
        for ticker in tickers:
            df = self._load_ticker(ticker)
            if df is not None and "Close" in df.columns:
                frames[ticker] = df["Close"]

        if not frames:
            return pd.DataFrame()

        prices = pd.DataFrame(frames)
        prices.index.name = "Date"

        # Apply date filters
        if start is not None:
            prices = prices[prices.index >= pd.Timestamp(start)]
        if end is not None:
            prices = prices[prices.index <= pd.Timestamp(end)]

        return prices

    def get_latest_prices(self, tickers: list[str] | None = None) -> pd.Series:
        """Most recent price for each ticker.

        Args:
            tickers: Ticker symbols. Defaults to universe.

        Returns:
            Series indexed by ticker with the latest close price.
        """
        prices = self.get_prices(tickers=tickers)
        if prices.empty:
            return pd.Series(dtype=float)
        return prices.ffill().iloc[-1]

    def get_features(self, ticker: str) -> pd.DataFrame:
        """Load pre-computed features for a ticker from parquet.

        Args:
            ticker: Ticker symbol.

        Returns:
            DataFrame of features with DatetimeIndex, or empty DataFrame.
        """
        safe_name = self._safe_filename(ticker)
        features_dir = self._processed_dir / "features"
        path = features_dir / f"{safe_name}.parquet"

        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                logger.error(f"Failed to load features for {ticker}: {exc}")

        return pd.DataFrame()

    def get_returns(
        self,
        tickers: list[str] | None = None,
        window: int = 252,
    ) -> pd.DataFrame:
        """Compute or load returns for the specified window.

        Args:
            tickers: Ticker symbols. Defaults to universe.
            window: Number of trading days to include.

        Returns:
            DataFrame of daily returns, tickers as columns.
        """
        prices = self.get_prices(tickers=tickers)
        if prices.empty:
            return pd.DataFrame()

        # Trim to requested window (plus one day for the return calc)
        if len(prices) > window + 1:
            prices = prices.iloc[-(window + 1):]

        returns = prices.pct_change().dropna(how="all")
        return returns

    def get_data_status(self) -> dict[str, Any]:
        """Return summary of available data.

        Returns:
            Dict with: last_updated, tickers_available, date_range,
            data_source, file_sizes.
        """
        tickers_available: list[str] = []
        file_sizes: dict[str, int] = {}
        last_updated: datetime | None = None
        date_range: tuple[str, str] | None = None

        # Scan processed directory first, fall back to raw
        scan_dir = (
            self._processed_dir
            if self._processed_dir.exists()
            else self._raw_dir
        )

        if scan_dir.exists():
            for path in sorted(scan_dir.glob("*.parquet")):
                ticker = path.stem.replace("_", ".")
                tickers_available.append(ticker)
                file_sizes[ticker] = path.stat().st_size

                # Track most recent modification
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if last_updated is None or mtime > last_updated:
                    last_updated = mtime

            # Read first available file for date range
            if tickers_available:
                first_file = scan_dir / f"{self._safe_filename(tickers_available[0])}.parquet"
                if first_file.exists():
                    try:
                        df = pd.read_parquet(first_file)
                        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                            date_range = (
                                str(df.index.min().date()),
                                str(df.index.max().date()),
                            )
                    except Exception:
                        pass

        data_source = self._config.get("data", {}).get("source", "unknown")

        return {
            "last_updated": last_updated,
            "tickers_available": tickers_available,
            "date_range": date_range,
            "data_source": data_source,
            "file_sizes": file_sizes,
        }

    # ── Internal helpers ─────────────────────────────────────

    def _load_ticker(self, ticker: str) -> pd.DataFrame | None:
        """Load a single ticker's OHLCV data, trying processed then raw.

        Args:
            ticker: Ticker symbol.

        Returns:
            OHLCV DataFrame or None if not found.
        """
        safe_name = self._safe_filename(ticker)

        # Try processed first
        for directory in (self._processed_dir, self._raw_dir):
            path = directory / f"{safe_name}.parquet"
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    return df
                except Exception as exc:
                    logger.error(
                        f"Failed to load {ticker} from {path}: {exc}"
                    )

        logger.debug(f"No data file found for {ticker}")
        return None

    @staticmethod
    def _safe_filename(ticker: str) -> str:
        """Convert a ticker symbol to a safe filename stem.

        Args:
            ticker: Ticker symbol (e.g. 'SHEL.L', '^FTSE').

        Returns:
            Filesystem-safe string.
        """
        return ticker.replace(".", "_").replace("^", "idx_")
