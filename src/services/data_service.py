"""Data service -- read-only access to pipeline outputs and status.

Provides a clean interface for the dashboard and other consumers to
query processed data, portfolio weights, and data status without
directly touching the filesystem.

Usage:
    from src.services.data_service import DataService
    service = DataService(config)
    prices = service.get_prices()
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataService:
    """Read-only service for accessing pipeline outputs and status.

    Args:
        config: Project configuration dict. If ``None``, uses empty defaults.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._config = config
        self._tickers: list[str] = list(
            config.get("universe", {}).get("tickers", [])
        )

        data_dir = config.get("general", {}).get("data_dir", "data")
        self._data_dir = Path(data_dir)
        self._processed_dir = self._data_dir / "processed"
        self._raw_dir = self._data_dir / "raw"

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def get_prices(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Load Close price data for the given tickers.

        Args:
            tickers: Ticker symbols to load. Defaults to configured universe.
            start: Start date string for filtering (inclusive).
            end: End date string for filtering (inclusive).

        Returns:
            DataFrame with DatetimeIndex and one column per ticker.
        """
        if tickers is None:
            tickers = self._tickers

        frames: dict[str, pd.Series] = {}
        for ticker in tickers:
            df = self._load_ticker(ticker)
            if df is not None and "Close" in df.columns:
                frames[ticker] = df["Close"]

        if not frames:
            return pd.DataFrame()

        result = pd.DataFrame(frames)
        result.index = pd.DatetimeIndex(result.index)

        if start is not None:
            result = result[result.index >= pd.Timestamp(start)]
        if end is not None:
            result = result[result.index <= pd.Timestamp(end)]

        return result

    def get_latest_prices(self) -> pd.Series:
        """Return the most recent price for each configured ticker.

        Returns:
            Series indexed by ticker symbol.
        """
        prices = self.get_prices()
        if prices.empty:
            return pd.Series(dtype=float)
        return prices.iloc[-1]

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def get_returns(self, window: int = 60) -> pd.DataFrame:
        """Compute daily percentage returns over a trailing window.

        Args:
            window: Number of trailing days to include.

        Returns:
            DataFrame of daily returns.
        """
        prices = self.get_prices()
        if prices.empty:
            return pd.DataFrame()
        returns = prices.pct_change().dropna()
        return returns.tail(window)

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def get_features(self, ticker: str) -> pd.DataFrame:
        """Load pre-computed features for a ticker.

        Args:
            ticker: Ticker symbol.

        Returns:
            Feature DataFrame, or empty DataFrame if not available.
        """
        safe_name = ticker.replace(".", "_").replace("^", "idx_")
        feature_path = self._processed_dir / f"{safe_name}_features.parquet"
        if feature_path.exists():
            return pd.read_parquet(feature_path)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_data_status(self) -> dict[str, Any]:
        """Return a summary of available data.

        Returns:
            Dict with keys: ``last_updated``, ``tickers_available``,
            ``date_range``, ``data_source``, ``file_sizes``.
        """
        if not self._processed_dir.exists():
            return {
                "last_updated": None,
                "tickers_available": [],
                "date_range": None,
                "data_source": self._config.get("data", {}).get(
                    "source", "unknown"
                ),
                "file_sizes": {},
            }

        tickers: list[str] = []
        file_sizes: dict[str, int] = {}
        latest_mtime: float | None = None

        for path in sorted(self._processed_dir.iterdir()):
            if path.suffix not in (".parquet", ".csv"):
                continue
            if path.stem.endswith("_features"):
                continue
            # Reverse the safe-name transformation
            name = path.stem.replace("idx_", "^").replace("_", ".")
            tickers.append(name)
            size = path.stat().st_size
            file_sizes[name] = size
            mtime = path.stat().st_mtime
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime

        # Determine date range from the first available ticker
        date_range = None
        if tickers:
            df = self._load_ticker(tickers[0])
            if df is not None and len(df) > 0:
                date_range = {
                    "start": str(df.index.min().date()),
                    "end": str(df.index.max().date()),
                }

        last_updated: str | None = None
        if latest_mtime is not None:
            last_updated = datetime.fromtimestamp(
                latest_mtime, tz=timezone.utc,
            ).isoformat()

        return {
            "last_updated": last_updated,
            "tickers_available": tickers,
            "date_range": date_range,
            "data_source": self._config.get("data", {}).get(
                "source", "unknown"
            ),
            "file_sizes": file_sizes,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_ticker(self, ticker: str) -> pd.DataFrame | None:
        """Load OHLCV data for a single ticker from processed directory."""
        safe_name = ticker.replace(".", "_").replace("^", "idx_")
        parquet_path = self._processed_dir / f"{safe_name}.parquet"
        csv_path = self._processed_dir / f"{safe_name}.csv"

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index.name = "Date"
            return df

        return None
