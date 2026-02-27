"""Data fetcher module — abstract interface and concrete implementations.

Separates the *what* (fetching market data) from the *how* (yfinance, OpenBB, etc.).
This makes it trivial to swap data sources without touching downstream code.

Usage:
    from src.data.fetcher import create_fetcher
    fetcher = create_fetcher("synthetic")  # or "yfinance" when you have network
    data = fetcher.fetch("AAPL", start="2020-01-01")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger
from src.utils.validators import validate_ohlcv

logger = get_logger(__name__)


class DataFetcher(ABC):
    """Abstract base class for all data fetchers.

    All fetchers must implement `fetch()` and `fetch_multiple()`.
    This ensures consistent behaviour regardless of the underlying data source.
    """

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single ticker.

        Args:
            ticker: Asset ticker symbol.
            start: Start date (inclusive).
            end: End date (inclusive). Defaults to today.
            interval: Data frequency ('1d', '1h', '5m').

        Returns:
            DataFrame with DatetimeIndex and [Open, High, Low, Close, Volume].
        """

    def fetch_multiple(
        self,
        tickers: list[str],
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple tickers.

        Default implementation calls fetch() in a loop.
        Subclasses can override for batch-optimised fetching.

        Args:
            tickers: List of ticker symbols.
            start: Start date.
            end: End date.
            interval: Data frequency.

        Returns:
            Dictionary mapping ticker → DataFrame.
        """
        results = {}
        for ticker in tickers:
            try:
                df = self.fetch(ticker, start=start, end=end, interval=interval)
                results[ticker] = df
                logger.info(f"Fetched {ticker}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
        return results

    def save(
        self,
        data: dict[str, pd.DataFrame],
        output_dir: Path,
        fmt: str = "parquet",
    ) -> list[Path]:
        """Save fetched data to disk.

        Args:
            data: Dictionary of ticker → DataFrame.
            output_dir: Directory to write files to.
            fmt: Output format ('parquet' or 'csv').

        Returns:
            List of saved file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        for ticker, df in data.items():
            safe_name = ticker.replace(".", "_").replace("^", "idx_")
            if fmt == "parquet":
                path = output_dir / f"{safe_name}.parquet"
                df.to_parquet(path)
            elif fmt == "csv":
                path = output_dir / f"{safe_name}.csv"
                df.to_csv(path)
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            saved.append(path)
            logger.info(f"Saved {ticker} → {path}")

        return saved


class SyntheticFetcher(DataFetcher):
    """Generates synthetic market data for offline testing.

    Uses geometric Brownian motion to produce realistic-looking OHLCV data.
    No network access required — perfect for CI/CD and development.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fetch(
        self,
        ticker: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for a ticker."""
        from src.data.synthetic import generate_synthetic_ohlcv

        start_str = str(start) if start else "2015-01-02"

        # Calculate approximate number of trading days
        if end:
            start_dt = pd.Timestamp(start_str)
            end_dt = pd.Timestamp(end)
            days = max(int((end_dt - start_dt).days * 252 / 365), 100)
        else:
            days = 2520

        # Deterministic per-ticker seed
        ticker_hash = hash(ticker) % (2**31)
        df = generate_synthetic_ohlcv(
            ticker=ticker,
            days=days,
            start_date=start_str,
            seed=self.seed + ticker_hash,
        )

        # Trim to requested date range
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]

        return validate_ohlcv(df, ticker)


class YFinanceFetcher(DataFetcher):
    """Fetches data from Yahoo Finance via the yfinance library.

    Requires: pip install yfinance
    Requires: Network access to Yahoo Finance API.
    """

    def fetch(
        self,
        ticker: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance not installed. Run: pip install yfinance\n"
                "Or use source='synthetic' for offline testing."
            )

        logger.info(f"Fetching {ticker} from Yahoo Finance ({start} → {end})")
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Standardise column names (yfinance sometimes varies)
        col_map = {
            "Adj Close": "Adj_Close",
            "Stock Splits": "Stock_Splits",
        }
        df = df.rename(columns=col_map)

        # Keep only standard OHLCV columns
        keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep_cols]

        return validate_ohlcv(df, ticker)

    def fetch_multiple(
        self,
        tickers: list[str],
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Batch fetch via yfinance's download() for efficiency."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        logger.info(f"Batch fetching {len(tickers)} tickers from Yahoo Finance")
        raw = yf.download(
            tickers, start=start, end=end, interval=interval, group_by="ticker"
        )

        results = {}
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                else:
                    df = raw[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()

                df = df.dropna(how="all")
                if not df.empty:
                    results[ticker] = validate_ohlcv(df, ticker)
                    logger.info(f"Fetched {ticker}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")

        return results


# ─────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────

_FETCHER_REGISTRY: dict[str, type[DataFetcher]] = {
    "synthetic": SyntheticFetcher,
    "yfinance": YFinanceFetcher,
}


def create_fetcher(source: str = "synthetic", **kwargs: Any) -> DataFetcher:
    """Create a data fetcher by source name.

    Args:
        source: Data source name ('synthetic', 'yfinance').
        **kwargs: Additional arguments passed to the fetcher constructor.

    Returns:
        Configured DataFetcher instance.

    Raises:
        ValueError: If source is not recognised.
    """
    if source not in _FETCHER_REGISTRY:
        raise ValueError(
            f"Unknown data source: '{source}'. "
            f"Available: {list(_FETCHER_REGISTRY.keys())}"
        )
    return _FETCHER_REGISTRY[source](**kwargs)
