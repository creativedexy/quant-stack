"""Data service — high-level orchestration for data operations.

Provides a unified interface for fetching historical data, live prices,
and triggering data refreshes. This is the main entry point for the
dashboard and other consumers that need market data.

Usage:
    from src.services.data_service import DataService
    svc = DataService(config=cfg)
    prices = svc.get_live_prices(["AAPL", "MSFT"])
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.data.fetcher import create_fetcher
from src.data.cleaner import DataCleaner
from src.data.live_price import LivePriceService
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataService:
    """High-level data service that coordinates fetchers, cleaners, and live prices.

    Args:
        config: Project configuration dict. If None, uses defaults.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        data_cfg = self._config.get("data", {})

        self._source = data_cfg.get("source", "synthetic")
        self._start_date = data_cfg.get("start_date")
        self._interval = data_cfg.get("interval", "1d")

        self._fetcher = create_fetcher(self._source, config=self._config)
        self._cleaner = DataCleaner()
        self._live_price_service = LivePriceService(config=self._config)

    @property
    def live_price_service(self) -> LivePriceService:
        """Access the underlying LivePriceService for advanced use."""
        return self._live_price_service

    def get_tickers(self) -> list[str]:
        """Return the configured universe tickers."""
        return list(
            self._config.get("universe", {}).get("tickers", [])
        )

    def get_live_prices(self, tickers: list[str] | None = None) -> pd.DataFrame:
        """Get current prices via LivePriceService.

        Args:
            tickers: Ticker symbols to fetch. Defaults to the configured universe.

        Returns:
            DataFrame with columns: price, source, timestamp, delayed.
            Index is the ticker symbol.
        """
        if tickers is None:
            tickers = self.get_tickers()

        if not tickers:
            logger.warning("No tickers specified for live price fetch")
            return pd.DataFrame(columns=["price", "source", "timestamp", "delayed"])

        logger.info(f"Fetching live prices for {len(tickers)} tickers")
        return self._live_price_service.get_prices(tickers)

    def get_intraday_prices(
        self,
        ticker: str,
        period: str = "1d",
    ) -> pd.DataFrame:
        """Get intraday price data if available (yfinance or Alpha Vantage).

        Falls back to fetching the last day of daily data if intraday is
        not supported by the configured source.

        Args:
            ticker: Asset ticker symbol.
            period: Period to fetch ('1d', '5d'). Interpretation depends on source.

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex.
        """
        logger.info(f"Fetching intraday data for {ticker} (period={period})")

        # Try yfinance first for intraday (it supports 1m/5m intervals)
        try:
            import yfinance as yf
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(period=period, interval="5m")
            if not df.empty:
                keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                return df[keep]
        except ImportError:
            logger.debug("yfinance not available for intraday")
        except Exception as e:
            logger.warning(f"yfinance intraday failed for {ticker}: {e}")

        # Fallback: use configured fetcher with daily interval
        logger.debug(f"Falling back to daily data for {ticker}")
        df = self._fetcher.fetch(ticker, interval="1d")
        # Return last few rows as a proxy
        return df.tail(5)

    def refresh_data(self) -> dict[str, Any]:
        """Trigger a data refresh for all configured tickers.

        Fetches the latest data, cleans it, and returns a status dict.

        Returns:
            Status dict with keys: status, tickers_updated, errors, timestamp.
        """
        tickers = self.get_tickers()
        if not tickers:
            return {
                "status": "warning",
                "tickers_updated": [],
                "errors": ["No tickers configured"],
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }

        logger.info(f"Refreshing data for {len(tickers)} tickers")
        errors: list[str] = []
        updated: list[str] = []

        try:
            raw_data = self._fetcher.fetch_multiple(
                tickers, start=self._start_date
            )
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            return {
                "status": "error",
                "tickers_updated": [],
                "errors": [str(e)],
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }

        for ticker in tickers:
            if ticker not in raw_data:
                errors.append(f"{ticker}: no data returned")
                continue
            try:
                self._cleaner.clean(raw_data[ticker], ticker=ticker)
                updated.append(ticker)
            except Exception as e:
                errors.append(f"{ticker}: {e}")

        status = "ok" if not errors else ("partial" if updated else "error")
        return {
            "status": status,
            "tickers_updated": updated,
            "errors": errors,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
