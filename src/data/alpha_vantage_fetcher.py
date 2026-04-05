"""Alpha Vantage data fetcher — fallback source when yfinance is unavailable.

Requires an API key from https://www.alphavantage.co/ (free tier: 5 calls/min).
Key is read from config['api_keys']['alpha_vantage'] or the ALPHA_VANTAGE_API_KEY
environment variable.

Usage:
    from src.data.fetcher import create_fetcher
    fetcher = create_fetcher("alpha_vantage")
    data = fetcher.fetch("AAPL", start="2020-01-01")
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import pandas as pd

from src.data.fetcher import DataFetcher, _FETCHER_REGISTRY
from src.utils.logging import get_logger
from src.utils.validators import validate_ohlcv

logger = get_logger(__name__)

# Alpha Vantage free tier allows 5 requests per minute.
_RATE_LIMIT_DELAY = 12.5  # seconds between calls (60 / 5 = 12, with margin)


class AlphaVantageFetcher(DataFetcher):
    """Fetcher for Alpha Vantage API. Used as fallback when yfinance fails.

    Handles rate limiting (5 calls/min on free tier) by sleeping between
    requests. Supports daily and intraday intervals.

    Args:
        config: Optional config dict. If provided, reads the API key from
            config['api_keys']['alpha_vantage'].
        api_key: Explicit API key override. Takes priority over config/env.
    """

    _BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, config: dict[str, Any] | None = None, api_key: str | None = None):
        if api_key:
            self._api_key = api_key
        elif config and config.get("api_keys", {}).get("alpha_vantage"):
            self._api_key = config["api_keys"]["alpha_vantage"]
        else:
            self._api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

        if not self._api_key:
            logger.warning(
                "No Alpha Vantage API key found. Set ALPHA_VANTAGE_API_KEY "
                "or pass via config/constructor."
            )

        self._last_call_time: float = 0.0

    def _wait_for_rate_limit(self) -> None:
        """Sleep if necessary to respect the free-tier rate limit."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < _RATE_LIMIT_DELAY:
            wait = _RATE_LIMIT_DELAY - elapsed
            logger.debug(f"Rate-limiting: sleeping {wait:.1f}s")
            time.sleep(wait)
        self._last_call_time = time.monotonic()

    def _request(self, params: dict[str, str]) -> dict[str, Any]:
        """Make a request to the Alpha Vantage API.

        Args:
            params: Query parameters (function, symbol, etc.).

        Returns:
            Parsed JSON response.

        Raises:
            ImportError: If requests is not installed.
            ValueError: If the API returns an error or no data.
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests not installed. Run: pip install requests\n"
                "Or use source='synthetic' for offline testing."
            )

        params["apikey"] = self._api_key
        self._wait_for_rate_limit()

        logger.debug(f"Alpha Vantage request: {params.get('function')} {params.get('symbol')}")
        response = requests.get(self._BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        if "Note" in data:
            raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")

        return data

    def fetch(
        self,
        ticker: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Alpha Vantage.

        Args:
            ticker: Asset ticker symbol.
            start: Start date (inclusive).
            end: End date (inclusive). Defaults to today.
            interval: Data frequency ('1d' for daily, '1h'/'5m' for intraday).

        Returns:
            DataFrame with DatetimeIndex and [Open, High, Low, Close, Volume].
        """
        if interval in ("1d", "daily"):
            return self._fetch_daily(ticker, start, end)
        else:
            return self._fetch_intraday(ticker, start, end, interval)

    def _fetch_daily(
        self,
        ticker: str,
        start: str | datetime | None,
        end: str | datetime | None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV from Alpha Vantage TIME_SERIES_DAILY."""
        logger.info(f"Fetching {ticker} daily from Alpha Vantage ({start} → {end})")

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "full",
        }
        data = self._request(params)

        time_series_key = "Time Series (Daily)"
        if time_series_key not in data:
            raise ValueError(f"No daily data returned for {ticker}")

        df = self._parse_time_series(data[time_series_key])
        df = self._trim_dates(df, start, end)
        return validate_ohlcv(df, ticker)

    def _fetch_intraday(
        self,
        ticker: str,
        start: str | datetime | None,
        end: str | datetime | None,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch intraday OHLCV from Alpha Vantage TIME_SERIES_INTRADAY."""
        # Map our interval names to Alpha Vantage's format
        interval_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "60min",
        }
        av_interval = interval_map.get(interval, interval)

        logger.info(
            f"Fetching {ticker} intraday ({av_interval}) from Alpha Vantage ({start} → {end})"
        )

        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker,
            "interval": av_interval,
            "outputsize": "full",
        }
        data = self._request(params)

        time_series_key = f"Time Series ({av_interval})"
        if time_series_key not in data:
            raise ValueError(f"No intraday data returned for {ticker}")

        df = self._parse_time_series(data[time_series_key])
        df = self._trim_dates(df, start, end)
        return validate_ohlcv(df, ticker)

    @staticmethod
    def _parse_time_series(raw: dict[str, dict[str, str]]) -> pd.DataFrame:
        """Parse Alpha Vantage time series JSON into a DataFrame.

        Args:
            raw: Dict of date/datetime strings → OHLCV dicts.

        Returns:
            DataFrame with DatetimeIndex and standard OHLCV columns.
        """
        records = []
        for date_str, values in raw.items():
            records.append({
                "Date": pd.Timestamp(date_str),
                "Open": float(values["1. open"]),
                "High": float(values["2. high"]),
                "Low": float(values["3. low"]),
                "Close": float(values["4. close"]),
                "Volume": int(values["5. volume"]),
            })

        df = pd.DataFrame(records)
        df = df.set_index("Date").sort_index()
        return df

    @staticmethod
    def _trim_dates(
        df: pd.DataFrame,
        start: str | datetime | None,
        end: str | datetime | None,
    ) -> pd.DataFrame:
        """Trim DataFrame to requested date range."""
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        if df.empty:
            raise ValueError("No data after trimming to requested date range")
        return df

    def fetch_multiple(
        self,
        tickers: list[str],
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple tickers.

        Handles rate limiting automatically between requests.

        Args:
            tickers: List of ticker symbols.
            start: Start date.
            end: End date.
            interval: Data frequency.

        Returns:
            Dictionary mapping ticker → DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                df = self.fetch(ticker, start=start, end=end, interval=interval)
                results[ticker] = df
                logger.info(f"Fetched {ticker}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to fetch {ticker} from Alpha Vantage: {e}")
        return results


# Register in the fetcher factory
_FETCHER_REGISTRY["alpha_vantage"] = AlphaVantageFetcher
