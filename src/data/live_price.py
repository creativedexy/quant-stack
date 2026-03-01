"""Live price service — current prices with a configurable fallback chain.

Tries sources in order (default: IB → yfinance → Alpha Vantage → cached)
and returns the first successful result. Each source has a configurable
timeout, and failures are logged before trying the next source.

Usage:
    from src.data.live_price import LivePriceService
    svc = LivePriceService(config=cfg)
    price = svc.get_price("AAPL")
    # {"price": 187.42, "source": "yfinance", "timestamp": ..., "delayed": True}
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class LivePriceService:
    """Provides current prices with a fallback chain.

    Tries sources in order:
        1. IB real-time (if connected)
        2. yfinance (delayed ~15 min)
        3. Alpha Vantage
        4. Last cached price from parquet files

    Each source has a timeout. If a source fails, the error is logged and the
    next source in the chain is attempted.

    Args:
        config: Project configuration dict. Used to read fallback_chain,
            cache TTL, request timeout, and data directories.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        live_cfg = self._config.get("data", {}).get("live", {})

        self._cache_ttl: float = float(live_cfg.get("price_cache_ttl", 60))
        self._timeout: float = float(live_cfg.get("request_timeout", 10))

        chain_names = live_cfg.get("fallback_chain", ["yfinance", "alpha_vantage", "cached"])
        self.sources: list[str] = list(chain_names)

        self._cache: dict[str, tuple[dict[str, Any], float]] = {}

    # ── Public API ────────────────────────────────────────────

    def get_price(self, ticker: str) -> dict[str, Any]:
        """Get the current price for a single ticker.

        Walks the fallback chain until a source returns a valid price.

        Args:
            ticker: Asset ticker symbol.

        Returns:
            Dict with keys: price (float), source (str),
            timestamp (datetime), delayed (bool).

        Raises:
            RuntimeError: If all sources fail.
        """
        # Check in-memory cache first
        cached = self._check_cache(ticker)
        if cached is not None:
            return cached

        errors: list[str] = []
        for source in self.sources:
            try:
                result = self._fetch_from_source(source, ticker)
                if result is not None:
                    self._update_cache(ticker, result)
                    return result
            except Exception as e:
                msg = f"{source} failed for {ticker}: {e}"
                logger.warning(msg)
                errors.append(msg)

        raise RuntimeError(
            f"All price sources failed for {ticker}. Errors: {'; '.join(errors)}"
        )

    def get_prices(self, tickers: list[str]) -> pd.DataFrame:
        """Batch price fetch for multiple tickers.

        Args:
            tickers: List of ticker symbols.

        Returns:
            DataFrame with columns: price, source, timestamp, delayed.
            Index is the ticker symbol.
        """
        records = []
        for ticker in tickers:
            try:
                result = self.get_price(ticker)
                records.append({"ticker": ticker, **result})
            except RuntimeError as e:
                logger.error(str(e))
                records.append({
                    "ticker": ticker,
                    "price": float("nan"),
                    "source": "unavailable",
                    "timestamp": datetime.now(tz=timezone.utc),
                    "delayed": True,
                })

        df = pd.DataFrame(records).set_index("ticker")
        return df

    def get_price_source_status(self) -> dict[str, dict[str, Any]]:
        """Report which price sources are currently available.

        Returns:
            Dict mapping source name → {"available": bool, "detail": str}.
        """
        status: dict[str, dict[str, Any]] = {}
        for source in self.sources:
            available, detail = self._check_source_available(source)
            status[source] = {"available": available, "detail": detail}
        return status

    # ── Cache management ──────────────────────────────────────

    def _check_cache(self, ticker: str) -> dict[str, Any] | None:
        """Return cached price if within TTL, else None."""
        if ticker not in self._cache:
            return None
        result, cached_at = self._cache[ticker]
        if (time.monotonic() - cached_at) < self._cache_ttl:
            logger.debug(f"Cache hit for {ticker}")
            return result
        return None

    def _update_cache(self, ticker: str, result: dict[str, Any]) -> None:
        """Store a price result in the in-memory cache."""
        self._cache[ticker] = (result, time.monotonic())

    # ── Source dispatching ────────────────────────────────────

    def _fetch_from_source(self, source: str, ticker: str) -> dict[str, Any] | None:
        """Dispatch a price fetch to the named source.

        Args:
            source: Source name ('ib', 'yfinance', 'alpha_vantage', 'cached').
            ticker: Asset ticker symbol.

        Returns:
            Price dict or None if the source cannot provide a price.
        """
        dispatch = {
            "ib": self._fetch_ib,
            "yfinance": self._fetch_yfinance,
            "alpha_vantage": self._fetch_alpha_vantage,
            "cached": self._fetch_cached,
        }
        handler = dispatch.get(source)
        if handler is None:
            logger.warning(f"Unknown price source: {source}")
            return None
        return handler(ticker)

    def _fetch_ib(self, ticker: str) -> dict[str, Any] | None:
        """Fetch real-time price from Interactive Brokers.

        Returns None if IB is not connected or ibapi is not installed.
        """
        try:
            from ibapi.client import EClient  # noqa: F401
        except ImportError:
            logger.debug("ibapi not installed — skipping IB source")
            return None

        # IB integration is Phase 6; return None until broker module is ready.
        logger.debug("IB source not yet implemented — skipping")
        return None

    def _fetch_yfinance(self, ticker: str) -> dict[str, Any] | None:
        """Fetch latest price from Yahoo Finance (delayed ~15 min)."""
        try:
            import yfinance as yf
        except ImportError:
            logger.debug("yfinance not installed — skipping")
            return None

        logger.debug(f"Fetching live price for {ticker} via yfinance")
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(period="1d")

        if hist.empty:
            return None

        last_row = hist.iloc[-1]
        return {
            "price": float(last_row["Close"]),
            "source": "yfinance",
            "timestamp": datetime.now(tz=timezone.utc),
            "delayed": True,
        }

    def _fetch_alpha_vantage(self, ticker: str) -> dict[str, Any] | None:
        """Fetch latest price from Alpha Vantage GLOBAL_QUOTE endpoint."""
        try:
            import requests
        except ImportError:
            logger.debug("requests not installed — skipping Alpha Vantage")
            return None

        api_key = self._config.get("api_keys", {}).get("alpha_vantage", "")
        if not api_key:
            import os
            api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            logger.debug("No Alpha Vantage API key — skipping")
            return None

        logger.debug(f"Fetching live price for {ticker} via Alpha Vantage")
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": api_key,
        }
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params=params,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        quote = data.get("Global Quote", {})
        price_str = quote.get("05. price")
        if not price_str:
            return None

        return {
            "price": float(price_str),
            "source": "alpha_vantage",
            "timestamp": datetime.now(tz=timezone.utc),
            "delayed": True,
        }

    def _fetch_cached(self, ticker: str) -> dict[str, Any] | None:
        """Return the last known price from local parquet files."""
        data_dir = self._config.get("general", {}).get("data_dir", "data")
        processed_dir = Path(data_dir) / "processed"

        safe_name = ticker.replace(".", "_").replace("^", "idx_")
        parquet_path = processed_dir / f"{safe_name}.parquet"

        if not parquet_path.exists():
            logger.debug(f"No cached parquet for {ticker} at {parquet_path}")
            return None

        try:
            df = pd.read_parquet(parquet_path)
            if df.empty:
                return None

            last_row = df.iloc[-1]
            last_date = df.index[-1]
            return {
                "price": float(last_row["Close"]),
                "source": "cached",
                "timestamp": pd.Timestamp(last_date).to_pydatetime().replace(
                    tzinfo=timezone.utc
                ),
                "delayed": True,
            }
        except Exception as e:
            logger.warning(f"Failed to read cached data for {ticker}: {e}")
            return None

    # ── Source availability checks ────────────────────────────

    def _check_source_available(self, source: str) -> tuple[bool, str]:
        """Check whether a given source is currently reachable.

        Returns:
            Tuple of (available, detail_message).
        """
        if source == "ib":
            try:
                from ibapi.client import EClient  # noqa: F401
                return False, "ibapi installed but broker not connected"
            except ImportError:
                return False, "ibapi not installed"

        if source == "yfinance":
            try:
                import yfinance  # noqa: F401
                return True, "yfinance installed"
            except ImportError:
                return False, "yfinance not installed"

        if source == "alpha_vantage":
            api_key = self._config.get("api_keys", {}).get("alpha_vantage", "")
            if not api_key:
                import os
                api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            if api_key:
                return True, "API key configured"
            return False, "No API key"

        if source == "cached":
            data_dir = self._config.get("general", {}).get("data_dir", "data")
            processed_dir = Path(data_dir) / "processed"
            if processed_dir.exists():
                return True, f"Parquet cache at {processed_dir}"
            return False, f"No processed data directory at {processed_dir}"

        return False, f"Unknown source: {source}"
