"""Binance price feed for live and historical OHLCV data.

Provides current price, midprice, historical OHLCV, and price change
calculations via the Binance REST API.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from src.utils.exceptions import DataFeedError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BinancePriceFeed:
    """Fetches price data from Binance spot markets.

    Args:
        client: An initialised python-binance Client instance.
        config: Settings dict with binance.request_timeout_seconds.
    """

    def __init__(self, client: object, config: dict) -> None:
        self._client = client
        self._config = config
        self._timeout = config.get("binance", {}).get("request_timeout_seconds", 10)
        self._max_retries = 2

    def get_current_price(self, symbol: str) -> float:
        """Return the latest trade price for a symbol.

        Args:
            symbol: Trading pair (e.g. 'BTCGBP').

        Returns:
            Latest trade price as float.

        Raises:
            DataFeedError: If the price cannot be fetched.
        """
        response = self._retry_call(self._client.get_symbol_ticker, symbol=symbol)
        return float(response["price"])

    def get_midprice(self, symbol: str) -> float:
        """Return (best_bid + best_ask) / 2 from the order book top.

        Args:
            symbol: Trading pair (e.g. 'BTCGBP').

        Returns:
            Midprice as float.

        Raises:
            DataFeedError: If the order book cannot be fetched.
        """
        book = self._retry_call(self._client.get_order_book, symbol=symbol, limit=1)
        best_bid = float(book["bids"][0][0])
        best_ask = float(book["asks"][0][0])
        return (best_bid + best_ask) / 2.0

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        lookback_periods: int,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame with DatetimeIndex in UTC.

        Args:
            symbol: Trading pair (e.g. 'BTCGBP').
            interval: Kline interval (e.g. '1h', '1d').
            lookback_periods: Number of periods to fetch.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
            Index is UTC-aware DatetimeIndex.

        Raises:
            DataFeedError: If data contains NaN values or fetch fails.
        """
        raw = self._retry_call(
            self._client.get_klines,
            symbol=symbol,
            interval=interval,
            limit=lookback_periods,
        )

        if not raw:
            raise DataFeedError(f"No OHLCV data returned for {symbol}")

        rows = []
        for kline in raw:
            ts = datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc)
            rows.append(
                {
                    "timestamp": ts,
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                }
            )

        df = pd.DataFrame(rows)
        df.index = pd.DatetimeIndex(df["timestamp"])

        if df[["open", "high", "low", "close", "volume"]].isna().any().any():
            nan_count = int(df.isna().sum().sum())
            raise DataFeedError(
                f"OHLCV data for {symbol} contains {nan_count} NaN values"
            )

        logger.debug(
            "ohlcv_fetched",
            symbol=symbol,
            interval=interval,
            rows=len(df),
        )

        return df

    def get_price_change_pct(self, symbol: str, hours: int) -> float:
        """Return percentage price change over the last N hours.

        Args:
            symbol: Trading pair (e.g. 'BTCGBP').
            hours: Number of hours to look back.

        Returns:
            Percentage change (negative = price fell).

        Raises:
            DataFeedError: If data cannot be fetched.
        """
        df = self.get_ohlcv(symbol, "1h", hours)
        if len(df) < 2:
            raise DataFeedError(
                f"Insufficient OHLCV data for {symbol}: need >= 2 rows, got {len(df)}"
            )

        first_close = df["close"].iloc[0]
        last_close = df["close"].iloc[-1]
        return ((last_close - first_close) / first_close) * 100.0

    def _retry_call(self, method: object, **kwargs: object) -> object:
        """Call a Binance client method with one retry on failure.

        Args:
            method: The client method to call.
            **kwargs: Arguments to pass to the method.

        Returns:
            The method's return value.

        Raises:
            DataFeedError: If both attempts fail.
        """
        last_error = None
        for attempt in range(self._max_retries):
            try:
                return method(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    logger.warning(
                        "binance_call_retry",
                        method=str(method),
                        attempt=attempt + 1,
                        error=str(exc),
                    )
                    time.sleep(2)

        raise DataFeedError(
            f"Binance API call failed after {self._max_retries} attempts: {last_error}"
        )
