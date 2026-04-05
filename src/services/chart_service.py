"""Chart service for the production UI.

Provides historical price data with technical overlays (Bollinger Bands,
RSI, MACD), ML confidence series, and forensic trade analysis.

All methods return plain dicts or lists -- no DataFrames leak to templates.
Uses ``src.features.technical`` compute functions exclusively.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.data.fetcher import create_fetcher
from src.data.cleaner import DataCleaner
from src.features.technical import (
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Period label -> approximate trading days
_PERIOD_DAYS: dict[str, int] = {
    "1M": 21,
    "3M": 63,
    "1Y": 252,
    "3Y": 756,
    "All": 9999,
}


class ChartService:
    """Provides chart data for the production UI.

    Args:
        config: Application configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._fetcher = create_fetcher("synthetic", seed=42)
        self._cleaner = DataCleaner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_price_history(
        self,
        ticker: str,
        period: str = "1Y",
    ) -> dict[str, Any]:
        """Historical OHLCV with Bollinger Bands and benchmark overlay.

        Args:
            ticker: Instrument ticker (e.g. ``"SHEL.L"``).
            period: One of ``"1M"``, ``"3M"``, ``"1Y"``, ``"3Y"``, ``"All"``.

        Returns:
            Dict with keys: ``ticker``, ``period``, ``dates``, ``open``,
            ``high``, ``low``, ``close``, ``volume``, ``bb_upper``,
            ``bb_middle``, ``bb_lower``, ``benchmark_dates``,
            ``benchmark_close``, ``trade_entries``.
        """
        df = self._fetch_ohlcv(ticker)
        n_days = _PERIOD_DAYS.get(period, 252)
        df = df.iloc[-n_days:]

        if df.empty:
            return self._empty_price_history(ticker, period)

        # Bollinger Bands
        bb = compute_bollinger_bands(df)
        bb = bb.reindex(df.index)

        # Benchmark
        benchmark_dates: list[str] = []
        benchmark_close: list[float] = []
        benchmark_ticker = self._config.get("universe", {}).get("benchmark", "^FTSE")
        try:
            bm_df = self._fetch_ohlcv(benchmark_ticker)
            bm_df = bm_df.loc[bm_df.index.isin(df.index) | (bm_df.index >= df.index[0])]
            bm_df = bm_df.loc[bm_df.index <= df.index[-1]]
            if not bm_df.empty and not df.empty:
                # Normalise so benchmark[0] == stock close[0]
                scale = float(df["Close"].iloc[0]) / float(bm_df["Close"].iloc[0])
                benchmark_dates = [d.strftime("%Y-%m-%d") for d in bm_df.index]
                benchmark_close = [round(float(v) * scale, 2) for v in bm_df["Close"]]
        except Exception as exc:
            logger.warning("Benchmark fetch failed for %s: %s", benchmark_ticker, exc)

        # Trade entries (from execution history if available)
        trade_entries = self._get_trade_entries(ticker, df.index[0], df.index[-1])

        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        return {
            "ticker": ticker,
            "period": period,
            "dates": dates,
            "open": [round(float(v), 2) for v in df["Open"]],
            "high": [round(float(v), 2) for v in df["High"]],
            "low": [round(float(v), 2) for v in df["Low"]],
            "close": [round(float(v), 2) for v in df["Close"]],
            "volume": [int(v) for v in df["Volume"]],
            "bb_upper": _safe_series(bb.get("bb_upper")),
            "bb_middle": _safe_series(bb.get("bb_middle")),
            "bb_lower": _safe_series(bb.get("bb_lower")),
            "benchmark_dates": benchmark_dates,
            "benchmark_close": benchmark_close,
            "trade_entries": trade_entries,
        }

    def get_rsi_series(
        self,
        ticker: str,
        period: str = "1Y",
        window: int = 14,
    ) -> dict[str, Any]:
        """RSI indicator values bounded [0, 100].

        Args:
            ticker: Instrument ticker.
            period: Period label.
            window: RSI look-back window.

        Returns:
            Dict with ``dates`` and ``rsi`` lists.
        """
        df = self._fetch_ohlcv(ticker)

        if df.empty:
            return {"dates": [], "rsi": [], "window": window}

        # Compute RSI on ALL available history so the indicator is
        # fully warmed up, then slice to the display period.
        rsi_df = compute_rsi(df, window=window)
        col = f"rsi_{window}"
        rsi_values = rsi_df[col] if col in rsi_df.columns else rsi_df.iloc[:, 0]

        n_days = _PERIOD_DAYS.get(period, 252)
        df = df.iloc[-n_days:]
        rsi_values = rsi_values.iloc[-n_days:]

        # Clamp to [0, 100] and handle NaN
        clamped: list[float | None] = []
        for v in rsi_values:
            if pd.isna(v):
                clamped.append(None)
            else:
                clamped.append(round(float(max(0.0, min(100.0, v))), 2))

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "rsi": clamped,
            "window": window,
        }

    def get_macd_series(
        self,
        ticker: str,
        period: str = "1Y",
    ) -> dict[str, Any]:
        """MACD line, signal, and histogram with colour coding.

        Returns:
            Dict with ``dates``, ``macd_line``, ``macd_signal``,
            ``macd_histogram``, and ``histogram_colours``.
        """
        df = self._fetch_ohlcv(ticker)

        if df.empty:
            return {
                "dates": [],
                "macd_line": [],
                "macd_signal": [],
                "macd_histogram": [],
                "histogram_colours": [],
            }

        # Compute MACD on ALL history (needs 26+9 bars warm-up),
        # then slice to display period.
        macd_df = compute_macd(df)
        macd_df = macd_df.reindex(df.index)

        n_days = _PERIOD_DAYS.get(period, 252)
        df = df.iloc[-n_days:]
        macd_df = macd_df.iloc[-n_days:]

        hist_colours: list[str] = []
        for v in macd_df["macd_histogram"]:
            if pd.isna(v):
                hist_colours.append("green")
            else:
                hist_colours.append("green" if float(v) >= 0 else "red")

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "macd_line": _safe_series(macd_df.get("macd_line")),
            "macd_signal": _safe_series(macd_df.get("macd_signal")),
            "macd_histogram": _safe_series(macd_df.get("macd_histogram")),
            "histogram_colours": hist_colours,
        }

    def get_ml_confidence_series(
        self,
        ticker: str,
        period: str = "1Y",
    ) -> dict[str, Any]:
        """ML model confidence scores with direction.

        Falls back to synthetic confidence if no real model exists.

        Returns:
            Dict with ``dates``, ``confidence``, ``direction``.
        """
        df = self._fetch_ohlcv(ticker)
        n_days = _PERIOD_DAYS.get(period, 252)
        df = df.iloc[-n_days:]

        if df.empty:
            return {"dates": [], "confidence": [], "direction": []}

        # Generate synthetic confidence based on price momentum
        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        close = df["Close"].values
        confidence: list[float] = []
        direction: list[str] = []

        rng = np.random.default_rng(hash(ticker) % (2**31))
        for i in range(len(close)):
            if i < 5:
                confidence.append(0.5)
                direction.append("NEUTRAL")
            else:
                # 5-day momentum as base signal
                momentum = (close[i] - close[i - 5]) / close[i - 5]
                noise = rng.normal(0, 0.02)
                raw = 0.5 + momentum * 5 + noise
                conf = round(float(max(0.0, min(1.0, raw))), 2)
                confidence.append(conf)
                if conf > 0.6:
                    direction.append("LONG")
                elif conf < 0.4:
                    direction.append("SHORT")
                else:
                    direction.append("NEUTRAL")

        return {
            "dates": dates,
            "confidence": confidence,
            "direction": direction,
        }

    def get_forensic_analysis(
        self,
        ticker: str,
        entry_date: str,
        entry_price: float,
    ) -> dict[str, Any]:
        """Forensic analysis of a trade entry.

        Looks at a +/-5 day window around the entry to assess timing.

        Args:
            ticker: Instrument ticker.
            entry_date: Entry date as ``"YYYY-MM-DD"``.
            entry_price: Price at entry.

        Returns:
            Dict with ``ticker``, ``entry_date``, ``entry_price``,
            ``window_dates``, ``window_close``, ``optimal_date``,
            ``optimal_price``, ``slippage_pct``, ``position_size``.
        """
        df = self._fetch_ohlcv(ticker)
        if df.empty:
            return self._empty_forensic(ticker, entry_date, entry_price)

        try:
            target = pd.Timestamp(entry_date)
        except Exception:
            return self._empty_forensic(ticker, entry_date, entry_price)

        # Find nearest trading day
        idx = df.index.get_indexer([target], method="nearest")[0]
        if idx < 0:
            return self._empty_forensic(ticker, entry_date, entry_price)

        start = max(0, idx - 5)
        end = min(len(df), idx + 6)
        window_df = df.iloc[start:end]

        window_dates = [d.strftime("%Y-%m-%d") for d in window_df.index]
        window_close = [round(float(v), 2) for v in window_df["Close"]]

        # Optimal entry = lowest close in window (assuming buy)
        min_idx = int(window_df["Close"].values.argmin())
        optimal_date = window_dates[min_idx]
        optimal_price = window_close[min_idx]

        slippage_pct = round(
            (entry_price - optimal_price) / optimal_price * 100, 2
        ) if optimal_price > 0 else 0.0

        # Position size from config
        risk_cfg = self._config.get("risk", {})
        position_size = risk_cfg.get("default_position_size_shares", 1000)

        return {
            "ticker": ticker,
            "entry_date": entry_date,
            "entry_price": round(entry_price, 2),
            "window_dates": window_dates,
            "window_close": window_close,
            "optimal_date": optimal_date,
            "optimal_price": optimal_price,
            "slippage_pct": slippage_pct,
            "position_size": position_size,
        }

    # ------------------------------------------------------------------
    # SVG path generation
    # ------------------------------------------------------------------

    def generate_price_svg(
        self,
        prices: pd.Series,
        width: int = 800,
        height: int = 300,
    ) -> dict[str, Any]:
        """Generate SVG path data from a price series.

        Args:
            prices: Series of close prices with DatetimeIndex.
            width: SVG viewBox width.
            height: SVG viewBox height.

        Returns:
            Dict with ``path_d`` (line), ``area_d`` (filled area),
            ``gradient_id``, ``colour`` (green if up, red if down),
            ``width``, ``height``.
        """
        if prices.empty or len(prices) < 2:
            return {
                "path_d": "",
                "area_d": "",
                "gradient_id": "grad-flat",
                "colour": "var(--t3)",
                "width": width,
                "height": height,
            }

        values = prices.dropna().values.astype(float)
        n = len(values)
        v_min, v_max = float(values.min()), float(values.max())
        v_range = v_max - v_min if v_max != v_min else 1.0
        pad = height * 0.05

        points: list[tuple[float, float]] = []
        for i, v in enumerate(values):
            x = (i / (n - 1)) * width
            y = pad + (1 - (v - v_min) / v_range) * (height - 2 * pad)
            points.append((round(x, 1), round(y, 1)))

        path_d = "M" + " L".join(f"{x},{y}" for x, y in points)
        area_d = path_d + f" L{points[-1][0]},{height} L{points[0][0]},{height} Z"

        colour = "var(--green)" if values[-1] >= values[0] else "var(--red)"
        gradient_id = "grad-up" if values[-1] >= values[0] else "grad-dn"

        return {
            "path_d": path_d,
            "area_d": area_d,
            "gradient_id": gradient_id,
            "colour": colour,
            "width": width,
            "height": height,
        }

    def generate_bollinger_svg(
        self,
        prices: pd.Series,
        window: int = 20,
        width: int = 800,
        height: int = 300,
    ) -> dict[str, str]:
        """Generate SVG path data for Bollinger Bands.

        Args:
            prices: Series of close prices.
            window: Bollinger band window.
            width: SVG viewBox width.
            height: SVG viewBox height.

        Returns:
            Dict with ``upper_d``, ``lower_d``, ``mid_d`` path strings.
        """
        if prices.empty or len(prices) < window:
            return {"upper_d": "", "lower_d": "", "mid_d": ""}

        df = pd.DataFrame({"Close": prices})
        bb = compute_bollinger_bands(df, window=window)
        bb = bb.reindex(prices.index)

        vals = prices.dropna().values.astype(float)
        all_vals = np.concatenate([
            vals,
            bb["bb_upper"].dropna().values.astype(float),
            bb["bb_lower"].dropna().values.astype(float),
        ])
        v_min, v_max = float(all_vals.min()), float(all_vals.max())
        v_range = v_max - v_min if v_max != v_min else 1.0
        pad = height * 0.05
        n = len(prices)

        def _to_path(series: pd.Series) -> str:
            pts: list[str] = []
            for i, v in enumerate(series):
                if pd.isna(v):
                    continue
                x = round((i / (n - 1)) * width, 1)
                y = round(pad + (1 - (float(v) - v_min) / v_range) * (height - 2 * pad), 1)
                pts.append(f"{x},{y}")
            if not pts:
                return ""
            return "M" + " L".join(pts)

        return {
            "upper_d": _to_path(bb["bb_upper"]),
            "lower_d": _to_path(bb["bb_lower"]),
            "mid_d": _to_path(bb["bb_middle"]),
        }

    def generate_trade_dots(
        self,
        trades: list[dict[str, Any]],
        prices: pd.Series,
        width: int = 800,
        height: int = 300,
    ) -> list[dict[str, Any]]:
        """Generate SVG circle positions for trade entry dots.

        Args:
            trades: List of trade dicts with ``date`` and ``price`` keys.
            prices: Series of close prices.
            width: SVG viewBox width.
            height: SVG viewBox height.

        Returns:
            List of dicts with ``cx``, ``cy``, ``colour``, ``trade_data``.
        """
        if prices.empty or not trades:
            return []

        values = prices.dropna().values.astype(float)
        n = len(values)
        v_min, v_max = float(values.min()), float(values.max())
        v_range = v_max - v_min if v_max != v_min else 1.0
        pad = height * 0.05

        dates = prices.index
        dots: list[dict[str, Any]] = []
        for trade in trades:
            try:
                t_date = pd.Timestamp(trade.get("date", ""))
                t_price = float(trade.get("price", 0))
                idx = dates.get_indexer([t_date], method="nearest")[0]
                if idx < 0 or idx >= n:
                    continue
                cx = round((idx / (n - 1)) * width, 1)
                cy = round(pad + (1 - (t_price - v_min) / v_range) * (height - 2 * pad), 1)
                colour = "var(--green)" if trade.get("side", "buy") == "buy" else "var(--red)"
                dots.append({"cx": cx, "cy": cy, "colour": colour, "trade_data": trade})
            except Exception:
                continue
        return dots

    def generate_prev_close_line(
        self,
        prev_close: float,
        price_range: tuple[float, float],
        height: int = 300,
    ) -> float:
        """Return the y-coordinate for a previous close dashed line.

        Args:
            prev_close: Previous close price.
            price_range: Tuple of (min_price, max_price).
            height: SVG viewBox height.

        Returns:
            y-coordinate in SVG viewBox units.
        """
        v_min, v_max = price_range
        v_range = v_max - v_min if v_max != v_min else 1.0
        pad = height * 0.05
        return round(pad + (1 - (prev_close - v_min) / v_range) * (height - 2 * pad), 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Fetch and clean OHLCV data, falling back to synthetic."""
        try:
            df = self._fetcher.fetch(ticker, start="2018-01-01")
            return self._cleaner.clean(df, ticker)
        except Exception as exc:
            logger.warning("Fetch failed for %s, using synthetic: %s", ticker, exc)
            return self._generate_synthetic(ticker)

    def _generate_synthetic(self, ticker: str) -> pd.DataFrame:
        """Generate synthetic OHLCV data as fallback."""
        from src.data.synthetic import generate_synthetic_ohlcv

        try:
            return generate_synthetic_ohlcv(
                ticker=ticker,
                start="2020-01-01",
                end="2024-12-31",
                seed=hash(ticker) % (2**31),
            )
        except Exception as exc:
            logger.error("Synthetic generation failed for %s: %s", ticker, exc)
            return pd.DataFrame()

    def _get_trade_entries(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        """Return trade entries for a ticker within a date range.

        Currently returns an empty list. Will integrate with
        ExecutionService when trade history is available.
        """
        return []

    def _empty_price_history(self, ticker: str, period: str) -> dict[str, Any]:
        """Return empty price history dict."""
        return {
            "ticker": ticker,
            "period": period,
            "dates": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "bb_upper": [],
            "bb_middle": [],
            "bb_lower": [],
            "benchmark_dates": [],
            "benchmark_close": [],
            "trade_entries": [],
        }

    def _empty_forensic(
        self,
        ticker: str,
        entry_date: str,
        entry_price: float,
    ) -> dict[str, Any]:
        """Return empty forensic analysis dict."""
        return {
            "ticker": ticker,
            "entry_date": entry_date,
            "entry_price": round(entry_price, 2),
            "window_dates": [],
            "window_close": [],
            "optimal_date": entry_date,
            "optimal_price": round(entry_price, 2),
            "slippage_pct": 0.0,
            "position_size": self._config.get("risk", {}).get(
                "default_position_size_shares", 1000
            ),
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _safe_series(series: pd.Series | None) -> list[float | None]:
    """Convert a pandas Series to a list, replacing NaN with None."""
    if series is None:
        return []
    result: list[float | None] = []
    for v in series:
        if pd.isna(v):
            result.append(None)
        else:
            result.append(round(float(v), 4))
    return result
