"""Market data service -- live/synthetic intraday prices and fundamentals.

Provides real-time market data for the watchlist UI via yfinance,
with synthetic fallback for offline development and testing.

Usage:
    from src.services.market_data_service import MarketDataService
    svc = MarketDataService(config, strategy_service)
    card = svc.get_watchlist_cards(["BARC.L", "SHEL.L"])
"""

from __future__ import annotations

import hashlib
import random
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Cache TTL in seconds (default 15 minutes)
_DEFAULT_CACHE_TTL = 900

# Synthetic company metadata keyed by ticker
_SYNTHETIC_COMPANIES: dict[str, dict[str, Any]] = {
    "BARC.L": {
        "name": "Barclays plc",
        "volume": 48_200_000,
        "market_cap": 37_200_000_000,
        "pe_ratio": 7.2,
        "div_yield": 0.038,
        "profit_margin": 0.284,
        "operating_margin": 0.341,
        "prev_close": 245.80,
        "summary": (
            "Barclays is a diversified UK bank with operations in retail, "
            "corporate, and investment banking. The stock trades on a low "
            "P/E with moderate dividend yield."
        ),
    },
    "SHEL.L": {
        "name": "Shell plc",
        "volume": 12_300_000,
        "market_cap": 178_500_000_000,
        "pe_ratio": 11.3,
        "div_yield": 0.041,
        "profit_margin": 0.092,
        "operating_margin": 0.138,
        "prev_close": 2684.00,
        "summary": (
            "Shell is a global energy and petrochemicals company. "
            "Revenue tied to commodity prices with steady capital returns."
        ),
    },
    "AZN.L": {
        "name": "AstraZeneca plc",
        "volume": 3_400_000,
        "market_cap": 185_600_000_000,
        "pe_ratio": 35.2,
        "div_yield": 0.021,
        "profit_margin": 0.157,
        "operating_margin": 0.224,
        "prev_close": 11520.00,
        "summary": (
            "AstraZeneca is a UK-Swedish pharma company with a strong "
            "oncology pipeline. Premium valuation reflects growth outlook."
        ),
    },
    "HSBA.L": {
        "name": "HSBC Holdings plc",
        "volume": 22_100_000,
        "market_cap": 152_300_000_000,
        "pe_ratio": 8.1,
        "div_yield": 0.052,
        "profit_margin": 0.312,
        "operating_margin": 0.384,
        "prev_close": 785.60,
        "summary": (
            "HSBC is one of the world's largest banking groups with "
            "strong Asia-Pacific exposure. High dividend yield."
        ),
    },
    "BP.L": {
        "name": "BP plc",
        "volume": 28_600_000,
        "market_cap": 82_400_000_000,
        "pe_ratio": 12.8,
        "div_yield": 0.045,
        "profit_margin": 0.048,
        "operating_margin": 0.071,
        "prev_close": 425.30,
        "summary": (
            "BP is a global energy major transitioning towards "
            "lower-carbon operations. Moderate yield with commodity risk."
        ),
    },
    "LLOY.L": {
        "name": "Lloyds Banking Group plc",
        "volume": 85_300_000,
        "market_cap": 42_800_000_000,
        "pe_ratio": 8.4,
        "div_yield": 0.047,
        "profit_margin": 0.198,
        "operating_margin": 0.267,
        "prev_close": 64.20,
        "summary": (
            "Lloyds is the UK's largest retail bank by deposit share. "
            "Domestic-focused with sensitivity to UK interest rates."
        ),
    },
}


def _is_lse_ticker(ticker: str) -> bool:
    """Check whether a ticker is London Stock Exchange listed."""
    return ticker.upper().endswith(".L")


def _currency_symbol(ticker: str) -> str:
    """Return the appropriate currency symbol for a ticker."""
    if _is_lse_ticker(ticker):
        return "\u00a3"
    return "$"


def _format_volume(vol: float | int | None) -> str:
    """Format volume with appropriate suffix.

    Rules:
        <1000     -> raw number
        <1M       -> "Xk"
        <1B       -> "X.XM"
        else      -> "X.XB"
    """
    if vol is None or vol == 0:
        return "N/A"
    v = float(vol)
    if v < 1_000:
        return str(int(v))
    if v < 1_000_000:
        return f"{v / 1_000:.0f}k"
    if v < 1_000_000_000:
        return f"{v / 1_000_000:.1f}M"
    return f"{v / 1_000_000_000:.1f}B"


def _format_market_cap(cap: float | int | None, ticker: str) -> str:
    """Format market cap with currency symbol and suffix."""
    if cap is None or cap == 0:
        return "N/A"
    sym = _currency_symbol(ticker)
    v = float(cap)
    if v < 1_000_000:
        return f"{sym}{v:,.0f}"
    if v < 1_000_000_000:
        return f"{sym}{v / 1_000_000:.1f}M"
    return f"{sym}{v / 1_000_000_000:.1f}B"


def _format_pe(pe: float | None) -> str:
    """Format P/E ratio to 1 decimal place, or N/A."""
    if pe is None:
        return "N/A"
    try:
        return f"{float(pe):.1f}x"
    except (TypeError, ValueError):
        return "N/A"


def _format_pct(val: float | None) -> str:
    """Format a decimal as a percentage, or N/A."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _ticker_seed(ticker: str) -> int:
    """Deterministic seed from ticker string for reproducible synthetic data."""
    return int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)


class MarketDataService:
    """Live and synthetic market data for the watchlist UI.

    Fetches intraday prices and fundamentals via yfinance with
    automatic synthetic fallback when network is unavailable.

    Args:
        config: Project configuration dict.
        strategy_service: Optional StrategyService for signal data.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        strategy_service: Any | None = None,
    ) -> None:
        config = config or {}
        self._config = config
        self._strategy_service = strategy_service

        ui_cfg = config.get("ui", {})
        watchlist_cfg = ui_cfg.get("watchlist", {})
        self._cache_ttl: int = watchlist_cfg.get(
            "cache_ttl", _DEFAULT_CACHE_TTL,
        )
        self._logo_colours: dict[str, str] = watchlist_cfg.get(
            "logo_colours", {},
        )

        # Per-ticker cache: {ticker: (timestamp, yf_ticker_obj)}
        self._ticker_cache: dict[str, tuple[float, Any]] = {}

    # ------------------------------------------------------------------
    # yfinance session management
    # ------------------------------------------------------------------

    def _get_yf_ticker(self, ticker: str) -> Any | None:
        """Get or create a cached yfinance Ticker object.

        Returns None if yfinance is unavailable or the call fails.
        A single Ticker object is reused for both intraday and
        fundamentals to avoid duplicate network requests.
        """
        now = time.time()

        # Return cached object if still fresh
        cached = self._ticker_cache.get(ticker)
        if cached is not None:
            ts, obj = cached
            if now - ts < self._cache_ttl:
                return obj

        try:
            import yfinance as yf
            obj = yf.Ticker(ticker)
            self._ticker_cache[ticker] = (now, obj)
            return obj
        except ImportError:
            logger.warning(
                "yfinance not available -> using synthetic fallback"
            )
            return None
        except Exception as exc:
            logger.error(
                "Failed to create yfinance Ticker for %s: %s",
                ticker, exc,
            )
            return None

    def _invalidate_cache(self, ticker: str | None = None) -> None:
        """Clear the ticker cache, optionally for a single ticker."""
        if ticker:
            self._ticker_cache.pop(ticker, None)
        else:
            self._ticker_cache.clear()

    # ------------------------------------------------------------------
    # Intraday prices
    # ------------------------------------------------------------------

    def get_intraday_prices(self, ticker: str) -> dict[str, Any]:
        """Fetch intraday price data for a single ticker.

        Args:
            ticker: Ticker symbol (e.g. "BARC.L").

        Returns:
            Dict with keys: prices, times, prev_close, direction, current.
        """
        try:
            yf_obj = self._get_yf_ticker(ticker)
            if yf_obj is not None:
                result = self._fetch_intraday_yf(ticker, yf_obj)
                if result is not None:
                    return result
        except Exception as exc:
            logger.error(
                "yfinance intraday failed for %s -> fallback: %s",
                ticker, exc,
            )

        return self._synthetic_intraday(ticker)

    def _fetch_intraday_yf(
        self, ticker: str, yf_obj: Any,
    ) -> dict[str, Any] | None:
        """Attempt to fetch intraday data from yfinance."""
        try:
            hist = yf_obj.history(period="1d", interval="30m")
            if hist.empty:
                return None

            info = yf_obj.info or {}
            prev_close = info.get(
                "previousClose",
                info.get("regularMarketPreviousClose"),
            )
            if prev_close is None and len(hist) > 0:
                prev_close = float(hist["Close"].iloc[0])

            prev_close = float(prev_close)
            prices = [round(float(p), 2) for p in hist["Close"].values]
            times = [
                t.isoformat() for t in hist.index.to_pydatetime()
            ]
            current = prices[-1] if prices else prev_close

            return {
                "prices": prices,
                "times": times,
                "prev_close": round(prev_close, 2),
                "direction": "up" if current >= prev_close else "dn",
                "current": round(current, 2),
            }
        except Exception as exc:
            logger.error(
                "Failed to parse yfinance intraday for %s: %s",
                ticker, exc,
            )
            return None

    def _synthetic_intraday(self, ticker: str) -> dict[str, Any]:
        """Generate synthetic intraday data for offline development."""
        meta = _SYNTHETIC_COMPANIES.get(ticker, {})
        prev_close = meta.get("prev_close", 100.0 + _ticker_seed(ticker) % 500)
        rng = random.Random(_ticker_seed(ticker) + int(time.time() // 3600))

        # Generate ~21 data points from 04:00 to current
        base_time = datetime.now(timezone.utc).replace(
            hour=4, minute=0, second=0, microsecond=0,
        )
        prices: list[float] = []
        times: list[str] = []
        price = prev_close

        for i in range(21):
            t = base_time + timedelta(minutes=30 * i)
            noise = rng.gauss(0, 0.002)
            price = price * (1 + noise)
            prices.append(round(price, 2))
            times.append(t.isoformat())

        current = prices[-1]
        return {
            "prices": prices,
            "times": times,
            "prev_close": round(prev_close, 2),
            "direction": "up" if current >= prev_close else "dn",
            "current": round(current, 2),
        }

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """Fetch fundamental data for a single ticker.

        Args:
            ticker: Ticker symbol (e.g. "BARC.L").

        Returns:
            Dict with keys: volume, market_cap, pe_ratio, div_yield,
            profit_margin, operating_margin, is_positive_margin,
            is_positive_op_margin.
        """
        try:
            yf_obj = self._get_yf_ticker(ticker)
            if yf_obj is not None:
                result = self._fetch_fundamentals_yf(ticker, yf_obj)
                if result is not None:
                    return result
        except Exception as exc:
            logger.error(
                "yfinance fundamentals failed for %s -> fallback: %s",
                ticker, exc,
            )

        return self._synthetic_fundamentals(ticker)

    def _fetch_fundamentals_yf(
        self, ticker: str, yf_obj: Any,
    ) -> dict[str, Any] | None:
        """Attempt to fetch fundamentals from yfinance."""
        try:
            info = yf_obj.info
            if not info:
                return None

            volume = info.get("volume")
            market_cap = info.get("marketCap")
            pe_ratio = info.get("trailingPE")
            div_yield = info.get("dividendYield")
            profit_margin = info.get("profitMargins")
            operating_margin = info.get("operatingMargins")

            pm_float = float(profit_margin) if profit_margin is not None else 0
            om_float = (
                float(operating_margin) if operating_margin is not None else 0
            )

            return {
                "volume": _format_volume(volume),
                "market_cap": _format_market_cap(market_cap, ticker),
                "pe_ratio": _format_pe(pe_ratio),
                "div_yield": _format_pct(div_yield),
                "profit_margin": _format_pct(profit_margin),
                "operating_margin": _format_pct(operating_margin),
                "is_positive_margin": pm_float > 0,
                "is_positive_op_margin": om_float > 0,
            }
        except Exception as exc:
            logger.error(
                "Failed to parse yfinance fundamentals for %s: %s",
                ticker, exc,
            )
            return None

    def _synthetic_fundamentals(self, ticker: str) -> dict[str, Any]:
        """Return plausible synthetic fundamentals."""
        meta = _SYNTHETIC_COMPANIES.get(ticker, {})

        volume = meta.get("volume", 10_000_000 + _ticker_seed(ticker) % 50_000_000)
        market_cap = meta.get("market_cap", 20_000_000_000)
        pe_ratio = meta.get("pe_ratio")
        div_yield = meta.get("div_yield")
        profit_margin = meta.get("profit_margin", 0.15)
        operating_margin = meta.get("operating_margin", 0.20)

        pm_float = float(profit_margin) if profit_margin is not None else 0
        om_float = float(operating_margin) if operating_margin is not None else 0

        return {
            "volume": _format_volume(volume),
            "market_cap": _format_market_cap(market_cap, ticker),
            "pe_ratio": _format_pe(pe_ratio),
            "div_yield": _format_pct(div_yield),
            "profit_margin": _format_pct(profit_margin),
            "operating_margin": _format_pct(operating_margin),
            "is_positive_margin": pm_float > 0,
            "is_positive_op_margin": om_float > 0,
        }

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------

    def get_summary(self, ticker: str) -> str:
        """Get a short summary for a ticker.

        Tries Claude API if available, then yfinance longBusinessSummary
        (truncated), then synthetic text.

        Args:
            ticker: Ticker symbol.

        Returns:
            2-3 sentence plain-language summary.
        """
        # Try yfinance business summary
        try:
            yf_obj = self._get_yf_ticker(ticker)
            if yf_obj is not None:
                info = yf_obj.info or {}
                summary = info.get("longBusinessSummary", "")
                if summary:
                    # Truncate to ~2 sentences
                    sentences = summary.split(". ")
                    short = ". ".join(sentences[:2])
                    if not short.endswith("."):
                        short += "."
                    return short
        except Exception:
            pass

        # Synthetic fallback
        meta = _SYNTHETIC_COMPANIES.get(ticker, {})
        return meta.get(
            "summary",
            f"{ticker} is listed on the exchange. "
            "No additional summary available at this time.",
        )

    # ------------------------------------------------------------------
    # FX & Commodities
    # ------------------------------------------------------------------

    def get_fx_commodities(self) -> list[dict[str, Any]]:
        """Get FX and commodity rates for the sidebar.

        Returns:
            List of dicts with keys: name, value, change, direction.
        """
        pairs = [
            ("GBP/USD", "GBPUSD=X"),
            ("Brent Crude", "BZ=F"),
            ("Gold", "GC=F"),
        ]
        results: list[dict[str, Any]] = []

        for name, symbol in pairs:
            try:
                yf_obj = self._get_yf_ticker(symbol)
                if yf_obj is not None:
                    info = yf_obj.info or {}
                    price = info.get(
                        "regularMarketPrice",
                        info.get("previousClose"),
                    )
                    prev = info.get(
                        "regularMarketPreviousClose",
                        info.get("previousClose"),
                    )
                    if price is not None and prev is not None:
                        change = (float(price) - float(prev)) / float(prev)
                        results.append({
                            "name": name,
                            "value": f"{float(price):.2f}",
                            "change": f"{change * 100:+.2f}%",
                            "direction": "up" if change >= 0 else "dn",
                        })
                        continue
            except Exception as exc:
                logger.error(
                    "FX/commodity fetch failed for %s: %s", name, exc,
                )

            # Synthetic fallback
            results.append(self._synthetic_fx_commodity(name))

        return results

    def _synthetic_fx_commodity(self, name: str) -> dict[str, Any]:
        """Return synthetic FX/commodity data."""
        defaults = {
            "GBP/USD": ("1.2734", "+0.12%", "up"),
            "Brent Crude": ("78.45", "-0.34%", "dn"),
            "Gold": ("2085.30", "+0.08%", "up"),
        }
        val, chg, direction = defaults.get(name, ("0.00", "0.00%", "up"))
        return {
            "name": name,
            "value": val,
            "change": chg,
            "direction": direction,
        }

    # ------------------------------------------------------------------
    # Watchlist cards (composite)
    # ------------------------------------------------------------------

    def get_watchlist_cards(
        self, tickers: list[str],
    ) -> list[dict[str, Any]]:
        """Build complete card data for each ticker in the watchlist.

        Combines intraday prices, fundamentals, signal data, and
        company name into a single dict per ticker.

        Args:
            tickers: List of ticker symbols.

        Returns:
            One dict per ticker with all card data.
        """
        cards: list[dict[str, Any]] = []

        for ticker in tickers:
            try:
                card = self._build_single_card(ticker)
                cards.append(card)
            except Exception as exc:
                logger.error(
                    "Failed to build card for %s: %s", ticker, exc,
                )
                # Still return a minimal card so the UI has something
                cards.append(self._fallback_card(ticker))

        return cards

    def _build_single_card(self, ticker: str) -> dict[str, Any]:
        """Build card data for a single ticker."""
        intraday = self.get_intraday_prices(ticker)
        fundamentals = self.get_fundamentals(ticker)
        summary = self.get_summary(ticker)

        # Company name from yfinance or synthetic
        company_name = self._get_company_name(ticker)

        # Signal data from strategy service
        signal_data = self._get_signal_data(ticker)

        # Price change
        current = intraday.get("current", 0)
        prev_close = intraday.get("prev_close", current)
        if prev_close > 0:
            change_val = current - prev_close
            change_pct = (change_val / prev_close) * 100
        else:
            change_val = 0
            change_pct = 0

        # Determine exchange
        upper = ticker.upper()
        if _is_lse_ticker(ticker):
            exchange = "LSE"
        elif upper.endswith("-USD") or upper.endswith("-GBP"):
            exchange = "Crypto"
        else:
            exchange = "NYSE"

        # Logo colour
        logo_colour = self._logo_colours.get(ticker, self._auto_colour(ticker))

        return {
            "ticker": ticker,
            "company_name": company_name,
            "exchange": exchange,
            "current": round(current, 2),
            "prev_close": round(prev_close, 2),
            "change_val": round(change_val, 2),
            "change_pct": round(change_pct, 2),
            "direction": intraday.get("direction", "up"),
            "prices": intraday.get("prices", []),
            "times": intraday.get("times", []),
            "logo_colour": logo_colour,
            "summary": summary,
            **fundamentals,
            **signal_data,
        }

    def _get_company_name(self, ticker: str) -> str:
        """Get human-readable company name."""
        # Try synthetic lookup first (fast)
        meta = _SYNTHETIC_COMPANIES.get(ticker)
        if meta:
            return meta["name"]

        # Try yfinance
        try:
            yf_obj = self._get_yf_ticker(ticker)
            if yf_obj is not None:
                info = yf_obj.info or {}
                name = info.get("longName") or info.get("shortName")
                if name:
                    return name
        except Exception:
            pass

        # Fallback: ticker string itself
        return ticker.replace(".L", "").replace(".", " ")

    def _get_signal_data(self, ticker: str) -> dict[str, Any]:
        """Get signal info from strategy service if available."""
        if self._strategy_service is None:
            return {
                "signal_direction": "NEUTRAL",
                "signal_confidence": 0.50,
            }

        try:
            strategies = self._strategy_service.get_available_strategies()
            for name in strategies:
                df = self._strategy_service.get_signals(name, n_days=5)
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    row_ticker = row.get("ticker", "")
                    if row_ticker == ticker:
                        return {
                            "signal_direction": row.get(
                                "direction",
                                row.get("signal", "NEUTRAL"),
                            ),
                            "signal_confidence": round(
                                float(row.get(
                                    "confidence",
                                    row.get("score", 0.5),
                                )),
                                2,
                            ),
                        }
        except Exception as exc:
            logger.error(
                "Failed to get signal data for %s: %s", ticker, exc,
            )

        return {
            "signal_direction": "NEUTRAL",
            "signal_confidence": 0.50,
        }

    def _auto_colour(self, ticker: str) -> str:
        """Generate a deterministic colour from ticker string."""
        colours = [
            "#0062cc", "#c8102e", "#7b2d8b", "#db0011",
            "#006eb7", "#006a4e", "#e65100", "#1a237e",
        ]
        idx = _ticker_seed(ticker) % len(colours)
        return colours[idx]

    def _fallback_card(self, ticker: str) -> dict[str, Any]:
        """Minimal card data when everything else fails."""
        upper = ticker.upper()
        if _is_lse_ticker(ticker):
            exchange = "LSE"
        elif upper.endswith("-USD") or upper.endswith("-GBP"):
            exchange = "Crypto"
        else:
            exchange = "NYSE"
        return {
            "ticker": ticker,
            "company_name": ticker,
            "exchange": exchange,
            "current": 0.0,
            "prev_close": 0.0,
            "change_val": 0.0,
            "change_pct": 0.0,
            "direction": "up",
            "prices": [],
            "times": [],
            "logo_colour": self._auto_colour(ticker),
            "summary": "Data unavailable.",
            "volume": "N/A",
            "market_cap": "N/A",
            "pe_ratio": "N/A",
            "div_yield": "N/A",
            "profit_margin": "N/A",
            "operating_margin": "N/A",
            "is_positive_margin": False,
            "is_positive_op_margin": False,
            "signal_direction": "NEUTRAL",
            "signal_confidence": 0.50,
        }
