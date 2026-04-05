"""Enrichment service -- adds fundamentals, technicals, and sentiment to candidates.

Reuses the existing FeaturePipeline for technical indicators.
Fetches fundamentals via yfinance. Converts prices to GBP.
"""

from __future__ import annotations

import time
from typing import Any

from src.deployment import AbstractEnricher, Candidate, EnrichedCandidate
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EnrichmentService(AbstractEnricher):
    """Enriches candidates with fundamentals, technicals, and sentiment.

    Args:
        config: Deployment config dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._fx_cache: dict[str, float] = {}

    def enrich(
        self, candidates: list[Candidate],
    ) -> list[EnrichedCandidate]:
        """Enrich all candidates. Drops those where data fetch fails.

        Args:
            candidates: Raw candidates from the scanner.

        Returns:
            List of EnrichedCandidate objects.
        """
        enriched: list[EnrichedCandidate] = []

        for candidate in candidates:
            try:
                result = self._enrich_single(candidate)
                if result is not None:
                    enriched.append(result)
            except Exception as e:
                logger.warning(
                    "Enrichment failed for %s: %s", candidate.ticker, e
                )

        logger.info(
            "Enriched %d/%d candidates",
            len(enriched),
            len(candidates),
        )
        return enriched

    def _enrich_single(
        self, candidate: Candidate,
    ) -> EnrichedCandidate | None:
        """Enrich a single candidate with all data sources."""
        fundamentals = self._fetch_fundamentals(candidate.ticker)
        technicals = self._calculate_technicals(candidate.ticker)
        news_sentiment = self._fetch_news_sentiment(candidate.ticker)
        gbp_price = self._get_gbp_price(
            candidate.ticker, candidate.currency
        )

        return EnrichedCandidate.from_candidate(
            candidate,
            fundamentals=fundamentals,
            technicals=technicals,
            news_sentiment=news_sentiment,
            gbp_price=gbp_price,
        )

    def _fetch_fundamentals(self, ticker: str) -> dict[str, Any]:
        """Fetch fundamental data via yfinance."""
        try:
            import yfinance as yf
            time.sleep(0.3)
            t = yf.Ticker(ticker)
            info = t.info or {}

            return {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "revenue_growth_yoy": info.get("revenueGrowth"),
                "earnings_growth_yoy": info.get("earningsGrowth"),
                "dividend_yield": info.get("dividendYield", 0) or 0,
                "payout_ratio": info.get("payoutRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "market_cap": info.get("marketCap", 0),
                "analyst_buy": info.get(
                    "numberOfAnalystOpinions", 0
                ),
                "analyst_target_mean": info.get("targetMeanPrice"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "current_price": info.get("regularMarketPrice", 0),
                "ex_dividend_date": str(
                    info.get("exDividendDate", "")
                ),
            }
        except Exception as e:
            logger.warning(
                "Fundamentals fetch failed for %s: %s", ticker, e
            )
            return {}

    def _calculate_technicals(self, ticker: str) -> dict[str, Any]:
        """Calculate technical indicators using the existing feature pipeline."""
        try:
            import yfinance as yf
            from src.features.technical import (
                compute_rsi,
                compute_macd,
                compute_bollinger_bands,
                compute_atr,
                compute_sma,
                compute_returns,
                compute_volatility,
            )

            time.sleep(0.3)
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")

            if hist.empty or len(hist) < 50:
                logger.warning(
                    "Insufficient history for technicals: %s (%d rows)",
                    ticker,
                    len(hist),
                )
                return {}

            # Compute individual indicators.
            rsi = compute_rsi(hist, window=14)
            macd = compute_macd(hist, fast=12, slow=26, signal=9)
            bb = compute_bollinger_bands(hist, window=20, num_std=2.0)
            atr = compute_atr(hist, window=14)
            sma_50 = compute_sma(hist, windows=[50])
            sma_200 = compute_sma(hist, windows=[200])
            vol = compute_volatility(hist, windows=[21])

            # Extract latest values.
            latest = {}

            if not rsi.empty:
                latest["rsi_14"] = float(rsi.iloc[-1].iloc[0])

            if not macd.empty:
                for col in macd.columns:
                    latest[col] = float(macd[col].iloc[-1])

            if not bb.empty:
                for col in bb.columns:
                    latest[col] = float(bb[col].iloc[-1])

            if not atr.empty:
                latest["atr_14"] = float(atr.iloc[-1].iloc[0])

            if not sma_50.empty:
                latest["sma_50"] = float(sma_50.iloc[-1].iloc[0])

            if not sma_200.empty and len(hist) >= 200:
                latest["sma_200"] = float(sma_200.iloc[-1].iloc[0])

            if not vol.empty:
                latest["volatility_21d"] = float(vol.iloc[-1].iloc[0])

            # Price position relative to moving averages.
            close = float(hist["Close"].iloc[-1])
            if "sma_50" in latest:
                latest["price_vs_sma50"] = (
                    close / latest["sma_50"] - 1.0
                )
            if "sma_200" in latest:
                latest["price_vs_sma200"] = (
                    close / latest["sma_200"] - 1.0
                )

            # Volume trend (last 5 days vs 20-day average).
            if len(hist) >= 20:
                vol_20d = float(hist["Volume"].tail(20).mean())
                vol_5d = float(hist["Volume"].tail(5).mean())
                if vol_20d > 0:
                    latest["volume_trend_ratio"] = vol_5d / vol_20d
                    latest["volume_trend"] = (
                        "rising" if vol_5d > vol_20d * 1.1 else
                        "falling" if vol_5d < vol_20d * 0.9 else
                        "flat"
                    )

            return latest

        except Exception as e:
            logger.warning(
                "Technicals calculation failed for %s: %s", ticker, e
            )
            return {}

    def _fetch_news_sentiment(self, ticker: str) -> dict[str, Any]:
        """Fetch news sentiment. Returns neutral if unavailable.

        Never blocks enrichment on news sentiment failure.
        """
        try:
            import os
            if not os.environ.get("ANTHROPIC_API_KEY"):
                return {
                    "score": 0.5,
                    "available": False,
                    "headline_count": 0,
                    "note": "No API key -- sentiment unavailable",
                }

            # Use the existing news service pattern if available.
            # For now, return neutral with a flag.
            return {
                "score": 0.5,
                "available": False,
                "headline_count": 0,
                "note": "News sentiment not yet wired to enricher",
            }

        except Exception:
            return {
                "score": 0.5,
                "available": False,
                "headline_count": 0,
            }

    def _get_gbp_price(self, ticker: str, currency: str) -> float:
        """Get the current price converted to GBP.

        Args:
            ticker: Ticker symbol.
            currency: Currency of the instrument.

        Returns:
            Price in GBP.
        """
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            info = t.info or {}
            price = float(info.get("regularMarketPrice", 0) or 0)

            if currency.upper() == "GBP" or currency.upper() == "GBp":
                # GBp is pence; convert to pounds.
                return price / 100 if currency == "GBp" else price

            # Convert to GBP.
            rate = self._get_fx_rate(currency.upper())
            return price / rate if rate > 0 else price

        except Exception as e:
            logger.warning(
                "GBP price fetch failed for %s: %s", ticker, e
            )
            return 0.0

    def _get_fx_rate(self, from_currency: str) -> float:
        """Get FX rate (units of from_currency per 1 GBP).

        Caches rates for the session to avoid repeated API calls.
        """
        if from_currency == "GBP":
            return 1.0

        if from_currency in self._fx_cache:
            return self._fx_cache[from_currency]

        try:
            import yfinance as yf

            pair = f"GBP{from_currency}=X"
            t = yf.Ticker(pair)
            hist = t.history(period="1d")
            if not hist.empty:
                rate = float(hist["Close"].iloc[-1])
                self._fx_cache[from_currency] = rate
                logger.info("FX rate GBP/%s = %.4f", from_currency, rate)
                return rate
        except Exception as e:
            logger.warning("FX rate fetch failed for %s: %s", from_currency, e)

        # Fallback to config defaults.
        fx_config = self._config.get("fx", {})
        key = f"default_gbp_{from_currency.lower()}"
        rate = fx_config.get(key, 1.27)  # Default USD rate.
        self._fx_cache[from_currency] = rate
        logger.info(
            "Using fallback FX rate GBP/%s = %.4f", from_currency, rate
        )
        return rate


# ------------------------------------------------------------------
# Synthetic enricher for offline testing
# ------------------------------------------------------------------


class SyntheticEnricher(AbstractEnricher):
    """Returns realistic enriched data for offline testing."""

    def enrich(
        self, candidates: list[Candidate],
    ) -> list[EnrichedCandidate]:
        """Enrich all candidates with synthetic data."""
        enriched: list[EnrichedCandidate] = []

        for i, c in enumerate(candidates):
            seed = hash(c.ticker) % 100
            enriched.append(EnrichedCandidate.from_candidate(
                c,
                fundamentals={
                    "pe_ratio": 15.0 + seed * 0.3,
                    "forward_pe": 12.0 + seed * 0.25,
                    "peg_ratio": 0.8 + seed * 0.02,
                    "revenue_growth_yoy": 0.05 + seed * 0.003,
                    "earnings_growth_yoy": 0.10 + seed * 0.005,
                    "dividend_yield": 0.02 + seed * 0.0005,
                    "payout_ratio": 0.30 + seed * 0.005,
                    "price_to_book": 2.0 + seed * 0.1,
                    "price_to_sales": 3.0 + seed * 0.15,
                    "market_cap": c.market_cap_usd,
                    "analyst_buy": 20 + seed % 20,
                    "analyst_target_mean": 130.0 + seed,
                    "fifty_two_week_high": 150.0 + seed,
                    "fifty_two_week_low": 80.0 + seed * 0.5,
                    "current_price": 120.0 + seed * 0.7,
                },
                technicals={
                    "rsi_14": 45.0 + seed * 0.3,
                    "macd_line": 1.5 + seed * 0.05,
                    "macd_signal": 1.0 + seed * 0.03,
                    "macd_histogram": 0.5 + seed * 0.02,
                    "bb_upper": 145.0 + seed,
                    "bb_lower": 115.0 + seed * 0.5,
                    "bb_middle": 130.0 + seed * 0.7,
                    "sma_50": 125.0 + seed * 0.6,
                    "sma_200": 110.0 + seed * 0.5,
                    "atr_14": 3.0 + seed * 0.1,
                    "volume_trend": "rising" if seed % 3 == 0 else "flat",
                    "volatility_21d": 0.20 + seed * 0.002,
                },
                news_sentiment={
                    "score": 0.5 + (seed % 30) * 0.01,
                    "available": True,
                    "headline_count": 5 + seed % 20,
                },
                gbp_price=100.0 + seed * 0.8,
            ))

        return enriched
