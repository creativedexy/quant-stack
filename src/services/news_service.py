"""News service -- RSS ingestion with optional Claude API relevance filtering.

Fetches articles from configured RSS feeds, optionally filters them using
Claude for relevance to watchlist tickers, and caches results in
``data/news/latest.json``.

Usage:
    from src.services.news_service import NewsService
    svc = NewsService(config)
    articles = svc.refresh_news(["BARC.L", "SHEL.L"])
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Ticker -> company name for keyword matching
_TICKER_COMPANY_MAP: dict[str, str] = {
    "BARC.L": "Barclays",
    "SHEL.L": "Shell",
    "AZN.L": "AstraZeneca",
    "HSBA.L": "HSBC",
    "BP.L": "BP",
    "LLOY.L": "Lloyds",
    "RIO.L": "Rio Tinto",
    "GSK.L": "GSK",
    "ULVR.L": "Unilever",
    "VOD.L": "Vodafone",
    "BT-A.L": "BT Group",
}


class NewsService:
    """RSS news ingestion with optional Claude API relevance filtering."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise the news service.

        Args:
            config: Project configuration dict.  If None, loads from
                    config/settings.yaml.
        """
        if config is None:
            from src.utils.config import load_config
            config = load_config()

        self._config = config
        news_cfg = config.get("news", {})

        self._feeds: list[str] = news_cfg.get("feeds", [])
        self._min_relevance: int = news_cfg.get("min_relevance_score", 7)
        self._max_displayed: int = news_cfg.get("max_articles_displayed", 20)
        self._claude_model: str = news_cfg.get(
            "claude_model", "claude-haiku-4-5-20251001"
        )
        self._cache_ttl: int = news_cfg.get("refresh_minutes", 15) * 60

        data_dir_name = config.get("general", {}).get("data_dir", "data")
        self._data_dir = _PROJECT_ROOT / data_dir_name / "news"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._latest_path = self._data_dir / "latest.json"
        self._sample_path = self._data_dir / "sample_articles.json"

        # In-memory cache
        self._cached_articles: list[dict[str, Any]] = []
        self._cache_time: float = 0.0

        # Startup API key check
        self._has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
        if not self._has_api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set -- news filtering will use "
                "keyword matching instead of Claude API"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_raw_articles(self, max_per_feed: int = 30) -> list[dict[str, Any]]:
        """Parse RSS feeds and return raw articles.

        Args:
            max_per_feed: Maximum articles to keep per feed.

        Returns:
            Deduplicated list of article dicts with keys:
            title, summary, link, published, source.
        """
        import feedparser

        articles: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        feeds_failed = 0

        for feed_url in self._feeds:
            try:
                parsed = feedparser.parse(feed_url)
                if parsed.bozo and not parsed.entries:
                    raise ValueError(
                        f"Feed parse error: {parsed.bozo_exception}"
                    )

                feed_title = getattr(parsed.feed, "title", feed_url)
                count = 0
                for entry in parsed.entries:
                    if count >= max_per_feed:
                        break
                    link = getattr(entry, "link", "")
                    if not link or link in seen_urls:
                        continue
                    seen_urls.add(link)

                    published_raw = getattr(entry, "published", "")
                    published = self._parse_published(published_raw)

                    articles.append({
                        "title": getattr(entry, "title", ""),
                        "summary": self._clean_summary(
                            getattr(entry, "summary", "")
                        ),
                        "link": link,
                        "published": published,
                        "source": feed_title,
                    })
                    count += 1

                logger.info(
                    "Fetched %d articles from %s", count, feed_title,
                )
            except Exception as exc:
                logger.error("Failed to fetch feed %s: %s", feed_url, exc)
                feeds_failed += 1

        # Fallback to sample data if ALL feeds failed
        if feeds_failed == len(self._feeds) and not articles:
            articles = self._load_sample_articles()
            logger.warning(
                "All feeds failed -- loaded %d sample articles", len(articles),
            )

        # Cap total
        max_total = max_per_feed * max(len(self._feeds), 1)
        return articles[:max_total]

    def filter_for_watchlist(
        self,
        articles: list[dict[str, Any]],
        tickers: list[str],
    ) -> list[dict[str, Any]]:
        """Filter articles for relevance to watchlist tickers.

        If ``ANTHROPIC_API_KEY`` is set, uses Claude API for classification.
        Otherwise falls back to keyword matching.

        Args:
            articles: Raw article list from :meth:`fetch_raw_articles`.
            tickers: Watchlist ticker symbols (e.g. ``["BARC.L", "SHEL.L"]``).

        Returns:
            Filtered and enriched article list with relevance_score,
            tickers_mentioned, sentiment, and sentiment_reason.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            return self._claude_filter(articles, tickers, api_key)
        return self._keyword_filter(articles, tickers)

    def get_signal_impact(
        self, article: dict[str, Any], signals: dict[str, str],
    ) -> str:
        """Describe how an article aligns with active trading signals.

        Args:
            article: Enriched article dict with ``sentiment`` key.
            signals: ``{ticker: direction}`` from StrategyService, e.g.
                     ``{"BARC.L": "LONG", "SHEL.L": "SHORT"}``.

        Returns:
            ASCII-safe impact string, e.g. "Strengthens BARC.L long".
        """
        sentiment = article.get("sentiment", "neutral").lower()
        mentioned = article.get("tickers_mentioned", [])
        impacts: list[str] = []

        for ticker in mentioned:
            direction = signals.get(ticker, "").upper()
            if not direction or direction == "NEUTRAL":
                continue

            is_bullish = sentiment in ("bullish",)
            is_bearish = sentiment in ("bearish",)
            is_long = direction == "LONG"
            is_short = direction == "SHORT"

            if (is_bullish and is_long) or (is_bearish and is_short):
                impacts.append(f"Strengthens {ticker} {direction.lower()}")
            elif (is_bearish and is_long) or (is_bullish and is_short):
                impacts.append(f"Contradicts {ticker} {direction.lower()}")
            elif sentiment == "mixed":
                impacts.append(
                    f"Mixed signal for {ticker} {direction.lower()}"
                )
            else:
                impacts.append(
                    f"Corroborates {ticker} {direction.lower()}"
                )

        result = " | ".join(impacts) if impacts else ""
        # Guarantee ASCII-safe output
        return result.encode("ascii", errors="replace").decode("ascii")

    def refresh_news(self, tickers: list[str]) -> list[dict[str, Any]]:
        """Full refresh: fetch -> filter -> enrich -> cache.

        Args:
            tickers: Watchlist ticker symbols.

        Returns:
            Filtered articles sorted by relevance_score descending.
        """
        raw = self.fetch_raw_articles()
        filtered = self.filter_for_watchlist(raw, tickers)

        # Enrich with signal impact
        signals = self._get_active_signals()
        for article in filtered:
            article["signal_impact"] = self.get_signal_impact(
                article, signals,
            )

        # Sort by relevance
        filtered.sort(
            key=lambda a: a.get("relevance_score", 0), reverse=True,
        )

        # Cap to max displayed
        filtered = filtered[: self._max_displayed]

        # Save to disk
        self._save_latest(filtered)

        # Update in-memory cache
        self._cached_articles = filtered
        self._cache_time = time.monotonic()

        return filtered

    def get_cached_news(self) -> list[dict[str, Any]]:
        """Load cached news from disk.

        Returns:
            List of enriched article dicts, or ``[]`` if no cache file.
        """
        # Try in-memory cache first
        if (
            self._cached_articles
            and (time.monotonic() - self._cache_time) < self._cache_ttl
        ):
            return self._cached_articles

        # Fall back to disk
        if self._latest_path.exists():
            try:
                with open(self._latest_path, "r", encoding="utf-8") as f:
                    articles = json.load(f)
                self._cached_articles = articles
                self._cache_time = time.monotonic()
                return articles
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Failed to load cached news: %s", exc)

        return []

    def get_market_news(
        self,
        tickers: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return news items formatted for the Execute page sidebar.

        Each item includes time, title, source, tickers, has_impact,
        ai_summary, impact_label, and impact_colour.

        Args:
            tickers: Ticker symbols to filter for. If None, returns all.
            limit: Maximum number of items.

        Returns:
            List of news dicts sorted newest first.
        """
        articles = self.get_cached_news()

        if articles:
            items: list[dict[str, Any]] = []
            for a in articles[:limit]:
                mentioned = a.get("tickers_mentioned", [])
                if tickers:
                    # Keep articles that mention at least one ticker
                    if not any(t in mentioned for t in tickers):
                        continue

                sentiment = a.get("sentiment", "neutral").lower()
                impact_map = {
                    "bullish": ("green", "Positive"),
                    "bearish": ("red", "Negative"),
                    "mixed": ("amber", "Mixed"),
                    "neutral": ("grey", "Neutral"),
                }
                colour, label_prefix = impact_map.get(
                    sentiment, ("grey", "Neutral"),
                )

                # Build impact label
                if mentioned and sentiment != "neutral":
                    impact_label = f"{label_prefix} for {mentioned[0]}"
                else:
                    impact_label = "Neutral"

                items.append({
                    "time": a.get("published", ""),
                    "title": a.get("title", ""),
                    "source": a.get("source", ""),
                    "tickers": mentioned,
                    "has_impact": sentiment in ("bullish", "bearish"),
                    "ai_summary": a.get("sentiment_reason", ""),
                    "impact_label": impact_label,
                    "impact_colour": colour,
                })
            return items[:limit]

        # Synthetic fallback
        return [
            {
                "time": "2026-03-15 08:15",
                "title": "Shell reports record quarterly earnings",
                "source": "Reuters",
                "tickers": ["SHEL.L"],
                "has_impact": True,
                "ai_summary": (
                    "Strong upstream performance drove earnings above "
                    "consensus. Dividend increase announced."
                ),
                "impact_label": "Positive for SHEL.L",
                "impact_colour": "green",
            },
            {
                "time": "2026-03-15 07:45",
                "title": "AstraZeneca cancer drug wins FDA approval",
                "source": "Financial Times",
                "tickers": ["AZN.L"],
                "has_impact": True,
                "ai_summary": (
                    "New lung cancer treatment approved ahead of schedule. "
                    "Expected to generate GBP2bn annual revenue."
                ),
                "impact_label": "Positive for AZN.L",
                "impact_colour": "green",
            },
            {
                "time": "2026-03-15 07:30",
                "title": "FTSE 100 opens flat amid tariff concerns",
                "source": "BBC News",
                "tickers": [],
                "has_impact": False,
                "ai_summary": (
                    "Markets cautious ahead of trade talks. "
                    "No direct impact on portfolio holdings."
                ),
                "impact_label": "Neutral",
                "impact_colour": "grey",
            },
            {
                "time": "2026-03-14 16:30",
                "title": "BP cuts capital spending forecast",
                "source": "Bloomberg",
                "tickers": ["BP.L"],
                "has_impact": True,
                "ai_summary": (
                    "Spending cuts may slow transition plans "
                    "but improve near-term cash flow."
                ),
                "impact_label": "Negative for BP.L",
                "impact_colour": "red",
            },
            {
                "time": "2026-03-14 14:00",
                "title": "Barclays upgrades UK growth forecast",
                "source": "CNBC",
                "tickers": ["BARC.L"],
                "has_impact": True,
                "ai_summary": (
                    "Higher growth expectations positive for "
                    "domestic banking revenue."
                ),
                "impact_label": "Positive for BARC.L",
                "impact_colour": "green",
            },
            {
                "time": "2026-03-14 11:00",
                "title": "GSK vaccine trial shows promising results",
                "source": "The Guardian",
                "tickers": ["GSK.L"],
                "has_impact": True,
                "ai_summary": (
                    "Phase 3 trial met primary endpoints. "
                    "Regulatory submission expected Q3."
                ),
                "impact_label": "Positive for GSK.L",
                "impact_colour": "green",
            },
            {
                "time": "2026-03-14 09:30",
                "title": "UK inflation eases to 2.1%",
                "source": "ONS",
                "tickers": [],
                "has_impact": False,
                "ai_summary": (
                    "Inflation near target supports case for "
                    "rate cuts. Broadly positive for equities."
                ),
                "impact_label": "Neutral",
                "impact_colour": "grey",
            },
        ]

    # ------------------------------------------------------------------
    # Claude API filter
    # ------------------------------------------------------------------

    def _claude_filter(
        self,
        articles: list[dict[str, Any]],
        tickers: list[str],
        api_key: str,
    ) -> list[dict[str, Any]]:
        """Use Claude API to classify article relevance and sentiment."""
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic package not installed -- using keyword filter")
            return self._keyword_filter(articles, tickers)

        client = anthropic.Anthropic(api_key=api_key)
        results: list[dict[str, Any]] = []
        batch_size = 10

        ticker_names = {
            t: _TICKER_COMPANY_MAP.get(t, t.replace(".L", ""))
            for t in tickers
        }

        for i in range(0, len(articles), batch_size):
            batch = articles[i: i + batch_size]
            batch_with_idx = [
                {
                    "index": j,
                    "title": a["title"],
                    "summary": a["summary"],
                }
                for j, a in enumerate(batch)
            ]

            prompt = (
                f"You are a financial news classifier. "
                f"The user's watchlist tickers are: {json.dumps(ticker_names)}.\n\n"
                f"For each article below, return a JSON array of objects with:\n"
                f"- article_index (int)\n"
                f"- relevance_score (1-10, 10 = directly relevant)\n"
                f"- tickers_mentioned (list of ticker symbols from the watchlist)\n"
                f"- sentiment ('bullish' | 'bearish' | 'neutral' | 'mixed')\n"
                f"- sentiment_reason (one sentence)\n\n"
                f"Only include articles with relevance_score >= {self._min_relevance}.\n\n"
                f"Articles:\n{json.dumps(batch_with_idx, indent=2)}\n\n"
                f"Return ONLY the JSON array, no other text."
            )

            try:
                response = client.messages.create(
                    model=self._claude_model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()

                # Parse JSON from response
                parsed = self._parse_json_response(text)
                if parsed is None:
                    logger.warning(
                        "Claude response parse failed for batch %d -- "
                        "falling back to keyword filter for this batch",
                        i // batch_size,
                    )
                    results.extend(
                        self._keyword_filter(batch, tickers)
                    )
                    continue

                for item in parsed:
                    idx = item.get("article_index", -1)
                    if 0 <= idx < len(batch):
                        article = {**batch[idx]}
                        article["relevance_score"] = item.get(
                            "relevance_score", 5
                        )
                        article["tickers_mentioned"] = item.get(
                            "tickers_mentioned", []
                        )
                        article["sentiment"] = item.get(
                            "sentiment", "neutral"
                        )
                        article["sentiment_reason"] = item.get(
                            "sentiment_reason", ""
                        )
                        results.append(article)
            except Exception as exc:
                logger.error(
                    "Claude API call failed for batch %d: %s",
                    i // batch_size, exc,
                )
                results.extend(self._keyword_filter(batch, tickers))

        return results

    # ------------------------------------------------------------------
    # Keyword filter (fallback)
    # ------------------------------------------------------------------

    def _keyword_filter(
        self,
        articles: list[dict[str, Any]],
        tickers: list[str],
    ) -> list[dict[str, Any]]:
        """Simple keyword matching when no API key is available."""
        results: list[dict[str, Any]] = []

        for article in articles:
            text = (
                article.get("title", "") + " " + article.get("summary", "")
            ).upper()

            matched_tickers: list[str] = []
            for ticker in tickers:
                # Match ticker without .L suffix
                bare = ticker.replace(".L", "").replace("-", "")
                company = _TICKER_COMPANY_MAP.get(ticker, "").upper()

                if bare.upper() in text or (company and company in text):
                    matched_tickers.append(ticker)

            if matched_tickers:
                enriched = {**article}
                enriched["relevance_score"] = 7
                enriched["tickers_mentioned"] = matched_tickers
                enriched["sentiment"] = "neutral"
                enriched["sentiment_reason"] = "Keyword match (no API key)"
                results.append(enriched)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_active_signals(self) -> dict[str, str]:
        """Load active signals from StrategyService if available."""
        try:
            from src.services.data_service import DataService
            from src.services.strategy_service import StrategyService

            data_svc = DataService(self._config)
            strategy_svc = StrategyService(data_svc, self._config)
            signals: dict[str, str] = {}

            for name in strategy_svc.get_available_strategies():
                df = strategy_svc.get_signals(name, n_days=5)
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    ticker = row.get("ticker", "")
                    direction = row.get(
                        "direction", row.get("signal", "NEUTRAL")
                    )
                    if ticker and direction:
                        signals[ticker] = direction.upper()

            return signals
        except Exception as exc:
            logger.debug("Could not load active signals: %s", exc)
            return {}

    def _save_latest(self, articles: list[dict[str, Any]]) -> None:
        """Persist filtered articles to data/news/latest.json."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._latest_path, "w", encoding="utf-8") as f:
                json.dump(articles, f, indent=2, default=str)
            logger.debug("Saved %d articles to %s", len(articles), self._latest_path)
        except OSError as exc:
            logger.error("Failed to save news cache: %s", exc)

    def _load_sample_articles(self) -> list[dict[str, Any]]:
        """Load synthetic sample articles for offline testing."""
        if self._sample_path.exists():
            try:
                with open(self._sample_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Failed to load sample articles: %s", exc)
        return []

    @staticmethod
    def _parse_published(raw: str) -> str:
        """Normalise published date to ISO format."""
        if not raw:
            return datetime.now(timezone.utc).isoformat()
        # feedparser often provides struct_time-compatible strings
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(raw)
            return dt.isoformat()
        except Exception:
            return raw

    @staticmethod
    def _clean_summary(html: str) -> str:
        """Strip HTML tags from summary text."""
        import re
        text = re.sub(r"<[^>]+>", "", html)
        return text.strip()[:500]

    @staticmethod
    def _parse_json_response(text: str) -> list[dict[str, Any]] | None:
        """Extract JSON array from Claude response text."""
        # Try direct parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        return None
