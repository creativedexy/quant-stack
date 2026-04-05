"""Tests for NewsService — RSS ingestion, filtering, and caching.

Unit tests run without network access or API keys.
The integration test (marked) requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.services.news_service import NewsService


# -- Fixtures ---------------------------------------------------------------

@pytest.fixture
def news_config(tmp_path: Path) -> dict[str, Any]:
    """Minimal config for NewsService tests."""
    return {
        "general": {"data_dir": str(tmp_path / "data")},
        "news": {
            "feeds": [
                "https://feeds.reuters.com/reuters/businessNews",
                "https://feeds.bbci.co.uk/news/business/rss.xml",
            ],
            "refresh_minutes": 15,
            "min_relevance_score": 7,
            "max_articles_displayed": 20,
            "claude_model": "claude-haiku-4-5-20251001",
        },
        "ui": {
            "watchlist": {
                "tickers": ["BARC.L", "SHEL.L", "AZN.L"],
            },
        },
    }


@pytest.fixture
def news_service(news_config: dict[str, Any]) -> NewsService:
    """NewsService instance with test config."""
    return NewsService(news_config)


@pytest.fixture
def sample_articles() -> list[dict[str, Any]]:
    """Synthetic raw articles for filter tests."""
    return [
        {
            "title": "Barclays beats earnings expectations",
            "summary": "Barclays reported strong Q4 results driven by investment banking.",
            "link": "https://example.com/article1",
            "published": "2026-03-06T10:00:00+00:00",
            "source": "Reuters",
        },
        {
            "title": "Shell announces new North Sea project",
            "summary": "Shell will invest in a new deepwater oil field.",
            "link": "https://example.com/article2",
            "published": "2026-03-06T11:00:00+00:00",
            "source": "BBC News",
        },
        {
            "title": "UK GDP growth slows in February",
            "summary": "The Office for National Statistics said GDP rose 0.1%.",
            "link": "https://example.com/article3",
            "published": "2026-03-06T09:00:00+00:00",
            "source": "FT",
        },
        {
            "title": "AstraZeneca wins FDA approval for new drug",
            "summary": "AstraZeneca received accelerated approval for its cancer therapy.",
            "link": "https://example.com/article4",
            "published": "2026-03-06T08:00:00+00:00",
            "source": "Reuters",
        },
    ]


# -- Unit tests -------------------------------------------------------------


class TestKeywordFilter:
    """Tests for the keyword-based fallback filter."""

    def test_matches_company_name(
        self, news_service: NewsService, sample_articles: list[dict],
    ) -> None:
        """Articles mentioning a company name are matched to the ticker."""
        tickers = ["BARC.L", "SHEL.L"]
        result = news_service._keyword_filter(sample_articles, tickers)

        matched_titles = {a["title"] for a in result}
        assert "Barclays beats earnings expectations" in matched_titles
        assert "Shell announces new North Sea project" in matched_titles

    def test_unrelated_articles_excluded(
        self, news_service: NewsService, sample_articles: list[dict],
    ) -> None:
        """Articles not mentioning any watchlist ticker are excluded."""
        tickers = ["HSBA.L"]
        result = news_service._keyword_filter(sample_articles, tickers)
        assert len(result) == 0

    def test_enriched_fields_present(
        self, news_service: NewsService, sample_articles: list[dict],
    ) -> None:
        """Matched articles have relevance_score, tickers_mentioned, sentiment."""
        tickers = ["BARC.L"]
        result = news_service._keyword_filter(sample_articles, tickers)
        assert len(result) >= 1

        article = result[0]
        assert article["relevance_score"] == 7
        assert "BARC.L" in article["tickers_mentioned"]
        assert article["sentiment"] == "neutral"
        assert "sentiment_reason" in article

    def test_multiple_tickers_in_one_article(
        self, news_service: NewsService,
    ) -> None:
        """An article mentioning two tickers lists both."""
        articles = [
            {
                "title": "Barclays and Shell in joint venture",
                "summary": "Barclays and Shell announced a green energy partnership.",
                "link": "https://example.com/multi",
                "published": "2026-03-06T12:00:00+00:00",
                "source": "Test",
            },
        ]
        tickers = ["BARC.L", "SHEL.L"]
        result = news_service._keyword_filter(articles, tickers)
        assert len(result) == 1
        assert set(result[0]["tickers_mentioned"]) == {"BARC.L", "SHEL.L"}


class TestSignalImpact:
    """Tests for signal impact classification."""

    def test_bullish_long_strengthens(self, news_service: NewsService) -> None:
        """Bullish sentiment + LONG signal -> Strengthens."""
        article = {
            "sentiment": "bullish",
            "tickers_mentioned": ["BARC.L"],
        }
        signals = {"BARC.L": "LONG"}
        impact = news_service.get_signal_impact(article, signals)
        assert "Strengthens" in impact
        assert "BARC.L" in impact

    def test_bearish_long_contradicts(self, news_service: NewsService) -> None:
        """Bearish sentiment + LONG signal -> Contradicts."""
        article = {
            "sentiment": "bearish",
            "tickers_mentioned": ["SHEL.L"],
        }
        signals = {"SHEL.L": "LONG"}
        impact = news_service.get_signal_impact(article, signals)
        assert "Contradicts" in impact

    def test_no_signals_returns_empty(self, news_service: NewsService) -> None:
        """No active signals -> empty impact string."""
        article = {
            "sentiment": "bullish",
            "tickers_mentioned": ["BARC.L"],
        }
        impact = news_service.get_signal_impact(article, {})
        assert impact == ""

    def test_ascii_safe_output(self, news_service: NewsService) -> None:
        """Impact string is ASCII-safe (no Unicode arrows or symbols)."""
        article = {
            "sentiment": "mixed",
            "tickers_mentioned": ["BARC.L"],
        }
        signals = {"BARC.L": "LONG"}
        impact = news_service.get_signal_impact(article, signals)
        assert impact.isascii()


class TestCaching:
    """Tests for the caching and persistence layer."""

    def test_save_and_load_cache(
        self, news_service: NewsService, sample_articles: list[dict],
    ) -> None:
        """Articles saved to disk can be loaded back."""
        enriched = [
            {**a, "relevance_score": 8, "tickers_mentioned": ["BARC.L"],
             "sentiment": "bullish", "sentiment_reason": "Test"}
            for a in sample_articles[:2]
        ]
        news_service._save_latest(enriched)

        # Clear in-memory cache to force disk read
        news_service._cached_articles = []
        news_service._cache_time = 0.0

        loaded = news_service.get_cached_news()
        assert len(loaded) == 2
        assert loaded[0]["relevance_score"] == 8

    def test_empty_cache_returns_list(self, news_service: NewsService) -> None:
        """get_cached_news returns [] when no cache exists."""
        result = news_service.get_cached_news()
        assert result == []


class TestFetchRawArticles:
    """Tests for RSS fetching with mocked feedparser."""

    def test_fetch_with_mocked_feeds(self, news_service: NewsService) -> None:
        """fetch_raw_articles parses RSS entries correctly."""
        mock_entry = MagicMock()
        mock_entry.title = "Test headline"
        mock_entry.summary = "<p>Test summary</p>"
        mock_entry.link = "https://example.com/test"
        mock_entry.published = "Thu, 06 Mar 2026 10:00:00 GMT"

        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = [mock_entry]
        mock_feed.feed.title = "Test Feed"

        with patch("feedparser.parse", return_value=mock_feed):
            articles = news_service.fetch_raw_articles()

        assert len(articles) >= 1
        assert articles[0]["title"] == "Test headline"
        assert "<p>" not in articles[0]["summary"]

    def test_failed_feeds_fallback_to_sample(
        self, news_service: NewsService, tmp_path: Path,
    ) -> None:
        """When all feeds fail, sample_articles.json is loaded as fallback."""
        # Write sample data
        sample = [{"title": "Sample article", "summary": "Test", "link": "https://example.com/s", "published": "2026-03-06", "source": "Sample"}]
        news_service._sample_path.parent.mkdir(parents=True, exist_ok=True)
        with open(news_service._sample_path, "w") as f:
            json.dump(sample, f)

        with patch("feedparser.parse", side_effect=Exception("Network error")):
            articles = news_service.fetch_raw_articles()

        assert len(articles) == 1
        assert articles[0]["title"] == "Sample article"


class TestJsonParsing:
    """Tests for Claude API response parsing."""

    def test_parse_direct_json(self) -> None:
        """Direct JSON array is parsed correctly."""
        text = '[{"article_index": 0, "relevance_score": 9}]'
        result = NewsService._parse_json_response(text)
        assert result is not None
        assert result[0]["relevance_score"] == 9

    def test_parse_markdown_code_block(self) -> None:
        """JSON inside markdown code fences is extracted."""
        text = '```json\n[{"article_index": 0, "relevance_score": 8}]\n```'
        result = NewsService._parse_json_response(text)
        assert result is not None
        assert result[0]["relevance_score"] == 8

    def test_parse_invalid_returns_none(self) -> None:
        """Invalid text returns None."""
        result = NewsService._parse_json_response("not json at all")
        assert result is None


# -- Integration test -------------------------------------------------------


@pytest.mark.integration
def test_claude_filter_end_to_end(
    news_service: NewsService, sample_articles: list[dict],
) -> None:
    """End-to-end test with real Claude API (requires ANTHROPIC_API_KEY)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    tickers = ["BARC.L", "SHEL.L", "AZN.L"]
    result = news_service._claude_filter(sample_articles, tickers, api_key)

    assert isinstance(result, list)
    for article in result:
        assert "relevance_score" in article
        assert "tickers_mentioned" in article
        assert "sentiment" in article
        assert article["sentiment"] in ("bullish", "bearish", "neutral", "mixed")
