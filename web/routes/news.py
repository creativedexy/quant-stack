"""FastAPI router for the news page (/ui/news).

Displays RSS news filtered by watchlist relevance, with optional
Claude API sentiment classification. HTMX refreshes the feed every
15 minutes.

Routes call NewsService from src/services/ only.
Never import from src/data, src/features, or src/models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.utils.config import load_config
from src.utils.logging import get_logger
from web.routes.ui import check_auth

logger = get_logger(__name__)

_WEB_DIR = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _WEB_DIR / "templates"

router = APIRouter(prefix="/ui", dependencies=[Depends(check_auth)])
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_css_path = _WEB_DIR / "static" / "css" / "theme.css"
templates.env.globals["css_v"] = (
    str(int(_css_path.stat().st_mtime)) if _css_path.exists() else "0"
)

# -- Service layer bootstrap -----------------------------------

_services: dict[str, Any] | None = None


def _get_services() -> dict[str, Any]:
    """Initialise and cache service instances."""
    global _services
    if _services is not None:
        return _services

    from src.services.news_service import NewsService

    try:
        config = load_config()
    except FileNotFoundError:
        config = {}

    news_svc = NewsService(config)

    _services = {
        "news": news_svc,
        "config": config,
    }
    return _services


# -- Helpers ---------------------------------------------------

def _get_tickers(config: dict[str, Any]) -> list[str]:
    """Read watchlist tickers from config."""
    return (
        config.get("ui", {})
        .get("watchlist", {})
        .get("tickers", [])
    )


def _filter_articles(
    articles: list[dict[str, Any]],
    ticker: str | None,
    sentiments: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter articles by ticker and/or sentiment."""
    result = articles

    if ticker and ticker.lower() != "all":
        result = [
            a for a in result
            if ticker in a.get("tickers_mentioned", [])
        ]

    if sentiments:
        sentiments_lower = [s.lower() for s in sentiments]
        result = [
            a for a in result
            if a.get("sentiment", "neutral").lower() in sentiments_lower
        ]

    return result


def _compute_counts(articles: list[dict[str, Any]]) -> dict[str, int]:
    """Compute summary counts for the sidebar."""
    total = len(articles)
    bullish = sum(1 for a in articles if a.get("sentiment") == "bullish")
    bearish = sum(1 for a in articles if a.get("sentiment") == "bearish")
    relevant = sum(
        1 for a in articles if a.get("relevance_score", 0) >= 7
    )
    return {
        "total": total,
        "relevant": relevant,
        "bullish": bullish,
        "bearish": bearish,
    }


def _ticker_sentiments(
    articles: list[dict[str, Any]],
    tickers: list[str],
) -> list[dict[str, str]]:
    """Compute dominant sentiment per ticker."""
    ticker_sent: dict[str, dict[str, int]] = {
        t: {"bullish": 0, "bearish": 0, "neutral": 0, "mixed": 0}
        for t in tickers
    }

    for article in articles:
        sentiment = article.get("sentiment", "neutral").lower()
        for t in article.get("tickers_mentioned", []):
            if t in ticker_sent and sentiment in ticker_sent[t]:
                ticker_sent[t][sentiment] += 1

    result: list[dict[str, str]] = []
    for t in tickers:
        counts = ticker_sent[t]
        if sum(counts.values()) == 0:
            dominant = "neutral"
        else:
            dominant = max(counts, key=lambda k: counts[k])
        result.append({"ticker": t, "sentiment": dominant})

    return result


# -- Routes ----------------------------------------------------


@router.get("/news", response_class=HTMLResponse)
async def news_page(
    request: Request,
    ticker: str = Query(default="all"),
    sentiment: list[str] | None = Query(default=None),
) -> HTMLResponse:
    """Render the full news page."""
    svcs = _get_services()
    config = svcs["config"]
    news_svc = svcs["news"]

    tickers = _get_tickers(config)
    news_cfg = config.get("news", {})
    refresh_seconds = news_cfg.get("refresh_minutes", 15) * 60

    articles = news_svc.get_cached_news()
    filtered = _filter_articles(articles, ticker, sentiment)
    counts = _compute_counts(articles)
    ticker_sents = _ticker_sentiments(articles, tickers)

    return templates.TemplateResponse(
        "news.html",
        {
            "request": request,
            "active_page": "news",
            "articles": filtered,
            "tickers": tickers,
            "active_ticker": ticker,
            "active_sentiments": sentiment or [],
            "counts": counts,
            "ticker_sentiments": ticker_sents,
            "refresh_seconds": refresh_seconds,
        },
    )


@router.get("/news/feed", response_class=HTMLResponse)
async def news_feed_partial(
    request: Request,
    ticker: str = Query(default="all"),
    sentiment: list[str] | None = Query(default=None),
) -> HTMLResponse:
    """Return news feed HTML fragment for HTMX swap."""
    svcs = _get_services()
    config = svcs["config"]
    news_svc = svcs["news"]

    articles = news_svc.get_cached_news()
    filtered = _filter_articles(articles, ticker, sentiment)

    return templates.TemplateResponse(
        "partials/news_feed.html",
        {
            "request": request,
            "articles": filtered,
        },
    )
