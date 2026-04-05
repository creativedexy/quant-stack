"""News refresh job -- periodic RSS fetch and classification.

Designed to run as an APScheduler job inside PipelineScheduler.

Usage:
    from src.scheduler.news_job import NewsRefreshJob
    job = NewsRefreshJob(news_service, settings)
    job.run()
"""

from __future__ import annotations

from typing import Any

from src.services.news_service import NewsService
from src.utils.logging import get_logger

logger = get_logger(__name__)


class NewsRefreshJob:
    """Refreshes news articles on a scheduled interval."""

    def __init__(
        self,
        news_service: NewsService,
        settings: dict[str, Any],
    ) -> None:
        """Initialise the news refresh job.

        Args:
            news_service: NewsService instance for fetching and filtering.
            settings: Project configuration dict.
        """
        self._news_service = news_service
        self._settings = settings

    def run(self) -> None:
        """Execute a news refresh cycle.

        Reads watchlist tickers from settings, calls
        ``news_service.refresh_news()``, and logs the result.
        Never raises -- all errors are caught and logged.
        """
        try:
            tickers = (
                self._settings
                .get("ui", {})
                .get("watchlist", {})
                .get("tickers", [])
            )
            if not tickers:
                logger.warning("No watchlist tickers configured -- skipping news refresh")
                return

            articles = self._news_service.refresh_news(tickers)
            logger.info(
                "news_refresh_complete",
                extra={"article_count": len(articles)},
            )
        except Exception as exc:
            logger.error("News refresh failed: %s", exc)
