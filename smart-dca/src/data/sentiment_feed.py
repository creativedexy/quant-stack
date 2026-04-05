"""Fear & Greed index feed from Alternative.me API.

Provides current and historical F&G values as a contrarian
sentiment indicator. Values below the configured threshold
indicate fear and are treated as buy signals.
"""

from __future__ import annotations

import httpx

from src.utils.exceptions import DataFeedError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FNG_API_URL = "https://api.alternative.me/fng/"


class FearGreedFeed:
    """Fetches Fear & Greed index data from Alternative.me.

    Args:
        http_client: An httpx.AsyncClient instance (injected for testability).
    """

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._client = http_client
        self._cached_value: int | None = None

    async def get_current_index(self) -> int:
        """Fetch the current Fear & Greed index (0-100).

        Returns:
            Integer index value.

        Raises:
            DataFeedError: If the API fails and no cached value is available.
        """
        try:
            response = await self._client.get(f"{_FNG_API_URL}?limit=1")
            response.raise_for_status()
            data = response.json()
            value = int(data["data"][0]["value"])
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            if self._cached_value is not None:
                logger.warning(
                    "fng_api_fallback_to_cache",
                    cached_value=self._cached_value,
                    error=str(exc),
                )
                return self._cached_value
            raise DataFeedError(f"Fear & Greed API failed and no cached value: {exc}") from exc

        if not (0 <= value <= 100):
            raise DataFeedError(f"Fear & Greed value out of range: {value}")

        self._cached_value = value
        logger.debug("fng_fetched", value=value)
        return value

    async def get_index_trend(self, days: int = 7) -> list[int]:
        """Fetch daily F&G values, most recent first.

        Args:
            days: Number of days of history to fetch.

        Returns:
            List of integer F&G values, most recent first.

        Raises:
            DataFeedError: If the API fails.
        """
        try:
            response = await self._client.get(f"{_FNG_API_URL}?limit={days}")
            response.raise_for_status()
            data = response.json()
            return [int(entry["value"]) for entry in data["data"]]
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            raise DataFeedError(f"Fear & Greed trend API failed: {exc}") from exc
