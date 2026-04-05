"""Tests for FearGreedFeed."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.data.sentiment_feed import FearGreedFeed
from src.utils.exceptions import DataFeedError


@pytest.fixture
def mock_http_client() -> AsyncMock:
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def feed(mock_http_client: AsyncMock) -> FearGreedFeed:
    return FearGreedFeed(mock_http_client)


def _make_response(value: str, status_code: int = 200) -> MagicMock:
    """Build a mock httpx Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"data": [{"value": value}]}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


class TestGetCurrentIndex:
    @pytest.mark.asyncio
    async def test_returns_integer_index(
        self, feed: FearGreedFeed, mock_http_client: AsyncMock
    ) -> None:
        """API returns {"data": [{"value": "23"}]} -> returns int 23."""
        mock_http_client.get.return_value = _make_response("23")
        result = await feed.get_current_index()
        assert result == 23
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_returns_cached_on_api_failure(
        self, feed: FearGreedFeed, mock_http_client: AsyncMock
    ) -> None:
        """First call succeeds (30), second call fails -> returns cached 30."""
        mock_http_client.get.return_value = _make_response("30")
        result1 = await feed.get_current_index()
        assert result1 == 30

        # Second call fails
        mock_http_client.get.side_effect = httpx.ConnectError("connection failed")
        result2 = await feed.get_current_index()
        assert result2 == 30

    @pytest.mark.asyncio
    async def test_raises_if_no_cache_and_api_fails(
        self, feed: FearGreedFeed, mock_http_client: AsyncMock
    ) -> None:
        """First call fails with no cached value -> DataFeedError."""
        mock_http_client.get.side_effect = httpx.ConnectError("connection failed")
        with pytest.raises(DataFeedError, match="no cached value"):
            await feed.get_current_index()

    @pytest.mark.asyncio
    async def test_validates_range(
        self, feed: FearGreedFeed, mock_http_client: AsyncMock
    ) -> None:
        """Value outside 0-100 raises DataFeedError."""
        mock_http_client.get.return_value = _make_response("150")
        with pytest.raises(DataFeedError, match="out of range"):
            await feed.get_current_index()
