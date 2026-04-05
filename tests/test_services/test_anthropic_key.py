"""Integration test -- verify ANTHROPIC_API_KEY is resolvable.

Requires ``ANTHROPIC_API_KEY`` to be set in the environment.
Skipped automatically in CI / offline runs via the integration marker.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def api_key() -> str:
    """Return the Anthropic API key or skip the test."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


class TestAnthropicKeyResolution:
    """Verify the Anthropic API key is valid and usable."""

    def test_key_is_non_empty(self, api_key: str) -> None:
        assert len(api_key) > 10, "API key looks too short"

    def test_key_format(self, api_key: str) -> None:
        """Anthropic keys start with 'sk-ant-'."""
        assert api_key.startswith("sk-ant-"), (
            f"Unexpected key prefix: {api_key[:8]}..."
        )

    def test_news_service_detects_key(self, api_key: str) -> None:
        """NewsService should detect the API key at init."""
        from src.services.news_service import NewsService

        svc = NewsService()
        assert svc._has_api_key is True

    def test_client_can_be_created(self, api_key: str) -> None:
        """Verify the anthropic package can create a client."""
        try:
            import anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        client = anthropic.Anthropic(api_key=api_key)
        assert client is not None
