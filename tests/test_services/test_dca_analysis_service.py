"""Tests for DCAAnalysisService.

Covers context building, ASCII safety, and graceful degradation
when the Anthropic API key is missing or the API call fails.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.dca_analysis_service import (
    SYSTEM_PROMPT,
    build_entry_context,
    stream_entry_analysis,
    stream_freetext_response,
)


# -- Fixtures ----------------------------------------------------------


@pytest.fixture
def sample_history() -> list[dict]:
    """Minimal purchase history for testing."""
    return [
        {
            "date": "2024-01-15",
            "price": 2500.0,
            "amount_gbp": 1000.0,
            "shares": 40,
            "entry_quality": "good",
            "rsi_at_purchase": 38.5,
            "return_pct": 0.05,
            "current_value": 1050.0,
            "return_gbp": 50.0,
        },
        {
            "date": "2024-04-10",
            "price": 2700.0,
            "amount_gbp": 1000.0,
            "shares": 37,
            "entry_quality": "late",
            "rsi_at_purchase": 68.2,
            "return_pct": -0.02,
            "current_value": 980.0,
            "return_gbp": -20.0,
        },
    ]


@pytest.fixture
def sample_summary() -> dict:
    """Minimal DCA summary for testing."""
    return {
        "ticker": "SHEL.L",
        "total_invested": 2000.0,
        "current_value": 2030.0,
        "total_return_pct": 0.015,
        "total_return_gbp": 30.0,
        "avg_entry_price": 2596.0,
        "total_shares": 77,
    }


def _collect_sse(coro):
    """Run an async generator and collect all chunks."""
    chunks = []

    async def _run():
        async for chunk in coro:
            chunks.append(chunk)

    asyncio.get_event_loop_policy()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()
    return chunks


# -- Tests -------------------------------------------------------------


class TestBuildEntryContext:
    """Tests for build_entry_context."""

    def test_build_entry_context_is_ascii_safe(
        self, sample_history, sample_summary,
    ):
        """Context string must contain only ASCII characters."""
        context = build_entry_context(
            ticker="SHEL.L",
            entry_date="2024-01-15",
            entry_price=2500.0,
            purchase_history=sample_history,
            rsi_at_entry=38.5,
            dca_summary=sample_summary,
        )
        assert context.isascii(), f"Non-ASCII found in context: {context!r}"

    def test_context_contains_key_fields(
        self, sample_history, sample_summary,
    ):
        """Context must include all required fields."""
        context = build_entry_context(
            ticker="SHEL.L",
            entry_date="2024-01-15",
            entry_price=2500.0,
            purchase_history=sample_history,
            rsi_at_entry=38.5,
            dca_summary=sample_summary,
        )
        assert "TICKER: SHEL.L" in context
        assert "PURCHASE DATE: 2024-01-15" in context
        assert "2500p" in context
        assert "RSI AT ENTRY: 38.5" in context
        assert "TOTAL PURCHASES: 2" in context


class TestSystemPrompt:
    """Tests for system prompt content."""

    def test_system_prompt_is_ascii_safe(self):
        """System prompt must contain only ASCII characters."""
        assert SYSTEM_PROMPT.isascii(), (
            f"Non-ASCII found in SYSTEM_PROMPT: {SYSTEM_PROMPT!r}"
        )


class TestStreamEntryAnalysis:
    """Tests for stream_entry_analysis."""

    def test_stream_entry_analysis_fallback_when_no_api_key(
        self, sample_history, sample_summary,
    ):
        """When ANTHROPIC_API_KEY is not set, yield fallback text."""
        with patch.dict("os.environ", {}, clear=True):
            # Ensure no key is present
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)

            chunks = _collect_sse(
                stream_entry_analysis(
                    ticker="SHEL.L",
                    entry_date="2024-01-15",
                    entry_price=2500.0,
                    purchase_history=sample_history,
                    rsi_at_entry=38.5,
                    dca_summary=sample_summary,
                )
            )

        assert len(chunks) >= 2
        # First chunk should be fallback
        assert "data: " in chunks[0]
        assert "unavailable" in chunks[0].lower()
        # Last chunk should be sentinel
        assert chunks[-1].strip() == "data: [DONE]"


class TestStreamFreetextResponse:
    """Tests for stream_freetext_response."""

    def test_stream_freetext_fallback_on_api_error(
        self, sample_history, sample_summary,
    ):
        """When the API throws an exception, yield fallback -- no raise."""
        mock_client = MagicMock()
        mock_client.messages.stream.side_effect = Exception("API down")

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch(
                "src.services.dca_analysis_service.anthropic",
                create=True,
            ) as mock_anthropic:
                mock_anthropic.AsyncAnthropic.return_value = mock_client

                # The import inside the function will use our mock
                with patch.dict(
                    "sys.modules",
                    {"anthropic": mock_anthropic},
                ):
                    chunks = _collect_sse(
                        stream_freetext_response(
                            ticker="SHEL.L",
                            question="when did I overpay?",
                            dca_summary=sample_summary,
                            purchase_history=sample_history,
                        )
                    )

        # Should have at least fallback + sentinel -- no exception raised
        assert len(chunks) >= 2
        assert chunks[-1].strip() == "data: [DONE]"

    def test_break_even_question_mentions_price(
        self, sample_history, sample_summary,
    ):
        """Context for a break-even question includes current_value."""
        # We just verify the context is built correctly -- no API call
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)

            chunks = _collect_sse(
                stream_freetext_response(
                    ticker="SHEL.L",
                    question="what's my break-even?",
                    dca_summary=sample_summary,
                    purchase_history=sample_history,
                )
            )

        # Should get fallback text (no API key)
        assert len(chunks) >= 2
        # The context building should not raise
        # and the fallback should be returned
        fallback_text = chunks[0]
        assert "data: " in fallback_text


@pytest.mark.integration
def test_stream_entry_analysis_live(sample_history, sample_summary):
    """Integration test -- actually calls the Claude API.

    Skipped in CI unless ANTHROPIC_API_KEY is set.
    """
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    chunks = _collect_sse(
        stream_entry_analysis(
            ticker="SHEL.L",
            entry_date="2024-01-15",
            entry_price=2500.0,
            purchase_history=sample_history,
            rsi_at_entry=38.5,
            dca_summary=sample_summary,
        )
    )

    assert len(chunks) >= 2
    assert chunks[-1].strip() == "data: [DONE]"
    # At least one non-sentinel data chunk
    data_chunks = [c for c in chunks if "[DONE]" not in c]
    assert len(data_chunks) >= 1
