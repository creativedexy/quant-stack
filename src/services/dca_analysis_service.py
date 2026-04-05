"""DCA analysis service -- Claude API integration for entry analysis.

Provides streaming analysis of individual DCA purchases and free-text
Q&A about a user's dollar-cost-averaging history.  Streams responses
as SSE-formatted strings for consumption by HTMX.

Graceful degradation: if ANTHROPIC_API_KEY is not set or the API call
fails, returns a plain-text fallback -- NEVER raises to the caller.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a DCA (dollar-cost averaging) investment analyst for a quant trading system "
    "focused on FTSE 100 stocks. The user is reviewing their purchase history.\n\n"
    "When analysing a specific purchase entry, you:\n"
    "1. Assess whether the timing was good based on RSI at entry "
    "(below 40 = strong, above 65 = expensive)\n"
    "2. Calculate what the alternative optimal entry would have been "
    "and the cost difference in pounds\n"
    "3. Give a brief, direct verdict: \"Strong entry\", \"Acceptable entry\", "
    "or \"Could have waited\"\n"
    "4. Suggest a specific maximum entry price for future purchases "
    "based on their avg cost basis\n\n"
    "Keep responses to 3-4 sentences maximum. Use British English. Be direct "
    "-- no padding.\n"
    "NEVER use special characters, curly quotes, em-dashes, or non-ASCII text."
)

_FREETEXT_SYSTEM_PROMPT = (
    "You are a DCA (dollar-cost averaging) investment analyst for a quant trading system "
    "focused on FTSE 100 stocks. The user is asking a question about their purchase history.\n\n"
    "Answer concisely (3-5 sentences). Use British English. Be direct.\n"
    "NEVER use special characters, curly quotes, em-dashes, or non-ASCII text.\n"
    "All monetary values in GBP. Prices in pence where appropriate."
)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 512

_FALLBACK_ENTRY = (
    "AI analysis is currently unavailable. "
    "Based on the rule-based assessment, review the entry quality label "
    "and RSI value shown above for a quick evaluation."
)

_FALLBACK_FREETEXT = (
    "AI analysis is currently unavailable. "
    "Please check the summary statistics and purchase history table "
    "for the information you need."
)

# Startup API key check
if not os.environ.get("ANTHROPIC_API_KEY", ""):
    logger.warning(
        "ANTHROPIC_API_KEY not set -- DCA chat analysis will use "
        "fallback responses instead of Claude API"
    )


def _classify_entry_quality(rsi: float | None) -> str:
    """Return quality label from RSI value."""
    if rsi is None:
        return "neutral"
    if rsi < 35.0:
        return "best"
    if rsi < 45.0:
        return "good"
    if rsi <= 60.0:
        return "neutral"
    return "late"


def build_entry_context(
    ticker: str,
    entry_date: str,
    entry_price: float,
    purchase_history: list[dict[str, Any]],
    rsi_at_entry: float,
    dca_summary: dict[str, Any],
) -> str:
    """Build a structured context string for the Claude API prompt.

    Args:
        ticker: Instrument ticker (e.g. "SHEL.L").
        entry_date: Purchase date as YYYY-MM-DD.
        entry_price: Purchase price in pence.
        purchase_history: List of enriched purchase dicts.
        rsi_at_entry: RSI-14 value at the entry date.
        dca_summary: Aggregated DCA statistics dict.

    Returns:
        ASCII-safe structured text block.
    """
    quality = _classify_entry_quality(rsi_at_entry)

    avg_entry = dca_summary.get("avg_entry_price", 0.0)
    total_invested = dca_summary.get("total_invested", 0.0)
    total_return_pct = dca_summary.get("total_return_pct", 0.0) * 100.0

    lines = [
        f"TICKER: {ticker}",
        f"PURCHASE DATE: {entry_date}",
        f"PURCHASE PRICE: {entry_price:.0f}p",
        f"RSI AT ENTRY: {rsi_at_entry:.1f}",
        f"ENTRY QUALITY (rule-based): {quality}",
        f"TOTAL PURCHASES: {len(purchase_history)}",
        f"AVERAGE COST BASIS: {avg_entry:.0f}p",
        f"TOTAL INVESTED: GBP{total_invested:,.0f}",
        f"CURRENT RETURN: {total_return_pct:+.1f}%",
    ]
    return "\n".join(lines)


def _build_freetext_context(
    ticker: str,
    dca_summary: dict[str, Any],
    purchase_history: list[dict[str, Any]],
) -> str:
    """Build context for free-text questions."""
    avg_entry = dca_summary.get("avg_entry_price", 0.0)
    total_invested = dca_summary.get("total_invested", 0.0)
    current_value = dca_summary.get("current_value", 0.0)
    total_return_pct = dca_summary.get("total_return_pct", 0.0) * 100.0
    total_shares = dca_summary.get("total_shares", 0)

    lines = [
        f"TICKER: {ticker}",
        f"TOTAL PURCHASES: {len(purchase_history)}",
        f"TOTAL INVESTED: GBP{total_invested:,.0f}",
        f"CURRENT VALUE: GBP{current_value:,.0f}",
        f"TOTAL SHARES: {total_shares}",
        f"AVERAGE COST BASIS: {avg_entry:.0f}p",
        f"CURRENT RETURN: {total_return_pct:+.1f}%",
        "",
        "PURCHASE HISTORY:",
    ]

    for p in purchase_history:
        quality = p.get("entry_quality", "unknown")
        rsi = p.get("rsi_at_purchase")
        rsi_str = f"{rsi:.1f}" if rsi is not None else "n/a"
        ret_pct = p.get("return_pct", 0.0) * 100.0
        lines.append(
            f"  {p['date']} | {p['price']:.0f}p | RSI {rsi_str} | "
            f"{quality} | return {ret_pct:+.1f}%"
        )

    return "\n".join(lines)


async def stream_entry_analysis(
    ticker: str,
    entry_date: str,
    entry_price: float,
    purchase_history: list[dict[str, Any]],
    rsi_at_entry: float,
    dca_summary: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    """Stream SSE-formatted analysis of a single DCA entry.

    Yields SSE-formatted strings: ``"data: {chunk}\\n\\n"``.
    On API error, yields a single fallback event then stops.

    Args:
        ticker: Instrument ticker.
        entry_date: Purchase date as YYYY-MM-DD.
        entry_price: Purchase price in pence.
        purchase_history: List of enriched purchase dicts.
        rsi_at_entry: RSI-14 value at entry date.
        dca_summary: Aggregated DCA statistics (optional).
    """
    if dca_summary is None:
        dca_summary = {}

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set -- returning fallback")
        yield f"data: {_FALLBACK_ENTRY}\n\n"
        yield "data: [DONE]\n\n"
        return

    context = build_entry_context(
        ticker, entry_date, entry_price,
        purchase_history, rsi_at_entry, dca_summary,
    )
    user_message = (
        f"Analyse this DCA purchase entry:\n\n{context}"
    )

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)
        async with client.messages.stream(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                yield f"data: {text}\n\n"
    except Exception as exc:
        logger.error("Claude API streaming failed: %s", exc)
        yield f"data: {_FALLBACK_ENTRY}\n\n"

    yield "data: [DONE]\n\n"


async def stream_freetext_response(
    ticker: str,
    question: str,
    dca_summary: dict[str, Any],
    purchase_history: list[dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """Stream SSE-formatted answer to a free-text DCA question.

    Yields SSE-formatted strings: ``"data: {chunk}\\n\\n"``.
    On API error, yields a single fallback event then stops.

    Args:
        ticker: Instrument ticker.
        question: The user's free-text question.
        dca_summary: Aggregated DCA statistics.
        purchase_history: List of enriched purchase dicts.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set -- returning fallback")
        yield f"data: {_FALLBACK_FREETEXT}\n\n"
        yield "data: [DONE]\n\n"
        return

    context = _build_freetext_context(ticker, dca_summary, purchase_history)
    user_message = (
        f"Context about the user's DCA for {ticker}:\n\n"
        f"{context}\n\n"
        f"User question: {question}"
    )

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)
        async with client.messages.stream(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_FREETEXT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                yield f"data: {text}\n\n"
    except Exception as exc:
        logger.error("Claude API streaming failed: %s", exc)
        yield f"data: {_FALLBACK_FREETEXT}\n\n"

    yield "data: [DONE]\n\n"
