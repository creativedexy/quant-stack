"""Guide Chat route -- Claude-powered contextual assistant.

POST /ui/guide/chat returns SSE stream with AI responses.
Falls back to keyword matching if ANTHROPIC_API_KEY is not set.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from src.utils.logging import get_logger
from web.routes.ui import check_auth

logger = get_logger(__name__)

router = APIRouter(prefix="/ui", dependencies=[Depends(check_auth)])

_SYSTEM = (
    "You are the Quant Guide, an assistant embedded in the Quant Stack "
    "algorithmic trading system. You help the user understand how to use "
    "the system and interpret what the data means.\n"
    "The system has these pages:\n"
    "- Watchlist: FTSE 100 tickers, intraday charts, fundamentals (P/E, margin, yield)\n"
    "- Chart Deep-Dive: Bollinger Bands, RSI, MACD, ML Confidence, Monte Carlo projections\n"
    "- Portfolio: Position heatmap, VaR, CVaR, DCA planner\n"
    "- Research: Upload papers, build strategies, walk-forward backtesting\n"
    "- News Feed: AI-filtered RSS news, sentiment, signal impact\n"
    "- Execution: Interactive Brokers paper/live order management\n"
    "Signal system: RandomForest + GradientBoosting models produce Long/Short/Neutral "
    "signals with confidence scores. RSI below 35 = oversold, above 65 = overbought.\n"
    "No lookahead bias -- all features use only past data. Walk-forward validation only.\n"
    "Keep answers to 2-4 sentences. Be direct. Use British English.\n"
    "NEVER use curly quotes, em dashes, or non-ASCII characters."
)

_FALLBACK_RESPONSES: list[tuple[list[str], str]] = [
    (
        ["barc", "signal"],
        "BARC is showing a momentum long signal at 87% confidence. RSI is 54 - "
        "not overbought. Trigger: 3-day close above 20-day SMA plus NIM guidance upgrade.",
    ),
    (
        ["backtest", "test"],
        "Go to Research and upload a strategy or use the parameter explorer. "
        "The engine uses walk-forward validation - no lookahead bias. Runs in ~30 seconds.",
    ),
    (
        ["rsi"],
        "RSI shows momentum on a 0-100 scale. Below 35 is oversold (potential buy), "
        "above 65 is overbought (potential exit). Toggle the RSI panel on the Chart page.",
    ),
    (
        ["risk", "portfolio"],
        "Your portfolio VaR (95%) is GBP 2,840 - 5% chance of losing more in a day. "
        "CVaR is GBP 3,210. Both within configured limits. Full detail in Portfolio.",
    ),
    (
        ["dca", "planner"],
        "The DCA Planner is under Portfolio > DCA tab. Add purchase history, set a "
        "monthly contribution and target, and Monte Carlo simulates 1,000 paths.",
    ),
]

_DEFAULT_FALLBACK = (
    "I can help with that. Open the relevant page from the tiles below - "
    "each section has built-in contextual data. What specifically would you like to know?"
)


def _keyword_fallback(message: str) -> str:
    """Match message against keyword groups and return best response."""
    lower = message.lower()
    for keywords, response in _FALLBACK_RESPONSES:
        if any(kw in lower for kw in keywords):
            return response
    return _DEFAULT_FALLBACK


async def _sse_fallback(message: str):
    """Generate SSE events from keyword fallback."""
    response = _keyword_fallback(message)
    yield f"data: {response}\n\n"
    yield "data: [DONE]\n\n"


async def _sse_claude(message: str):
    """Generate SSE events from Claude API streaming."""
    try:
        import anthropic

        client = anthropic.Anthropic()
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=_SYSTEM,
            messages=[{"role": "user", "content": message}],
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {text}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.error("Claude API error: %s", exc)
        fallback = _keyword_fallback(message)
        yield f"data: {fallback}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/guide/chat")
async def guide_chat(request: Request, message: str = Form("")) -> StreamingResponse:
    """Chat endpoint -- returns SSE stream with AI response."""
    if not message.strip():
        async def empty():
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        return StreamingResponse(
            _sse_claude(message),
            media_type="text/event-stream",
        )
    else:
        return StreamingResponse(
            _sse_fallback(message),
            media_type="text/event-stream",
        )
