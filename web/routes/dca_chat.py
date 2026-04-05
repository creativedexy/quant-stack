"""FastAPI router for DCA AI chat (/ui/portfolio/dca/).

Provides SSE streaming endpoints for:
- Entry analysis: Claude analyses a specific DCA purchase
- Free-text chat: user asks questions about their DCA history

Both endpoints return ``text/event-stream`` responses consumed by
HTMX.  Each chunk is an HTML ``<span>`` so fragments concatenate
naturally in the DOM.  The final chunk is ``data: [DONE]\\n\\n``
which signals HTMX to close the SSE connection.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import StreamingResponse

from src.services.dca_analysis_service import (
    stream_entry_analysis,
    stream_freetext_response,
)
from src.utils.config import load_config
from src.utils.logging import get_logger
from web.routes.ui import check_auth

logger = get_logger(__name__)

router = APIRouter(prefix="/ui/portfolio/dca", dependencies=[Depends(check_auth)])


# -- Service bootstrap ------------------------------------------------

_services: dict[str, Any] | None = None


def _get_services() -> dict[str, Any]:
    """Initialise and cache service instances."""
    global _services
    if _services is not None:
        return _services

    from src.services.chart_service import ChartService
    from src.services.data_service import DataService
    from src.services.dca_service import DCAService
    from src.services.market_data_service import MarketDataService

    try:
        config = load_config()
    except FileNotFoundError:
        config = {}

    data_svc = DataService(config)
    market_svc = MarketDataService(config)
    chart_svc = ChartService(config)
    dca_svc = DCAService(config, chart_svc, data_svc, market_svc)

    _services = {
        "dca": dca_svc,
        "chart": chart_svc,
        "config": config,
    }
    return _services


# -- SSE helpers -------------------------------------------------------


async def _wrap_sse_as_html(sse_generator):
    """Wrap SSE data chunks as HTML span elements.

    Passes through the ``[DONE]`` sentinel unchanged so HTMX can
    detect stream completion.
    """
    async for chunk in sse_generator:
        # chunk format: "data: {text}\n\n"
        if not chunk.startswith("data: "):
            yield chunk
            continue

        payload = chunk[6:].rstrip("\n")

        if payload == "[DONE]":
            yield "data: [DONE]\n\n"
            continue

        # Escape HTML entities in the text chunk
        safe = (
            payload
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        yield f"data: <span>{safe}</span>\n\n"


# -- Routes ------------------------------------------------------------


@router.get("/analyse")
async def analyse_entry(
    request: Request,
    ticker: str = Query(..., description="Ticker symbol"),
    date: str = Query(..., description="Purchase date YYYY-MM-DD"),
    price: float = Query(..., description="Purchase price in pence"),
) -> StreamingResponse:
    """Stream AI analysis of a specific DCA purchase entry.

    Returns SSE with HTML span fragments injected into
    ``#dca-chat-messages`` via ``hx-swap="beforeend"``.
    """
    svcs = _get_services()
    dca_svc = svcs["dca"]
    chart_svc = svcs["chart"]

    # Load purchase history and summary
    try:
        history = dca_svc.get_purchase_history(ticker)
    except Exception:
        history = []

    try:
        summary = dca_svc.get_dca_summary(ticker)
    except Exception:
        summary = {}

    # Look up RSI at the entry date
    rsi_at_entry = 50.0  # default neutral
    try:
        rsi_data = chart_svc.get_rsi_series(ticker, "All")
        dates = rsi_data.get("dates", [])
        rsi_vals = rsi_data.get("rsi", [])
        for d, r in zip(dates, rsi_vals):
            if d == date and r is not None:
                rsi_at_entry = float(r)
                break
    except Exception as exc:
        logger.warning("RSI lookup failed for %s on %s: %s", ticker, date, exc)

    sse_stream = stream_entry_analysis(
        ticker=ticker,
        entry_date=date,
        entry_price=price,
        purchase_history=history,
        rsi_at_entry=rsi_at_entry,
        dca_summary=summary,
    )

    return StreamingResponse(
        _wrap_sse_as_html(sse_stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def freetext_chat(
    request: Request,
    message: str = Form(..., description="User question"),
    ticker: str = Form(..., description="Ticker symbol"),
) -> StreamingResponse:
    """Stream AI answer to a free-text DCA question.

    Prepends a user message bubble before the AI response.
    Returns SSE with HTML fragments.
    """
    svcs = _get_services()
    dca_svc = svcs["dca"]

    try:
        history = dca_svc.get_purchase_history(ticker)
    except Exception:
        history = []

    try:
        summary = dca_svc.get_dca_summary(ticker)
    except Exception:
        summary = {}

    async def _generate():
        # Emit user message bubble first
        safe_msg = (
            message
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        yield f'data: <div class="msg user">{safe_msg}</div>\n\n'

        # Stream the AI response
        sse_stream = stream_freetext_response(
            ticker=ticker,
            question=message,
            dca_summary=summary,
            purchase_history=history,
        )
        async for chunk in _wrap_sse_as_html(sse_stream):
            yield chunk

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
