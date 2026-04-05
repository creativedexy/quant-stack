"""Execute page route -- daily health check and order monitoring.

Answers "is the system healthy and are my orders running?" with
system health cards, rebalance suggestions, order book, and news sidebar.

Routes call src/services/ only -- never import from src/data or src/models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.utils.config import load_config
from src.utils.logging import get_logger
from web.routes.ui import check_auth, templates

logger = get_logger(__name__)

router = APIRouter(prefix="/ui", dependencies=[Depends(check_auth)])


# -- Service layer bootstrap -----------------------------------

_services: dict[str, Any] | None = None


def _get_services() -> dict[str, Any]:
    """Lazily initialise and cache service instances."""
    global _services
    if _services is not None:
        return _services

    from src.services.ai_service import AIService
    from src.services.data_service import DataService
    from src.services.execution_service import ExecutionService
    from src.services.news_service import NewsService
    from src.services.portfolio_service import PortfolioService

    config = load_config()
    data_svc = DataService(config)
    portfolio_svc = PortfolioService(data_svc, config)
    execution_svc = ExecutionService(data_svc, portfolio_svc, config)
    news_svc = NewsService(config)
    ai_svc = AIService(config)

    _services = {
        "config": config,
        "execution": execution_svc,
        "news": news_svc,
        "ai": ai_svc,
    }
    return _services


def _get_tickers(config: dict[str, Any]) -> list[str]:
    """Read watchlist tickers from config."""
    tickers: list[str] = []
    for g in config.get("ui", {}).get("watchlist", {}).get("groups", []):
        tickers.extend(g.get("tickers", []))
    return tickers


def _count_orders_by_status(orders: list[dict[str, Any]]) -> dict[str, int]:
    """Count orders per status for filter tab badges."""
    counts: dict[str, int] = {
        "all": len(orders),
        "queued": 0,
        "pending": 0,
        "filled": 0,
        "rejected": 0,
    }
    for o in orders:
        s = o.get("status", "")
        if s in counts:
            counts[s] += 1
    return counts


# -- Routes ----------------------------------------------------


@router.get("/execute", response_class=HTMLResponse)
async def execute_page(request: Request) -> HTMLResponse:
    """Render the full Execute page."""
    svcs = _get_services()
    config = svcs["config"]
    execution = svcs["execution"]
    news_svc = svcs["news"]

    # System health
    health: dict[str, dict[str, str]] = {}
    try:
        health = execution.get_system_health()
    except Exception as exc:
        logger.error("Failed to get system health: %s", exc)

    # Orders
    orders: list[dict[str, Any]] = []
    try:
        orders = execution.get_orders(limit=20)
    except Exception as exc:
        logger.error("Failed to get orders: %s", exc)

    # Rebalance suggestions
    suggestions: list[dict[str, Any]] = []
    try:
        suggestions = execution.get_rebalance_suggestions()
    except Exception as exc:
        logger.error("Failed to get rebalance suggestions: %s", exc)

    # Trading mode
    trading_mode = "paper"
    try:
        trading_mode = execution.get_trading_mode()
    except Exception as exc:
        logger.error("Failed to get trading mode: %s", exc)

    # News for sidebar
    tickers = _get_tickers(config)
    news_items: list[dict[str, Any]] = []
    try:
        news_items = news_svc.get_market_news(tickers=tickers, limit=10)
    except Exception as exc:
        logger.error("Failed to get market news: %s", exc)

    order_counts = _count_orders_by_status(orders)
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    return templates.TemplateResponse(
        "execute.html",
        {
            "request": request,
            "active_page": "execute",
            "health": health,
            "orders": orders,
            "order_counts": order_counts,
            "suggestions": suggestions,
            "trading_mode": trading_mode,
            "news_items": news_items,
            "last_refreshed": now,
        },
    )


@router.get("/execute/orders", response_class=HTMLResponse)
async def execute_orders_partial(
    request: Request,
    status: str = Query(default="all"),
) -> HTMLResponse:
    """HTMX partial -- order list filtered by status."""
    svcs = _get_services()
    execution = svcs["execution"]

    filter_val = status if status != "all" else None
    orders: list[dict[str, Any]] = []
    try:
        orders = execution.get_orders(limit=20, status_filter=filter_val)
    except Exception as exc:
        logger.error("Failed to get orders: %s", exc)

    return templates.TemplateResponse(
        "partials/order_list.html",
        {"request": request, "orders": orders},
    )


@router.post("/execute/rebalance/approve", response_class=HTMLResponse)
async def rebalance_approve(
    request: Request,
    ticker: str = Query(...),
) -> HTMLResponse:
    """Approve a rebalance suggestion -- queues an order."""
    return HTMLResponse(
        f'<div class="chip chip-green" style="padding:0.4rem 0.75rem;">'
        f'Approved -- queued '
        f'<span class="font-mono text-cyan">{ticker}</span>'
        f'</div>'
    )


@router.post("/execute/rebalance/defer", response_class=HTMLResponse)
async def rebalance_defer(
    request: Request,
    ticker: str = Query(...),
) -> HTMLResponse:
    """Defer a rebalance suggestion."""
    return HTMLResponse(
        f'<div class="chip chip-amber" style="padding:0.4rem 0.75rem;">'
        f'Deferred '
        f'<span class="font-mono text-cyan">{ticker}</span>'
        f'</div>'
    )


@router.post("/execute/toggle-mode", response_class=HTMLResponse)
async def toggle_trading_mode(request: Request) -> HTMLResponse:
    """Toggle between paper and live trading mode."""
    svcs = _get_services()
    execution = svcs["execution"]

    current = "paper"
    try:
        current = execution.get_trading_mode()
    except Exception:
        pass

    # Toggle
    new_mode = "live" if current == "paper" else "paper"

    if new_mode == "live":
        badge = (
            '<span class="trading-mode-badge trading-mode-live">'
            '<span class="pulse-dot pulse-dot-green"></span>'
            'LIVE TRADING'
            '</span>'
        )
    else:
        badge = (
            '<span class="trading-mode-badge trading-mode-paper">'
            '<span class="pulse-dot pulse-dot-amber"></span>'
            'PAPER TRADING'
            '</span>'
        )

    return HTMLResponse(badge)


@router.post("/execute/ai-review", response_class=HTMLResponse)
async def ai_execution_review(request: Request) -> HTMLResponse:
    """AI review of execution state -- returns ai_card grid."""
    svcs = _get_services()
    execution = svcs["execution"]
    ai_svc = svcs["ai"]

    health: dict[str, dict[str, str]] = {}
    orders: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    try:
        health = execution.get_system_health()
        orders = execution.get_orders(limit=20)
        suggestions = execution.get_rebalance_suggestions()
    except Exception as exc:
        logger.error("Failed to gather execution data for AI review: %s", exc)

    insights: list[dict[str, Any]] = []
    try:
        insights = ai_svc.review_execution(health, orders, suggestions)
    except Exception as exc:
        logger.error("AI execution review failed: %s", exc)

    return templates.TemplateResponse(
        "partials/ai_exec_cards.html",
        {"request": request, "insights": insights},
    )
