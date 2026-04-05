"""FastAPI router for the redesigned Portfolio page (/ui/portfolio).

The user's primary operational hub. Answers: "What do I own, how is
it doing, and what should I change?"

Server-rendered heatmap grid, HTMX-loaded drawer, always-visible DCA
sidebar.  No Plotly treemap, no JavaScript modals.

Routes call service functions from src/services/ only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.utils.logging import get_logger
from web.routes.ui import (
    _gbp,
    _pct,
    _pct_plain,
    _signal_label,
    _lse_market_context,
    check_auth,
)

logger = get_logger(__name__)

_WEB_DIR = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _WEB_DIR / "templates"

router = APIRouter(prefix="/ui", dependencies=[Depends(check_auth)])
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_css_path = _WEB_DIR / "static" / "css" / "theme.css"
templates.env.globals["css_v"] = (
    str(int(_css_path.stat().st_mtime)) if _css_path.exists() else "0"
)
templates.env.globals["lse_market"] = _lse_market_context

templates.env.filters["gbp"] = _gbp
templates.env.filters["pct"] = _pct
templates.env.filters["pct_plain"] = _pct_plain
templates.env.filters["signal_label"] = _signal_label


# -- Service layer bootstrap -----------------------------------

_services: dict[str, Any] | None = None


def _get_services() -> dict[str, Any]:
    """Initialise and cache service instances."""
    global _services
    if _services is not None:
        return _services

    from src.services.ai_service import AIService
    from src.services.data_service import DataService
    from src.services.execution_service import ExecutionService
    from src.services.news_service import NewsService
    from src.services.portfolio_service import PortfolioService
    from src.services.strategy_service import StrategyService
    from src.utils.config import load_config

    try:
        config = load_config()
    except FileNotFoundError:
        config = {}

    data_svc = DataService(config)
    portfolio_svc = PortfolioService(data_svc, config)
    strategy_svc = StrategyService(data_svc, config)
    execution_svc = ExecutionService(data_svc, portfolio_svc, config)
    news_svc = NewsService(config)
    ai_svc = AIService(config)

    _services = {
        "config": config,
        "data": data_svc,
        "portfolio": portfolio_svc,
        "strategy": strategy_svc,
        "execution": execution_svc,
        "news": news_svc,
        "ai": ai_svc,
    }
    return _services


# -- Helpers ---------------------------------------------------


def _get_active_signals(strategy_svc: Any) -> dict[str, Any]:
    """Build ticker-keyed dict of active signals."""
    signals: dict[str, Any] = {}
    try:
        strategies = strategy_svc.get_available_strategies()
        for name in strategies:
            df = strategy_svc.get_signals(name, n_days=5)
            if df.empty:
                continue
            for _, row in df.iterrows():
                ticker = row.get("ticker", "")
                direction = row.get(
                    "direction", row.get("signal", "HOLD"),
                )
                if direction == "NEUTRAL" or direction == "HOLD":
                    continue
                confidence = float(
                    row.get("confidence", row.get("score", 0.5))
                )
                signals[ticker] = {
                    "direction": direction,
                    "confidence": round(confidence, 2),
                    "strategy": name,
                    "label": _signal_label(confidence),
                }
    except Exception as exc:
        logger.error("Failed to get active signals: %s", exc)
    return signals


# -- Icon SVG snippets for AI cards ----------------------------

_ICON_SVGS = {
    "check": (
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none">'
        '<path d="M13 4L6 12L3 9" stroke="currentColor" stroke-width="2"'
        ' stroke-linecap="round"/></svg>'
    ),
    "alert": (
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none">'
        '<path d="M8 3v6M8 12v1" stroke="currentColor" stroke-width="2"'
        ' stroke-linecap="round"/></svg>'
    ),
    "chart": (
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none">'
        '<path d="M2 14l4-5 3 2 5-7" stroke="currentColor" stroke-width="2"'
        ' stroke-linecap="round"/></svg>'
    ),
    "info": (
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none">'
        '<circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.5"/>'
        '<path d="M8 7v4M8 5v1" stroke="currentColor" stroke-width="2"'
        ' stroke-linecap="round"/></svg>'
    ),
}


# -- Routes ----------------------------------------------------


@router.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request) -> HTMLResponse:
    """Portfolio hub -- heatmap, drawer, DCA sidebar, risk alert."""
    svcs = _get_services()
    portfolio_svc = svcs["portfolio"]
    strategy_svc = svcs["strategy"]
    execution_svc = svcs["execution"]

    # Positions + summary
    positions = portfolio_svc.get_positions()
    summary = portfolio_svc.get_summary()

    # Active signals keyed by ticker
    signals = _get_active_signals(strategy_svc)

    # Heatmap enrichment
    heatmap = portfolio_svc.get_heatmap_data(positions, signals)

    # Risk alert
    risk_metrics = portfolio_svc.get_risk_metrics()
    risk_alert = portfolio_svc.get_risk_alert(risk_metrics)

    # DCA plans + summary
    dca_plans = execution_svc.get_dca_plans()
    dca_summary = execution_svc.get_dca_summary(dca_plans)

    return templates.TemplateResponse(
        "portfolio.html",
        {
            "request": request,
            "active_page": "portfolio",
            "summary": summary,
            "heatmap": heatmap,
            "signals": signals,
            "risk_alert": risk_alert,
            "dca_plans": dca_plans,
            "dca_summary": dca_summary,
        },
    )


@router.get("/portfolio/drawer", response_class=HTMLResponse)
async def portfolio_drawer(
    request: Request,
    ticker: str = Query(..., description="Ticker code"),
) -> HTMLResponse:
    """HTMX partial -- position drawer for a single ticker."""
    svcs = _get_services()
    portfolio_svc = svcs["portfolio"]
    strategy_svc = svcs["strategy"]

    # Find position data
    positions = portfolio_svc.get_positions()
    pos = next((p for p in positions if p["ticker"] == ticker), None)

    if pos is None:
        return HTMLResponse(
            '<div class="text-muted" style="padding:1rem;">'
            "Position not found.</div>"
        )

    # Signal for this ticker
    signals = _get_active_signals(strategy_svc)
    signal = signals.get(ticker)
    signal_label = signal["label"] if signal else "Neutral"

    # Plain English summary
    weight_pct = pos["weight"] * 100
    mv = pos["market_value"]
    name = pos["name"]
    summary_text = (
        f"You hold GBP{mv:,.0f} worth of {name}, "
        f"making it {'your largest position' if weight_pct > 15 else 'a position'} "
        f"at {weight_pct:.0f}% of your portfolio. "
    )
    ret = pos["daily_return"]
    if abs(ret) < 0.001:
        summary_text += "It has barely moved today."
    elif ret > 0:
        summary_text += f"It is up {ret * 100:.1f}% today."
    else:
        summary_text += f"It is down {abs(ret) * 100:.1f}% today."

    # News context (only if relevant -- ticker moved significantly)
    news_context = None
    if abs(ret) > 0.02:
        try:
            news_svc = svcs["news"]
            articles = news_svc.get_cached_news()
            for article in articles:
                if ticker.split(".")[0].lower() in article.get(
                    "title", ""
                ).lower():
                    news_context = article.get("title", "")
                    break
        except Exception:
            pass

    return templates.TemplateResponse(
        "partials/portfolio_drawer.html",
        {
            "request": request,
            "pos": pos,
            "signal_label": signal_label,
            "summary_text": summary_text,
            "news_context": news_context,
        },
    )


@router.post("/portfolio/ai-review", response_class=HTMLResponse)
async def portfolio_ai_review(request: Request) -> HTMLResponse:
    """HTMX endpoint -- AI portfolio review cards."""
    svcs = _get_services()
    ai_svc = svcs["ai"]
    portfolio_svc = svcs["portfolio"]
    strategy_svc = svcs["strategy"]
    execution_svc = svcs["execution"]

    positions = portfolio_svc.get_positions()
    signals = _get_active_signals(strategy_svc)
    dca_plans = execution_svc.get_dca_plans()

    insights = ai_svc.review_portfolio(positions, signals, dca_plans)

    # Map icon names to SVG
    for card in insights:
        card["icon_svg"] = _ICON_SVGS.get(card.get("icon", "info"), "")

    return templates.TemplateResponse(
        "partials/ai_portfolio_cards.html",
        {"request": request, "insights": insights},
    )
