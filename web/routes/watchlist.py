"""FastAPI router for the watchlist page (/ui/watchlist).

Displays live stock cards with intraday charts, fundamentals,
and signal data.  HTMX refreshes individual cards every 30s.

Routes call MarketDataService from src/services/ only.
Never import from src/data, src/features, or src/models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.utils.config import load_config
from src.utils.logging import get_logger
from web.routes.ui import check_auth

logger = get_logger(__name__)

_WEB_DIR = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _WEB_DIR / "templates"

router = APIRouter(prefix="/ui", dependencies=[Depends(check_auth)])
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_css_path = _WEB_DIR / "static" / "css" / "theme.css"
templates.env.globals["css_v"] = (
    str(int(_css_path.stat().st_mtime)) if _css_path.exists() else "0"
)

# -- Service layer bootstrap -----------------------------------

_services: dict[str, Any] | None = None


def _get_services() -> dict[str, Any]:
    """Initialise and cache service instances."""
    global _services
    if _services is not None:
        return _services

    from src.services.market_data_service import MarketDataService
    from src.services.strategy_service import StrategyService
    from src.services.data_service import DataService
    from src.utils.config import load_config

    try:
        config = load_config()
    except FileNotFoundError:
        config = {}

    data_svc = DataService(config)
    strategy_svc = StrategyService(data_svc, config)
    market_data_svc = MarketDataService(config, strategy_svc)

    _services = {
        "market_data": market_data_svc,
        "config": config,
    }
    return _services


# -- Routes ----------------------------------------------------


@router.get("/watchlist", response_class=HTMLResponse)
async def watchlist_page(request: Request) -> HTMLResponse:
    """Render the full watchlist page.

    Cards are NOT fetched on initial load -- each card triggers an
    HTMX GET on ``load`` so the browser parallelises yfinance requests
    instead of blocking the whole page for 50+ tickers.
    """
    svcs = _get_services()
    config = svcs["config"]
    market_data = svcs["market_data"]

    ui_cfg = config.get("ui", {})
    watchlist_cfg = ui_cfg.get("watchlist", {})
    refresh_seconds = watchlist_cfg.get("refresh_seconds", 30)
    chart_cfg = ui_cfg.get("chart", {})

    # Build groups from config (no yfinance calls here)
    groups = _build_groups(watchlist_cfg)

    # FX & Commodities (only 3 tickers -- fast)
    try:
        fx_commodities = market_data.get_fx_commodities()
    except Exception as exc:
        logger.error("Failed to get FX/commodities: %s", exc)
        fx_commodities = []

    # Active signals
    try:
        active_signals = _get_active_signals(market_data)
    except Exception as exc:
        logger.error("Failed to get active signals: %s", exc)
        active_signals = []

    return templates.TemplateResponse(
        "watchlist.html",
        {
            "request": request,
            "active_page": "watchlist",
            "groups": groups,
            "fx_commodities": fx_commodities,
            "active_signals": active_signals,
            "refresh_seconds": refresh_seconds,
            "chart_cfg": chart_cfg,
            "chart_cfg_json": json.dumps(chart_cfg),
        },
    )


@router.get("/watchlist/card/{ticker}", response_class=HTMLResponse)
async def watchlist_card_partial(
    request: Request,
    ticker: str,
) -> HTMLResponse:
    """Return a single stock card HTML fragment for HTMX swap."""
    svcs = _get_services()
    config = svcs["config"]
    market_data = svcs["market_data"]

    ui_cfg = config.get("ui", {})
    chart_cfg = ui_cfg.get("chart", {})

    try:
        cards = market_data.get_watchlist_cards([ticker])
        card = cards[0] if cards else _empty_card(ticker)
    except Exception as exc:
        logger.error("Failed to build card for %s: %s", ticker, exc)
        card = _empty_card(ticker)

    return templates.TemplateResponse(
        "partials/stock_card.html",
        {
            "request": request,
            "card": card,
            "chart_cfg": chart_cfg,
            "chart_cfg_json": json.dumps(chart_cfg),
        },
    )


@router.post("/ask")
async def ask_watchlist(request: Request, query: str = Form("")) -> HTMLResponse:
    """Stub ask endpoint -- returns a plain HTML response fragment."""
    if not query.strip():
        return HTMLResponse("")
    # Production: forward to Claude API. For now return a loading indicator.
    return HTMLResponse(
        f'<span style="color:#f59e0b;font-family:monospace;font-size:11px">'
        f'Analysing: {query[:60]}...</span>'
    )


_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


@router.post("/watchlist/remove/{ticker}", response_class=HTMLResponse)
async def watchlist_remove(request: Request, ticker: str) -> HTMLResponse:
    """Remove a ticker from the watchlist and save settings.yaml."""
    _remove_ticker_from_settings(ticker)
    return HTMLResponse("", status_code=200)


@router.post("/watchlist/refresh/{ticker}", response_class=HTMLResponse)
async def watchlist_refresh(request: Request, ticker: str) -> HTMLResponse:
    """Refresh a single watchlist row and return the wl_row partial."""
    svcs = _get_services()
    market_data = svcs["market_data"]

    try:
        cards = market_data.get_watchlist_cards([ticker])
        card = cards[0] if cards else _empty_card(ticker)
    except Exception as exc:
        logger.error("Failed to refresh card for %s: %s", ticker, exc)
        card = _empty_card(ticker)

    row = card_to_wl_row(card)
    return templates.TemplateResponse(
        "partials/wl_row.html",
        {"request": request, "row": row},
    )


@router.post("/watchlist/add", response_class=HTMLResponse)
async def watchlist_add(request: Request, ticker: str = Form("")) -> HTMLResponse:
    """Add a ticker to the watchlist and return a wl_row partial."""
    ticker = ticker.strip().upper()
    if not ticker:
        return HTMLResponse("", status_code=400)

    # Check for duplicates
    all_tickers = _get_all_tickers_from_settings()
    if ticker in all_tickers:
        return HTMLResponse(
            '<span style="color:var(--red);font-size:11px;">Already in watchlist</span>',
            status_code=409,
        )

    # Append to settings
    _add_ticker_to_settings(ticker)

    # Build the row
    svcs = _get_services()
    market_data = svcs["market_data"]
    try:
        cards = market_data.get_watchlist_cards([ticker])
        card = cards[0] if cards else _empty_card(ticker)
    except Exception:
        card = _empty_card(ticker)

    row = card_to_wl_row(card)
    return templates.TemplateResponse(
        "partials/wl_row.html",
        {"request": request, "row": row},
    )


def card_to_wl_row(card: dict[str, Any]) -> dict[str, Any]:
    """Convert a watchlist card dict to a wl_row template dict."""
    ticker = card.get("ticker", "")
    current = card.get("current", 0.0)
    change_pct = card.get("change_pct", 0.0)
    direction = card.get("direction", "up")

    # Price formatting
    if ticker.upper().endswith(".L"):
        price_fmt = f"{current:.1f}p" if current <= 100 else f"{current:.0f}p"
    elif card.get("exchange") == "Crypto":
        price_fmt = f"${current:,.2f}"
    else:
        price_fmt = f"${current:.2f}"

    change_fmt = f"{change_pct:+.2f}%"

    # Sparkline: take last 8 prices from the card
    prices = card.get("prices", [])
    if len(prices) >= 8:
        step = len(prices) // 8
        spark_points = [prices[i * step] for i in range(8)]
    elif prices:
        spark_points = prices[:8]
    else:
        spark_points = []

    # Stable ID from ticker
    row_id = ticker.replace(".", "-").replace("$", "").replace("-", "_")

    return {
        "id": row_id,
        "ticker_raw": ticker,
        "ticker_display": ticker.replace(".L", ""),
        "exchange": card.get("exchange", "NYSE"),
        "company_name": card.get("company_name", ticker),
        "logo_colour": card.get("logo_colour", "#6366f1"),
        "price_fmt": price_fmt,
        "change_fmt": change_fmt,
        "direction": direction,
        "spark_points": spark_points,
    }


def _get_all_tickers_from_settings() -> list[str]:
    """Read all tickers across all watchlist groups from settings.yaml."""
    try:
        config = load_config()
    except FileNotFoundError:
        return []
    groups = config.get("ui", {}).get("watchlist", {}).get("groups", [])
    all_tickers: list[str] = []
    for g in groups:
        all_tickers.extend(g.get("tickers", []))
    return all_tickers


def _remove_ticker_from_settings(ticker: str) -> None:
    """Remove a ticker from settings.yaml and save to disk."""
    if not _SETTINGS_PATH.exists():
        return
    with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    groups = raw.get("ui", {}).get("watchlist", {}).get("groups", [])
    for g in groups:
        tickers_list = g.get("tickers", [])
        if ticker in tickers_list:
            tickers_list.remove(ticker)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _add_ticker_to_settings(ticker: str) -> None:
    """Append a ticker to the first watchlist group in settings.yaml."""
    if not _SETTINGS_PATH.exists():
        return
    with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    groups = raw.get("ui", {}).get("watchlist", {}).get("groups", [])
    if groups:
        groups[0].setdefault("tickers", []).append(ticker)
    else:
        raw.setdefault("ui", {}).setdefault("watchlist", {})["groups"] = [
            {"name": "Watchlist", "tickers": [ticker]}
        ]
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _build_groups(watchlist_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse ``ui.watchlist.groups`` from config into a template-friendly list.

    Falls back to a flat ``tickers`` key for backwards compatibility.
    """
    groups_raw = watchlist_cfg.get("groups", [])
    if groups_raw:
        return [
            {
                "name": g.get("name", "Watchlist"),
                "tickers": g.get("tickers", []),
            }
            for g in groups_raw
            if g.get("tickers")
        ]

    # Legacy flat list
    flat = watchlist_cfg.get("tickers", [])
    if flat:
        return [{"name": "Watchlist", "tickers": flat}]

    return []


def _get_active_signals(market_data: Any) -> list[dict[str, Any]]:
    """Extract active non-NEUTRAL signals from cards."""
    signals: list[dict[str, Any]] = []
    try:
        if market_data._strategy_service is None:
            return signals
        strategies = market_data._strategy_service.get_available_strategies()
        for name in strategies:
            df = market_data._strategy_service.get_signals(name, n_days=5)
            if df.empty:
                continue
            for _, row in df.iterrows():
                direction = row.get("direction", row.get("signal", "NEUTRAL"))
                if direction and direction.upper() != "NEUTRAL":
                    signals.append({
                        "ticker": row.get("ticker", ""),
                        "direction": direction,
                        "confidence": round(
                            float(row.get("confidence", row.get("score", 0.5))),
                            2,
                        ),
                        "strategy": name,
                    })
    except Exception as exc:
        logger.error("Failed to extract active signals: %s", exc)
    return signals


def _empty_card(ticker: str) -> dict[str, Any]:
    """Minimal card when data is unavailable."""
    if ticker.upper().endswith(".L"):
        exchange = "LSE"
    elif ticker.upper().endswith("-USD"):
        exchange = "Crypto"
    else:
        exchange = "NYSE"
    return {
        "ticker": ticker,
        "company_name": ticker,
        "exchange": exchange,
        "current": 0.0,
        "prev_close": 0.0,
        "change_val": 0.0,
        "change_pct": 0.0,
        "direction": "up",
        "prices": [],
        "times": [],
        "logo_colour": "#6366f1",
        "summary": "Data unavailable.",
        "volume": "N/A",
        "market_cap": "N/A",
        "pe_ratio": "N/A",
        "div_yield": "N/A",
        "profit_margin": "N/A",
        "operating_margin": "N/A",
        "is_positive_margin": False,
        "is_positive_op_margin": False,
        "signal_direction": "NEUTRAL",
        "signal_confidence": 0.50,
    }
