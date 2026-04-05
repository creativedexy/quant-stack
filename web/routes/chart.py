"""FastAPI router for the chart page (/ui/chart).

Displays historical price charts with toggleable overlays:
Bollinger Bands, RSI, MACD, ML confidence, and forensic analysis.

SVG charts are server-rendered via Jinja2.  HTMX loads overlay
panels as HTML fragments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
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

# -- Jinja2 filters ------------------------------------------------


def _to_svg_points(
    dates_or_indices: list,
    values: list,
    width: int = 900,
    height: int = 220,
    pad_x: int = 40,
    pad_y: int = 10,
) -> str:
    """Convert parallel lists into an SVG polyline points string.

    Skips None values.  Maps data to viewBox coordinates.
    """
    pairs: list[tuple[int, float]] = []
    for i, v in enumerate(values):
        if v is not None:
            pairs.append((i, float(v)))

    if len(pairs) < 2:
        return ""

    n = len(values)
    vals = [p[1] for p in pairs]
    min_v = min(vals)
    max_v = max(vals)
    v_range = max_v - min_v if max_v != min_v else 1.0

    pts: list[str] = []
    for idx, val in pairs:
        x = pad_x + (idx / max(n - 1, 1)) * (width - 2 * pad_x)
        y = pad_y + (1 - (val - min_v) / v_range) * (height - 2 * pad_y)
        pts.append(f"{x:.1f},{y:.1f}")

    return " ".join(pts)


def _to_svg_bars(
    values: list,
    colours: list,
    width: int = 900,
    height: int = 100,
    pad_x: int = 40,
) -> list[dict[str, Any]]:
    """Convert histogram values into SVG rect definitions.

    Returns a list of dicts with keys: x, y, w, h, colour.
    """
    if not values:
        return []

    n = len(values)
    bar_w = max(1.0, (width - 2 * pad_x) / n * 0.8)
    gap = (width - 2 * pad_x) / max(n, 1)
    mid_y = height / 2

    # Find max absolute value for scaling
    abs_vals = [abs(float(v)) for v in values if v is not None]
    max_abs = max(abs_vals) if abs_vals else 1.0

    bars: list[dict[str, Any]] = []
    for i, v in enumerate(values):
        if v is None:
            continue
        fv = float(v)
        colour = colours[i] if i < len(colours) else "green"
        scaled_h = abs(fv) / max_abs * (height / 2 - 5)
        x = pad_x + i * gap
        if fv >= 0:
            y = mid_y - scaled_h
        else:
            y = mid_y
        bars.append({
            "x": round(x, 1),
            "y": round(y, 1),
            "w": round(bar_w, 1),
            "h": round(max(scaled_h, 0.5), 1),
            "colour": colour,
        })

    return bars


templates.env.filters["to_svg_points"] = _to_svg_points
templates.env.filters["to_svg_bars"] = _to_svg_bars
templates.env.globals["to_svg_points"] = _to_svg_points
templates.env.globals["to_svg_bars"] = _to_svg_bars


# -- Service bootstrap ---------------------------------------------

_services: dict[str, Any] | None = None


def _get_services() -> dict[str, Any]:
    """Initialise and cache service instances."""
    global _services
    if _services is not None:
        return _services

    from src.services.chart_service import ChartService
    from src.services.data_service import DataService
    from src.services.market_data_service import MarketDataService
    from src.services.monte_carlo_service import MonteCarloService

    try:
        config = load_config()
    except FileNotFoundError:
        config = {}

    data_svc = DataService(config)
    market_svc = MarketDataService(config)

    _services = {
        "chart": ChartService(config),
        "monte_carlo": MonteCarloService(config, data_svc, market_svc),
        "config": config,
    }
    return _services


# -- Routes --------------------------------------------------------


@router.get("/chart", response_class=HTMLResponse)
async def chart_page(
    request: Request,
    ticker: str = Query("SHEL.L", description="Ticker symbol"),
    period: str = Query("1Y", description="Period: 1M, 3M, 1Y, 3Y, All"),
) -> HTMLResponse:
    """Render the full chart page."""
    svcs = _get_services()
    config = svcs["config"]
    chart_svc = svcs["chart"]

    # Available tickers from config
    ui_cfg = config.get("ui", {})
    watchlist_cfg = ui_cfg.get("watchlist", {})
    tickers = watchlist_cfg.get(
        "tickers",
        config.get("universe", {}).get("tickers", ["SHEL.L"]),
    )

    # Fetch main price data
    try:
        price_data = chart_svc.get_price_history(ticker, period)
    except Exception as exc:
        logger.error("Chart price fetch failed for %s: %s", ticker, exc)
        price_data = chart_svc._empty_price_history(ticker, period)

    return templates.TemplateResponse(
        "chart.html",
        {
            "request": request,
            "active_page": "chart",
            "ticker": ticker,
            "period": period,
            "tickers": tickers,
            "periods": ["1M", "3M", "1Y", "3Y", "All"],
            "price_data": price_data,
            "price_data_json": json.dumps(price_data, default=str),
        },
    )


@router.get("/chart/overlay/rsi", response_class=HTMLResponse)
async def chart_rsi_overlay(
    request: Request,
    ticker: str = Query("SHEL.L"),
    period: str = Query("1Y"),
    window: int = Query(14),
) -> HTMLResponse:
    """Return RSI panel HTML fragment."""
    svcs = _get_services()
    chart_svc = svcs["chart"]

    try:
        rsi_data = chart_svc.get_rsi_series(ticker, period, window)
    except Exception as exc:
        logger.error("RSI overlay failed for %s: %s", ticker, exc)
        rsi_data = {"dates": [], "rsi": [], "window": window}

    return templates.TemplateResponse(
        "partials/chart_rsi.html",
        {"request": request, "rsi_data": rsi_data},
    )


@router.get("/chart/overlay/macd", response_class=HTMLResponse)
async def chart_macd_overlay(
    request: Request,
    ticker: str = Query("SHEL.L"),
    period: str = Query("1Y"),
) -> HTMLResponse:
    """Return MACD panel HTML fragment."""
    svcs = _get_services()
    chart_svc = svcs["chart"]

    try:
        macd_data = chart_svc.get_macd_series(ticker, period)
    except Exception as exc:
        logger.error("MACD overlay failed for %s: %s", ticker, exc)
        macd_data = {
            "dates": [],
            "macd_line": [],
            "macd_signal": [],
            "macd_histogram": [],
            "histogram_colours": [],
        }

    return templates.TemplateResponse(
        "partials/chart_macd.html",
        {"request": request, "macd_data": macd_data},
    )


@router.get("/chart/overlay/ml", response_class=HTMLResponse)
async def chart_ml_overlay(
    request: Request,
    ticker: str = Query("SHEL.L"),
    period: str = Query("1Y"),
) -> HTMLResponse:
    """Return ML confidence panel HTML fragment."""
    svcs = _get_services()
    chart_svc = svcs["chart"]

    try:
        ml_data = chart_svc.get_ml_confidence_series(ticker, period)
    except Exception as exc:
        logger.error("ML overlay failed for %s: %s", ticker, exc)
        ml_data = {"dates": [], "confidence": [], "direction": []}

    return templates.TemplateResponse(
        "partials/chart_ml.html",
        {"request": request, "ml_data": ml_data},
    )


@router.get("/chart/overlay/forensic", response_class=HTMLResponse)
async def chart_forensic_overlay(
    request: Request,
    ticker: str = Query("SHEL.L"),
    entry_date: str = Query("2024-06-15"),
    entry_price: float = Query(100.0),
) -> HTMLResponse:
    """Return forensic analysis panel HTML fragment."""
    svcs = _get_services()
    chart_svc = svcs["chart"]

    try:
        forensic_data = chart_svc.get_forensic_analysis(
            ticker, entry_date, entry_price
        )
    except Exception as exc:
        logger.error("Forensic overlay failed for %s: %s", ticker, exc)
        forensic_data = chart_svc._empty_forensic(ticker, entry_date, entry_price)

    return templates.TemplateResponse(
        "partials/chart_forensic.html",
        {"request": request, "forensic_data": forensic_data},
    )


@router.get("/chart/monte-carlo", response_class=HTMLResponse)
async def chart_monte_carlo_overlay(
    request: Request,
    ticker: str = Query("SHEL.L"),
    period: int = Query(1, description="Projection horizon in years"),
    simulations: int = Query(1000),
) -> HTMLResponse:
    """Return Monte Carlo fan chart HTML fragment."""
    svcs = _get_services()
    mc_svc = svcs["monte_carlo"]

    try:
        mc_data = mc_svc.run_portfolio_projection(
            ticker, period_years=period, n_simulations=simulations,
        )
    except Exception as exc:
        logger.error("Monte Carlo overlay failed for %s: %s", ticker, exc)
        mc_data = {
            "dates_forward": [],
            "p5": [],
            "p50": [],
            "p95": [],
            "current_price": 0.0,
            "svg_points": {"p5": "", "p50": "", "p95": ""},
        }

    return templates.TemplateResponse(
        "partials/chart_monte_carlo.html",
        {"request": request, "mc_data": mc_data},
    )
