"""Analyse page route -- focused research workbench.

Merged chart + strategy page. User selects a ticker, sees an SVG chart
with toggleable overlays on the left, plain-English signal summary with
AI deep-dive on the right. No sliders, no jargon.

Routes call src/services/ only -- never import from src/data or src/models.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.utils.config import load_config
from src.utils.logging import get_logger
from web.routes.ui import check_auth, templates
from web.routes.chart import _to_svg_points, _to_svg_bars

logger = get_logger(__name__)

# Register SVG helper filters on the ui templates instance so that
# included sub-panel partials (chart_rsi, chart_macd, chart_ml) can
# call to_svg_points / to_svg_bars as Jinja2 globals.
if "to_svg_points" not in templates.env.globals:
    templates.env.globals["to_svg_points"] = _to_svg_points
    templates.env.globals["to_svg_bars"] = _to_svg_bars
    templates.env.filters["to_svg_points"] = _to_svg_points
    templates.env.filters["to_svg_bars"] = _to_svg_bars

router = APIRouter(prefix="/ui", dependencies=[Depends(check_auth)])


# -- Period mapping --------------------------------------------------------

_PERIOD_DAYS: dict[str, int] = {
    "1D": 1,
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}


# -- Service helpers -------------------------------------------------------

_services: dict[str, Any] | None = None


def _get_services() -> dict[str, Any]:
    """Lazily initialise and cache service instances."""
    global _services
    if _services is not None:
        return _services

    from src.services.ai_service import AIService
    from src.services.chart_service import ChartService
    from src.services.data_service import DataService
    from src.services.market_data_service import MarketDataService
    from src.services.monte_carlo_service import MonteCarloService
    from src.services.strategy_service import StrategyService

    config = load_config()
    data_svc = DataService(config)
    market_data_svc = MarketDataService(config)
    chart_svc = ChartService(config)
    strategy_svc = StrategyService(data_svc, config)
    ai_svc = AIService(config)
    mc_svc = MonteCarloService(config, data_svc, market_data_svc)

    _services = {
        "config": config,
        "data": data_svc,
        "chart": chart_svc,
        "strategy": strategy_svc,
        "ai": ai_svc,
        "monte_carlo": mc_svc,
    }
    return _services


# -- Helpers ---------------------------------------------------------------


def _default_ticker() -> str:
    """Return the first ticker from the watchlist config."""
    cfg = load_config()
    groups = cfg.get("ui", {}).get("watchlist", {}).get("groups", [])
    for g in groups:
        tickers = g.get("tickers", [])
        if tickers:
            return tickers[0]
    return "CNDX.L"


def _watchlist_groups() -> list[dict[str, Any]]:
    """Return watchlist groups with tickers and display names."""
    svcs = _get_services()
    cfg = svcs["config"]
    groups = cfg.get("ui", {}).get("watchlist", {}).get("groups", [])
    result = []
    for g in groups:
        tickers = g.get("tickers", [])
        entries = [
            {"ticker": t, "name": _ticker_display_name(t)}
            for t in tickers
        ]
        result.append({"name": g.get("name", ""), "entries": entries})
    return result


def _ticker_display_name(ticker: str) -> str:
    """Return a human-friendly name for a ticker."""
    cfg = load_config()
    # Check portfolio holdings (supports both dict and list formats)
    for account in ("chip_isa", "revolut_gia", "crypto"):
        account_cfg = cfg.get("portfolio", {}).get(account, [])
        if isinstance(account_cfg, dict):
            holdings = account_cfg.get("holdings", []) or []
        else:
            holdings = account_cfg or []
        for h in holdings:
            if h.get("ticker") == ticker:
                return h.get("name", ticker)
    return ticker


_LOGO_COLOURS = [
    "#6366f1", "#ec4899", "#14b8a6", "#f97316", "#8b5cf6",
    "#06b6d4", "#84cc16", "#ef4444", "#eab308", "#22c55e",
]


def _logo_colour(ticker: str) -> str:
    """Deterministic colour for a ticker logo."""
    return _LOGO_COLOURS[hash(ticker) % len(_LOGO_COLOURS)]


def _build_y_labels(
    close_values: list[float], height: int, n_labels: int = 3,
) -> list[dict[str, Any]]:
    """Generate Y-axis price labels with percentage positions."""
    if not close_values or len(close_values) < 2:
        return []

    v_min = min(close_values)
    v_max = max(close_values)
    v_range = v_max - v_min
    if v_range == 0:
        return []

    labels = []
    for i in range(n_labels):
        frac = i / (n_labels - 1)  # 0.0 = top, 1.0 = bottom
        price = v_max - frac * v_range
        pct = frac * 100
        # Format price nicely
        if price >= 1000:
            text = f"{price:,.0f}"
        elif price >= 10:
            text = f"{price:,.2f}"
        else:
            text = f"{price:,.4f}"
        labels.append({"text": text, "pct": pct})
    return labels


def _build_chart_context(
    ticker: str, period: str, overlays: list[str],
    log_scale: bool = False,
) -> dict[str, Any]:
    """Build all SVG chart data for the analyse page."""
    svcs = _get_services()
    chart_svc = svcs["chart"]

    # Get price data
    price_data = chart_svc.get_price_history(ticker, period=_map_period(period))
    close_values = price_data.get("close", [])
    dates = price_data.get("dates", [])

    if not close_values or len(close_values) < 2:
        return {
            "svg": {
                "path_d": "", "area_d": "", "colour": "var(--t3)",
                "width": 800, "height": 400, "gradient_id": "grad-empty",
                "y_labels": [],
            },
            "bollinger": None,
            "trade_dots": [],
            "prev_close_y": None,
            "overlays": overlays,
            "learn_strips": [],
            "log_scale": log_scale,
            "rsi_data": None,
            "macd_data": None,
            "ml_data": None,
        }

    # Apply log scale if requested
    if log_scale:
        close_for_svg = [math.log(max(v, 0.01)) for v in close_values]
    else:
        close_for_svg = close_values

    # Build a pandas Series for SVG generation
    idx = pd.to_datetime(dates)
    prices = pd.Series(close_for_svg, index=idx, dtype=float)

    svg = chart_svc.generate_price_svg(prices, width=800, height=400)

    # Add Y-axis labels (use original prices for labels, not log)
    svg["y_labels"] = _build_y_labels(close_values, height=400)

    # Bollinger overlay
    bollinger = None
    if "bollinger" in overlays:
        bollinger = chart_svc.generate_bollinger_svg(prices, width=800, height=400)

    # Trade dots
    trade_entries = price_data.get("trade_entries", [])
    trade_dots = chart_svc.generate_trade_dots(trade_entries, prices, width=800, height=400)

    # Previous close line
    prev_close_y = None
    if len(close_for_svg) >= 2:
        prev = close_for_svg[-2]
        v_min = min(close_for_svg)
        v_max = max(close_for_svg)
        prev_close_y = chart_svc.generate_prev_close_line(prev, (v_min, v_max), height=400)

    # RSI overlay
    rsi_data = None
    if "rsi" in overlays:
        rsi_data = chart_svc.get_rsi_series(ticker, period=_map_period(period))

    # MACD overlay
    macd_data = None
    if "macd" in overlays:
        macd_data = chart_svc.get_macd_series(ticker, period=_map_period(period))

    # ML signal overlay
    ml_data = None
    if "ml" in overlays:
        ml_data = chart_svc.get_ml_confidence_series(ticker, period=_map_period(period))

    # Monte Carlo overlay
    mc_data = None
    if "monte_carlo" in overlays:
        mc_svc = svcs["monte_carlo"]
        mc_data = mc_svc.run_portfolio_projection(ticker, period_years=1)

    # Learn strips -- plain English explanation of active overlays
    learn_strips = _build_learn_strips(overlays, rsi_data, macd_data, ml_data, bollinger, ticker)

    return {
        "svg": svg,
        "bollinger": bollinger,
        "trade_dots": trade_dots,
        "prev_close_y": prev_close_y,
        "overlays": overlays,
        "learn_strips": learn_strips,
        "log_scale": log_scale,
        "rsi_data": rsi_data,
        "macd_data": macd_data,
        "ml_data": ml_data,
        "mc_data": mc_data,
    }


def _chart_layout(
    n_rows: int,
    row_heights: list[float] | None = None,
    log_scale: bool = False,
) -> dict[str, Any]:
    """Build the Plotly layout dict for the analyse chart."""
    if row_heights is None:
        row_heights = [1.0]

    layout: dict[str, Any] = {
        "paper_bgcolor": "transparent",
        "plot_bgcolor": "transparent",
        "font": {"color": "#f9fafb", "size": 11},
        "margin": {"l": 8, "r": 60, "t": 8, "b": 40},
        "showlegend": False,
        "hovermode": "x unified",
        "dragmode": "zoom",
        "shapes": [],
        "xaxis": {
            "gridcolor": "#2d2d2d",
            "linecolor": "#2d2d2d",
            "rangeslider": {"visible": False},
            "type": "date",
        },
        "yaxis": {
            "gridcolor": "#2d2d2d",
            "linecolor": "#2d2d2d",
            "side": "right",
            "type": "log" if log_scale else "linear",
        },
    }

    if n_rows == 1:
        layout["xaxis"]["domain"] = [0, 1]
        layout["yaxis"]["domain"] = [0, 1]
    elif n_rows == 2:
        # Contiguous domains -- no gap between panels.
        layout["yaxis"]["domain"] = [0.25, 1.0]
        layout["yaxis2"] = {
            "gridcolor": "#2d2d2d", "linecolor": "#2d2d2d",
            "side": "right", "domain": [0, 0.23],
            "anchor": "x",
        }
        # Separator line between panels.
        layout["shapes"].append({
            "type": "line", "xref": "paper", "yref": "paper",
            "x0": 0, "x1": 1, "y0": 0.24, "y1": 0.24,
            "line": {"color": "#2d2d2d", "width": 1},
        })
    else:
        # Contiguous domains -- no gap between panels.
        layout["yaxis"]["domain"] = [0.42, 1.0]
        layout["yaxis2"] = {
            "gridcolor": "#2d2d2d", "linecolor": "#2d2d2d",
            "side": "right", "domain": [0.21, 0.40],
            "anchor": "x",
        }
        layout["yaxis3"] = {
            "gridcolor": "#2d2d2d", "linecolor": "#2d2d2d",
            "side": "right", "domain": [0, 0.19],
            "anchor": "x",
        }
        # Separator lines between panels.
        for y in [0.41, 0.20]:
            layout["shapes"].append({
                "type": "line", "xref": "paper", "yref": "paper",
                "x0": 0, "x1": 1, "y0": y, "y1": y,
                "line": {"color": "#2d2d2d", "width": 1},
            })

    return layout


def _build_chart_json(
    ticker: str,
    period: str,
    overlays: list[str],
    log_scale: bool = False,
) -> dict[str, Any]:
    """Build a Plotly JSON spec for the interactive analyse chart.

    Returns a dict with ``data`` (list of traces) and ``layout``
    ready for ``Plotly.react()`` on the client.
    """
    svcs = _get_services()
    chart_svc = svcs["chart"]

    price_data = chart_svc.get_price_history(ticker, period=_map_period(period))
    dates = price_data.get("dates", [])
    opens = price_data.get("open", [])
    highs = price_data.get("high", [])
    lows = price_data.get("low", [])
    closes = price_data.get("close", [])

    if not dates or len(dates) < 2:
        return {"data": [], "layout": _chart_layout(1)}

    has_rsi = "rsi" in overlays
    has_macd = "macd" in overlays
    has_ml = "ml" in overlays

    n_sub = 1 + int(has_rsi or has_ml) + int(has_macd)
    if n_sub == 1:
        row_heights = [1.0]
    elif n_sub == 2:
        row_heights = [0.75, 0.25]
    else:
        row_heights = [0.60, 0.20, 0.20]

    traces: list[dict[str, Any]] = []

    # Row 1: Candlestick
    traces.append({
        "type": "candlestick",
        "x": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "name": ticker,
        "increasing": {"line": {"color": "#23c55e"}},
        "decreasing": {"line": {"color": "#ef4444"}},
        "xaxis": "x",
        "yaxis": "y",
    })

    # Bollinger bands
    if "bollinger" in overlays:
        bb_upper = price_data.get("bb_upper", [])
        bb_middle = price_data.get("bb_middle", [])
        bb_lower = price_data.get("bb_lower", [])
        if bb_upper:
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": dates, "y": bb_upper, "name": "BB Upper",
                "line": {"color": "#f59e0b", "width": 1, "dash": "dot"},
                "opacity": 0.6, "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": dates, "y": bb_lower, "name": "BB Lower",
                "line": {"color": "#f59e0b", "width": 1, "dash": "dot"},
                "opacity": 0.6, "fill": "tonexty",
                "fillcolor": "rgba(245,158,11,0.06)",
                "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": dates, "y": bb_middle, "name": "BB Mid",
                "line": {"color": "#f59e0b", "width": 1, "dash": "dash"},
                "opacity": 0.35, "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })

    # Monte Carlo projection
    if "monte_carlo" in overlays:
        mc_svc = svcs["monte_carlo"]
        mc_data = mc_svc.run_portfolio_projection(ticker, period_years=1)
        if mc_data.get("dates_forward"):
            mc_dates = mc_data["dates_forward"]
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": mc_dates, "y": mc_data["p95"], "name": "P95",
                "line": {"color": "#7c3aed", "width": 1, "dash": "dash"},
                "opacity": 0.5, "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": mc_dates, "y": mc_data["p5"], "name": "P05",
                "line": {"color": "#7c3aed", "width": 0.5, "dash": "dash"},
                "opacity": 0.4, "fill": "tonexty",
                "fillcolor": "rgba(124,58,237,0.08)",
                "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": mc_dates, "y": mc_data["p50"], "name": "P50 (median)",
                "line": {"color": "#7c3aed", "width": 2},
                "opacity": 0.8, "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })

    # Row 2: RSI or ML
    sub_row = 2
    if has_rsi:
        rsi_data = chart_svc.get_rsi_series(ticker, period=_map_period(period))
        if rsi_data.get("rsi"):
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": rsi_data["dates"],
                "y": list(rsi_data["rsi"]),
                "name": f"RSI ({rsi_data['window']})",
                "line": {"color": "#06b6d4", "width": 1.5},
                "xaxis": "x", "yaxis": f"y{sub_row}",
            })
    elif has_ml:
        ml_data = chart_svc.get_ml_confidence_series(
            ticker, period=_map_period(period),
        )
        if ml_data.get("confidence"):
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": ml_data["dates"],
                "y": ml_data["confidence"],
                "name": "ML Confidence",
                "line": {"color": "#23c55e", "width": 1.5},
                "xaxis": "x", "yaxis": f"y{sub_row}",
            })

    # Row 3: MACD
    if has_macd:
        macd_row = sub_row + (1 if (has_rsi or has_ml) else 0)
        if macd_row == sub_row:
            macd_row = 2
        macd_data = chart_svc.get_macd_series(
            ticker, period=_map_period(period),
        )
        if macd_data.get("macd_histogram"):
            hist_colors = [
                "#23c55e" if (v or 0) >= 0 else "#ef4444"
                for v in macd_data["macd_histogram"]
            ]
            traces.append({
                "type": "bar",
                "x": macd_data["dates"],
                "y": macd_data["macd_histogram"],
                "name": "Histogram",
                "marker": {"color": hist_colors, "opacity": 0.5},
                "xaxis": "x", "yaxis": f"y{macd_row}",
                "showlegend": False,
            })
        if macd_data.get("macd_line"):
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": macd_data["dates"],
                "y": macd_data["macd_line"],
                "name": "MACD",
                "line": {"color": "#06b6d4", "width": 1.2},
                "xaxis": "x", "yaxis": f"y{macd_row}",
            })
        if macd_data.get("macd_signal"):
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": macd_data["dates"],
                "y": macd_data["macd_signal"],
                "name": "Signal",
                "line": {"color": "#f59e0b", "width": 1, "dash": "dash"},
                "xaxis": "x", "yaxis": f"y{macd_row}",
            })

    # Layout
    layout = _chart_layout(n_sub, row_heights, log_scale)

    # Prev close horizontal line
    if len(closes) >= 2:
        layout.setdefault("shapes", []).append({
            "type": "line", "xref": "paper", "yref": "y",
            "x0": 0, "x1": 1,
            "y0": closes[-2], "y1": closes[-2],
            "line": {"color": "#f59e0b", "width": 1, "dash": "dash"},
            "opacity": 0.5,
        })

    # RSI threshold lines
    if has_rsi and n_sub >= 2:
        for level, color in [(70, "#ef4444"), (30, "#23c55e"), (50, "#666")]:
            layout["shapes"].append({
                "type": "line", "xref": "paper", "yref": "y2",
                "x0": 0, "x1": 1, "y0": level, "y1": level,
                "line": {"color": color, "width": 0.5, "dash": "dot"},
                "opacity": 0.4,
            })

    # MACD zero line
    if has_macd:
        macd_yref = f"y{n_sub}"
        layout["shapes"].append({
            "type": "line", "xref": "paper", "yref": macd_yref,
            "x0": 0, "x1": 1, "y0": 0, "y1": 0,
            "line": {"color": "#666", "width": 0.5, "dash": "dot"},
            "opacity": 0.4,
        })

    return {"data": traces, "layout": layout}


def _build_learn_strips(
    overlays: list[str],
    rsi_data: dict | None,
    macd_data: dict | None,
    ml_data: dict | None,
    bollinger: dict | None,
    ticker: str = "",
) -> list[dict[str, str]]:
    """Build plain English learn strip content for active overlays."""
    strips: list[dict[str, str]] = []

    if "bollinger" in overlays:
        strips.append({
            "name": "Bollinger Bands",
            "dot_colour": "var(--amber)",
            "explanation": (
                "The amber lines show the normal price range. "
                "When the price touches the top line, it might be getting expensive. "
                "When it touches the bottom, it might be a bargain."
                + (f" {ticker} is currently in the upper half -- healthy, with room to move."
                   if ticker else "")
            ),
        })

    if "rsi" in overlays:
        desc = "In the middle ground -- neither buyers nor sellers are in total control."
        if rsi_data:
            rsi_values = rsi_data.get("rsi", [])
            current_rsi = None
            for v in reversed(rsi_values):
                if v is not None:
                    current_rsi = v
                    break
            if current_rsi is not None:
                if current_rsi > 70:
                    desc = "Overbought -- buyers have been very active and the price may cool off."
                elif current_rsi < 30:
                    desc = "Oversold -- sellers have dominated and there could be a bounce coming."
        strips.append({
            "name": "Relative Strength",
            "dot_colour": "var(--cyan)",
            "explanation": desc,
        })

    if "macd" in overlays:
        strips.append({
            "name": "Trend Detector (MACD)",
            "dot_colour": "var(--purple)",
            "explanation": (
                "This shows whether the short-term trend is stronger or "
                "weaker than the longer-term trend. Green bars mean "
                "upward momentum, red bars mean downward."
            ),
        })

    if "ml" in overlays:
        conf_str = "50%"
        if ml_data:
            conf = ml_data.get("confidence", [])
            if conf:
                conf_str = f"{conf[-1]:.0%}"
        strips.append({
            "name": "AI Model Signal",
            "dot_colour": "var(--green)",
            "explanation": (
                f"Our model's confidence in the current direction: {conf_str}. "
                "Higher means more confident the trend will continue."
            ),
        })

    if "monte_carlo" in overlays:
        strips.append({
            "name": "Monte Carlo Projection",
            "dot_colour": "var(--purple)",
            "explanation": (
                "Thousands of possible future paths based on past behaviour. "
                "The shaded area shows where the price is most likely to end up."
            ),
        })

    return strips


def _map_period(period: str) -> str:
    """Map short period labels to ChartService period labels."""
    mapping = {"1D": "1M", "1W": "1M", "1M": "1M", "3M": "3M", "6M": "1Y", "1Y": "1Y"}
    return mapping.get(period, "1M")


def _build_signal_context(ticker: str) -> dict[str, Any]:
    """Build signal and feature data for the signal zone."""
    svcs = _get_services()
    strategy_svc = svcs["strategy"]

    signal = strategy_svc.get_signal(ticker)
    features = signal.get("features", {})
    direction = signal.get("direction", "Neutral")
    confidence = signal.get("confidence", 0.5)

    # Build plain English summary
    name = _ticker_display_name(ticker)
    if direction == "Looking Good":
        summary = (
            f'The model thinks <span class="text-cyan font-mono">{ticker}</span> is likely to '
            f'<span class="text-green">go up</span> over the next rebalancing period. '
            f"The price recently bounced from a low point and "
            f'<span class="text-amber">momentum is building</span>. '
            f"Volume is slightly below average, so the model is watching for "
            f"stronger buying before raising its confidence further."
        )
    elif direction == "Be Careful":
        summary = (
            f'The model is flagging some <span class="text-red">caution</span> '
            f'for <span class="text-cyan font-mono">{ticker}</span>. '
            f"Recent patterns for {name} suggest some weakness. "
            "This does not mean you should panic, but it is worth keeping a close eye on things."
        )
    elif direction == "Steady":
        summary = (
            f'<span class="text-cyan font-mono">{ticker}</span> is showing a '
            f'<span class="text-amber">balanced</span> picture right now. '
            f"{name} is neither strongly positive nor negative. "
            "The model sees a mixed bag of signals."
        )
    else:
        summary = (
            f'The signals for <span class="text-cyan font-mono">{ticker}</span> are mixed. '
            f"{name} does not have a strong lean in either direction right now. "
            "This is a wait-and-see situation."
        )

    # Build feature drivers with strength labels
    drivers: list[dict[str, Any]] = []
    for name_key, feat in features.items():
        drivers.append({
            "name": name_key,
            "strength": feat.get("strength", "Moderate"),
            "explanation": feat.get("explanation", ""),
            "bar_width": _strength_to_width(feat.get("strength", "Moderate")),
        })

    return {
        "signal": signal,
        "direction": direction,
        "confidence": confidence,
        "summary": summary,
        "drivers": drivers,
    }


def _strength_to_width(strength: str) -> int:
    """Convert a strength label to a percentage bar width."""
    return {"Strong": 90, "Moderate": 60, "Mild": 35, "Weak": 15}.get(strength, 50)


def _build_glance(ticker: str, close_values: list[float]) -> dict[str, str]:
    """Build the At a Glance strip data from fundamentals + price data."""
    svcs = _get_services()

    # Try to get MarketDataService if available
    mds = svcs.get("market_data")
    if mds is None:
        from src.services.market_data_service import MarketDataService
        mds = MarketDataService(svcs["config"])
        svcs["market_data"] = mds

    fundamentals = mds.get_fundamentals(ticker)

    # Day range and 52-week range from close values
    day_range = "--"
    year_range = "--"
    if close_values and len(close_values) >= 2:
        day_low = min(close_values[-1:])
        day_high = max(close_values[-1:])
        if len(close_values) >= 2:
            day_low = min(close_values[-2], close_values[-1])
            day_high = max(close_values[-2], close_values[-1])
        day_range = f"{day_low:,.2f} - {day_high:,.2f}"
        year_low = min(close_values)
        year_high = max(close_values)
        year_range = f"{year_low:,.2f} - {year_high:,.2f}"

    return {
        "market_cap": fundamentals.get("market_cap", "--"),
        "dividend_yield": fundamentals.get("div_yield", "--"),
        "day_range": day_range,
        "year_range": year_range,
        "volume": fundamentals.get("volume", "--"),
        "pe_ratio": fundamentals.get("pe_ratio", "--"),
    }


def _get_change_pct(close_values: list[float]) -> float:
    """Calculate daily change percentage from close values."""
    if len(close_values) < 2:
        return 0.0
    prev = close_values[-2]
    current = close_values[-1]
    if prev == 0:
        return 0.0
    return (current - prev) / prev


# -- Routes ----------------------------------------------------------------


@router.get("/analyse", response_class=HTMLResponse)
async def analyse_page(
    request: Request,
    ticker: str = Query(default=None),
    period: str = Query(default="1M"),
) -> HTMLResponse:
    """Main analyse page -- chart + signal workbench."""
    if ticker is None:
        ticker = _default_ticker()

    if period not in ("1D", "1W", "1M", "3M", "6M", "1Y"):
        period = "1M"
    overlays: list[str] = []

    chart_json = _build_chart_json(ticker, period, overlays)
    learn_strips = _build_learn_strips(overlays, None, None, None, None, ticker)
    signal_ctx = _build_signal_context(ticker)

    # Get price and change_pct from price data
    svcs = _get_services()
    price_data = svcs["chart"].get_price_history(ticker, period=_map_period(period))
    close_values = price_data.get("close", [])
    current_price = close_values[-1] if close_values else 0.0
    change_pct = _get_change_pct(close_values)

    glance = _build_glance(ticker, close_values)

    return templates.TemplateResponse(
        "analyse.html",
        {
            "request": request,
            "active_page": "analyse",
            "ticker": ticker,
            "company_name": _ticker_display_name(ticker),
            "logo_colour": _logo_colour(ticker),
            "period": period,
            "price": current_price,
            "change_pct": change_pct,
            "log_scale": False,
            "glance": glance,
            "watchlist_groups": _watchlist_groups(),
            "chart_json": chart_json,
            "overlays": overlays,
            "learn_strips": learn_strips,
            **signal_ctx,
        },
    )


@router.get("/analyse/chart", response_class=HTMLResponse)
async def analyse_chart_partial(
    request: Request,
    ticker: str = Query(default="CNDX.L"),
    period: str = Query(default="1M"),
    overlays: str = Query(default=""),
    log_scale: int = Query(default=0),
) -> HTMLResponse:
    """HTMX partial -- returns chart fragment only."""
    overlay_list = [o.strip() for o in overlays.split(",") if o.strip()]

    chart_json = _build_chart_json(
        ticker, period, overlay_list, log_scale=bool(log_scale),
    )
    learn_strips = _build_learn_strips(
        overlay_list, None, None, None, None, ticker,
    )

    return templates.TemplateResponse(
        "partials/analyse_chart.html",
        {
            "request": request,
            "ticker": ticker,
            "period": period,
            "overlays": overlay_list,
            "log_scale": bool(log_scale),
            "chart_json": chart_json,
            "learn_strips": learn_strips,
        },
    )


@router.get("/analyse/signal", response_class=HTMLResponse)
async def analyse_signal_partial(
    request: Request,
    ticker: str = Query(default="CNDX.L"),
) -> HTMLResponse:
    """HTMX partial -- returns signal zone fragment."""
    signal_ctx = _build_signal_context(ticker)

    return templates.TemplateResponse(
        "partials/analyse_signal.html",
        {
            "request": request,
            "ticker": ticker,
            "company_name": _ticker_display_name(ticker),
            **signal_ctx,
        },
    )


@router.post("/analyse/ai-dive", response_class=HTMLResponse)
async def analyse_ai_dive(
    request: Request,
    ticker: str = Query(default="CNDX.L"),
) -> HTMLResponse:
    """HTMX partial -- calls AIService and returns analysis."""
    svcs = _get_services()
    ai_svc = svcs["ai"]
    strategy_svc = svcs["strategy"]

    signal = strategy_svc.get_signal(ticker)
    features = signal.get("features", {})

    analysis = ai_svc.analyse_ticker(ticker, signal, features)

    return templates.TemplateResponse(
        "partials/ai_result.html",
        {
            "request": request,
            "ticker": ticker,
            "analysis": analysis,
        },
    )
