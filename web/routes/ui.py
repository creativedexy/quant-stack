"""FastAPI router for the production UI (/ui/ paths).

All pages are rendered server-side via Jinja2. HTMX handles
interactivity. No business logic in templates -- route handlers
do all computation and pass plain dicts to the template context.

Routes call service functions from src/services/ only.
Never import from src/data, src/features, or src/models.
"""

from __future__ import annotations

import json
import math
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# -- HTTP Basic Auth -------------------------------------------
# This is a single-user tool, not a multi-tenant SaaS product.
# HTTP Basic Auth over HTTPS is sufficient for protecting the
# dashboard from casual access.  Credentials are read from
# config/settings.yaml -> ui.auth.{username, password}.
# The JSON API routes (/api/) remain unprotected so that
# programmatic consumers (notebooks, scripts) are not affected.

_security = HTTPBasic(auto_error=False)
_cfg = load_config()
_AUTH_USER: str = _cfg.get("ui", {}).get("auth", {}).get("username", "admin")
_AUTH_PASS: str = _cfg.get("ui", {}).get("auth", {}).get("password", "") or "changeme"

_WEAK_PASSWORDS = {"changeme", "admin", "password", ""}
if _AUTH_PASS in _WEAK_PASSWORDS:
    logger.warning(
        "UI password is weak or unset. "
        "Set the UI_PASSWORD environment variable before deploying."
    )

_REALM_HEADER = {"WWW-Authenticate": 'Basic realm="Quant Stack"'}


def check_auth(
    credentials: HTTPBasicCredentials | None = Depends(_security),
) -> str:
    """Validate HTTP Basic credentials against config values.

    Uses ``secrets.compare_digest`` for timing-safe comparison so
    that an attacker cannot infer the password length from response
    latency.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers=_REALM_HEADER,
        )
    user_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        _AUTH_USER.encode("utf-8"),
    )
    pass_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        _AUTH_PASS.encode("utf-8"),
    )
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers=_REALM_HEADER,
        )
    return credentials.username


_WEB_DIR = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _WEB_DIR / "templates"

router = APIRouter(prefix="/ui", dependencies=[Depends(check_auth)])
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Cache-bust static CSS — mtime changes when the file is edited
_css_path = _WEB_DIR / "static" / "css" / "theme.css"
templates.env.globals["css_v"] = (
    str(int(_css_path.stat().st_mtime)) if _css_path.exists() else "0"
)

# -- Jinja2 filters -------------------------------------------


def _gbp(value: float) -> str:
    """Format a number as GBP currency: GBP1,234.50."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v < 0:
        return f"-GBP{abs(v):,.2f}"
    return f"GBP{v:,.2f}"


def _pct(value: float) -> str:
    """Format a decimal as a signed percentage with colour class span."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if v >= 0 else ""
    css = "text-green" if v >= 0 else "text-red"
    return f'<span class="{css}">{sign}{v * 100:.2f}%</span>'


def _pct_plain(value: float) -> str:
    """Format a decimal as a signed percentage string without HTML."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}%"


def _ago(dt: datetime | None) -> str:
    """Return a human-readable relative time string."""
    if dt is None:
        return ""
    try:
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = now - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "Just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        days = hours // 24
        return f"{days} day{'s' if days != 1 else ''} ago"
    except (TypeError, ValueError):
        return str(dt)


def _signal_label(confidence: float) -> str:
    """Convert a numeric confidence to a plain English label.

    >0.65 -> Looking Good, <0.35 -> Be Careful,
    0.45-0.55 -> Steady, else -> Neutral.
    """
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return "Neutral"
    if c > 0.65:
        return "Looking Good"
    if c < 0.35:
        return "Be Careful"
    if 0.45 <= c <= 0.55:
        return "Steady"
    return "Neutral"


templates.env.filters["gbp"] = _gbp
templates.env.filters["pct"] = _pct
templates.env.filters["pct_plain"] = _pct_plain
templates.env.filters["ago"] = _ago
templates.env.filters["signal_label"] = _signal_label


def _lse_market_context() -> dict[str, Any]:
    """Return LSE open/closed status for the current moment."""
    try:
        import zoneinfo

        london = zoneinfo.ZoneInfo("Europe/London")
    except Exception:
        return {"lse_open": False, "lse_time": ""}
    now = datetime.now(london)
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour_min = now.hour * 100 + now.minute
    is_open = weekday < 5 and 800 <= hour_min <= 1630
    return {
        "lse_open": is_open,
        "lse_time": now.strftime("%H:%M"),
    }


# Make LSE status available to every template via a context processor.
# Jinja2 globals are evaluated once at import time, but we need a callable
# that is resolved per-request.  We inject it as a global function.
templates.env.globals["lse_market"] = _lse_market_context


# -- Service layer bootstrap -----------------------------------
# Lazily initialised on first request to avoid import-time side effects.

_services: dict[str, Any] | None = None


def _get_services() -> dict[str, Any]:
    """Initialise and cache service instances."""
    global _services
    if _services is not None:
        return _services

    from src.services.chart_service import ChartService
    from src.services.data_service import DataService
    from src.services.dca_service import DCAService
    from src.services.execution_service import ExecutionService
    from src.services.market_data_service import MarketDataService
    from src.services.news_service import NewsService
    from src.services.portfolio_service import PortfolioService
    from src.services.strategy_service import StrategyService
    from src.utils.config import load_config

    try:
        config = load_config()
    except FileNotFoundError:
        config = {}

    data_svc = DataService(config)
    chart_svc = ChartService(config)
    market_svc = MarketDataService(config)
    portfolio_svc = PortfolioService(data_svc, config)
    strategy_svc = StrategyService(data_svc, config)
    execution_svc = ExecutionService(data_svc, portfolio_svc, config)
    dca_svc = DCAService(config, chart_svc, data_svc, market_svc)
    news_svc = NewsService(config)

    # ResearchLog is optional (fails gracefully if log path missing)
    research_log = None
    try:
        from src.notebooks.research_log import ResearchLog

        research_log = ResearchLog(config)
    except Exception:
        pass

    _services = {
        "config": config,
        "data": data_svc,
        "chart": chart_svc,
        "market": market_svc,
        "portfolio": portfolio_svc,
        "strategy": strategy_svc,
        "execution": execution_svc,
        "dca": dca_svc,
        "news": news_svc,
        "research_log": research_log,
    }
    return _services


# -- Helper: build overview metrics ----------------------------


def _build_overview_context() -> dict[str, Any]:
    """Compute all data needed by the overview page.

    Returns a dict with keys: metrics, positions, signals,
    heatmap_json.  On error any section falls back to safe defaults.
    """
    svcs = _get_services()
    portfolio: Any = svcs["portfolio"]
    strategy: Any = svcs["strategy"]
    data_svc: Any = svcs["data"]

    # -- Metrics (Level 1) -------------------------------------
    metrics = _build_metrics(portfolio, data_svc)

    # -- Positions (Level 2 heatmap) ---------------------------
    positions, heatmap_json = _build_positions(portfolio, data_svc)

    # -- Active signals (Level 3) ------------------------------
    signals = _build_signals(strategy)

    return {
        "metrics": metrics,
        "positions": positions,
        "signals": signals,
        "heatmap_json": heatmap_json,
    }


def _build_metrics(
    portfolio: Any, data_svc: Any,
) -> list[dict[str, str]]:
    """Build the five metric cards for the metrics bar."""
    try:
        risk = portfolio.get_risk_metrics()
        weights = portfolio.get_current_weights()
        prices = data_svc.get_prices()

        # Portfolio value: sum of latest prices * assumed 200 units each
        if not prices.empty and not weights.empty:
            latest = prices.iloc[-1]
            portfolio_value = float((latest * 200).sum())
        else:
            portfolio_value = 0.0

        # Daily P&L
        if not prices.empty and len(prices) >= 2:
            prev = prices.iloc[-2]
            curr = prices.iloc[-1]
            daily_pnl = float(((curr - prev) * 200).sum())
            daily_pnl_pct = daily_pnl / max(portfolio_value, 1)
        else:
            daily_pnl = 0.0
            daily_pnl_pct = 0.0

        sharpe = risk.get("sharpe_ratio", float("nan"))
        if np.isnan(sharpe):
            sharpe_str = "--"
        else:
            sharpe_str = f"{sharpe:.2f}"

        pnl_css = "badge-positive" if daily_pnl >= 0 else "badge-negative"

        # Signal count
        svcs = _get_services()
        strat_svc = svcs["strategy"]
        all_signals = _build_signals(strat_svc)
        signal_count = len(all_signals)

        return [
            {
                "label": "Portfolio Value",
                "value": _gbp(portfolio_value),
                "css_class": "",
            },
            {
                "label": "Daily P&L",
                "value": f"{_gbp(daily_pnl)} ({_pct(daily_pnl_pct)})",
                "css_class": pnl_css,
            },
            {
                "label": "Sharpe 30d",
                "value": sharpe_str,
                "css_class": "font-mono",
            },
            {
                "label": "Active Signals",
                "value": str(signal_count),
                "css_class": "font-mono",
            },
            {
                "label": "VIX",
                "value": "--",
                "css_class": "font-mono badge-neutral",
            },
        ]
    except Exception as exc:
        logger.error("Failed to build metrics: %s", exc)
        return [
            {"label": "Portfolio Value", "value": "--", "css_class": ""},
            {"label": "Daily P&L", "value": "--", "css_class": ""},
            {"label": "Sharpe 30d", "value": "--", "css_class": "font-mono"},
            {"label": "Active Signals", "value": "--", "css_class": "font-mono"},
            {"label": "VIX", "value": "--", "css_class": "font-mono badge-neutral"},
        ]


def _build_positions(
    portfolio: Any, data_svc: Any,
) -> tuple[list[dict[str, Any]], str]:
    """Build position list and Plotly treemap JSON for the heatmap."""
    try:
        weights = portfolio.get_current_weights()
        prices = data_svc.get_prices()

        if weights.empty or prices.empty:
            return [], json.dumps({"data": [], "layout": {}})

        common = weights.index.intersection(prices.columns)
        if len(common) == 0:
            return [], json.dumps({"data": [], "layout": {}})

        weights = weights[common]
        weights = weights / weights.sum()

        latest = prices[common].iloc[-1]
        if len(prices) >= 2:
            prev = prices[common].iloc[-2]
            daily_ret = (latest - prev) / prev
        else:
            daily_ret = pd.Series(0.0, index=common)

        positions = []
        labels = []
        values = []
        colors = []
        customdata = []

        for ticker in common:
            w = float(weights[ticker])
            ret = float(daily_ret[ticker])
            mv = float(latest[ticker]) * 200
            positions.append({
                "ticker": ticker,
                "weight": w,
                "daily_return": ret,
                "market_value": mv,
                "price": float(latest[ticker]),
            })
            labels.append(ticker)
            values.append(round(w * 100, 2))
            colors.append(round(ret * 100, 2))
            customdata.append([_gbp(mv), _pct(ret)])

        # Plotly treemap spec
        treemap = {
            "data": [
                {
                    "type": "treemap",
                    "labels": labels,
                    "parents": [""] * len(labels),
                    "values": values,
                    "marker": {
                        "colors": colors,
                        "colorscale": [
                            [0, "#ef4444"],
                            [0.5, "#2d2d2d"],
                            [1, "#22c55e"],
                        ],
                        "cmid": 0,
                    },
                    "customdata": customdata,
                    "texttemplate": "%{label}<br>%{customdata[1]}",
                    "hovertemplate": (
                        "<b>%{label}</b><br>"
                        "Weight: %{value:.1f}%<br>"
                        "Value: %{customdata[0]}<br>"
                        "Return: %{customdata[1]}"
                        "<extra></extra>"
                    ),
                }
            ],
            "layout": {
                "paper_bgcolor": "transparent",
                "plot_bgcolor": "transparent",
                "font": {"color": "#f9fafb"},
                "margin": {"l": 4, "r": 4, "t": 4, "b": 4},
            },
        }

        return positions, json.dumps(treemap)
    except Exception as exc:
        logger.error("Failed to build positions: %s", exc)
        return [], json.dumps({"data": [], "layout": {}})


def _build_signals(strategy_svc: Any) -> list[dict[str, Any]]:
    """Aggregate active signals from all strategies."""
    try:
        strategies = strategy_svc.get_available_strategies()
        all_signals: list[dict[str, Any]] = []

        for name in strategies:
            df = strategy_svc.get_signals(name, n_days=5)
            if df.empty:
                continue
            # Take the most recent row per ticker
            for _, row in df.iterrows():
                signal: dict[str, Any] = {
                    "strategy": name,
                    "ticker": row.get("ticker", ""),
                    "direction": row.get("direction", row.get("signal", "HOLD")),
                    "confidence": round(
                        float(row.get("confidence", row.get("score", 0.5))), 2
                    ),
                    "features": [],
                }
                # Extract top features if present
                if "top_features" in row:
                    signal["features"] = row["top_features"][:2]
                elif "feature_1" in row:
                    signal["features"] = [
                        f for f in [row.get("feature_1"), row.get("feature_2")]
                        if f
                    ]
                all_signals.append(signal)

        return all_signals
    except Exception as exc:
        logger.error("Failed to build signals: %s", exc)
        return []


def _build_position_detail(
    ticker: str,
) -> dict[str, Any]:
    """Build Level 4 detail for a single position."""
    svcs = _get_services()
    data_svc = svcs["data"]
    config = svcs["config"]

    detail: dict[str, Any] = {
        "ticker": ticker,
        "sparkline_json": "{}",
        "features": [],
        "interpretation": None,
    }

    try:
        # 7-day equity sparkline
        prices = data_svc.get_prices(tickers=[ticker])
        if not prices.empty and len(prices) >= 2:
            recent = prices[ticker].iloc[-7:] if len(prices) >= 7 else prices[ticker]
            spark = {
                "data": [
                    {
                        "type": "scatter",
                        "x": [d.strftime("%Y-%m-%d") for d in recent.index],
                        "y": [round(float(v), 2) for v in recent.values],
                        "fill": "tozeroy",
                        "fillcolor": "rgba(99, 102, 241, 0.15)",
                        "line": {"color": "#6366f1", "width": 2},
                        "mode": "lines",
                    }
                ],
                "layout": {
                    "paper_bgcolor": "transparent",
                    "plot_bgcolor": "transparent",
                    "font": {"color": "#f9fafb", "size": 10},
                    "margin": {"l": 32, "r": 8, "t": 8, "b": 24},
                    "xaxis": {"gridcolor": "#2d2d2d", "showgrid": False},
                    "yaxis": {"gridcolor": "#2d2d2d"},
                    "height": 150,
                },
            }
            detail["sparkline_json"] = json.dumps(spark)

        # Feature values
        features_df = data_svc.get_features(ticker)
        if not features_df.empty:
            last_row = features_df.iloc[-1]
            detail["features"] = [
                {"name": col, "value": round(float(last_row[col]), 4)}
                for col in features_df.columns[:8]
                if pd.notna(last_row[col])
            ]

        # Most recent Claude interpretation
        try:
            from src.notebooks.research_log import ResearchLog

            rlog = ResearchLog(config)
            entries = rlog.load_recent(20)
            # Filter for this ticker
            for entry in reversed(entries):
                ds = entry.get("data_summary", {})
                if ds.get("ticker") == ticker or ticker in str(ds):
                    detail["interpretation"] = entry.get("interpretation")
                    break
        except Exception:
            pass  # Research log not available

    except Exception as exc:
        logger.error("Failed to build position detail for %s: %s", ticker, exc)

    return detail


# -- Helper: build strategy context ----------------------------


def _build_strategy_context() -> dict[str, Any]:
    """Compute all data needed by the strategy page."""
    svcs = _get_services()
    strategy_svc = svcs["strategy"]

    try:
        strategies = strategy_svc.get_available_strategies()
    except Exception:
        strategies = []

    signals = _build_signals(strategy_svc)

    return {
        "strategies": strategies,
        "signals": signals,
    }


def _build_strategy_detail(strategy_name: str) -> dict[str, Any]:
    """Build detail view for a single strategy."""
    svcs = _get_services()
    strategy_svc = svcs["strategy"]
    portfolio_svc = svcs["portfolio"]

    detail: dict[str, Any] = {
        "strategy": strategy_name,
        "summary": {},
        "signals": [],
        "equity_json": "{}",
    }

    try:
        # Summary metrics from backtest results
        results = strategy_svc.get_backtest_results(strategy_name)
        if results is not None and isinstance(results.get("data"), pd.DataFrame):
            df = results["data"]
            if not df.empty:
                detail["summary"] = _extract_strategy_summary(df, strategy_name)

        # If no backtest data, compute from risk metrics
        if not detail["summary"]:
            risk = portfolio_svc.get_risk_metrics()
            detail["summary"] = {
                "sharpe": f"{risk.get('sharpe_ratio', 0):.2f}",
                "cagr": _pct(risk.get("annual_return", 0)),
                "max_drawdown": _pct(risk.get("max_drawdown", 0)),
                "win_rate": "--",
            }

        # Signal history (last 20)
        df = strategy_svc.get_signals(strategy_name, n_days=20)
        if not df.empty:
            for idx, row in df.iterrows():
                detail["signals"].append({
                    "date": idx.strftime("%Y-%m-%d")
                    if hasattr(idx, "strftime")
                    else str(idx),
                    "ticker": row.get("ticker", ""),
                    "direction": row.get(
                        "direction", row.get("signal", "HOLD"),
                    ),
                    "confidence": round(
                        float(row.get("confidence", row.get("score", 0.5))), 2,
                    ),
                })

        # Equity curve
        eq = portfolio_svc.get_equity_curve(strategy_name)
        if not eq.empty:
            recent = eq.iloc[-60:] if len(eq) > 60 else eq
            eq_chart = {
                "data": [
                    {
                        "type": "scatter",
                        "x": [d.strftime("%Y-%m-%d") for d in recent.index],
                        "y": [round(float(v), 4) for v in recent.values],
                        "line": {"color": "#6366f1", "width": 2},
                        "mode": "lines",
                    }
                ],
                "layout": {
                    "paper_bgcolor": "transparent",
                    "plot_bgcolor": "transparent",
                    "font": {"color": "#f9fafb", "size": 10},
                    "margin": {"l": 40, "r": 8, "t": 8, "b": 24},
                    "xaxis": {"gridcolor": "#2d2d2d"},
                    "yaxis": {"gridcolor": "#2d2d2d"},
                    "height": 200,
                },
            }
            detail["equity_json"] = json.dumps(eq_chart)

    except Exception as exc:
        logger.error(
            "Failed to build strategy detail for %s: %s",
            strategy_name, exc,
        )

    return detail


def _extract_strategy_summary(
    df: pd.DataFrame, name: str,
) -> dict[str, str]:
    """Extract summary metrics from a backtest results DataFrame."""
    summary: dict[str, str] = {
        "sharpe": "--",
        "cagr": "--",
        "max_drawdown": "--",
        "win_rate": "--",
    }
    try:
        # Various column name conventions
        for col in ("sharpe_ratio", "sharpe", "Sharpe"):
            if col in df.columns:
                summary["sharpe"] = f"{float(df[col].iloc[-1]):.2f}"
                break

        for col in ("cagr", "CAGR", "annual_return"):
            if col in df.columns:
                summary["cagr"] = _pct(float(df[col].iloc[-1]))
                break

        for col in ("max_drawdown", "Max Drawdown", "max_dd"):
            if col in df.columns:
                summary["max_drawdown"] = _pct(float(df[col].iloc[-1]))
                break

        for col in ("win_rate", "Win Rate", "hit_rate"):
            if col in df.columns:
                summary["win_rate"] = f"{float(df[col].iloc[-1]) * 100:.1f}%"
                break
    except Exception:
        pass

    return summary


# -- Helper: build portfolio context ----------------------------


def _build_portfolio_context() -> dict[str, Any]:
    """Compute all data needed by the portfolio page."""
    svcs = _get_services()
    portfolio: Any = svcs["portfolio"]
    data_svc: Any = svcs["data"]

    # Risk metrics
    risk: dict[str, float] = {}
    try:
        risk = portfolio.get_risk_metrics()
    except Exception as exc:
        logger.error("Failed to get risk metrics: %s", exc)

    # Allocation pie chart
    allocation_json = "{}"
    try:
        alloc = portfolio.get_allocation_chart_data()
        if alloc.get("labels"):
            _colors = [
                "#6366f1", "#22c55e", "#f59e0b", "#ef4444",
                "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6",
            ]
            pie = {
                "data": [
                    {
                        "type": "pie",
                        "labels": alloc["labels"],
                        "values": [round(v, 2) for v in alloc["values"]],
                        "marker": {"colors": _colors[: len(alloc["labels"])]},
                        "hole": 0.4,
                        "textinfo": "label+percent",
                        "textfont": {"color": "#f9fafb", "size": 11},
                    }
                ],
                "layout": {
                    "paper_bgcolor": "transparent",
                    "plot_bgcolor": "transparent",
                    "font": {"color": "#f9fafb"},
                    "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
                    "showlegend": False,
                    "height": 250,
                },
            }
            allocation_json = json.dumps(pie)
    except Exception as exc:
        logger.error("Failed to build allocation chart: %s", exc)

    # Equity curve + drawdown
    equity_json = "{}"
    drawdown_json = "{}"
    try:
        eq = portfolio.get_equity_curve()
        if not eq.empty:
            recent = eq.iloc[-120:] if len(eq) > 120 else eq
            eq_chart = {
                "data": [
                    {
                        "type": "scatter",
                        "x": [d.strftime("%Y-%m-%d") for d in recent.index],
                        "y": [round(float(v), 2) for v in recent.values],
                        "line": {"color": "#6366f1", "width": 2},
                        "mode": "lines",
                        "fill": "tozeroy",
                        "fillcolor": "rgba(99, 102, 241, 0.1)",
                    }
                ],
                "layout": {
                    "paper_bgcolor": "transparent",
                    "plot_bgcolor": "transparent",
                    "font": {"color": "#f9fafb", "size": 10},
                    "margin": {"l": 48, "r": 8, "t": 8, "b": 24},
                    "xaxis": {"gridcolor": "#2d2d2d"},
                    "yaxis": {"gridcolor": "#2d2d2d", "title": "Value"},
                    "height": 280,
                },
            }
            equity_json = json.dumps(eq_chart)

            # Drawdown from equity curve
            peak = recent.cummax()
            dd = (recent - peak) / peak.replace(0, 1)
            dd_chart = {
                "data": [
                    {
                        "type": "scatter",
                        "x": [d.strftime("%Y-%m-%d") for d in dd.index],
                        "y": [round(float(v) * 100, 2) for v in dd.values],
                        "line": {"color": "#ef4444", "width": 1},
                        "fill": "tozeroy",
                        "fillcolor": "rgba(239, 68, 68, 0.2)",
                        "mode": "lines",
                    }
                ],
                "layout": {
                    "paper_bgcolor": "transparent",
                    "plot_bgcolor": "transparent",
                    "font": {"color": "#f9fafb", "size": 10},
                    "margin": {"l": 48, "r": 8, "t": 8, "b": 24},
                    "xaxis": {"gridcolor": "#2d2d2d"},
                    "yaxis": {"gridcolor": "#2d2d2d", "title": "Drawdown %"},
                    "height": 200,
                },
            }
            drawdown_json = json.dumps(dd_chart)
    except Exception as exc:
        logger.error("Failed to build equity/drawdown charts: %s", exc)

    # Position table
    positions = _build_position_table_data(portfolio, data_svc)

    # Portfolio summary for the header
    total_value = sum(p.get("value", 0) for p in positions) if positions else 0
    daily_pnl = sum(p.get("daily_pnl", 0) for p in positions) if positions else 0
    daily_pnl_pct = daily_pnl / total_value if total_value else 0

    best_today = None
    worst_today = None
    for p in positions:
        ret = p.get("daily_return", 0) or 0
        if best_today is None or ret > best_today.get("return", 0):
            best_today = {"ticker": p.get("ticker", ""), "return": ret}
        if worst_today is None or ret < worst_today.get("return", 0):
            worst_today = {"ticker": p.get("ticker", ""), "return": ret}

    accounts = {p.get("account", p.get("sector", "default")) for p in positions}

    summary = {
        "total_value": total_value,
        "daily_pnl": daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "position_count": len(positions),
        "sector_count": max(len(accounts), 1),
        "best_today": best_today,
        "worst_today": worst_today,
    }

    return {
        "risk": risk,
        "positions": positions,
        "summary": summary,
        "allocation_json": allocation_json,
        "equity_json": equity_json,
        "drawdown_json": drawdown_json,
    }


def _build_position_table_data(
    portfolio: Any,
    data_svc: Any,
    sort_by: str = "weight",
) -> list[dict[str, Any]]:
    """Build sortable position table rows."""
    try:
        weights = portfolio.get_current_weights()
        prices = data_svc.get_prices()

        if weights.empty or prices.empty:
            return []

        common = weights.index.intersection(prices.columns)
        if len(common) == 0:
            return []

        weights = weights[common]
        weights = weights / weights.sum()

        latest = prices[common].iloc[-1]
        if len(prices) >= 2:
            prev = prices[common].iloc[-2]
            daily_ret = (latest - prev) / prev
        else:
            daily_ret = pd.Series(0.0, index=common)

        # Total return (first vs last)
        first = prices[common].iloc[0]
        total_ret = (latest - first) / first.replace(0, 1)

        rows: list[dict[str, Any]] = []
        for ticker in common:
            rows.append(
                {
                    "ticker": ticker,
                    "weight": float(weights[ticker]),
                    "price": float(latest[ticker]),
                    "pnl_today": float(daily_ret[ticker]),
                    "pnl_total": float(total_ret[ticker]),
                }
            )

        # Sort
        if sort_by == "ticker":
            rows.sort(key=lambda r: r["ticker"])
        elif sort_by == "pnl_today":
            rows.sort(key=lambda r: r["pnl_today"], reverse=True)
        elif sort_by == "pnl_total":
            rows.sort(key=lambda r: r["pnl_total"], reverse=True)
        else:  # weight (default)
            rows.sort(key=lambda r: r["weight"], reverse=True)

        return rows
    except Exception as exc:
        logger.error("Failed to build position table: %s", exc)
        return []


# -- Helper: build research context ----------------------------


def _build_research_context(notebook_filter: str = "") -> dict[str, Any]:
    """Compute all data needed by the research page."""
    svcs = _get_services()
    research_log = svcs.get("research_log")

    entries: list[dict[str, Any]] = []
    notebooks: list[str] = []

    if research_log is not None:
        try:
            raw = research_log.load_recent(50)
            # Reverse so newest first
            raw = list(reversed(raw))

            # Collect unique notebook names for filter
            notebooks = sorted(
                set(e.get("notebook", "") for e in raw if e.get("notebook"))
            )

            # Apply filter
            if notebook_filter:
                raw = [e for e in raw if e.get("notebook") == notebook_filter]

            # Format entries for display
            for entry in raw:
                ts = entry.get("timestamp", "")
                interp = entry.get("interpretation", {})
                summary = ""
                if isinstance(interp, dict):
                    summary = interp.get("summary", "")
                elif interp:
                    summary = str(interp)
                entries.append(
                    {
                        "id": ts,
                        "timestamp": ts[:19] if len(ts) >= 19 else ts,
                        "notebook": entry.get("notebook", "unknown"),
                        "task": entry.get("task", ""),
                        "summary": summary[:120],
                    }
                )
        except Exception as exc:
            logger.error("Failed to load research log: %s", exc)

    return {
        "entries": entries,
        "notebooks": notebooks,
        "active_filter": notebook_filter,
    }


def _build_research_entry(entry_id: str) -> dict[str, Any] | None:
    """Build detail view for a single research log entry."""
    svcs = _get_services()
    research_log = svcs.get("research_log")

    if research_log is None:
        return None

    try:
        raw = research_log.load_recent(100)
        for entry in raw:
            if entry.get("timestamp") == entry_id:
                interp = entry.get("interpretation", {})
                return {
                    "id": entry_id,
                    "timestamp": entry_id[:19]
                    if len(entry_id) >= 19
                    else entry_id,
                    "notebook": entry.get("notebook", "unknown"),
                    "task": entry.get("task", ""),
                    "data_summary": entry.get("data_summary", {}),
                    "interpretation": interp
                    if isinstance(interp, dict)
                    else {"summary": str(interp)},
                    "notes": entry.get("notes", ""),
                }
    except Exception as exc:
        logger.error("Failed to load research entry %s: %s", entry_id, exc)

    return None


# -- Helper: build execution context ----------------------------


def _build_execution_context() -> dict[str, Any]:
    """Compute all data needed by the execution page."""
    svcs = _get_services()
    execution: Any = svcs["execution"]

    # Broker status
    broker_status: dict[str, Any] = {
        "connected": False,
        "mode": "paper",
        "account_value": 0,
        "cash": 0,
        "invested": 0,
        "positions_count": 0,
    }
    try:
        broker_status = execution.get_broker_status()
    except Exception as exc:
        logger.error("Failed to get broker status: %s", exc)

    # Execution history (most recent)
    history: list[dict[str, Any]] = []
    try:
        history = execution.get_execution_history(n=50)
    except Exception as exc:
        logger.error("Failed to get execution history: %s", exc)

    is_live = broker_status.get("mode", "paper") == "live"

    return {
        "broker_status": broker_status,
        "is_live": is_live,
        "history": history[:50],
    }


def _build_order_history(
    page: int = 1, per_page: int = 20,
) -> dict[str, Any]:
    """Build paginated order history."""
    svcs = _get_services()
    execution: Any = svcs["execution"]

    try:
        all_history = execution.get_execution_history(n=100)
    except Exception:
        all_history = []

    total = len(all_history)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    start = (page - 1) * per_page
    end = start + per_page

    return {
        "orders": all_history[start:end],
        "page": page,
        "total_pages": total_pages,
        "total": total,
    }


# -- Routes: Overview ------------------------------------------


@router.get("/overview", response_class=HTMLResponse)
async def overview(request: Request) -> HTMLResponse:
    """Overview dashboard with chat, nav tiles, watchlist, and news snapshots."""
    svcs = _get_services()
    config = svcs["config"]
    market_svc = svcs["market"]
    news_svc = svcs["news"]
    strategy_svc = svcs["strategy"]

    # Watchlist rows for snapshot panel
    ui_cfg = config.get("ui", {})
    watchlist_cfg = ui_cfg.get("watchlist", {})
    all_tickers: list[str] = []
    for g in watchlist_cfg.get("groups", []):
        all_tickers.extend(g.get("tickers", []))

    watchlist_rows: list[dict[str, Any]] = []
    try:
        from web.routes.watchlist import card_to_wl_row
        cards = market_svc.get_watchlist_cards(all_tickers[:20])
        watchlist_rows = [card_to_wl_row(c) for c in cards]
    except Exception as exc:
        logger.error("Failed to build watchlist rows: %s", exc)

    # News snapshot
    news_items: list[dict[str, Any]] = []
    try:
        news_items = news_svc.get_cached_news()[:5]
    except Exception as exc:
        logger.error("Failed to load news: %s", exc)

    # Signal count for tiles
    signal_count = 0
    try:
        strategies = strategy_svc.get_available_strategies()
        for name in strategies:
            df = strategy_svc.get_signals(name, n_days=5)
            if df.empty:
                continue
            for _, row in df.iterrows():
                d = row.get("direction", row.get("signal", "NEUTRAL"))
                if d and d.upper() != "NEUTRAL":
                    signal_count += 1
    except Exception:
        pass

    # Unread news count
    unread_news = len(news_items)

    # Execution mode
    exec_mode = config.get("execution", {}).get("mode", "paper")

    # Tab statuses
    from web.routes.utils import get_tab_statuses
    tab_statuses = get_tab_statuses(config, signal_count, unread_news)

    return templates.TemplateResponse(
        "overview.html",
        {
            "request": request,
            "active_page": "overview",
            "watchlist_rows": watchlist_rows,
            "news_items": news_items,
            "signal_count": signal_count,
            "unread_news": unread_news,
            "exec_mode": exec_mode,
            "tab_statuses": tab_statuses,
        },
    )


@router.get("/partials/metrics-bar", response_class=HTMLResponse)
async def metrics_bar_partial(request: Request) -> HTMLResponse:
    """HTMX partial -- returns just the metrics bar fragment."""
    try:
        svcs = _get_services()
        metrics = _build_metrics(svcs["portfolio"], svcs["data"])
    except Exception:
        metrics = [
            {"label": "Portfolio Value", "value": "--", "css_class": ""},
            {"label": "Daily P&L", "value": "--", "css_class": ""},
            {"label": "Sharpe 30d", "value": "--", "css_class": "font-mono"},
            {"label": "Active Signals", "value": "--", "css_class": "font-mono"},
            {"label": "VIX", "value": "--", "css_class": "font-mono badge-neutral"},
        ]
    return templates.TemplateResponse(
        "partials/metric_bar.html",
        {"request": request, "metrics": metrics},
    )


@router.get("/partials/signals", response_class=HTMLResponse)
async def signals_partial(request: Request) -> HTMLResponse:
    """HTMX partial -- active signals panel."""
    try:
        svcs = _get_services()
        signals = _build_signals(svcs["strategy"])
    except Exception:
        signals = []
    return templates.TemplateResponse(
        "partials/signals.html",
        {"request": request, "signals": signals},
    )


@router.get("/partials/position-detail", response_class=HTMLResponse)
async def position_detail_partial(
    request: Request,
    ticker: str = Query("", description="Ticker symbol"),
) -> HTMLResponse:
    """HTMX partial -- Level 4 position detail."""
    if not ticker:
        return templates.TemplateResponse(
            "partials/position_detail.html",
            {"request": request, "detail": None},
        )
    detail = _build_position_detail(ticker)
    return templates.TemplateResponse(
        "partials/position_detail.html",
        {"request": request, "detail": detail},
    )


# -- Routes: Strategy ------------------------------------------


@router.get("/strategy", response_class=HTMLResponse)
async def strategy_page(request: Request) -> HTMLResponse:
    """Strategy browser page."""
    ctx = _build_strategy_context()
    return templates.TemplateResponse(
        "strategy.html",
        {
            "request": request,
            "active_page": "strategy",
            **ctx,
        },
    )


@router.get("/partials/strategy-detail", response_class=HTMLResponse)
async def strategy_detail_partial(
    request: Request,
    strategy: str = Query("", description="Strategy name"),
) -> HTMLResponse:
    """HTMX partial -- strategy detail panel."""
    if not strategy:
        return templates.TemplateResponse(
            "partials/strategy_detail.html",
            {"request": request, "detail": None},
        )
    detail = _build_strategy_detail(strategy)
    return templates.TemplateResponse(
        "partials/strategy_detail.html",
        {"request": request, "detail": detail},
    )


# -- Actions ---------------------------------------------------


@router.post("/actions/interpret", response_class=HTMLResponse)
async def interpret_action(
    request: Request,
    ticker: str = Query("", description="Ticker to interpret"),
) -> HTMLResponse:
    """Run Claude interpretation on demand for a ticker."""
    svcs = _get_services()
    config = svcs["config"]
    data_svc = svcs["data"]

    interpretation: dict[str, Any] | None = None
    error: str | None = None

    if not ticker:
        error = "No ticker specified"
    else:
        try:
            from src.notebooks.claude_interpreter import QuantInterpreter
            from src.notebooks.research_log import ResearchLog

            # Gather data for interpretation
            features_df = data_svc.get_features(ticker)
            risk = svcs["portfolio"].get_risk_metrics()

            data_summary = {
                "ticker": ticker,
                "risk_metrics": {
                    k: round(v, 4) if not np.isnan(v) else None
                    for k, v in risk.items()
                },
            }
            if not features_df.empty:
                last = features_df.iloc[-1]
                data_summary["features"] = {
                    col: round(float(last[col]), 4)
                    for col in features_df.columns[:6]
                    if pd.notna(last[col])
                }

            interpreter = QuantInterpreter(config)
            interpretation = interpreter.interpret("risk_metrics", data_summary)

            # Log it
            rlog = ResearchLog(config)
            rlog.log_entry(
                notebook="web_ui",
                task="risk_metrics",
                data_summary=data_summary,
                interpretation=interpretation,
            )
        except EnvironmentError:
            error = "ANTHROPIC_API_KEY not set -- interpretation unavailable"
            interpretation = {
                "summary": "Interpretation unavailable",
                "observations": [],
                "warnings": [error],
                "suggestions": [],
                "confidence": "low",
            }
        except Exception as exc:
            logger.error("Interpretation failed for %s: %s", ticker, exc)
            error = f"Interpretation failed: {exc}"
            interpretation = {
                "summary": "Interpretation unavailable",
                "observations": [],
                "warnings": [str(exc)],
                "suggestions": [],
                "confidence": "low",
            }

    # Re-build position detail with updated interpretation
    detail = _build_position_detail(ticker)
    if interpretation:
        detail["interpretation"] = interpretation
    if error and not interpretation:
        detail["interpretation"] = {
            "summary": error,
            "observations": [],
            "warnings": [],
            "suggestions": [],
            "confidence": "low",
        }

    return templates.TemplateResponse(
        "partials/position_detail.html",
        {"request": request, "detail": detail},
    )


# -- Routes: Portfolio -----------------------------------------
# Main /portfolio GET is in web/routes/portfolio.py (redesigned page).


@router.get("/partials/position-table", response_class=HTMLResponse)
async def position_table_partial(
    request: Request,
    sort: str = Query("weight", description="Sort column"),
) -> HTMLResponse:
    """HTMX partial -- sortable position table."""
    svcs = _get_services()
    positions = _build_position_table_data(
        svcs["portfolio"], svcs["data"], sort_by=sort,
    )
    return templates.TemplateResponse(
        "partials/position_table.html",
        {"request": request, "positions": positions},
    )


@router.get("/portfolio/dca/ticker/{ticker}", response_class=HTMLResponse)
async def portfolio_dca_detail(
    request: Request,
    ticker: str,
) -> HTMLResponse:
    """HTMX partial -- DCA detail panel for a single ticker."""
    svcs = _get_services()
    dca_svc = svcs["dca"]

    try:
        summary = dca_svc.get_dca_summary(ticker)
    except Exception:
        summary = dca_svc._empty_summary(ticker)

    try:
        chart_data = dca_svc.compute_dca_chart_data(ticker)
    except Exception:
        chart_data = dca_svc._empty_chart_data()

    try:
        history = dca_svc.get_purchase_history(ticker)
    except Exception:
        history = []

    try:
        limit = dca_svc.get_recommended_limit(ticker)
    except Exception:
        limit = {
            "recommended_limit_pence": 0,
            "recommended_limit_display": "N/A",
            "rationale": "",
            "current_vs_limit": 0.0,
        }

    # Get watchlist tickers for the selector
    config = svcs["config"]
    tickers = (
        config.get("ui", {})
        .get("watchlist", {})
        .get("tickers", config.get("universe", {}).get("tickers", []))
    )

    return templates.TemplateResponse(
        "partials/portfolio_dca.html",
        {
            "request": request,
            "ticker": ticker,
            "tickers": tickers,
            "summary": summary,
            "chart_data": chart_data,
            "history": history,
            "limit": limit,
        },
    )


@router.post("/portfolio/dca/purchase", response_class=HTMLResponse)
async def portfolio_dca_add_purchase(
    request: Request,
    ticker: str = Form(...),
    date: str = Form(...),
    price: float = Form(...),
    amount_gbp: float = Form(...),
    note: str = Form(""),
) -> HTMLResponse:
    """Add a DCA purchase via form POST.

    Returns a purchase row partial for HTMX out-of-band swap.
    """
    svcs = _get_services()
    dca_svc = svcs["dca"]

    try:
        record = dca_svc.add_purchase(ticker, date, price, amount_gbp, note)
    except ValueError as exc:
        return HTMLResponse(
            f'<tr><td colspan="8" class="muted">{exc}</td></tr>',
            status_code=422,
        )

    # Enrich the record with current data for the row partial
    history = dca_svc.get_purchase_history(ticker)
    # Find the just-added record
    new_row = next(
        (p for p in history if p["date"] == date),
        record,
    )

    return templates.TemplateResponse(
        "partials/dca_purchase_row.html",
        {"request": request, "purchase": new_row, "ticker": ticker},
    )


# -- Routes: Research ------------------------------------------


@router.get("/research", response_class=HTMLResponse)
async def research_page(request: Request) -> HTMLResponse:
    """Research log browser with trading personas."""
    ctx = _build_research_context()

    # Inject persona data
    from src.services.persona_service import PersonaService

    persona_svc = PersonaService(_cfg)
    personas = persona_svc.get_all_personas()
    _state_classes = {
        "active": "badge-green",
        "pending": "badge-amber",
        "dormant": "badge-grey",
        "paused": "badge-blue",
    }
    for p in personas:
        p["state_class"] = _state_classes.get(p.get("state", ""), "badge-grey")
    ctx["personas"] = personas

    return templates.TemplateResponse(
        "research.html",
        {"request": request, "active_page": "research", **ctx},
    )


@router.get("/partials/research-entries", response_class=HTMLResponse)
async def research_entries_partial(
    request: Request,
    notebook: str = Query("", description="Filter by notebook"),
) -> HTMLResponse:
    """HTMX partial -- filtered research entry list."""
    ctx = _build_research_context(notebook_filter=notebook)
    return templates.TemplateResponse(
        "partials/research_entries.html",
        {"request": request, **ctx},
    )


@router.get("/partials/research-entry", response_class=HTMLResponse)
async def research_entry_partial(
    request: Request,
    id: str = Query("", description="Entry timestamp ID"),
) -> HTMLResponse:
    """HTMX partial -- single research entry detail."""
    entry = _build_research_entry(id) if id else None
    return templates.TemplateResponse(
        "partials/research_entry.html",
        {"request": request, "entry": entry},
    )


@router.get("/partials/modal-upload-research", response_class=HTMLResponse)
async def modal_upload_research(request: Request) -> HTMLResponse:
    """HTMX partial -- upload research modal."""
    return templates.TemplateResponse(
        "partials/modal_upload_research.html",
        {"request": request},
    )


@router.post("/upload/research", response_class=HTMLResponse)
async def upload_research(
    request: Request,
    file: UploadFile,
    ticker: str = Form(""),
    notes: str = Form(""),
) -> HTMLResponse:
    """Handle research document upload.

    Saves file to data/research/ and appends an entry to the research log.
    """
    # Validate extension
    allowed = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        return HTMLResponse(
            '<div class="modal-card"><p class="modal-error">'
            f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md</p></div>",
            status_code=422,
        )

    # Save file
    research_dir = Path("data/research")
    research_dir.mkdir(parents=True, exist_ok=True)
    dest = research_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    # Append to research log
    svcs = _get_services()
    research_log = svcs.get("research_log")
    if research_log is not None:
        data_summary: dict[str, Any] = {
            "file": file.filename,
            "size_bytes": len(content),
        }
        if ticker:
            data_summary["ticker"] = ticker
        research_log.log_entry(
            notebook="web_upload",
            task="document_upload",
            data_summary=data_summary,
            interpretation={"summary": notes or f"Uploaded {file.filename}"},
            notes=notes,
        )

    return HTMLResponse(
        '<div class="modal-card"><p class="modal-success">'
        f"Successfully uploaded {file.filename}</p>"
        '<button class="btn btn-sm" onclick="document.getElementById(\'modal-container\').innerHTML=\'\'">'
        "Close</button></div>"
    )


# -- Routes: Import Purchases ----------------------------------


@router.get("/partials/modal-import-purchases", response_class=HTMLResponse)
async def modal_import_purchases(request: Request) -> HTMLResponse:
    """HTMX partial -- CSV import modal."""
    return templates.TemplateResponse(
        "partials/modal_import_purchases.html",
        {"request": request},
    )


@router.post("/upload/purchases/preview", response_class=HTMLResponse)
async def upload_purchases_preview(
    request: Request,
    file: UploadFile,
) -> HTMLResponse:
    """Parse uploaded CSV and return a preview table."""
    import csv as csv_mod
    import io as io_mod

    content = (await file.read()).decode("utf-8-sig")
    reader = csv_mod.DictReader(io_mod.StringIO(content))

    required = {"date", "ticker", "price", "amount_gbp"}
    if not required.issubset(set(reader.fieldnames or [])):
        missing = required - set(reader.fieldnames or [])
        return templates.TemplateResponse(
            "partials/import_preview_table.html",
            {"request": request, "error": f"Missing columns: {', '.join(sorted(missing))}", "rows": []},
        )

    rows = []
    skipped_cash = 0
    for row in reader:
        try:
            price_str = row.get("price", "").strip()
            if not price_str:
                # Cash balance rows have no price — skip them
                skipped_cash += 1
                continue
            rows.append({
                "date": row["date"].strip(),
                "ticker": row["ticker"].strip(),
                "price": float(price_str),
                "amount_gbp": float(row["amount_gbp"]),
                "note": row.get("note", "").strip(),
            })
        except (ValueError, KeyError):
            continue

    if not rows:
        return templates.TemplateResponse(
            "partials/import_preview_table.html",
            {"request": request, "error": "No valid rows found in CSV.", "rows": []},
        )

    return templates.TemplateResponse(
        "partials/import_preview_table.html",
        {"request": request, "rows": rows, "csv_raw": content, "error": None},
    )


@router.post("/upload/purchases/confirm", response_class=HTMLResponse)
async def upload_purchases_confirm(
    request: Request,
    csv_data: str = Form(...),
) -> HTMLResponse:
    """Bulk-import purchases from pre-validated CSV data."""
    import csv as csv_mod
    import io as io_mod

    svcs = _get_services()
    dca_svc = svcs["dca"]

    reader = csv_mod.DictReader(io_mod.StringIO(csv_data))
    imported = 0
    skipped = 0

    for row in reader:
        try:
            dca_svc.add_purchase(
                ticker=row["ticker"].strip(),
                date=row["date"].strip(),
                price=float(row["price"]),
                amount_gbp=float(row["amount_gbp"]),
                note=row.get("note", "").strip(),
            )
            imported += 1
        except ValueError:
            skipped += 1

    return HTMLResponse(
        '<div class="modal-card">'
        f'<p class="modal-success">{imported} purchase(s) imported.</p>'
        + (f'<p class="muted" style="font-size:0.75rem;">{skipped} duplicate(s) skipped.</p>' if skipped else "")
        + '<button class="btn btn-sm" onclick="document.getElementById(\'modal-container\').innerHTML=\'\'">'
        "Close</button></div>"
    )


@router.get("/partials/modal-add-purchase", response_class=HTMLResponse)
async def modal_add_purchase(request: Request) -> HTMLResponse:
    """HTMX partial -- quick-add purchase modal."""
    return templates.TemplateResponse(
        "partials/modal_add_purchase.html",
        {"request": request},
    )


# -- Routes: Execution -----------------------------------------


@router.get("/execution", response_class=HTMLResponse)
async def execution_page(request: Request) -> HTMLResponse:
    """Execution monitoring page."""
    ctx = _build_execution_context()
    return templates.TemplateResponse(
        "execution.html",
        {"request": request, "active_page": "execution", **ctx},
    )


@router.get("/partials/open-orders", response_class=HTMLResponse)
async def open_orders_partial(request: Request) -> HTMLResponse:
    """HTMX partial -- open orders (paper mode has no order queue)."""
    return templates.TemplateResponse(
        "partials/open_orders.html",
        {"request": request, "orders": []},
    )


@router.get("/partials/order-history", response_class=HTMLResponse)
async def order_history_partial(
    request: Request,
    page: int = Query(1, description="Page number"),
) -> HTMLResponse:
    """HTMX partial -- paginated order history."""
    history = _build_order_history(page=page)
    return templates.TemplateResponse(
        "partials/order_history.html",
        {"request": request, **history},
    )
