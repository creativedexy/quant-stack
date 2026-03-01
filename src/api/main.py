"""FastAPI application — Quant Stack API.

Wraps the service layer so a web frontend can be built on top.
All endpoints return JSON with a consistent envelope:
    {"data": ..., "timestamp": "...", "status": "ok"}

Run via:
    python -m scripts.run_api [--port 8000]
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ApiResponse,
    BacktestResult,
    EquityCurvePoint,
    ExecuteResult,
    ExecutePlanRequest,
    ExecutionRecord,
    ExecutionStatus,
    FeatureData,
    FeaturePoint,
    HealthData,
    LatestPrice,
    OHLCVPoint,
    PipelineRunRequest,
    PriceHistory,
    PipelineRunResult,
    PipelineStageStatus,
    PipelineStatus,
    PortfolioOverview,
    Position,
    RebalanceOrder,
    RebalancePlan,
    RebalancePlanRequest,
    RiskMetrics,
    StrategyComparison,
    StrategyInfo,
)
from src.data.cleaner import DataCleaner, compute_returns
from src.data.fetcher import create_fetcher
from src.data.synthetic import generate_multi_asset_data, generate_synthetic_ohlcv
from src.utils.logging import get_logger

import src as _src_root

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Application bootstrap
# ─────────────────────────────────────────────

_START_TIME = time.monotonic()

app = FastAPI(
    title="Quant Stack API",
    version=_src_root.__version__,
    description="REST API for the Quant Stack quantitative trading system.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Shared state (synthetic data seeded on startup)
# ─────────────────────────────────────────────

_DEMO_TICKERS = ["SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L"]
_fetcher = create_fetcher("synthetic", seed=42)
_cleaner = DataCleaner()


def _get_demo_data() -> dict[str, pd.DataFrame]:
    """Fetch (or cache) demo synthetic data for all demo tickers."""
    if not hasattr(_get_demo_data, "_cache"):
        raw = _fetcher.fetch_multiple(_DEMO_TICKERS, start="2020-01-01")
        _get_demo_data._cache = _cleaner.clean_multiple(raw)
    return _get_demo_data._cache


def _wrap(data: Any) -> dict[str, Any]:
    """Wrap payload in the standard API envelope."""
    return {"data": data, "timestamp": datetime.utcnow().isoformat(), "status": "ok"}


def _df_to_ohlcv(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of OHLCV dicts."""
    records = []
    for date, row in df.iterrows():
        records.append(
            OHLCVPoint(
                date=date.strftime("%Y-%m-%d"),
                open=round(float(row["Open"]), 2),
                high=round(float(row["High"]), 2),
                low=round(float(row["Low"]), 2),
                close=round(float(row["Close"]), 2),
                volume=int(row["Volume"]),
            ).model_dump()
        )
    return records


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────


@app.get("/api/health")
def health() -> dict:
    """Service status check."""
    modules = {
        "data": "available",
        "features": "placeholder",
        "models": "placeholder",
        "portfolio": "placeholder",
        "backtest": "placeholder",
        "execution": "placeholder",
    }
    return _wrap(
        HealthData(
            version=_src_root.__version__,
            uptime_seconds=round(time.monotonic() - _START_TIME, 2),
            modules=modules,
        ).model_dump()
    )


# ─────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────


@app.get("/api/portfolio/overview")
def portfolio_overview() -> dict:
    """Key portfolio KPI metrics (synthetic demo data)."""
    data = _get_demo_data()
    # Build a simple equal-weight portfolio from Close prices
    closes = pd.DataFrame({t: df["Close"] for t, df in data.items()})
    portfolio_value = closes.mean(axis=1)

    total_ret = float((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1)
    daily_rets = portfolio_value.pct_change().dropna()
    daily_ret = float(daily_rets.iloc[-1])

    ann_vol = float(daily_rets.std() * np.sqrt(252))
    sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() else 0
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    max_dd = float(drawdown.min())

    overview = PortfolioOverview(
        total_value=round(float(portfolio_value.iloc[-1]) * 1000, 2),
        daily_pnl=round(daily_ret * float(portfolio_value.iloc[-1]) * 1000, 2),
        daily_return_pct=round(daily_ret * 100, 4),
        total_return_pct=round(total_ret * 100, 2),
        sharpe_ratio=round(sharpe, 3),
        max_drawdown_pct=round(max_dd * 100, 2),
        num_positions=len(data),
        cash=10000.0,
    )
    return _wrap(overview.model_dump())


@app.get("/api/portfolio/positions")
def portfolio_positions() -> dict:
    """Current positions (synthetic demo)."""
    data = _get_demo_data()
    total = sum(float(df["Close"].iloc[-1]) for df in data.values()) * 200
    positions = []
    for ticker, df in data.items():
        price = float(df["Close"].iloc[-1])
        cost = float(df["Close"].iloc[0])
        qty = 200.0
        mv = price * qty
        positions.append(
            Position(
                ticker=ticker,
                quantity=qty,
                avg_cost=round(cost, 2),
                current_price=round(price, 2),
                market_value=round(mv, 2),
                unrealised_pnl=round((price - cost) * qty, 2),
                weight_pct=round(mv / total * 100, 2) if total else 0,
            ).model_dump()
        )
    return _wrap(positions)


@app.get("/api/portfolio/equity-curve")
def portfolio_equity_curve(
    period: str = Query("1y", description="Lookback period: 1m, 3m, 6m, 1y, max"),
) -> dict:
    """Equity curve data as a JSON array."""
    data = _get_demo_data()
    closes = pd.DataFrame({t: df["Close"] for t, df in data.items()})
    portfolio_value = closes.mean(axis=1) * 1000

    period_map = {"1m": 21, "3m": 63, "6m": 126, "1y": 252, "max": len(portfolio_value)}
    n = period_map.get(period, 252)
    portfolio_value = portfolio_value.iloc[-n:]

    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max

    curve = [
        EquityCurvePoint(
            date=date.strftime("%Y-%m-%d"),
            value=round(float(val), 2),
            drawdown_pct=round(float(drawdown.loc[date]) * 100, 2),
        ).model_dump()
        for date, val in portfolio_value.items()
    ]
    return _wrap(curve)


@app.get("/api/portfolio/risk")
def portfolio_risk() -> dict:
    """Portfolio risk metrics."""
    data = _get_demo_data()
    closes = pd.DataFrame({t: df["Close"] for t, df in data.items()})
    daily_rets = closes.mean(axis=1).pct_change().dropna()

    var_95 = float(np.percentile(daily_rets, 5))
    var_99 = float(np.percentile(daily_rets, 1))
    cvar_95 = float(daily_rets[daily_rets <= var_95].mean())
    cvar_99 = float(daily_rets[daily_rets <= var_99].mean())

    running_max = closes.mean(axis=1).cummax()
    drawdown = (closes.mean(axis=1) - running_max) / running_max

    metrics = RiskMetrics(
        var_95=round(var_95 * 100, 4),
        var_99=round(var_99 * 100, 4),
        cvar_95=round(cvar_95 * 100, 4),
        cvar_99=round(cvar_99 * 100, 4),
        max_drawdown_pct=round(float(drawdown.min()) * 100, 2),
        volatility_annual=round(float(daily_rets.std() * np.sqrt(252)) * 100, 2),
    )
    return _wrap(metrics.model_dump())


# ─────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────

_DEMO_STRATEGIES = [
    StrategyInfo(
        name="momentum_12_1",
        description="12-month momentum with 1-month reversal exclusion",
        status="backtest_only",
        last_run="2026-02-28T18:30:00",
        total_return_pct=34.5,
        sharpe_ratio=1.12,
    ),
    StrategyInfo(
        name="mean_reversion_bb",
        description="Mean reversion using Bollinger Band breakouts",
        status="backtest_only",
        last_run="2026-02-28T18:30:00",
        total_return_pct=21.8,
        sharpe_ratio=0.87,
    ),
    StrategyInfo(
        name="quality_value",
        description="Quality + value factor combination",
        status="inactive",
        total_return_pct=None,
        sharpe_ratio=None,
    ),
]


@app.get("/api/strategies")
def list_strategies() -> dict:
    """List available strategies."""
    return _wrap([s.model_dump() for s in _DEMO_STRATEGIES])


@app.get("/api/strategies/{name}/results")
def strategy_results(name: str) -> dict:
    """Backtest results for a strategy."""
    strategy = next((s for s in _DEMO_STRATEGIES if s.name == name), None)
    if strategy is None:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    if strategy.total_return_pct is None:
        raise HTTPException(status_code=404, detail=f"No backtest results for '{name}'")

    # Generate a synthetic equity curve for the result
    days = 504  # ~2 years
    rng = np.random.default_rng(hash(name) % (2**31))
    daily_rets = rng.normal(0.0003, 0.01, days)
    equity = 100_000 * np.cumprod(1 + daily_rets)
    dates = pd.bdate_range("2024-01-02", periods=days, freq="B")

    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max

    curve = [
        EquityCurvePoint(
            date=dates[i].strftime("%Y-%m-%d"),
            value=round(float(equity[i]), 2),
            drawdown_pct=round(float(dd[i]) * 100, 2),
        ).model_dump()
        for i in range(days)
    ]

    result = BacktestResult(
        strategy=name,
        start_date=dates[0].strftime("%Y-%m-%d"),
        end_date=dates[-1].strftime("%Y-%m-%d"),
        total_return_pct=strategy.total_return_pct,
        annual_return_pct=round(strategy.total_return_pct / 2, 2),
        sharpe_ratio=strategy.sharpe_ratio,
        max_drawdown_pct=round(float(dd.min()) * 100, 2),
        win_rate_pct=round(float(np.mean(daily_rets > 0)) * 100, 1),
        num_trades=rng.integers(50, 200),
        equity_curve=curve,
    )
    return _wrap(result.model_dump())


@app.get("/api/strategies/compare")
def compare_strategies() -> dict:
    """Comparison table of all strategies with results."""
    active = [s for s in _DEMO_STRATEGIES if s.total_return_pct is not None]
    return _wrap(StrategyComparison(strategies=active).model_dump())


# ─────────────────────────────────────────────
# Prices
# ─────────────────────────────────────────────


@app.get("/api/prices/latest")
def latest_prices() -> dict:
    """Current prices for the universe."""
    data = _get_demo_data()
    prices = []
    for ticker, df in data.items():
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        change = (float(last["Close"]) - float(prev["Close"])) / float(prev["Close"])
        prices.append(
            LatestPrice(
                ticker=ticker,
                price=round(float(last["Close"]), 2),
                change_pct=round(change * 100, 2),
                volume=int(last["Volume"]),
                timestamp=last.name.strftime("%Y-%m-%dT%H:%M:%S"),
            ).model_dump()
        )
    return _wrap(prices)


@app.get("/api/prices/{ticker}/history")
def price_history(
    ticker: str,
    start: str | None = Query(None, description="Start date YYYY-MM-DD"),
    end: str | None = Query(None, description="End date YYYY-MM-DD"),
    interval: str = Query("1d", description="Data interval"),
) -> dict:
    """Price history for a single ticker."""
    try:
        df = _fetcher.fetch(ticker, start=start or "2020-01-01", end=end)
        df = _cleaner.clean(df, ticker)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    history = PriceHistory(
        ticker=ticker,
        interval=interval,
        data=_df_to_ohlcv(df),
    )
    return _wrap(history.model_dump())


@app.get("/api/prices/{ticker}/features")
def price_features(
    ticker: str,
    windows: str = Query("1,5,21", description="Return windows (comma-separated)"),
) -> dict:
    """Feature data (returns) for a ticker."""
    try:
        df = _fetcher.fetch(ticker, start="2020-01-01")
        df = _cleaner.clean(df, ticker)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    window_list = [int(w.strip()) for w in windows.split(",")]
    returns_df = compute_returns(df, windows=window_list, log_returns=True)

    feature_names = list(returns_df.columns)
    points = []
    for date, row in returns_df.iterrows():
        values = {}
        for col in feature_names:
            val = row[col]
            values[col] = round(float(val), 6) if pd.notna(val) else None
        points.append(
            FeaturePoint(date=date.strftime("%Y-%m-%d"), values=values).model_dump()
        )

    result = FeatureData(ticker=ticker, features=feature_names, data=points)
    return _wrap(result.model_dump())


# ─────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────


@app.get("/api/execution/status")
def execution_status() -> dict:
    """Broker connection status."""
    status = ExecutionStatus(
        broker="synthetic",
        connected=True,
        mode="paper",
        pending_orders=0,
        last_heartbeat=datetime.utcnow().isoformat(),
    )
    return _wrap(status.model_dump())


@app.post("/api/execution/plan")
def execution_plan(request: RebalancePlanRequest) -> dict:
    """Generate a rebalance plan from target weights."""
    if abs(sum(request.target_weights.values()) - 1.0) > 0.01:
        raise HTTPException(status_code=422, detail="Target weights must sum to 1.0")

    data = _get_demo_data()
    total_value = request.total_value or 100_000.0
    n_tickers = len(data)
    current_weight = 1.0 / n_tickers if n_tickers else 0

    orders = []
    for ticker, target_w in request.target_weights.items():
        if ticker in data:
            price = float(data[ticker]["Close"].iloc[-1])
        else:
            price = 100.0  # fallback for unknown tickers

        delta_value = (target_w - current_weight) * total_value
        qty = abs(delta_value / price)
        if abs(delta_value) < 1.0:
            continue

        orders.append(
            RebalanceOrder(
                ticker=ticker,
                side="buy" if delta_value > 0 else "sell",
                quantity=round(qty, 2),
                estimated_cost=round(abs(delta_value), 2),
                current_weight_pct=round(current_weight * 100, 2),
                target_weight_pct=round(target_w * 100, 2),
            ).model_dump()
        )

    plan = RebalancePlan(
        orders=orders,
        estimated_commission=round(len(orders) * 1.50, 2),
        estimated_slippage=round(sum(o["estimated_cost"] for o in orders) * 0.001, 2),
    )
    return _wrap(plan.model_dump())


@app.post("/api/execution/execute")
def execution_execute(request: ExecutePlanRequest) -> dict:
    """Execute a rebalance plan (paper mode only)."""
    if request.mode != "paper":
        raise HTTPException(
            status_code=403,
            detail="Live trading is disabled. Only paper mode is allowed via the API.",
        )

    order_ids = [f"PAPER-{uuid.uuid4().hex[:8].upper()}" for _ in request.orders]
    result = ExecuteResult(
        submitted=len(request.orders),
        mode="paper",
        order_ids=order_ids,
    )
    return _wrap(result.model_dump())


@app.get("/api/execution/history")
def execution_history(limit: int = Query(20, ge=1, le=100)) -> dict:
    """Recent execution history (synthetic demo)."""
    rng = np.random.default_rng(99)
    records = []
    for i in range(min(limit, 10)):
        ticker = _DEMO_TICKERS[i % len(_DEMO_TICKERS)]
        records.append(
            ExecutionRecord(
                order_id=f"PAPER-{rng.integers(10000, 99999)}",
                ticker=ticker,
                side=rng.choice(["buy", "sell"]),
                quantity=float(rng.integers(10, 500)),
                price=round(float(rng.uniform(50, 300)), 2),
                commission=1.50,
                timestamp=f"2026-02-{28 - i:02d}T14:{rng.integers(0, 59):02d}:00",
                status="filled",
            ).model_dump()
        )
    return _wrap(records)


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

_ALL_STAGES = ["data", "features", "models", "portfolio", "backtest"]


@app.get("/api/pipeline/status")
def pipeline_status() -> dict:
    """Pipeline stage status."""
    stages = [
        PipelineStageStatus(name="data", status="completed", last_run="2026-02-28T18:00:00"),
        PipelineStageStatus(name="features", status="idle"),
        PipelineStageStatus(name="models", status="idle"),
        PipelineStageStatus(name="portfolio", status="idle"),
        PipelineStageStatus(name="backtest", status="idle"),
    ]
    status = PipelineStatus(running=False, stages=stages)
    return _wrap(status.model_dump())


@app.post("/api/pipeline/run")
def pipeline_run(request: PipelineRunRequest) -> dict:
    """Trigger a pipeline run."""
    stages = request.stages or _ALL_STAGES
    invalid = [s for s in stages if s not in _ALL_STAGES]
    if invalid:
        raise HTTPException(status_code=422, detail=f"Unknown stages: {invalid}")

    job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
    result = PipelineRunResult(
        job_id=job_id,
        stages_queued=stages,
        message=f"Pipeline queued with {len(stages)} stage(s) using source='{request.source}'",
    )
    return _wrap(result.model_dump())


# ─────────────────────────────────────────────
# WebSocket — live price stream
# ─────────────────────────────────────────────


@app.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket) -> None:
    """Stream synthetic price ticks for all demo tickers."""
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/prices")

    rng = np.random.default_rng()
    data = _get_demo_data()
    # Initialise from last known close
    current_prices = {t: float(df["Close"].iloc[-1]) for t, df in data.items()}

    try:
        while True:
            # Simulate small price changes
            ticks = []
            for ticker, price in current_prices.items():
                change = price * rng.normal(0, 0.001)
                new_price = round(price + change, 2)
                current_prices[ticker] = new_price
                ticks.append({
                    "ticker": ticker,
                    "price": new_price,
                    "change_pct": round(change / price * 100, 4),
                    "timestamp": datetime.utcnow().isoformat(),
                })

            await websocket.send_json({"data": ticks, "status": "ok"})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
