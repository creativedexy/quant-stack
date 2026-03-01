"""Pydantic models for API request/response shapes.

All responses follow a consistent envelope:
    {"data": ..., "timestamp": "...", "status": "ok"}
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ─────────────────────────────────────────────
# Response envelope
# ─────────────────────────────────────────────


class ApiResponse(BaseModel, Generic[T]):
    """Standard wrapper for every API response."""

    data: T
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "ok"


class ErrorDetail(BaseModel):
    """Body returned on error responses."""

    detail: str


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────


class HealthData(BaseModel):
    service: str = "quant-stack"
    version: str
    uptime_seconds: float
    modules: dict[str, str]


# ─────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────


class PortfolioOverview(BaseModel):
    total_value: float
    daily_pnl: float
    daily_return_pct: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_positions: int
    cash: float
    currency: str = "GBP"


class Position(BaseModel):
    ticker: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealised_pnl: float
    weight_pct: float


class EquityCurvePoint(BaseModel):
    date: str
    value: float
    drawdown_pct: float


class RiskMetrics(BaseModel):
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown_pct: float
    volatility_annual: float
    beta: float | None = None
    correlation_to_benchmark: float | None = None


# ─────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────


class StrategyInfo(BaseModel):
    name: str
    description: str
    status: str  # "active" | "inactive" | "backtest_only"
    last_run: str | None = None
    total_return_pct: float | None = None
    sharpe_ratio: float | None = None


class BacktestResult(BaseModel):
    strategy: str
    start_date: str
    end_date: str
    total_return_pct: float
    annual_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    equity_curve: list[EquityCurvePoint]


class StrategyComparison(BaseModel):
    strategies: list[StrategyInfo]


# ─────────────────────────────────────────────
# Prices
# ─────────────────────────────────────────────


class OHLCVPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class LatestPrice(BaseModel):
    ticker: str
    price: float
    change_pct: float
    volume: int
    timestamp: str


class PriceHistory(BaseModel):
    ticker: str
    interval: str
    data: list[OHLCVPoint]


class FeaturePoint(BaseModel):
    date: str
    values: dict[str, float | None]


class FeatureData(BaseModel):
    ticker: str
    features: list[str]
    data: list[FeaturePoint]


# ─────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────


class ExecutionStatus(BaseModel):
    broker: str
    connected: bool
    mode: str  # "paper" | "live"
    pending_orders: int
    last_heartbeat: str | None = None


class RebalancePlanRequest(BaseModel):
    target_weights: dict[str, float]
    total_value: float | None = None


class RebalanceOrder(BaseModel):
    ticker: str
    side: str  # "buy" | "sell"
    quantity: float
    estimated_cost: float
    current_weight_pct: float
    target_weight_pct: float


class RebalancePlan(BaseModel):
    orders: list[RebalanceOrder]
    estimated_commission: float
    estimated_slippage: float


class ExecutePlanRequest(BaseModel):
    orders: list[RebalanceOrder]
    mode: str = "paper"


class ExecuteResult(BaseModel):
    submitted: int
    mode: str
    order_ids: list[str]


class ExecutionRecord(BaseModel):
    order_id: str
    ticker: str
    side: str
    quantity: float
    price: float
    commission: float
    timestamp: str
    status: str  # "filled" | "cancelled" | "pending"


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────


class PipelineStageStatus(BaseModel):
    name: str
    status: str  # "idle" | "running" | "completed" | "failed"
    last_run: str | None = None
    message: str | None = None


class PipelineStatus(BaseModel):
    running: bool
    stages: list[PipelineStageStatus]


class PipelineRunRequest(BaseModel):
    stages: list[str] | None = None  # None = run all
    tickers: list[str] | None = None
    source: str = "synthetic"


class PipelineRunResult(BaseModel):
    job_id: str
    stages_queued: list[str]
    message: str
