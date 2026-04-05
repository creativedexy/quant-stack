"""Shared fixtures for the deployment engine test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure src is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.deployment import (
    Candidate,
    EnrichedCandidate,
    ScoredCandidate,
    SizedPosition,
    TradeOrder,
)


@pytest.fixture
def deployment_config() -> dict:
    """Minimal deployment config for testing (no file dependency)."""
    return {
        "evaluation": {
            "weights": {
                "growth_score": 0.40,
                "diversification_score": 0.25,
                "risk_adjusted_score": 0.20,
                "income_score": 0.15,
            },
            "thresholds": {
                "minimum_composite_score": 0.55,
                "maximum_position_pct": 0.25,
                "minimum_position_gbp": 50,
                "maximum_correlation_to_existing": 0.60,
            },
        },
        "scanner": {
            "universes": {
                "us_growth": {
                    "source": "yfinance",
                    "filter": {
                        "market_cap_min_usd": 10_000_000_000,
                        "sector_exclude": ["Energy", "Consumer Staples"],
                        "exchange": ["NYSE", "NASDAQ"],
                    },
                    "max_candidates": 20,
                },
                "sector_etfs": {
                    "source": "yfinance",
                    "tickers": ["SMH", "SOXX", "XLK"],
                    "max_candidates": 12,
                },
                "contrarian_value": {
                    "source": "yfinance",
                    "filter": {
                        "pe_ratio_max": 12,
                        "dividend_yield_min": 0.03,
                        "price_to_book_max": 1.5,
                        "market_cap_min_usd": 5_000_000_000,
                        "sector_include": [
                            "Industrials",
                            "Materials",
                            "Utilities",
                            "Real Estate",
                        ],
                    },
                    "max_candidates": 10,
                },
            },
            "global_exclusions": {"structural": []},
            "min_avg_daily_volume": 500_000,
            "verify_ib_tradeable": False,
        },
        "sizing": {
            "method": "fractional_kelly",
            "kelly_fraction": 0.25,
            "constraints": {
                "max_position_pct": 0.25,
                "min_position_gbp": 50,
                "max_positions": 8,
                "max_single_sector_pct": 0.35,
                "cash_reserve_pct": 0.05,
            },
            "scaling": {
                "tier_1": {
                    "label": "GBP 0 - 1,000",
                    "min_capital": 0,
                    "max_capital": 1000,
                    "max_positions": 5,
                    "min_position_gbp": 50,
                    "prefer_etfs": True,
                },
                "tier_2": {
                    "label": "GBP 1,000 - 10,000",
                    "min_capital": 1000,
                    "max_capital": 10000,
                    "max_positions": 8,
                    "min_position_gbp": 200,
                    "prefer_etfs": False,
                },
                "tier_3": {
                    "label": "GBP 10,000 - 100,000",
                    "min_capital": 10000,
                    "max_capital": 100000,
                    "max_positions": 15,
                    "min_position_gbp": 500,
                    "prefer_etfs": False,
                },
                "tier_4": {
                    "label": "GBP 100,000+",
                    "min_capital": 100000,
                    "max_capital": 999_999_999,
                    "max_positions": 25,
                    "min_position_gbp": 2000,
                    "prefer_etfs": False,
                },
            },
        },
        "execution": {
            "mode": "LIVE",
            "confirm_live": True,
            "order_type_default": "LIMIT",
            "limit_price_buffer_pct": 0.005,
            "market_order_volume_threshold": 1_000_000,
        },
        "risk": {
            "daily_loss_limit_pct": 0.10,
            "price_drift_max_pct": 0.02,
            "slippage_warning_pct": 0.005,
            "sdrt_rate": 0.005,
            "commission_estimate_gbp": 1.00,
        },
        "opus": {
            "model": "claude-opus-4-20250514",
            "max_tokens": 4096,
            "timeout": 120,
            "max_retries": 3,
            "cost_warning_gbp": 5.00,
        },
        "memory": {
            "path": "data/deployment/memory.json",
            "cycles_dir": "data/deployment/cycles",
            "cooldown_days": 7,
            "max_cycles_detail": 20,
        },
        "pipeline": {
            "max_retries": 3,
            "approval_timeout_minutes": 1440,
        },
        "fx": {
            "default_gbp_usd": 1.27,
            "default_gbp_eur": 1.17,
        },
    }


@pytest.fixture
def synthetic_candidates() -> list[Candidate]:
    """Five fake candidates spanning different sectors and geographies."""
    return [
        Candidate(
            ticker="NVDA",
            name="NVIDIA Corporation",
            exchange="NASDAQ",
            sector="Technology",
            market_cap_usd=2_800_000_000_000,
            source_universe="us_growth",
            avg_daily_volume=45_000_000,
            currency="USD",
            asset_type="stock",
        ),
        Candidate(
            ticker="SMH",
            name="VanEck Semiconductor ETF",
            exchange="NASDAQ",
            sector="Technology",
            market_cap_usd=25_000_000_000,
            source_universe="sector_etfs",
            avg_daily_volume=8_000_000,
            currency="USD",
            asset_type="etf",
        ),
        Candidate(
            ticker="IEMG",
            name="iShares Core MSCI Emerging Markets ETF",
            exchange="NYSE",
            sector="Diversified",
            market_cap_usd=70_000_000_000,
            source_universe="sector_etfs",
            avg_daily_volume=15_000_000,
            currency="USD",
            asset_type="etf",
        ),
        Candidate(
            ticker="RIO",
            name="Rio Tinto plc",
            exchange="NYSE",
            sector="Materials",
            market_cap_usd=100_000_000_000,
            source_universe="contrarian_value",
            avg_daily_volume=3_000_000,
            currency="USD",
            asset_type="stock",
        ),
        Candidate(
            ticker="NEE",
            name="NextEra Energy Inc",
            exchange="NYSE",
            sector="Utilities",
            market_cap_usd=150_000_000_000,
            source_universe="contrarian_value",
            avg_daily_volume=6_000_000,
            currency="USD",
            asset_type="stock",
        ),
    ]


@pytest.fixture
def synthetic_enriched(synthetic_candidates) -> list[EnrichedCandidate]:
    """Enriched versions of synthetic candidates with realistic data."""
    enriched = []
    for c in synthetic_candidates[:3]:
        enriched.append(
            EnrichedCandidate.from_candidate(
                c,
                fundamentals={
                    "pe_ratio": 28.5,
                    "forward_pe": 24.0,
                    "peg_ratio": 1.2,
                    "revenue_growth_yoy": 0.22,
                    "earnings_growth_yoy": 0.35,
                    "dividend_yield": 0.004,
                    "payout_ratio": 0.10,
                    "price_to_book": 45.0,
                    "price_to_sales": 30.0,
                    "analyst_buy": 35,
                    "analyst_hold": 8,
                    "analyst_sell": 2,
                    "fifty_two_week_high": 150.0,
                    "fifty_two_week_low": 80.0,
                },
                technicals={
                    "rsi_14": 62.0,
                    "macd_line": 2.5,
                    "macd_signal": 1.8,
                    "macd_histogram": 0.7,
                    "bb_upper": 145.0,
                    "bb_lower": 120.0,
                    "bb_middle": 132.5,
                    "sma_50": 130.0,
                    "sma_200": 115.0,
                    "atr_14": 4.5,
                    "volume_trend": "rising",
                },
                news_sentiment={
                    "score": 0.65,
                    "available": True,
                    "headline_count": 15,
                },
                gbp_price=105.0,
            )
        )
    return enriched


@pytest.fixture
def mock_opus_response() -> dict:
    """Valid JSON matching the Opus output schema."""
    return {
        "candidates": [
            {
                "ticker": "NVDA",
                "scores": {
                    "growth": 0.88,
                    "diversification": 0.72,
                    "risk_adjusted": 0.62,
                    "income": 0.05,
                    "composite": 0.658,
                },
                "reasoning": {
                    "growth": "Strong AI capex cycle with multiple catalysts.",
                    "diversification": "US tech absent from current holdings.",
                    "risk_adjusted": "High vol but strong momentum offsets.",
                    "income": "Minimal dividend, buyback programme instead.",
                },
                "conviction": "HIGH",
                "risks": [
                    "Valuation extended at 35x forward",
                    "China export controls",
                ],
                "position_suggestion": "2-3% of capital as starter position",
            }
        ],
        "portfolio_level_commentary": "Portfolio is UK-heavy, needs US tech exposure.",
        "regime_assessment": "Risk-on, late-cycle expansion.",
    }


@pytest.fixture
def fresh_memory() -> dict:
    """Empty memory state for testing."""
    return {
        "version": "1.0.0",
        "last_cycle_id": None,
        "last_cycle_timestamp": None,
        "capital_available_gbp": 0.0,
        "capital_deployed_gbp": 0.0,
        "positions_opened": [],
        "universe_state": {
            "included_tickers": [],
            "excluded_tickers_structural": [],
            "rejected_candidates": [],
            "watchlist": [],
        },
        "evaluation_history": [],
        "config_overrides": {},
        "error_log": [],
        "opus_usage": [],
        "sector_trends": {},
    }


@pytest.fixture
def synthetic_portfolio() -> list[dict]:
    """Realistic existing portfolio for testing diversification scoring."""
    return [
        {
            "ticker": "VUSA.L",
            "name": "Vanguard S&P 500 UCITS ETF",
            "weight": 0.20,
            "sector": "Diversified",
            "geography": "US",
            "asset_type": "etf",
            "market_value_gbp": 100.0,
            "daily_return": 0.005,
        },
        {
            "ticker": "CNDX.L",
            "name": "iShares NASDAQ 100 UCITS ETF",
            "weight": 0.15,
            "sector": "Technology",
            "geography": "US",
            "asset_type": "etf",
            "market_value_gbp": 75.0,
            "daily_return": -0.002,
        },
        {
            "ticker": "VHYL.L",
            "name": "Vanguard FTSE All-World High Dividend Yield",
            "weight": 0.12,
            "sector": "Diversified",
            "geography": "Global",
            "asset_type": "etf",
            "market_value_gbp": 60.0,
            "daily_return": 0.001,
        },
        {
            "ticker": "ABBV",
            "name": "AbbVie Inc",
            "weight": 0.10,
            "sector": "Healthcare",
            "geography": "US",
            "asset_type": "stock",
            "market_value_gbp": 50.0,
            "daily_return": 0.008,
        },
        {
            "ticker": "BTI",
            "name": "British American Tobacco",
            "weight": 0.08,
            "sector": "Consumer Staples",
            "geography": "UK",
            "asset_type": "stock",
            "market_value_gbp": 40.0,
            "daily_return": -0.003,
        },
    ]
