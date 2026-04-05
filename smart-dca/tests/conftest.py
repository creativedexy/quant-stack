"""Shared test fixtures for smart-dca."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_settings() -> dict:
    """Return a minimal valid settings dict (no file I/O)."""
    return {
        "binance": {
            "testnet": True,
            "request_timeout_seconds": 10,
        },
        "schedule": {
            "evaluation_interval_minutes": 15,
            "weekly_reset_day": "monday",
            "weekly_reset_hour_utc": 9,
        },
        "scoring": {
            "buy_threshold": 65,
            "notify_threshold": 55,
            "signal_weights": {
                "seasonality": 0.20,
                "mean_reversion": 0.35,
                "funding_rate": 0.25,
                "sentiment": 0.20,
            },
        },
        "signals": {
            "mean_reversion_dip_pct": 4.0,
            "mean_reversion_window_hours": 24,
            "funding_rate_negative_threshold": -0.01,
            "sentiment_fear_threshold": 30,
            "seasonality_scores": {
                "peak_hours": [14, 15, 16, 17, 18],
                "low_hours": [0, 1, 2, 3, 4, 5],
            },
        },
        "risk": {
            "max_single_order_gbp": 50.0,
            "min_balance_buffer_gbp": 10.0,
            "max_daily_loss_pct": 5.0,
            "halt_on_loss": True,
        },
        "execution": {
            "state_file": "data/budget_state.json",
        },
        "monitor": {
            "db_path": "data/performance.db",
        },
    }


@pytest.fixture
def sample_assets() -> dict:
    """Return assets dict with one enabled and one disabled asset."""
    return {
        "assets": {
            "BTCGBP": {
                "weekly_allocation_gbp": 10.0,
                "min_order_gbp": 5.0,
                "enabled": True,
                "funding_proxy": "BTCUSDT",
            },
            "ETHGBP": {
                "weekly_allocation_gbp": 10.0,
                "min_order_gbp": 5.0,
                "enabled": False,
                "funding_proxy": "ETHUSDT",
            },
            "XRPGBP": {
                "weekly_allocation_gbp": 5.0,
                "min_order_gbp": 2.0,
                "enabled": True,
                "funding_proxy": None,
            },
        }
    }


@pytest.fixture
def mock_binance_client() -> MagicMock:
    """Return a MagicMock shaped like the python-binance Client."""
    client = MagicMock()
    client.get_order_book.return_value = {
        "bids": [["49900.00", "0.5"]],
        "asks": [["50100.00", "0.5"]],
    }
    client.get_symbol_ticker.return_value = {"price": "50000.00"}
    client.get_klines.return_value = []
    client.get_account.return_value = {
        "balances": [
            {"asset": "GBP", "free": "100.00", "locked": "0.00"},
            {"asset": "BTC", "free": "0.001", "locked": "0.00"},
        ]
    }
    client.create_order.return_value = {
        "orderId": "12345",
        "symbol": "BTCGBP",
        "status": "NEW",
        "price": "50000.00",
        "executedQty": "0.00020000",
        "cummulativeQuoteQty": "10.00",
        "transactTime": 1700000000000,
    }
    client.get_order.return_value = {
        "orderId": "12345",
        "symbol": "BTCGBP",
        "status": "FILLED",
        "price": "50000.00",
        "executedQty": "0.00020000",
        "cummulativeQuoteQty": "10.00",
        "time": 1700000000000,
    }
    client.cancel_order.return_value = {
        "orderId": "12345",
        "symbol": "BTCGBP",
        "status": "CANCELLED",
    }
    client.get_symbol_info.return_value = {
        "filters": [
            {"filterType": "LOT_SIZE", "minQty": "0.00001000", "stepSize": "0.00001000"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "1.00"},
        ]
    }
    # Futures client mock
    client.futures_funding_rate.return_value = [
        {
            "symbol": "BTCUSDT",
            "fundingRate": "-0.00010000",
            "fundingTime": 1700000000000,
        }
    ]
    return client


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Return a DataFrame of 48 rows of synthetic hourly OHLCV data.

    Mean price ~50000, std ~500. Includes a deliberate 5% dip at row 36.
    """
    np.random.seed(42)
    n = 48
    base_price = 50000.0
    noise = np.random.normal(0, 500, n)
    closes = base_price + np.cumsum(noise * 0.1)

    # Inject a 5% dip at row 36
    closes[36] = closes[35] * 0.95

    opens = closes + np.random.normal(0, 50, n)
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 100, n))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 100, n))
    volumes = np.random.uniform(10, 100, n)

    timestamps = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=n,
        freq="h",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=timestamps,
    )
