"""Shared fixtures for persona tests."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest


@pytest.fixture()
def sample_config() -> dict:
    """Minimal config with persona definitions."""
    return {
        "general": {"data_dir": "data", "base_currency": "GBP"},
        "personas": {
            "enabled": True,
            "model": "claude-sonnet-4-6-20250514",
            "brief_schedule": "18:00",
            "event_check_interval_minutes": 30,
            "definitions": {
                "core": {
                    "name": "Core Portfolio",
                    "type": "always_on",
                    "capital_pct": 0.80,
                    "strategy": "momentum",
                    "strategy_config": {"sma_window": 50},
                    "universe": ["VUSA.L", "HMWO.L"],
                    "risk": {"max_drawdown": 0.15, "max_weight": 0.25},
                    "signal_method": "equal_weight_filtered",
                    "mandate": "You are the Core Portfolio analyst.",
                },
                "crisis": {
                    "name": "Crisis Alpha",
                    "type": "event_triggered",
                    "capital_pct": 0.10,
                    "strategy": "mean_reversion",
                    "strategy_config": {"oversold_threshold": 25},
                    "universe": ["VUSA.L"],
                    "risk": {"max_drawdown": 0.25, "max_weight": 0.40},
                    "activation": {
                        "detector": "drawdown",
                        "params": {
                            "drawdown_threshold": 0.10,
                            "lookback_days": 21,
                        },
                    },
                    "mandate": "You are Crisis Alpha.",
                },
                "crypto_winter": {
                    "name": "Crypto Winter",
                    "type": "event_triggered",
                    "capital_pct": 0.05,
                    "strategy": "dca",
                    "universe": ["BTC-USD"],
                    "risk": {"max_drawdown": 0.40},
                    "activation": {
                        "detector": "crypto_drawdown",
                        "params": {
                            "index_ticker": "BTC-USD",
                            "drawdown_threshold": 0.30,
                            "lookback_days": 60,
                        },
                    },
                    "mandate": "You are the Crypto Winter analyst.",
                },
            },
        },
    }


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Generate a simple OHLCV DataFrame for testing."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, 100))
    return pd.DataFrame(
        {
            "Open": close - rng.uniform(0, 1, 100),
            "High": close + rng.uniform(0, 2, 100),
            "Low": close - rng.uniform(0, 2, 100),
            "Close": close,
            "Volume": rng.integers(1000, 10000, 100),
        },
        index=dates,
    )


@pytest.fixture()
def drawdown_ohlcv() -> pd.DataFrame:
    """OHLCV data with a significant drawdown for detector testing."""
    dates = pd.bdate_range("2024-01-02", periods=60, freq="B")
    # Price rises to 120, then drops to 95 (21% drawdown from peak)
    close = np.concatenate([
        np.linspace(100, 120, 30),
        np.linspace(120, 95, 30),
    ])
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": rng.integers(1000, 10000, 60),
        },
        index=dates,
    )
