"""Shared test fixtures for the quant-stack test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic import generate_synthetic_ohlcv, generate_multi_asset_data


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Single-ticker synthetic OHLCV data, ~5 years."""
    return generate_synthetic_ohlcv("TEST", days=1260, seed=42)


@pytest.fixture
def multi_asset_data() -> dict[str, pd.DataFrame]:
    """Multi-ticker synthetic data for portfolio tests."""
    tickers = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E"]
    return generate_multi_asset_data(tickers, days=1260, seed=42)


@pytest.fixture
def sample_config() -> dict:
    """Minimal config for testing (no file dependency)."""
    return {
        "general": {
            "base_currency": "GBP",
            "timezone": "Europe/London",
            "log_level": "WARNING",
            "data_dir": "data",
            "random_seed": 42,
        },
        "universe": {
            "name": "test",
            "tickers": ["TEST_A", "TEST_B", "TEST_C"],
            "benchmark": "^TEST",
        },
        "data": {
            "source": "synthetic",
            "start_date": "2020-01-01",
            "end_date": None,
            "interval": "1d",
            "fields": ["Open", "High", "Low", "Close", "Volume"],
            "adjust_prices": True,
            "output_format": "parquet",
        },
        "features": {
            "technical": {
                "sma_windows": [5, 20, 50],
                "ema_windows": [12, 26],
                "rsi_window": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_window": 20,
                "bb_num_std": 2.0,
                "atr_window": 14,
                "volatility_windows": [21, 63],
            },
            "returns": {
                "windows": [1, 5, 21],
                "log_returns": True,
            },
        },
        "portfolio": {
            "default_method": "mean_variance",
            "methods": [
                "mean_variance",
                "min_cvar",
                "risk_parity",
                "equal_weight",
            ],
            "constraints": {
                "max_weight": 0.20,
                "min_weight": 0.00,
            },
            "risk_free_rate": 0.045,
            "covariance_method": "ledoit",
            "risk": {
                "confidence_level": 0.95,
                "correlation_flag_threshold": 0.85,
            },
        },
        "backtest": {
            "initial_capital": 100000,
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
            "strategies": {
                "mean_reversion": {
                    "rsi_lower": 30,
                    "rsi_upper": 70,
                },
                "momentum": {
                    "sma_window": 50,
                },
            },
        },
        "models": {
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 3,
                "min_samples_leaf": 5,
            },
            "gradient_boosting": {
                "n_estimators": 20,
                "max_depth": 2,
                "learning_rate": 0.1,
            },
            "evaluation": {
                "n_splits": 5,
                "gap": 5,
                "min_train_size": 252,
            },
        },
        "execution": {
            "mode": "paper",
            "broker": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1,
                "timeout": 10,
                "max_retries": 3,
                "retry_delay": 2,
            },
            "risk": {
                "max_trade_pct": 0.10,
                "max_daily_trades": 20,
            },
        },
    }


@pytest.fixture
def ohlcv_with_missing(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV data with injected missing values for testing cleaner."""
    df = sample_ohlcv.copy()
    rng = np.random.default_rng(99)
    # Inject ~2% missing data
    mask = rng.random(df.shape) < 0.02
    df = df.mask(mask)
    # Ensure index stays clean
    df.index = sample_ohlcv.index
    return df


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """50-row OHLCV DataFrame where every price is constant at 100."""
    n = 50
    dates = pd.bdate_range(start="2024-01-02", periods=n, freq="B")
    price = np.full(n, 100.0)
    return pd.DataFrame(
        {
            "Open": price.copy(),
            "High": price.copy(),
            "Low": price.copy(),
            "Close": price.copy(),
            "Volume": np.full(n, 10_000, dtype=int),
        },
        index=dates,
    )


@pytest.fixture
def single_row_ohlcv() -> pd.DataFrame:
    """Single-row OHLCV DataFrame."""
    dates = pd.bdate_range(start="2024-01-02", periods=1, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [102.0],
            "Volume": [50_000],
        },
        index=dates,
    )
