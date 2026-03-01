"""Feature engineering — technical indicators, alpha factors, and pipeline."""

from src.features.technical import (
    # Primary compute_* API
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_returns,
    compute_volatility,
    compute_all_technical,
    # Backward-compatible add_* API
    add_sma,
    add_ema,
    add_rsi,
    add_macd,
    add_bollinger_bands,
    add_atr,
    add_all_indicators,
)
from src.features.pipeline import FeaturePipeline

__all__ = [
    "compute_sma",
    "compute_ema",
    "compute_rsi",
    "compute_macd",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_returns",
    "compute_volatility",
    "compute_all_technical",
    "add_sma",
    "add_ema",
    "add_rsi",
    "add_macd",
    "add_bollinger_bands",
    "add_atr",
    "add_all_indicators",
    "FeaturePipeline",
]
