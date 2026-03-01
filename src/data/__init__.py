"""Data module — fetching, cleaning, and synthetic generation."""

from src.data.fetcher import create_fetcher, DataFetcher, SyntheticFetcher, YFinanceFetcher
from src.data.alpha_vantage_fetcher import AlphaVantageFetcher
from src.data.cleaner import DataCleaner, compute_returns
from src.data.synthetic import generate_synthetic_ohlcv, generate_multi_asset_data

__all__ = [
    "create_fetcher",
    "DataFetcher",
    "SyntheticFetcher",
    "YFinanceFetcher",
    "AlphaVantageFetcher",
    "DataCleaner",
    "compute_returns",
    "generate_synthetic_ohlcv",
    "generate_multi_asset_data",
]
