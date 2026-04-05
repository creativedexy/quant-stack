"""Data feed modules for smart-dca."""

from src.data.funding_feed import BinanceFundingFeed
from src.data.price_feed import BinancePriceFeed
from src.data.sentiment_feed import FearGreedFeed

__all__ = ["BinanceFundingFeed", "BinancePriceFeed", "FearGreedFeed"]
