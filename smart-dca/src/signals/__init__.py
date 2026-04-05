"""Signal modules for smart-dca."""

from src.signals.base import BaseSignal, SignalResult, clamp
from src.signals.funding_signal import FundingRateSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.scorer import CompositeScore, SignalScorer
from src.signals.seasonality import SeasonalitySignal
from src.signals.sentiment_signal import SentimentSignal

# Signal name constants
SEASONALITY = "seasonality"
MEAN_REVERSION = "mean_reversion"
FUNDING_RATE = "funding_rate"
SENTIMENT = "sentiment"

__all__ = [
    "BaseSignal",
    "CompositeScore",
    "SignalResult",
    "SignalScorer",
    "clamp",
    "FundingRateSignal",
    "MeanReversionSignal",
    "SeasonalitySignal",
    "SentimentSignal",
    "SEASONALITY",
    "MEAN_REVERSION",
    "FUNDING_RATE",
    "SENTIMENT",
]
