"""Base signal class and result dataclass for smart-dca signals.

All concrete signals inherit from BaseSignal and produce SignalResult
instances with scores clamped to [0.0, 1.0].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value to [low, high] range."""
    return max(low, min(high, value))


@dataclass
class SignalResult:
    """Result of a signal computation.

    Attributes:
        name: Signal identifier string.
        score: Signal strength from 0.0 (no signal) to 1.0 (strongest).
        weight: Weight from config for weighted averaging.
        reason: Human-readable explanation of the score.
        timestamp: UTC-aware datetime of computation.
    """

    name: str
    score: float
    weight: float
    reason: str
    timestamp: datetime


class BaseSignal(ABC):
    """Abstract base class for all smart-dca signals.

    Concrete signals must implement compute() and name. The compute method
    must never raise — return score=0.0 with reason on failure.

    Args:
        config: Full settings dict from config_loader.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def compute(self, symbol: str, **kwargs: object) -> SignalResult:
        """Compute signal for a given symbol.

        Must never raise. On internal failure, return SignalResult with
        score=0.0 and reason explaining the failure.

        Args:
            symbol: Trading pair (e.g. 'BTCGBP').
            **kwargs: Signal-specific extra arguments.

        Returns:
            SignalResult with score clamped to [0.0, 1.0].
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Signal identifier string (e.g. 'seasonality')."""
