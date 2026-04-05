"""Return calculation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def simple_return(prices: ArrayLike) -> NDArray[np.float64]:
    """Compute simple (arithmetic) period returns from a price series."""
    prices = np.asarray(prices, dtype=np.float64)
    return np.diff(prices) / prices[:-1]


def log_return(prices: ArrayLike) -> NDArray[np.float64]:
    """Compute logarithmic period returns from a price series."""
    prices = np.asarray(prices, dtype=np.float64)
    return np.diff(np.log(prices))


def cumulative_returns(returns: ArrayLike) -> NDArray[np.float64]:
    """Compute cumulative returns from a series of simple returns."""
    returns = np.asarray(returns, dtype=np.float64)
    return np.cumprod(1.0 + returns) - 1.0
