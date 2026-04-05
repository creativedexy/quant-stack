"""Option pricing models."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]


def _d1(s: float, k: float, t: float, r: float, sigma: float) -> float:
    return (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))


def _d2(s: float, k: float, t: float, r: float, sigma: float) -> float:
    return _d1(s, k, t, r, sigma) - sigma * math.sqrt(t)


def black_scholes_call(
    s: float, k: float, t: float, r: float, sigma: float
) -> float:
    """Price a European call option using the Black-Scholes formula.

    Parameters
    ----------
    s : Spot price
    k : Strike price
    t : Time to expiry in years
    r : Risk-free interest rate (annualised, continuous)
    sigma : Volatility (annualised)
    """
    d1 = _d1(s, k, t, r, sigma)
    d2 = _d2(s, k, t, r, sigma)
    return float(s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2))


def black_scholes_put(
    s: float, k: float, t: float, r: float, sigma: float
) -> float:
    """Price a European put option using the Black-Scholes formula.

    Parameters
    ----------
    s : Spot price
    k : Strike price
    t : Time to expiry in years
    r : Risk-free interest rate (annualised, continuous)
    sigma : Volatility (annualised)
    """
    d1 = _d1(s, k, t, r, sigma)
    d2 = _d2(s, k, t, r, sigma)
    return float(k * math.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1))
