"""Monte Carlo simulation for price path projection.

Monte Carlo simulation generates many possible future price paths by
randomly sampling from the historical return distribution.  Each path
represents one plausible future trajectory of an asset's price.

This module uses **log returns** (not arithmetic returns) for multi-period
simulation.  The reason is mathematical:  arithmetic returns do not compound
correctly over multiple periods.  If an asset falls 50% and then rises 50%,
arithmetic compounding would suggest the price is unchanged, but the true
price is 75% of the original.  Log returns are additive over time -- their
sum can be exponentiated once to recover the final price -- which makes them
the correct basis for simulating cumulative price paths.

Method:
    1. Convert historical returns to log returns: ``ln(1 + r)``.
    2. Estimate ``mu`` (mean) and ``sigma`` (std) of the log return series.
    3. Draw random samples from ``N(mu, sigma)`` for each day and path.
    4. Cumulative-sum the samples and exponentiate to produce price paths
       normalised to start at 1.0.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_SIMULATIONS = 10_000


def simulate_paths(
    returns: np.ndarray,
    n_simulations: int = 1000,
    n_days: int = 252,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate price paths using parametric bootstrap of log returns.

    Args:
        returns: Historical daily returns as a 1-D array.
        n_simulations: Number of Monte Carlo paths to generate.
        n_days: Length of each simulated path in trading days.
        random_seed: Optional seed for reproducibility.

    Returns:
        Array of shape ``(n_days, n_simulations)`` where each column is a
        normalised price path starting at 1.0.  Multiply by the current
        price to obtain absolute values.

    Raises:
        ValueError: If *n_simulations* exceeds the safety limit (10 000).
    """
    if n_simulations > _MAX_SIMULATIONS:
        raise ValueError(
            f"n_simulations={n_simulations} exceeds maximum of {_MAX_SIMULATIONS} "
            "(prevent out-of-memory)"
        )

    if random_seed is not None:
        np.random.seed(random_seed)

    # Protect against log(0) -- clip returns so 1 + r > 0
    clipped = np.clip(returns, -0.9999, None)
    log_returns = np.log(1 + clipped)

    mu = float(np.mean(log_returns))
    sigma = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0

    # Draw random log returns: shape (n_days, n_simulations)
    simulated_log = np.random.normal(
        loc=mu, scale=sigma, size=(n_days, n_simulations),
    )

    # Cumulative sum then exponentiate to get price paths
    cum_log = np.cumsum(simulated_log, axis=0)
    paths = np.exp(cum_log)

    # Normalise so paths[0, :] == 1.0
    # (first row is exp of one day's return -- we want paths starting at 1.0)
    # Prepend a row of ones and remove the last row to keep shape
    ones = np.ones((1, n_simulations))
    paths = np.vstack([ones, paths[:-1, :]])  # shift down, keeping shape

    # Actually, cleaner: rebuild from scratch with day-0 = 0.0 in cumsum
    # Re-do: insert a zero row at the top of simulated_log
    zero_row = np.zeros((1, n_simulations))
    padded_log = np.vstack([zero_row, simulated_log])  # shape (n_days+1, n_sim)
    cum_log_full = np.cumsum(padded_log, axis=0)
    paths = np.exp(cum_log_full[:n_days, :])  # take first n_days rows

    return paths


def compute_percentile_paths(
    paths: np.ndarray,
    percentiles: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute percentile envelopes across simulated paths.

    Args:
        paths: Array of shape ``(n_days, n_simulations)``.
        percentiles: List of percentile values (default ``[5, 50, 95]``).

    Returns:
        Dict mapping ``"p5"``, ``"p50"``, ``"p95"`` (etc.) to 1-D arrays
        of shape ``(n_days,)``.
    """
    if percentiles is None:
        percentiles = [5, 50, 95]

    result: dict[str, np.ndarray] = {}
    for p in percentiles:
        key = f"p{int(p)}"
        result[key] = np.percentile(paths, p, axis=1)
    return result


def simulate_dca(
    returns: np.ndarray,
    monthly_contribution: float,
    n_months: int,
    starting_value: float = 0.0,
    n_simulations: int = 1000,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate dollar-cost averaging into an asset.

    Each month the contribution is added at the start of the month and
    subject to that month's full simulated return.

    Args:
        returns: Historical daily returns as a 1-D array.
        monthly_contribution: Fixed amount added each month.
        n_months: Number of months to simulate.
        starting_value: Existing position value at simulation start.
        n_simulations: Number of Monte Carlo paths.
        random_seed: Optional seed for reproducibility.

    Returns:
        Array of shape ``(n_months, n_simulations)`` -- portfolio value
        at the end of each month.

    Raises:
        ValueError: If *n_simulations* exceeds the safety limit.
    """
    if n_simulations > _MAX_SIMULATIONS:
        raise ValueError(
            f"n_simulations={n_simulations} exceeds maximum of {_MAX_SIMULATIONS} "
            "(prevent out-of-memory)"
        )

    if random_seed is not None:
        np.random.seed(random_seed)

    # Resample daily returns to monthly by compounding 21 trading days
    clipped = np.clip(returns, -0.9999, None)
    log_daily = np.log(1 + clipped)
    mu_daily = float(np.mean(log_daily))
    sigma_daily = float(np.std(log_daily, ddof=1)) if len(log_daily) > 1 else 0.0

    # Monthly log return ~ N(mu_daily * 21, sigma_daily * sqrt(21))
    mu_monthly = mu_daily * 21
    sigma_monthly = sigma_daily * np.sqrt(21)

    # Simulate monthly returns: shape (n_months, n_simulations)
    monthly_log_returns = np.random.normal(
        loc=mu_monthly, scale=sigma_monthly, size=(n_months, n_simulations),
    )
    monthly_returns = np.exp(monthly_log_returns) - 1  # convert back to simple returns

    # Build portfolio values month by month
    portfolio = np.zeros((n_months, n_simulations))

    for m in range(n_months):
        if m == 0:
            prev_value = starting_value
        else:
            prev_value = portfolio[m - 1, :]

        # Add contribution at start of month, then apply month's return
        value_with_contribution = prev_value + monthly_contribution
        portfolio[m, :] = value_with_contribution * (1 + monthly_returns[m, :])

    return portfolio


def estimate_target_date(
    paths: np.ndarray,
    target_value: float,
    starting_month: int = 0,
) -> dict:
    """Estimate when a DCA portfolio reaches a target value.

    Args:
        paths: Array from :func:`simulate_dca`, shape ``(n_months, n_simulations)``.
        target_value: The portfolio value to reach.
        starting_month: Month offset for date labelling.

    Returns:
        Dict with keys:
        - ``p50_months``: Median months to reach target (``None`` if not reached).
        - ``p90_probability``: Probability of reaching target within *n_months*.
        - ``p50_date_label``: Human-readable date label (e.g. ``"Dec 2028"``).
    """
    n_months, n_simulations = paths.shape

    # For each simulation, find the first month where value >= target
    first_hit = np.full(n_simulations, np.nan)
    for sim in range(n_simulations):
        hits = np.where(paths[:, sim] >= target_value)[0]
        if len(hits) > 0:
            first_hit[sim] = hits[0]

    reached = ~np.isnan(first_hit)
    n_reached = int(np.sum(reached))
    probability = n_reached / n_simulations

    if n_reached == 0:
        return {
            "p50_months": None,
            "p90_probability": probability,
            "p50_date_label": "Not reached",
        }

    p50_months = int(np.nanmedian(first_hit))

    # Build a date label from now + p50_months
    target_date = datetime.now() + timedelta(days=30 * (starting_month + p50_months))
    date_label = target_date.strftime("%b %Y")

    return {
        "p50_months": p50_months,
        "p90_probability": round(probability, 4),
        "p50_date_label": date_label,
    }
