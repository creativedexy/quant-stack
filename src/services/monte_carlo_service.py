"""Monte Carlo projection service for the production UI.

Wraps :mod:`src.analysis.monte_carlo` functions and provides pre-computed
SVG coordinate strings ready for Jinja2 templates.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.monte_carlo import (
    compute_percentile_paths,
    estimate_target_date,
    simulate_dca,
    simulate_paths,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MonteCarloService:
    """Monte Carlo projections for chart overlays and DCA planning.

    Args:
        config: Application configuration dict.
        data_service: DataService instance for historical returns.
        market_data_service: MarketDataService instance for current prices.
    """

    def __init__(
        self,
        config: dict[str, Any],
        data_service: Any,
        market_data_service: Any,
    ) -> None:
        self._config = config
        self._data_service = data_service
        self._market_data_service = market_data_service

        mc_cfg = config.get("analysis", {}).get("monte_carlo", {})
        self._default_simulations: int = mc_cfg.get("default_simulations", 1000)
        self._max_simulations: int = mc_cfg.get("max_simulations", 10000)
        self._history_years: int = mc_cfg.get("return_history_years", 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_portfolio_projection(
        self,
        ticker: str,
        period_years: int = 1,
        n_simulations: int | None = None,
    ) -> dict[str, Any]:
        """Project future price paths for a single ticker.

        Args:
            ticker: Instrument ticker (e.g. ``"SHEL.L"``).
            period_years: Projection horizon in years.
            n_simulations: Override for number of simulations.

        Returns:
            Dict with keys: ``dates_forward``, ``p5``, ``p50``, ``p95``,
            ``current_price``, ``svg_points``.
        """
        if n_simulations is None:
            n_simulations = self._default_simulations
        n_simulations = min(n_simulations, self._max_simulations)

        returns = self._get_historical_returns(ticker)
        if returns is None or len(returns) < 20:
            return self._empty_projection(ticker)

        n_days = period_years * 252
        paths = simulate_paths(
            returns, n_simulations=n_simulations, n_days=n_days, random_seed=42,
        )

        # Scale by current price
        current_price = self._get_current_price(ticker)
        scaled_paths = paths * current_price

        pct = compute_percentile_paths(scaled_paths, [5, 50, 95])

        # Build forward date strings
        today = datetime.now()
        dates_forward = [
            (today + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n_days)
        ]

        p5_list = [round(float(v), 2) for v in pct["p5"]]
        p50_list = [round(float(v), 2) for v in pct["p50"]]
        p95_list = [round(float(v), 2) for v in pct["p95"]]

        return {
            "dates_forward": dates_forward,
            "p5": p5_list,
            "p50": p50_list,
            "p95": p95_list,
            "current_price": round(current_price, 2),
            "svg_points": {
                "p5": _to_svg_points(list(range(n_days)), p5_list),
                "p50": _to_svg_points(list(range(n_days)), p50_list),
                "p95": _to_svg_points(list(range(n_days)), p95_list),
            },
        }

    def run_dca_projection(
        self,
        ticker: str,
        monthly_contribution: float,
        n_months: int,
        starting_value: float,
        target_value: float,
        n_simulations: int | None = None,
    ) -> dict[str, Any]:
        """Project DCA portfolio growth with target date estimation.

        Args:
            ticker: Instrument ticker.
            monthly_contribution: Fixed monthly contribution amount.
            n_months: Number of months to simulate.
            starting_value: Existing position value.
            target_value: Target portfolio value.
            n_simulations: Override for number of simulations.

        Returns:
            Dict with keys: ``months``, ``p5``, ``p50``, ``p95``,
            ``target_date``, ``svg_points``, ``today_divider_x``.
        """
        if n_simulations is None:
            n_simulations = self._default_simulations
        n_simulations = min(n_simulations, self._max_simulations)

        returns = self._get_historical_returns(ticker)
        if returns is None or len(returns) < 20:
            return self._empty_dca()

        portfolio = simulate_dca(
            returns,
            monthly_contribution=monthly_contribution,
            n_months=n_months,
            starting_value=starting_value,
            n_simulations=n_simulations,
            random_seed=42,
        )

        target_info = estimate_target_date(portfolio, target_value)
        pct = compute_percentile_paths(portfolio, [5, 50, 95])

        months_list = list(range(n_months))
        p5_list = [round(float(v), 2) for v in pct["p5"]]
        p50_list = [round(float(v), 2) for v in pct["p50"]]
        p95_list = [round(float(v), 2) for v in pct["p95"]]

        # TODAY divider -- x coordinate at month 0 in the SVG
        # For DCA there is no historical section, so divider is at x=0
        today_divider_x = 40.0  # pad_x from _to_svg_points

        return {
            "months": months_list,
            "p5": p5_list,
            "p50": p50_list,
            "p95": p95_list,
            "target_date": target_info,
            "svg_points": {
                "p5": _to_svg_points(months_list, p5_list),
                "p50": _to_svg_points(months_list, p50_list),
                "p95": _to_svg_points(months_list, p95_list),
            },
            "today_divider_x": today_divider_x,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_historical_returns(self, ticker: str) -> np.ndarray | None:
        """Load daily returns from DataService (last N years)."""
        try:
            history_days = self._history_years * 252
            prices = self._data_service.get_prices(
                tickers=[ticker],
            )
            if prices.empty or ticker not in prices.columns:
                return None

            series = prices[ticker].dropna()
            series = series.tail(history_days)
            if len(series) < 20:
                return None

            returns = series.pct_change().dropna().values
            return returns
        except Exception as exc:
            logger.error(
                "Failed to load returns for %s: %s", ticker, exc,
            )
            return None

    def _get_current_price(self, ticker: str) -> float:
        """Get current price from MarketDataService."""
        try:
            intraday = self._market_data_service.get_intraday_prices(ticker)
            return float(intraday.get("current", 100.0))
        except Exception as exc:
            logger.warning(
                "Current price unavailable for %s, using 100.0: %s",
                ticker, exc,
            )
            return 100.0

    def _empty_projection(self, ticker: str) -> dict[str, Any]:
        """Return empty projection when data is insufficient."""
        return {
            "dates_forward": [],
            "p5": [],
            "p50": [],
            "p95": [],
            "current_price": 0.0,
            "svg_points": {"p5": "", "p50": "", "p95": ""},
        }

    def _empty_dca(self) -> dict[str, Any]:
        """Return empty DCA projection."""
        return {
            "months": [],
            "p5": [],
            "p50": [],
            "p95": [],
            "target_date": {
                "p50_months": None,
                "p90_probability": 0.0,
                "p50_date_label": "N/A",
            },
            "svg_points": {"p5": "", "p50": "", "p95": ""},
            "today_divider_x": 40.0,
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _to_svg_points(
    indices: list,
    values: list,
    width: int = 900,
    height: int = 220,
    pad_x: int = 40,
    pad_y: int = 10,
) -> str:
    """Convert index/value pairs into an SVG polyline points string.

    Matches the convention in ``web/routes/chart.py:_to_svg_points``.
    """
    pairs: list[tuple[int, float]] = []
    for i, v in enumerate(values):
        if v is not None:
            pairs.append((i, float(v)))

    if len(pairs) < 2:
        return ""

    n = len(values)
    vals = [p[1] for p in pairs]
    min_v = min(vals)
    max_v = max(vals)
    v_range = max_v - min_v if max_v != min_v else 1.0

    pts: list[str] = []
    for idx, val in pairs:
        x = pad_x + (idx / max(n - 1, 1)) * (width - 2 * pad_x)
        y = pad_y + (1 - (val - min_v) / v_range) * (height - 2 * pad_y)
        pts.append(f"{x:.1f},{y:.1f}")

    return " ".join(pts)
