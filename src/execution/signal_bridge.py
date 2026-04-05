"""Bridge between strategy signals and portfolio target weights.

Converts discrete trading signals ({-1, 0, 1}) from one or more strategies
into target portfolio weights suitable for the :class:`OrderManagementSystem`.

Three weighting methods are supported:

* ``equal_weight_filtered`` -- equal weight among long signals, zero the rest.
* ``signal_weighted`` -- use signals as expected-return direction input
  to :class:`PortfolioOptimiser`.
* ``long_short`` -- positive weight for longs, negative for shorts
  (market-neutral).

Usage::

    from src.execution.signal_bridge import SignalBridge
    bridge = SignalBridge(config=cfg)
    weights = bridge.signals_to_weights(signals, returns)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.portfolio.optimiser import PortfolioOptimiser
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_METHODS = frozenset({
    "equal_weight_filtered",
    "signal_weighted",
    "long_short",
})

# Scale factor: signals (+/-1) get multiplied by this to produce
# expected-return estimates for the optimiser.  Small enough to keep
# the optimiser numerically stable.
_SIGNAL_SCALE = 0.01


class SignalBridge:
    """Convert strategy signals into portfolio target weights.

    Args:
        config: Full project configuration dict.
        optimiser: Optional pre-built :class:`PortfolioOptimiser`.  Created
            automatically from *config* when ``None``.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        optimiser: PortfolioOptimiser | None = None,
    ) -> None:
        self._config = config or {}
        self._optimiser = optimiser or PortfolioOptimiser(config=config)

        strat_cfg = self._config.get("strategies", {})
        self._default_method: str = strat_cfg.get(
            "signal_method", "equal_weight_filtered",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def signals_to_weights(
        self,
        signals: dict[str, int],
        returns: pd.DataFrame,
        method: str | None = None,
    ) -> pd.Series:
        """Convert discrete signals to portfolio target weights.

        Args:
            signals: Mapping of ticker -> signal value where each value
                is one of {-1, 0, 1}.
            returns: Historical returns DataFrame for risk estimation.
                Columns are tickers, index is a DatetimeIndex.
            method: Weight computation method.  One of
                ``equal_weight_filtered``, ``signal_weighted``,
                ``long_short``.  Defaults to the config value.

        Returns:
            :class:`pd.Series` of target weights (ticker -> float).

        Raises:
            ValueError: If *method* is not recognised or *signals* is empty.
        """
        method = method or self._default_method

        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown signal method '{method}'. "
                f"Supported: {sorted(_SUPPORTED_METHODS)}",
            )

        if not signals:
            raise ValueError("No signals provided")

        if method == "equal_weight_filtered":
            return self._equal_weight_filtered(signals)
        if method == "signal_weighted":
            return self._signal_weighted(signals, returns)
        return self._long_short(signals)

    def aggregate_strategy_signals(
        self,
        strategy_signals: dict[str, dict[str, int]],
        weights: dict[str, float] | None = None,
    ) -> dict[str, int]:
        """Combine signals from multiple strategies via weighted voting.

        Args:
            strategy_signals: Mapping of strategy_name -> {ticker: signal}.
            weights: Optional mapping of strategy_name -> weight for the
                weighted average.  Defaults to equal weighting.

        Returns:
            Consensus signal dict {ticker: int} with values in {-1, 0, 1}.

        Raises:
            ValueError: If *strategy_signals* is empty.
        """
        if not strategy_signals:
            raise ValueError("No strategy signals to aggregate")

        strategy_names = list(strategy_signals.keys())

        if weights is None:
            w = {name: 1.0 / len(strategy_names) for name in strategy_names}
        else:
            total = sum(weights.get(n, 0.0) for n in strategy_names)
            if total <= 0:
                w = {name: 1.0 / len(strategy_names) for name in strategy_names}
            else:
                w = {n: weights.get(n, 0.0) / total for n in strategy_names}

        # Collect all tickers across strategies
        all_tickers: set[str] = set()
        for sigs in strategy_signals.values():
            all_tickers.update(sigs.keys())

        # Weighted vote per ticker
        consensus: dict[str, int] = {}
        for ticker in sorted(all_tickers):
            weighted_sum = 0.0
            for name in strategy_names:
                sig = strategy_signals[name].get(ticker, 0)
                weighted_sum += sig * w[name]

            # Round to nearest integer in {-1, 0, 1}
            consensus[ticker] = int(np.clip(np.round(weighted_sum), -1, 1))

        logger.info(
            "Aggregated %d strategies across %d tickers",
            len(strategy_names),
            len(consensus),
        )
        return consensus

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _equal_weight_filtered(
        self,
        signals: dict[str, int],
    ) -> pd.Series:
        """Equal weight among long signals, zero for short/flat.

        The simplest approach -- any ticker with signal=1 gets an equal
        share; everything else is excluded.
        """
        long_tickers = [t for t, s in signals.items() if s == 1]

        if not long_tickers:
            logger.warning("No long signals -- returning empty weights")
            return pd.Series(
                {t: 0.0 for t in signals}, name="weights", dtype=float,
            )

        weight = 1.0 / len(long_tickers)
        weights = {t: weight if t in long_tickers else 0.0 for t in signals}
        result = pd.Series(weights, name="weights", dtype=float)

        logger.info(
            "Equal-weight-filtered: %d long out of %d tickers",
            len(long_tickers),
            len(signals),
        )
        return result

    def _signal_weighted(
        self,
        signals: dict[str, int],
        returns: pd.DataFrame,
    ) -> pd.Series:
        """Use signals as expected-return input to the portfolio optimiser.

        Signals are scaled by :data:`_SIGNAL_SCALE` to produce small
        expected-return estimates.  Tickers with signal=0 are excluded
        from optimisation.
        """
        # Filter to non-zero signals with available return data
        active = {
            t: s for t, s in signals.items()
            if s != 0 and t in returns.columns
        }

        if not active:
            logger.warning(
                "No active signals with return data -- falling back to equal weight",
            )
            return self._equal_weight_filtered(signals)

        active_tickers = sorted(active.keys())
        expected_returns = pd.Series(
            {t: active[t] * _SIGNAL_SCALE for t in active_tickers},
            name="expected_returns",
        )

        active_returns = returns[active_tickers]

        weights = self._optimiser.optimise(
            active_returns,
            method="mean_variance",
            expected_returns=expected_returns,
        )

        # Fill inactive tickers with zero
        full_weights = pd.Series(
            {t: weights.get(t, 0.0) for t in signals}, name="weights", dtype=float,
        )

        logger.info(
            "Signal-weighted: %d active tickers optimised", len(active),
        )
        return full_weights

    def _long_short(
        self,
        signals: dict[str, int],
    ) -> pd.Series:
        """Equal weight longs and shorts for a market-neutral portfolio.

        Long signals get positive weight, short signals get negative weight,
        flat signals get zero.  Weights are normalised so that the sum of
        absolute weights equals 1.0.
        """
        longs = [t for t, s in signals.items() if s == 1]
        shorts = [t for t, s in signals.items() if s == -1]

        if not longs and not shorts:
            logger.warning("No long or short signals -- returning empty weights")
            return pd.Series(
                {t: 0.0 for t in signals}, name="weights", dtype=float,
            )

        n_active = len(longs) + len(shorts)
        w = 1.0 / n_active

        weights: dict[str, float] = {}
        for t in signals:
            if t in longs:
                weights[t] = w
            elif t in shorts:
                weights[t] = -w
            else:
                weights[t] = 0.0

        result = pd.Series(weights, name="weights", dtype=float)

        logger.info(
            "Long-short: %d long, %d short, %d flat",
            len(longs),
            len(shorts),
            len(signals) - len(longs) - len(shorts),
        )
        return result
