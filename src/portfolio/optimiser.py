"""Portfolio optimisation wrapper around riskfolio-lib.

Provides a config-driven PortfolioOptimiser class that supports multiple
optimisation methods (mean-variance, minimum CVaR, risk parity, equal weight)
with position-size constraints.

Usage:
    from src.portfolio.optimiser import PortfolioOptimiser
    opt = PortfolioOptimiser(config=cfg)
    weights = opt.optimise(returns, method="mean_variance")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import riskfolio as rp

    _HAS_RISKFOLIO = True
except ImportError:  # pragma: no cover
    _HAS_RISKFOLIO = False

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_METHODS = frozenset({
    "mean_variance",
    "min_cvar",
    "risk_parity",
    "equal_weight",
})


class PortfolioOptimiser:
    """Config-driven portfolio optimiser backed by riskfolio-lib.

    Args:
        config: Full project configuration dict.  If ``None`` the default
            ``config/settings.yaml`` is loaded automatically.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()

        portfolio_cfg = config.get("portfolio", {})

        self.default_method: str = portfolio_cfg.get("default_method", "mean_variance")
        self.max_weight: float = float(portfolio_cfg.get("constraints", {}).get("max_weight", 0.20))
        self.min_weight: float = float(portfolio_cfg.get("constraints", {}).get("min_weight", 0.00))
        self.risk_free_rate: float = float(portfolio_cfg.get("risk_free_rate", 0.045))
        self.covariance_method: str = portfolio_cfg.get("covariance_method", "ledoit")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimise(
        self,
        returns: pd.DataFrame,
        method: str | None = None,
        expected_returns: pd.Series | None = None,
    ) -> pd.Series:
        """Compute target portfolio weights.

        Args:
            returns: Historical returns DataFrame with tickers as columns
                and a DatetimeIndex.
            method: Optimisation method.  One of ``mean_variance``,
                ``min_cvar``, ``risk_parity``, or ``equal_weight``.
                Falls back to the config default when ``None``.
            expected_returns: Optional alpha signals / expected-return
                overrides.  When supplied these replace the sample-mean
                estimates for mean-variance optimisation.

        Returns:
            Pandas Series mapping ticker → weight, summing to 1.0.

        Raises:
            ValueError: If *method* is not recognised or *returns* is empty.
        """
        method = method or self.default_method

        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown optimisation method '{method}'. "
                f"Supported: {sorted(_SUPPORTED_METHODS)}"
            )

        if returns.empty:
            raise ValueError("Returns DataFrame is empty.")

        logger.info(
            "Running portfolio optimisation",
            extra={"method": method, "assets": returns.shape[1]},
        )

        if method == "equal_weight":
            return self._equal_weight(returns)

        return self._riskfolio_optimise(returns, method, expected_returns)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _equal_weight(returns: pd.DataFrame) -> pd.Series:
        """Return equal weights for all tickers."""
        n = returns.shape[1]
        weights = pd.Series(1.0 / n, index=returns.columns, name="weights")
        return weights

    def _riskfolio_optimise(
        self,
        returns: pd.DataFrame,
        method: str,
        expected_returns: pd.Series | None,
    ) -> pd.Series:
        """Run riskfolio-lib optimisation with constraints."""
        if not _HAS_RISKFOLIO:  # pragma: no cover
            raise ImportError(
                "riskfolio-lib is required for optimisation methods other than "
                "equal_weight.  Install with: pip install 'quant-stack[portfolio]'"
            )

        port = rp.Portfolio(returns=returns)

        # Estimate statistics
        port.assets_stats(method_mu="hist", method_cov=self.covariance_method)

        # Override expected returns if alpha signals provided
        if expected_returns is not None:
            port.mu = expected_returns.reindex(returns.columns).values.reshape(1, -1)

        # Position-size constraints
        port.upperlng = self.max_weight
        if self.min_weight > 0:
            port.lowerlng = self.min_weight

        if method == "mean_variance":
            weights = self._mean_variance(port)
        elif method == "min_cvar":
            weights = self._min_cvar(port)
        elif method == "risk_parity":
            weights = self._risk_parity(port)
        else:  # pragma: no cover
            raise ValueError(f"Unhandled method: {method}")

        if weights is None:
            raise RuntimeError(
                f"Optimisation failed for method '{method}'. "
                "The problem may be infeasible with the current constraints."
            )

        return self._to_series(weights, returns.columns)

    def _mean_variance(self, port: Any) -> Any:
        """Mean-variance (minimum variance) optimisation."""
        return port.optimization(
            model="Classic",
            rm="MV",
            obj="MinRisk",
            rf=self.risk_free_rate,
            hist=True,
        )

    def _min_cvar(self, port: Any) -> Any:
        """Minimum Conditional Value-at-Risk optimisation."""
        return port.optimization(
            model="Classic",
            rm="CVaR",
            obj="MinRisk",
            rf=self.risk_free_rate,
            hist=True,
        )

    def _risk_parity(self, port: Any) -> Any:
        """Risk-parity (equal risk contribution) optimisation."""
        return port.rp_optimization(
            model="Classic",
            rm="MV",
            rf=self.risk_free_rate,
            hist=True,
        )

    @staticmethod
    def _to_series(weights_df: pd.DataFrame, columns: pd.Index) -> pd.Series:
        """Convert riskfolio weight DataFrame to a clean Series."""
        w = weights_df.squeeze()
        if isinstance(w, pd.DataFrame):
            w = w.iloc[:, 0]
        w.index = columns
        w.name = "weights"
        # Normalise to exactly 1.0 to avoid floating-point drift
        w = w / w.sum()
        return w
