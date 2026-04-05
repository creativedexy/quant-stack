"""Tests for src.portfolio.optimiser.

Covers:
- PortfolioOptimiser.optimise() for equal_weight (no riskfolio dependency)
- PortfolioOptimiser.rebalance() with and without max_turnover
- Standalone equal_weight() and inverse_volatility() functions
- Validation and error handling
- Riskfolio-backed methods (skipped if riskfolio not installed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.optimiser import (
    PortfolioOptimiser,
    equal_weight,
    inverse_volatility,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def returns_df() -> pd.DataFrame:
    """Simple 5-asset returns DataFrame for deterministic tests."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=500, freq="B")
    data = rng.normal(0.0003, 0.015, (500, 5))
    return pd.DataFrame(
        data,
        index=dates,
        columns=["A", "B", "C", "D", "E"],
    )


@pytest.fixture
def portfolio_config() -> dict:
    """Minimal portfolio config for tests."""
    return {
        "portfolio": {
            "default_method": "mean_variance",
            "constraints": {"max_weight": 0.20, "min_weight": 0.00},
            "risk_free_rate": 0.045,
            "covariance_method": "ledoit",
        },
    }


# -------------------------------------------------------------------
# Equal-weight via optimiser
# -------------------------------------------------------------------

class TestEqualWeightOptimise:

    def test_weights_sum_to_one(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="equal_weight")
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_weights_are_equal(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="equal_weight")
        expected = 1.0 / returns_df.shape[1]
        for w in weights:
            assert abs(w - expected) < 1e-10

    def test_index_matches_columns(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="equal_weight")
        assert list(weights.index) == list(returns_df.columns)

    def test_two_assets(self, portfolio_config: dict) -> None:
        dates = pd.bdate_range("2023-01-02", periods=100, freq="B")
        ret = pd.DataFrame(np.zeros((100, 2)), index=dates, columns=["X", "Y"])
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(ret, method="equal_weight")
        assert abs(weights["X"] - 0.5) < 1e-10
        assert abs(weights["Y"] - 0.5) < 1e-10


# -------------------------------------------------------------------
# Standalone functions
# -------------------------------------------------------------------

class TestEqualWeightStandalone:

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        weights = equal_weight(returns_df)
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_all_equal(self, returns_df: pd.DataFrame) -> None:
        weights = equal_weight(returns_df)
        expected = 1.0 / returns_df.shape[1]
        assert (np.abs(weights - expected) < 1e-10).all()


class TestInverseVolatility:

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        weights = inverse_volatility(returns_df)
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_all_positive(self, returns_df: pd.DataFrame) -> None:
        weights = inverse_volatility(returns_df)
        assert (weights > 0).all()

    def test_lower_vol_higher_weight(self) -> None:
        """Less volatile asset should get a higher weight."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2022-01-03", periods=500, freq="B")
        low_vol = rng.normal(0, 0.005, 500)
        high_vol = rng.normal(0, 0.03, 500)
        ret = pd.DataFrame({"LowVol": low_vol, "HighVol": high_vol}, index=dates)
        weights = inverse_volatility(ret)
        assert weights["LowVol"] > weights["HighVol"]

    def test_zero_volatility_raises(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        ret = pd.DataFrame(
            {"A": np.zeros(100), "B": np.ones(100) * 0.01},
            index=dates,
        )
        with pytest.raises(ValueError, match="zero volatility"):
            inverse_volatility(ret)

    def test_index_matches_columns(self, returns_df: pd.DataFrame) -> None:
        weights = inverse_volatility(returns_df)
        assert list(weights.index) == list(returns_df.columns)


# -------------------------------------------------------------------
# Rebalance
# -------------------------------------------------------------------

class TestRebalance:

    def test_no_turnover_limit(self, portfolio_config: dict) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        current = pd.Series([0.2, 0.3, 0.5], index=["A", "B", "C"])
        target = pd.Series([0.4, 0.4, 0.2], index=["A", "B", "C"])
        result = opt.rebalance(current, target)
        assert abs(result.sum() - 1.0) < 1e-10
        # Without constraint, should match target
        np.testing.assert_allclose(result.values, target.values, atol=1e-10)

    def test_turnover_constraint_applied(self, portfolio_config: dict) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.0, 1.0], index=["A", "B"])
        result = opt.rebalance(current, target, max_turnover=0.20)
        # Max turnover = 0.20, actual diff = 1.0, so scale = 0.20
        # Result (before normalisation): [0.5 - 0.5*0.2, 0.5 + 0.5*0.2] = [0.4, 0.6]
        assert abs(result.sum() - 1.0) < 1e-10
        # Actual turnover in result should be <= 0.20 + normalisation effects
        actual_change = (result - current).abs().sum()
        assert actual_change < 0.25  # Some tolerance for normalisation

    def test_turnover_below_limit(self, portfolio_config: dict) -> None:
        """When actual turnover is below limit, target is used directly."""
        opt = PortfolioOptimiser(config=portfolio_config)
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.45, 0.55], index=["A", "B"])
        result = opt.rebalance(current, target, max_turnover=0.50)
        np.testing.assert_allclose(result.values, target.values, atol=1e-10)

    def test_rebalance_sums_to_one(self, portfolio_config: dict) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        current = pd.Series([0.3, 0.3, 0.4], index=["A", "B", "C"])
        target = pd.Series([0.1, 0.5, 0.4], index=["A", "B", "C"])
        result = opt.rebalance(current, target, max_turnover=0.10)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_new_ticker_in_target(self, portfolio_config: dict) -> None:
        """Rebalancing should handle tickers present only in target."""
        opt = PortfolioOptimiser(config=portfolio_config)
        current = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.33, 0.33, 0.34], index=["A", "B", "C"])
        result = opt.rebalance(current, target)
        assert "C" in result.index
        assert abs(result.sum() - 1.0) < 1e-10


# -------------------------------------------------------------------
# Validation & error handling
# -------------------------------------------------------------------

class TestValidation:

    def test_unknown_method_raises(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        with pytest.raises(ValueError, match="Unknown optimisation method"):
            opt.optimise(returns_df, method="magic_method")

    def test_empty_returns_raises(self, portfolio_config: dict) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            opt.optimise(empty, method="equal_weight")

    def test_default_method_from_config(self, portfolio_config: dict) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        assert opt.default_method == "mean_variance"

    def test_constraint_values_from_config(self, portfolio_config: dict) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        assert opt.max_weight == 0.20
        assert opt.min_weight == 0.00
        assert opt.risk_free_rate == 0.045


# -------------------------------------------------------------------
# Riskfolio-backed methods (skipped if riskfolio not installed)
# -------------------------------------------------------------------

try:
    import riskfolio  # noqa: F401
    _HAS_RISKFOLIO = True
except ImportError:
    _HAS_RISKFOLIO = False


@pytest.mark.skipif(not _HAS_RISKFOLIO, reason="riskfolio-lib not installed")
class TestMeanVariance:

    def test_weights_sum_to_one(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="mean_variance")
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_max_weight_respected(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="mean_variance")
        assert (weights <= opt.max_weight + 1e-6).all()

    def test_no_negative_weights(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="mean_variance")
        assert (weights >= -1e-6).all()

    def test_with_alpha_signals(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        alpha = pd.Series(
            [0.10, 0.05, 0.08, 0.03, 0.12],
            index=returns_df.columns,
        )
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(
            returns_df, method="mean_variance", expected_returns=alpha,
        )
        assert abs(weights.sum() - 1.0) < 1e-6


@pytest.mark.skipif(not _HAS_RISKFOLIO, reason="riskfolio-lib not installed")
class TestMinCVaR:

    def test_weights_sum_to_one(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="min_cvar")
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_max_weight_respected(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="min_cvar")
        assert (weights <= opt.max_weight + 1e-6).all()


@pytest.mark.skipif(not _HAS_RISKFOLIO, reason="riskfolio-lib not installed")
class TestRiskParity:

    def test_weights_sum_to_one(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="risk_parity")
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_all_weights_positive(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="risk_parity")
        assert (weights > -1e-6).all()
