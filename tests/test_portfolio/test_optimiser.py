"""Tests for src.portfolio.optimiser."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.optimiser import PortfolioOptimiser


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
# Equal-weight tests
# -------------------------------------------------------------------

class TestEqualWeight:
    """Tests for the equal_weight method (no riskfolio dependency)."""

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame, portfolio_config: dict) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="equal_weight")
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_weights_are_equal(self, returns_df: pd.DataFrame, portfolio_config: dict) -> None:
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

    def test_two_assets_equal(self, portfolio_config: dict) -> None:
        dates = pd.bdate_range("2023-01-02", periods=100, freq="B")
        ret = pd.DataFrame(
            np.zeros((100, 2)),
            index=dates,
            columns=["X", "Y"],
        )
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(ret, method="equal_weight")
        assert abs(weights["X"] - 0.5) < 1e-10
        assert abs(weights["Y"] - 0.5) < 1e-10


# -------------------------------------------------------------------
# Validation & error handling
# -------------------------------------------------------------------

class TestValidation:
    """Tests for input validation and error handling."""

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

riskfolio_available = pytest.importorskip("riskfolio", reason="riskfolio-lib not installed")


class TestMeanVariance:
    """Tests for mean-variance optimisation via riskfolio-lib."""

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame, portfolio_config: dict) -> None:
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
        weights = opt.optimise(returns_df, method="mean_variance", expected_returns=alpha)
        assert abs(weights.sum() - 1.0) < 1e-6


class TestMinCVaR:
    """Tests for minimum CVaR optimisation."""

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame, portfolio_config: dict) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="min_cvar")
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_max_weight_respected(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="min_cvar")
        assert (weights <= opt.max_weight + 1e-6).all()


class TestRiskParity:
    """Tests for risk-parity optimisation."""

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame, portfolio_config: dict) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="risk_parity")
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_all_weights_positive(
        self, returns_df: pd.DataFrame, portfolio_config: dict
    ) -> None:
        opt = PortfolioOptimiser(config=portfolio_config)
        weights = opt.optimise(returns_df, method="risk_parity")
        assert (weights > -1e-6).all()
